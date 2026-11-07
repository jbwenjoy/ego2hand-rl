"""Frozen geometric-intent shaping for an explicitly labeled A2 ablation.

This is a soft geometric prior/anticipation term. It is NOT a qualified
hand-object contact predictor, force estimator or verified dynamics gradient.
No MLP contact head, on-policy target construction or finger-identity penalty
is used. The caller adds dt * bonus only inside the intended imagination path.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .contact_geometry import ContactGeometryProxy, weighted_wrench_support


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _checked_file(path, expected, label):
    path = Path(path).resolve()
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"frozen {label} SHA256 mismatch or missing file: {path}")
    return path


def official_tolerance_support_match(current, target, *, tolerance=.1, var=.1):
    """Pure-torch FlashCHORD contact.py:120 reduction, no data-dependent checks.

    Shapes (...,hand,body,basis) -> (...). Inputs/constants are checked when
    constructing the frozen objective. There is no isfinite branch in forward.
    """
    commanded = target > 1e-3
    live = current > 1e-3
    low = ((1-tolerance)*target-current).clamp_min(0)
    high = (current-(1+tolerance)*target).clamp_min(0)
    score = torch.where(commanded & live, torch.exp(-(low.square()+high.square())/var), 0.)
    active_bodies = commanded.any(-1)
    body_score = score.sum(-1)/commanded.sum(-1).clamp_min(1)
    active_hands = active_bodies.any(-1)
    hand_score = body_score.sum(-1)/active_bodies.sum(-1).clamp_min(1)
    return hand_score.sum(-1)/active_hands.sum(-1).clamp_min(1)


class GeometricContactObjective(nn.Module):
    """Independent frozen bundle -> unscaled geometric prior on predicted x1.

    Instantiate on CPU, then migrate through Module.to or from_bundle(device=).
    Every target/phase/geometry constant is a non-optimizable registered buffer.
    The bundle is single-right-hand/single-pen. Phase indexing matches PenMDP.
    """
    def __init__(self, bundle_dir, *, reference_file=None, model_file=None,
                 expected_reference_sha256=None, bandwidth_m=.001, tolerance=.1, var=.1,
                 dtype=torch.float32):
        super().__init__()
        import mujoco

        if not math.isfinite(bandwidth_m) or bandwidth_m <= 0 or not math.isfinite(tolerance) or tolerance < 0 or not math.isfinite(var) or var <= 0:
            raise ValueError("finite positive bandwidth/variance and nonnegative tolerance required")
        bundle = Path(bundle_dir).resolve()
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("schema") != "supermimic_chord_contact_target_manifest:1" or manifest.get("source_kind") != "kinematic_reference_geometry":
            raise ValueError("expected an independent frozen kinematic-reference geometry target manifest")
        if manifest.get("target_changes_frozen_reference") is not False or manifest.get("target_defined_from_policy_contacts") is not False or manifest.get("no_real_contact_force_claim") is not True:
            raise ValueError("policy-defined targets, modified reference or physical-force claim are not allowed")
        if expected_reference_sha256 is not None and expected_reference_sha256 != manifest["reference_npz_sha256"]:
            raise ValueError("requested frozen reference hash disagrees with contact target manifest")
        target_path = _checked_file(bundle / "contact_targets.npz", manifest["target_npz_sha256"], "target")
        reference_path = _checked_file(reference_file or manifest["reference_npz_file"], manifest["reference_npz_sha256"], "reference")
        reference_json = _checked_file(reference_path.with_suffix(".json"), manifest["reference_json_sha256"], "reference metadata")
        model_path = _checked_file(model_file or manifest["model_file"], manifest["model_sha256"], "collision model")
        checked = {"manifest_sha256": _sha256(manifest_path), "target_npz_sha256": manifest["target_npz_sha256"],
                   "reference_npz_sha256": manifest["reference_npz_sha256"], "reference_json_sha256": manifest["reference_json_sha256"],
                   "model_sha256": manifest["model_sha256"]}
        for field, filename in (("rules", "rules.json"), ("raw_geometry", "raw_geometry.json"), ("time_windows", "time_windows.json")):
            _checked_file(bundle / filename, manifest[field+"_sha256"], field)
            checked[field+"_sha256"] = manifest[field+"_sha256"]
        reference_meta = json.loads(reference_json.read_text(encoding="utf-8-sig"))
        with np.load(target_path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        with np.load(reference_path, allow_pickle=False) as data:
            reference = {key: data[key].copy() for key in data.files}
        if not all(np.issubdtype(value.dtype, np.number) or np.issubdtype(value.dtype, np.bool_) for value in arrays.values()):
            raise ValueError("contact bundle must contain numeric non-pickled arrays")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("contact bundle contains nonfinite arrays")
        basis, support = arrays["support_basis"], arrays["support512"]
        frames = len(arrays["time_s"])
        if basis.shape != (512, 6) or support.shape != (frames, 1, 1, 512) or np.any(support < 0):
            raise ValueError("expected nonnegative (T,1,1,512) support and a (512,6) basis")
        if not np.allclose(np.linalg.norm(basis, axis=-1), 1, atol=1e-6):
            raise ValueError("contact wrench basis directions must have unit norm")
        for contact_key, reference_key in (("time_s", "time_s"), ("input_joint_pos_rad", "joint_pos_rad"),
                                            ("input_object_pose_w", "object_pose_w"), ("input_wrist_pose_w", "wrist_pose_w")):
            if arrays[contact_key].shape != reference[reference_key].shape or not np.allclose(arrays[contact_key], reference[reference_key], rtol=0, atol=1e-8):
                raise ValueError(f"contact target input does not match frozen reference field {reference_key}")
        known, absent, phase = arrays["contact_known"], arrays["known_no_contact"], arrays["phase"]
        if known.shape != (frames,) or absent.shape != (frames,) or phase.shape != (frames,) or not np.isin(phase, [0, 1, 2, 3, 4]).all():
            raise ValueError("invalid known/no-contact/phase frame layout")
        if not np.isin(known, [0, 1]).all() or not np.isin(absent, [0, 1]).all():
            raise ValueError("known and known_no_contact masks must be binary")
        model = mujoco.MjModel.from_binary_path(str(model_path))
        joint_names = reference_meta["joint_names"]
        if len(joint_names) != 20 or len(set(joint_names)) != 20:
            raise ValueError("expected the fixed 20-joint right-hand mapping")
        radius = float(reference_meta["object_radius_m"])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("invalid COM support radius")
        object_body = model.body("object/cube").id
        com = model.body_ipos[object_body]
        object_geoms = np.flatnonzero(model.geom_bodyid == object_body)
        computed_radius = max(float(np.linalg.norm(model.geom_pos[geom]-com)+model.geom_rbound[geom]) for geom in object_geoms)
        if not np.isclose(radius, computed_radius, rtol=0, atol=1e-9) or not np.allclose(arrays["object_com_offset_o_m"], com, rtol=0, atol=1e-9):
            raise ValueError("reference COM/radius does not agree with the hashed collision model")
        subset = np.arange(0, 512, 8, dtype=np.int64)
        self.geometry_proxy = ContactGeometryProxy(model, joint_names, basis[subset], radius,
                                                   points_per_geom=64, dtype=dtype)
        self.register_buffer("basis512", torch.tensor(basis, dtype=dtype))
        self.register_buffer("basis_subset_indices", torch.tensor(subset, dtype=torch.long))
        self.register_buffer("reference_support512", torch.tensor(support, dtype=dtype))
        self.register_buffer("reference_support64", torch.tensor(support[..., subset], dtype=dtype))
        self.register_buffer("contact_known", torch.tensor(known, dtype=torch.bool))
        self.register_buffer("known_no_contact", torch.tensor(absent, dtype=torch.bool))
        self.register_buffer("phase", torch.tensor(phase, dtype=torch.long))
        phase_weight = np.where(np.isin(phase, [2, 3]), 1., np.where(phase == 4, .5, 0.))
        phase_weight *= known.astype(bool) & ~absent.astype(bool)
        self.register_buffer("phase_weight", torch.tensor(phase_weight, dtype=dtype))
        self.bandwidth_m, self.tolerance, self.var = float(bandwidth_m), float(tolerance), float(var)
        self.frame_count = frames
        checked.update(basis512_sha256=hashlib.sha256(np.ascontiguousarray(basis).tobytes()).hexdigest(),
                       training_basis64_sha256=hashlib.sha256(np.ascontiguousarray(basis[subset]).tobytes()).hexdigest())
        self.metadata = {"schema": "supermimic_chord_geometric_intent_shaping:1",
                         "scientific_role": "soft geometric prior / anticipation shaping ablation only",
                         "qualified_contact_predictor": False, "physical_force_prediction": False,
                         "dynamics_gradient_qualified": False, "MLP_contact_head_used": False,
                         "independent_target_source_kind": manifest["source_kind"],
                         "target_defined_from_policy_contacts": False, "target_frozen": True,
                         **checked, "bundle_directory": str(bundle), "reference_file": str(reference_path),
                         "model_file": str(model_path), "target_file": str(target_path),
                         "joint_names": joint_names, "points_per_geom": 64,
                         "sampled_hand_surface_points": self.geometry_proxy.metadata["sample_count"],
                         "geometry": self.geometry_proxy.metadata,
                         "basis": {"training_directions": 64, "evaluation_directions": 512,
                                   "training_indices": subset.tolist(), "subset_rule": "fixed stride 8, indices 0:512:8",
                                   "full_evaluation_required": "physical evaluation reports the independent full 512-direction target; training64 is not evaluation512"},
                         "scoring": {"kind": "official tolerance-band object-centric union-support score",
                                     "tolerance": self.tolerance, "variance": self.var, "active_threshold": 1e-3,
                                     "support_coefficient": 1., "bandwidth_m": self.bandwidth_m,
                                     "no_finger_identity_or_extra_contact_penalty": True},
                         "phase_weights": {"0_unknown": 0., "1_known_no_contact": 0., "2_contact_slow": 1.,
                                           "3_contact_moving": 1., "4_switch": .5},
                         "known_mask_override": "support attraction requires contact_known AND NOT known_no_contact; deep-penetration prior remains active in all phases",
                         "phase_counts": {str(int(key)): int(value) for key, value in zip(*np.unique(phase, return_counts=True))},
                         "penetration_prior": {"allowance_m": .003, "normalization_m": .001, "coefficient": .1,
                                               "formula": "square(max(relu(-sample_sdf-.003))/.001)",
                                               "phase_masked": False, "meaning": "always-active sampled geometric prior, not full penetration measurement"},
                         "formula": "phase_weight * support_match - .1*penetration_prior",
                         "dt_applied_inside_module": False, "caller_dt_s": .02,
                         "frame_index": "(start + clamp_min(time,0)).minimum(end-1).long().squeeze(-1), same as PenMDP",
                         "known_limitations": ["distance-band proximity can anticipate or hallucinate solver-active contacts",
                                               "local proxy qualification failed; this ablation does not erase that result",
                                               "correct geometric derivatives do not prove useful or correct WM/action dynamics gradients",
                                               "sampling can miss contact or penetration extrema; true physical evaluation remains independent",
                                               "a missing reference thumb contact never forbids use of the thumb for stable support"]}

    @classmethod
    def from_bundle(cls, bundle_dir, *, reference_file=None, model_file=None, expected_reference_sha256=None,
                    bandwidth_m=.001, tolerance=.1, var=.1, device="cpu", dtype=torch.float32):
        return cls(bundle_dir, reference_file=reference_file, model_file=model_file,
                   expected_reference_sha256=expected_reference_sha256, bandwidth_m=bandwidth_m,
                   tolerance=tolerance, var=var, dtype=dtype).to(device=device, dtype=dtype)

    def forward(self, x1, memory):
        frame = (memory["start"]+memory["time"].clamp_min(0)).minimum(memory["end"]-1).long().squeeze(-1)
        geometry = self.geometry_proxy(x1, self.bandwidth_m, return_geometry=True)
        target = self.reference_support64[frame]
        support_match = official_tolerance_support_match(geometry["support"][..., None, None, :], target,
                                                        tolerance=self.tolerance, var=self.var)
        excess = (-geometry["sampled_signed_distance_m"]-.003).clamp_min(0).amax(-1)/.001
        penetration = excess.square()
        weight = self.phase_weight[frame]
        bonus = weight*support_match-.1*penetration
        terms = {"geometry_intent_support_match": support_match, "geometry_intent_penetration_prior": penetration,
                 "geometry_intent_phase_weight": weight, "geometry_intent_bonus_unscaled": bonus,
                 "geometry_intent_min_sdf_m": geometry["minimum_signed_distance_m"],
                 "geometry_intent_reference_frame": frame.to(x1.dtype), "geometry_intent_phase": self.phase[frame].to(x1.dtype)}
        return bonus, terms

    def full_basis_support_for_evaluation(self, x1):
        """Optional geometric diagnostic at all 512 directions, never task success."""
        with torch.autocast(device_type=x1.device.type, enabled=False):
            geometry = self.geometry_proxy(x1, self.bandwidth_m, return_geometry=True)
            return weighted_wrench_support(geometry["projected_contact_com_o_m"], geometry["inward_normal_o"],
                                           geometry["soft_contact_weight"], self.basis512, self.geometry_proxy.radius)


ContactObjective = GeometricContactObjective
