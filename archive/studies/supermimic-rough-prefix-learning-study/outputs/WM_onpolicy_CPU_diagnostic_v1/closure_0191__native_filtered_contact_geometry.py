"""Kinematic overlap residuals from MuJoCo's original filtered contact list.

No geom-distance query, contact attraction, observed contact labels, dynamics,
forces or physical grasp qualification. An absent native contact has unknown
gap: its optimization residual is zero, while its reported distance is NaN.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import mujoco
import numpy as np

from retarget_palmroot_cpu import RobotKinematics

ALLOWANCE_M = .0005
NORMALIZATION_M = .001
DIAGNOSTIC_TIE_TOLERANCE_M = 1e-9
NORMAL_UNIT_TOLERANCE = 1e-6


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _readonly(value):
    value = np.asarray(value)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class ContactEvaluation:
    generated_mask: np.ndarray
    signed_distance_m: np.ndarray
    selected_distance_jacobian: np.ndarray
    contact_count: np.ndarray
    tie_count: np.ndarray
    second_contact_gap_m: np.ndarray
    selected_contact_index: np.ndarray
    selected_normal_a_to_b_w: np.ndarray
    selected_position_w: np.ndarray
    selected_native_geom_order: np.ndarray
    residual: np.ndarray
    residual_jacobian: np.ndarray
    active_mask: np.ndarray
    native_contact_records: tuple
    native_contact_count: int
    uncatalogued_native_contact_count: int


class NativeFilteredContactGeometry:
    """Fixed-catalog residual/Jacobian for twenty robot hinge coordinates.

    Call set_object_pose before evaluate. All distance/Jacobian diagnostics for
    an ungenerated pair are NaN, not measured zero clearance. Only the residual
    and its Jacobian are zero there. At a tie, the first deepest native contact
    is selected deterministically; diagnostics expose ties and do not certify
    differentiability. The original MJB numerical collision options are kept.
    """
    def __init__(self, model_path, context_path):
        self.robot = RobotKinematics(model_path)
        self.model, self.data = self.robot.model, self.robot.data
        m = self.model
        nominal_geometry = {name: getattr(m, name).copy() for name in ("geom_size", "geom_pos")}
        with np.load(context_path, allow_pickle=False) as context:
            for name in ("geom_size", "geom_pos"):
                value = context[name]
                if value.ndim != getattr(m, name).ndim + 1 or not len(value) or value.shape[1:] != getattr(m, name).shape:
                    raise ValueError("Expected fixed per-world geometry shape: " + name)
                if not np.isfinite(value).all() or not np.array_equal(value, np.broadcast_to(value[:1], value.shape)):
                    raise ValueError("Expected finite shared fixed geometry: " + name)
                getattr(m, name)[:] = value[0]
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        hand_candidates = [i for i, name in enumerate(self.names) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", name)]
        enabled = lambda i: bool(m.geom_contype[i] or m.geom_conaffinity[i])
        self.hand_ids = _readonly(np.array([i for i in hand_candidates if enabled(i)], dtype=int))
        self.disabled_hand_ids = _readonly(np.array([i for i in hand_candidates if not enabled(i)], dtype=int))
        object_candidates = [int(m.geom(name).id) for name in ("object/cube", "object/A_bottom", "object/A_top")]
        self.object_ids = _readonly(np.array([i for i in object_candidates if enabled(i)], dtype=int))
        if not len(self.hand_ids) or not len(self.object_ids):
            raise ValueError("No enabled named hand/object collision geometry")
        compatible = lambda a, b: bool((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a]))
        hand_self = [(int(a), int(b)) for j, a in enumerate(self.hand_ids) for b in self.hand_ids[j + 1:]]
        hand_object = [tuple(sorted((int(h), int(o)))) for h in self.hand_ids for o in self.object_ids if compatible(h, o)]
        self.pairs = _readonly(np.array(hand_self + hand_object, dtype=int).reshape(-1, 2))
        self.pair_kind = _readonly(np.array(["hand_self"] * len(hand_self) + ["hand_object"] * len(hand_object)))
        self._lookup = {tuple(pair): i for i, pair in enumerate(self.pairs.tolist())}
        if len(self._lookup) != len(self.pairs):
            raise ValueError("Duplicate fixed collision pair")
        joint = m.joint("object/cube_freejoint")
        if m.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected explicit fixed-pose freejoint object")
        self.object_qa = int(joint.qposadr[0])
        self._object_pose = None
        self.last_q = None
        self.cached = None
        self.collision_query_count = 0
        self.query_options = {"ccd_tolerance": float(m.opt.ccd_tolerance), "ccd_iterations": int(m.opt.ccd_iterations),
                              "disableflags": int(m.opt.disableflags), "enableflags": int(m.opt.enableflags),
                              "scope": "Original MJB collision numerical settings; kinematics/comPos/mj_collision only"}
        self.metadata = {
            "schema": "native_filtered_kinematic_contact_geometry:1", "model_sha256": _sha(model_path),
            "context_sha256": _sha(context_path), "module_sha256": _sha(__file__),
            "hand_geometry_enabled_count": len(self.hand_ids), "object_geometry_enabled_count": len(self.object_ids),
            "hand_self_potential_pairs": len(hand_self), "hand_object_mask_compatible_pairs": len(hand_object),
            "disabled_hand_names": [self.names[i] for i in self.disabled_hand_ids],
            "omitted_object_names_disabled": [self.names[i] for i in object_candidates if not enabled(i)],
            "omitted_hand_object_mask_incompatible_pairs": [[self.names[h], self.names[o]] for h in self.hand_ids for o in self.object_ids if not compatible(h, o)],
            "catalogue_same_body_pairs": int(sum(m.geom_bodyid[a] == m.geom_bodyid[b] for a, b in self.pairs)),
            "catalogue_same_weld_pairs": int(sum(m.body_weldid[m.geom_bodyid[a]] == m.body_weldid[m.geom_bodyid[b]] for a, b in self.pairs)),
            "native_parent_filter_enabled": not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)),
            "model_excluded_body_signatures": m.exclude_signature.tolist(), "explicit_model_pair_count": int(m.npair),
            "query_options": self.query_options, "allowance_m": ALLOWANCE_M, "normalization_m": NORMALIZATION_M,
            "normal_unit_tolerance": NORMAL_UNIT_TOLERANCE, "diagnostic_tie_tolerance_m": DIAGNOSTIC_TIE_TOLERANCE_M,
            "selection": "Exact argmin contactdist; first native contact wins exact ties. Diagnostic tie tolerance does not change selected value or objective.",
            "absent_contact": "generated=false; distance and selected distanceJac/normal/position=NaN; residual and residualJac=0. No gap or no-overlap certificate.",
            "distance_derivative": "n_geom0_to_geom1 dot (J_body1_at_contactpos - J_body0_at_contactpos), restricted to robot20 hinge DOFs; independent fixed object contributes zero robotJac.",
            "nominal_to_context_geometry_max_abs": {name: float(np.max(np.abs(getattr(m, name) - value))) for name, value in nominal_geometry.items()},
            "contact_attraction": False, "known_observed_contact_labels": False, "physical_reference_qualified": False,
            "scope": "Hand-object plus hand-self overlap residual only; filters, feature/tie nonsmoothness, object scale/palm mapping and dynamic/force feasibility remain unqualified."}

    def set_object_pose(self, pose):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1.) > 1e-6:
            raise ValueError("Expected finite SI/wxyz pose shape(7,) with unit quaternion")
        if self._object_pose is not None and np.array_equal(pose, self._object_pose):
            return
        self.data.qpos[self.object_qa:self.object_qa + 7] = pose
        self._object_pose = pose.copy()
        self.robot.last_q = None
        self.last_q = None
        self.cached = None

    def evaluate(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError("Expected finite joint vector shape(20,)")
        if self._object_pose is None:
            raise ValueError("Call set_object_pose before contact evaluation")
        m, d = self.model, self.data
        if d.time != 0 or np.any(d.qvel):
            raise ValueError("Geometry-only query requires clock0 and qvel0")
        if (float(m.opt.ccd_tolerance), int(m.opt.ccd_iterations), int(m.opt.disableflags), int(m.opt.enableflags)) != (self.query_options["ccd_tolerance"], self.query_options["ccd_iterations"], self.query_options["disableflags"], self.query_options["enableflags"]):
            raise ValueError("Frozen collision options were mutated")
        if self.last_q is not None and np.array_equal(q, self.last_q):
            return self.cached
        self.robot.forward(q)
        expected_qpos = d.qpos.copy()
        mujoco.mj_collision(m, d)
        self.collision_query_count += 1
        if not np.array_equal(expected_qpos, d.qpos) or d.time != 0 or np.any(d.qvel):
            raise ValueError("Collision query mutated physical state")
        count = len(self.pairs)
        distances = np.full(count, np.nan)
        derivative = np.full((count, 20), np.nan)
        generated = np.zeros(count, dtype=bool)
        contact_count = np.zeros(count, dtype=int)
        tie_count = np.zeros(count, dtype=int)
        second_gap = np.full(count, np.nan)
        selected_index = np.full(count, -1, dtype=int)
        normals = np.full((count, 3), np.nan)
        positions = np.full((count, 3), np.nan)
        orders = np.full((count, 2), -1, dtype=int)
        raw = []
        buckets = [[] for _ in range(count)]
        for index, contact in enumerate(d.contact):
            a, b = map(int, contact.geom)
            row = self._lookup.get(tuple(sorted((a, b))))
            if row is None:
                continue
            distance = float(contact.dist)
            normal, position = contact.frame[:3].copy(), contact.pos.copy()
            if not np.isfinite(distance) or not np.isfinite(normal).all() or not np.isfinite(position).all() or abs(np.linalg.norm(normal) - 1) > NORMAL_UNIT_TOLERANCE:
                raise ValueError("Invalid native geometric contact normal/distance/position")
            raw.append((row, index, a, b, distance, tuple(position), tuple(normal)))
            buckets[row].append((distance, index, a, b, position, normal))
        for row, contacts in enumerate(buckets):
            if not contacts:
                continue
            values = np.array([c[0] for c in contacts])
            selected = contacts[int(np.argmin(values))]
            distance, index, a, b, position, normal = selected
            generated[row] = True
            distances[row] = distance
            selected_index[row] = index
            contact_count[row] = len(contacts)
            tie_count[row] = int(np.count_nonzero(np.abs(values - distance) <= DIAGNOSTIC_TIE_TOLERANCE_M))
            if len(values) > 1:
                second_gap[row] = float(np.partition(values, 1)[1] - distance)
            jp_a, jr_a = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            jp_b, jr_b = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp_a, jr_a, position, int(m.geom_bodyid[a]))
            mujoco.mj_jac(m, d, jp_b, jr_b, position, int(m.geom_bodyid[b]))
            derivative[row] = normal @ (jp_b[:, self.robot.va] - jp_a[:, self.robot.va])
            normals[row] = normal if (a, b) == tuple(self.pairs[row]) else -normal
            positions[row], orders[row] = position, (a, b)
        residual = np.zeros(count)
        residual_jac = np.zeros((count, 20))
        active = generated & (distances < -ALLOWANCE_M)
        residual[active] = (-distances[active] - ALLOWANCE_M) / NORMALIZATION_M
        residual_jac[active] = -derivative[active] / NORMALIZATION_M
        if not np.isfinite(residual).all() or not np.isfinite(residual_jac).all():
            raise ValueError("Nonfinite native overlap residual/Jacobian")
        self.last_q = q.copy()
        self.cached = ContactEvaluation(*[_readonly(v) for v in (generated, distances, derivative, contact_count, tie_count, second_gap, selected_index, normals, positions, orders, residual, residual_jac, active)], tuple(raw), int(d.ncon), int(d.ncon) - len(raw))
        return self.cached

    def penetration_residual(self, q):
        value = self.evaluate(q)
        return value.residual, value.residual_jacobian

    def distance_and_jacobian(self, q):
        """Selected native contactdist only. Ungenerated rows are NaN."""
        value = self.evaluate(q)
        return value.signed_distance_m, value.selected_distance_jacobian


ContactGeometry = NativeFilteredContactGeometry
