"""Build fixed kinematic-reference geometry contact intentions on CPU.

No simulation rollout or policy contacts define these targets. Every frame is
independently positioned at the frozen robot/object reference and queried with
native MuJoCo collision geometry. Penetration-inconsistent frames stay unknown.
The resulting geometric support is not measured force, a teacher, or ego data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

import mujoco
import numpy as np
import torch

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))
from chord_study.contacts import sample_basis, wrench_support
from chord_study.geometry import quat_matrix, rotation_error
from chord_study.reference import Reference
from controlled_pen_rollout import sha256

RULES = {
    "schema": "supermimic_chord_reference_contact_rules:1",
    "source_kind": "kinematic_reference_geometry",
    "geometry_query": "independent native mj_forward at each frozen reference q/object pose, no integration",
    "contact_detection": "MuJoCo detected hand-object contacts with signed distance <=0; no invented proximity contacts",
    "contact_distance_threshold_m": 0.,
    "max_penetration_m": .003,
    "invalid_frame_rules": ["any hand-object, hand-self or other geometry penetration exceeds 3mm",
                            "joint position outside hard limits plus 1mrad tolerance", "object contacts a non-hand geom",
                            "nonfinite geometry or invalid active normal"],
    "joint_limit_margin_rad": .001,
    "known_no_contact": "valid geometry frame with no detected hand-object contact, not evidence of actual human release",
    "unknown": "geometry inconsistent under fixed thresholds; labels and reward must be masked, not treated as no contact",
    "confidence": "deterministic consistency under configured collision pairs only; not a calibrated probability or video observation confidence",
    "normal_convention": "force-on-object inward, world contact frame normal sign selected using modern contact.geom[2]",
    "support": {"basis_count": 512, "basis_seed": 0, "mu": .1, "cone_edges": 8,
                "include_normal_ray": True, "force_ray_norm": "unit", "moment_origin": "object_center_of_mass",
                "torque_normalization": "divide by bounding radius once", "aggregation": "nonnegative maximum across unit rays and contacts",
                "limitations": "assumed diagnostic friction, no force amplitudes or torsional contact wrench"},
    "phase_rules": {"0": "unknown geometry", "1": "known no contact", "2": "contact with shaft speed below 15deg/s",
                    "3": "contact with shaft speed >=15deg/s", "4": "contact signature changes between adjacent known frames; label both neighboring contact frames"},
    "phase_limitations": "fixed-reference geometric stages only; not observed grasp/release intent, unknown gaps are never bridged",
    "independence": "no A0/A1/A2 rollout contact or policy result is used to define targets",
}


def intervals(values, time_s):
    """Inclusive samples, exclusive frame stop; no interval bridge across labels."""
    values = np.asarray(values)
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return [{"label": int(values[b]), "frame_begin": int(b), "frame_stop_exclusive": int(e),
             "time_first_sample_s": float(time_s[b]), "time_last_sample_s": float(time_s[e - 1])}
            for b, e in zip(edges[:-1], edges[1:])]


def build_phases(known, finger_active, object_pose, dt):
    axes = quat_matrix(object_pose[:, 3:])[:, :, 0]
    angular = np.rad2deg(np.arccos(np.clip((axes[1:] * axes[:-1]).sum(-1), -1, 1))) / dt
    speed = np.r_[angular[0], angular]
    signature = (finger_active.astype(np.int64) * (1 << np.arange(6))).sum(-1)
    has_contact = finger_active.any(-1)
    phase = np.zeros(len(known), dtype=np.int32)
    phase[known & ~has_contact] = 1
    phase[known & has_contact & (speed < 15)] = 2
    phase[known & has_contact & (speed >= 15)] = 3
    switches = np.flatnonzero(known[1:] & known[:-1] & (signature[1:] != signature[:-1])) + 1
    for k in switches:
        for j in (k - 1, k):
            if known[j] and has_contact[j]:
                phase[j] = 4
    return phase, signature, speed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    reference_path, model_path = a.reference.resolve(strict=True), a.model.resolve(strict=True)
    reference_json = reference_path.with_suffix(".json")
    input_hashes = {"reference_npz_sha256": sha256(reference_path), "reference_json_sha256": sha256(reference_json),
                    "model_sha256": sha256(model_path)}
    ref = Reference.load(reference_path)
    ref.validate()
    arrays = ref.arrays
    time_s = np.asarray(arrays["time_s"], dtype=float)
    if not np.allclose(np.diff(time_s), .02, atol=1e-8):
        raise ValueError("this contact-target experiment requires the frozen 50Hz reference")
    model = mujoco.MjModel.from_binary_path(str(model_path))
    object_body = model.body("object/cube").id
    palm_body = model.body("robot/right_palm_link").id
    object_joint = model.joint("object/cube_freejoint").id
    oq = int(model.jnt_qposadr[object_joint])
    joint_ids = np.asarray([model.joint("robot/" + name).id for name in ref.meta["joint_names"]])
    qadr = model.jnt_qposadr[joint_ids]
    actuator_ids = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item() for jid in joint_ids])
    lower, upper = model.jnt_range[joint_ids].T
    object_geoms = set(np.flatnonzero(model.geom_bodyid == object_body).tolist())
    hand_geoms = {i for i in range(model.ngeom) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", model.geom(i).name)}
    if not object_geoms or not hand_geoms:
        raise ValueError("could not identify the object/hand geometry")
    hand_body_ids = sorted({int(model.geom_bodyid[g]) for g in hand_geoms})
    body_to_link = {body: i for i, body in enumerate(hand_body_ids)}
    link_names = [model.body(i).name for i in hand_body_ids]
    com_offset = model.body_ipos[object_body].copy()
    radius = max(float(np.linalg.norm(model.geom_pos[g] - com_offset) + model.geom_rbound[g]) for g in object_geoms)
    if not np.isclose(radius, ref.meta["object_radius_m"], atol=1e-6):
        raise ValueError("model and frozen reference object radius differ")
    out.mkdir(parents=True, exist_ok=True)
    rules = {**RULES, **input_hashes, "script_sha256": sha256(__file__)}
    (out / "rules.json").write_text(json.dumps(rules, indent=2), encoding="utf-8")
    started = time.perf_counter()
    frames = []
    known = np.ones(len(time_s), dtype=bool)
    for k in range(len(time_s)):
        d = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, d, model.key("init_state").id)
        else:
            mujoco.mj_resetData(model, d)
        d.qpos[qadr] = arrays["joint_pos_rad"][k]
        d.qpos[oq:oq + 7] = arrays["object_pose_w"][k]
        d.qvel[:] = 0
        d.ctrl[actuator_ids] = arrays["joint_pos_rad"][k]
        d.xfrc_applied[:] = 0
        d.qfrc_applied[:] = 0
        mujoco.mj_forward(model, d)
        if not np.allclose(d.xpos[palm_body], arrays["wrist_pose_w"][k, :3], atol=1e-6) or rotation_error(d.xquat[palm_body], arrays["wrist_pose_w"][k, 3:]) > 1e-5:
            raise ValueError("model wrist frame does not match the frozen reference")
        depth = {"hand_object": 0., "hand_self": 0., "other": 0.}
        contacts, nonhand_contacts, reasons = [], [], []
        rt = d.xmat[object_body].reshape(3, 3).T
        for j in range(d.ncon):
            c = d.contact[j]
            g1, g2 = (int(g) for g in c.geom)
            has_object = g1 in object_geoms or g2 in object_geoms
            hand_count = int(g1 in hand_geoms) + int(g2 in hand_geoms)
            category = "hand_object" if has_object and hand_count else "hand_self" if hand_count == 2 else "other"
            depth[category] = max(depth[category], max(0., -float(c.dist)))
            if not has_object:
                continue
            if category != "hand_object":
                if c.dist <= RULES["contact_distance_threshold_m"]:
                    nonhand_contacts.append([g1, g2])
                continue
            hand_geom = g1 if g1 in hand_geoms else g2
            sign = 1. if g2 in object_geoms else -1.
            normal_w = sign * np.asarray(c.frame).reshape(3, 3)[0]
            normal_o = rt @ normal_w
            pos_w = np.asarray(c.pos).copy()
            pos_o = rt @ (pos_w - d.xpos[object_body])
            body_id = int(model.geom_bodyid[hand_geom])
            finger_match = re.search(r"right_finger([1-5])_", model.geom(hand_geom).name)
            finger = int(finger_match.group(1)) if finger_match else 0
            item = {"point_w_m": pos_w.tolist(), "point_o_m": pos_o.tolist(), "point_com_o_m": (pos_o - com_offset).tolist(),
                    "normal_inward_w": normal_w.tolist(), "normal_inward_o": normal_o.tolist(),
                    "signed_distance_m": float(c.dist), "penetration_m": max(0., -float(c.dist)),
                    "geom_pair": [g1, g2], "hand_geom_id": hand_geom, "hand_body_id": body_id,
                    "hand_link_id": body_to_link[body_id], "finger_id": finger,
                    "geometry_contact": bool(c.dist <= RULES["contact_distance_threshold_m"])}
            if not np.isfinite(np.r_[pos_o, normal_o, c.dist]).all() or np.linalg.norm(normal_o) < 1e-3:
                reasons.append("invalid_contact_geometry_or_normal")
            contacts.append(item)
        for category, value in depth.items():
            if value > RULES["max_penetration_m"]:
                reasons.append(category + "_penetration_over_3mm")
        q = arrays["joint_pos_rad"][k]
        if not np.all((q >= lower - .001) & (q <= upper + .001)):
            reasons.append("joint_limits")
        if nonhand_contacts:
            reasons.append("object_nonhand_contact")
        known[k] = not reasons
        frames.append({"frame": k, "time_s": float(time_s[k]), "known": bool(known[k]),
                       "unknown_reasons": sorted(set(reasons)), "max_penetration_m": depth,
                       "contacts": contacts, "object_nonhand_geom_pairs": nonhand_contacts})
    max_slots = max((len(f["contacts"]) for f in frames), default=0)
    n = len(frames)
    def zeros(tail=(), dtype=float):
        return np.zeros((n, max_slots) + tuple(tail), dtype=dtype)
    bundle = {"time_s": time_s, "contact_pos_w_m": zeros((3,)), "contact_pos_o_m": zeros((3,)),
              "contact_pos_com_o_m": zeros((3,)), "contact_normal_inward_w": zeros((3,)), "contact_normal_o": zeros((3,)),
              "raw_contact_present": zeros(dtype=bool), "raw_geometry_contact": zeros(dtype=bool),
              "raw_signed_distance_m": zeros(), "raw_penetration_m": zeros(),
              "raw_geom_pair": np.full((n, max_slots, 2), -1, dtype=np.int32),
              "hand_link_id": np.full((n, max_slots), -1, dtype=np.int32), "finger_id": np.full((n, max_slots), -1, dtype=np.int32),
              "contact_known": known, "contact_active": zeros(dtype=bool), "contact_slot_known": zeros(dtype=bool),
              "input_joint_pos_rad": arrays["joint_pos_rad"], "input_object_pose_w": arrays["object_pose_w"],
              "input_wrist_pose_w": arrays["wrist_pose_w"], "object_com_offset_o_m": com_offset,
              "finger_known": np.repeat(known[:, None], 6, axis=1), "finger_active": np.zeros((n, 6), dtype=bool),
              "frame_max_penetration_m": np.asarray([[f["max_penetration_m"][c] for c in ("hand_object", "hand_self", "other")] for f in frames])}
    for k, frame in enumerate(frames):
        for j, c in enumerate(frame["contacts"]):
            for out_key, in_key in (("contact_pos_w_m", "point_w_m"), ("contact_pos_o_m", "point_o_m"),
                                    ("contact_pos_com_o_m", "point_com_o_m"), ("contact_normal_inward_w", "normal_inward_w"),
                                    ("contact_normal_o", "normal_inward_o"), ("raw_signed_distance_m", "signed_distance_m"),
                                    ("raw_penetration_m", "penetration_m"), ("raw_geom_pair", "geom_pair"),
                                    ("hand_link_id", "hand_link_id"), ("finger_id", "finger_id")):
                bundle[out_key][k, j] = c[in_key]
            bundle["raw_contact_present"][k, j] = True
            bundle["raw_geometry_contact"][k, j] = c["geometry_contact"]
            bundle["contact_slot_known"][k, j] = known[k]
            bundle["contact_active"][k, j] = known[k] and c["geometry_contact"]
            if known[k] and c["geometry_contact"]:
                bundle["finger_active"][k, c["finger_id"]] = True
    bundle["known_no_contact"] = known & ~bundle["contact_active"].any(-1)
    bundle["finger_known_no_contact"] = bundle["finger_known"] & ~bundle["finger_active"]
    bundle["geometry_consistency_margin_m"] = RULES["max_penetration_m"] - bundle["frame_max_penetration_m"].max(-1)
    phase, signatures, shaft_speed = build_phases(known, bundle["finger_active"], arrays["object_pose_w"], .02)
    bundle["phase"] = phase
    bundle["contact_finger_signature"] = signatures
    bundle["reference_shaft_speed_deg_s"] = shaft_speed
    basis = sample_basis(512, 0, dtype=torch.float64, device="cpu")
    computable = (np.isfinite(bundle["contact_pos_com_o_m"]).all(-1) & np.isfinite(bundle["contact_normal_o"]).all(-1)
                  & (np.linalg.norm(bundle["contact_normal_o"], axis=-1) > 1e-3))
    bundle["support_geometry_computable"] = computable
    # Preserve invalid raw geometry verbatim, but do not let it manufacture a
    # finite support target. Such frames are unknown and never rewarded.
    points = torch.from_numpy(np.nan_to_num(bundle["contact_pos_com_o_m"]))
    normals = torch.from_numpy(np.nan_to_num(bundle["contact_normal_o"]))
    raw_active = torch.from_numpy(bundle["raw_contact_present"] & bundle["raw_geometry_contact"] & computable)
    active = torch.from_numpy(bundle["contact_active"] & computable)
    bundle["support_basis"] = basis.numpy()
    bundle["raw_geometry_support512"] = wrench_support(points, normals, raw_active, basis, radius, mu=.1, edges=8).numpy()
    support = wrench_support(points, normals, active, basis, radius, mu=.1, edges=8).numpy()
    bundle["support512"] = support[:, None, None, :]
    bundle["finger_support512"] = np.stack([wrench_support(points, normals, active & torch.from_numpy(bundle["finger_id"] == f),
                                                           basis, radius, mu=.1, edges=8).numpy() for f in range(6)], axis=1)
    target_path = out / "contact_targets.npz"
    np.savez_compressed(target_path, **bundle)
    (out / "raw_geometry.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")
    (out / "time_windows.json").write_text(json.dumps({"phase": intervals(phase, time_s),
        "frame_contact_state": intervals(np.where(~known, 0, np.where(bundle["known_no_contact"], 1, 2)), time_s),
        "finger_state": {str(f): intervals(np.where(~known, 0, np.where(bundle["finger_active"][:, f], 2, 1)), time_s) for f in range(6)},
        "state_labels": {"0": "unknown", "1": "known_no_contact", "2": "known_contact"}}, indent=2), encoding="utf-8")
    from collections import Counter
    summary = {"schema": "supermimic_chord_reference_contacts:1", "source_kind": RULES["source_kind"],
               "frames": n, "known_frames": int(known.sum()), "known_coverage": float(known.mean()),
               "known_contact_frames": int((known & ~bundle["known_no_contact"]).sum()),
               "known_no_contact_frames": int(bundle["known_no_contact"].sum()), "unknown_frames": int((~known).sum()),
               "unknown_reason_frame_counts": dict(Counter(reason for f in frames for reason in f["unknown_reasons"])),
               "unknown_windows": [w for w in intervals(known.astype(int), time_s) if w["label"] == 0],
               "max_penetration_m": {c: float(bundle["frame_max_penetration_m"][:, i].max()) for i, c in enumerate(("hand_object", "hand_self", "other"))},
               "maximum_raw_contact_slots": max_slots, "hand_link_names": link_names,
               "hand_link_model_body_ids": hand_body_ids, "finger_id_names": ["palm", "thumb", "index", "middle", "ring", "little"],
               "finger_known_contact_frame_counts": bundle["finger_active"].sum(0).tolist(),
               "object_radius_m": radius, "object_com_offset_o_m": com_offset.tolist(),
               "normal_convention": RULES["normal_convention"], "support_shape": list(bundle["support512"].shape),
               "confidence_source": RULES["confidence"], "human_keypoints_available": "source_mediapipe_landmarks_w" in arrays,
               "human_keypoints_used_for_contact_labels": False,
               "human_keypoints_limitation": "no human mesh/contact sensor confidence; do not infer human forces or mesh contacts from landmarks alone",
               "reference_source_id": ref.meta["source_id"], "split_group": ref.meta["split_group"],
               **input_hashes, "target_npz_sha256": sha256(target_path), "rules_sha256": sha256(out / "rules.json"),
               "raw_geometry_sha256": sha256(out / "raw_geometry.json"), "script_sha256": sha256(__file__),
               "mujoco_version": mujoco.__version__, "wall_seconds": time.perf_counter() - started,
               "frozen_reference_modified": False, "policy_contacts_used": False,
               "scientific_boundary": "geometry intent only; unknown frames excluded from reward, known absence is not observed release; no contact-dynamics/teacher/ego success claim"}
    if sha256(reference_path) != input_hashes["reference_npz_sha256"] or sha256(reference_json) != input_hashes["reference_json_sha256"]:
        raise RuntimeError("frozen reference was modified during target construction")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "contact_targets.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
