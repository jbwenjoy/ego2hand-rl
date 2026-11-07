"""Prespecified CPU-only initial-grasp screening over all single-axis clips.

This is training-initialization selection, never a held-out performance test.
All candidate outcomes are retained. Each trial initializes once, holds the
hand at its initial q target for 0.3s, and lets the object move freely.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import time

import mujoco
import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_ROOT / "src"))
from chord_study.geometry import quat_matrix, rotation_error
from chord_study.reference import Reference
from controlled_pen_rollout import mul_quat, sha256, velocity_at_start, verify_contact_sign, write_bundle


PROTOCOL = {
    "schema": "supermimic_chord_initial_grasp_protocol:2",
    "scope": "training initialization selection only; not held-out, not a teacher, not an ego result",
    "source_group": "01_single_axis", "candidate_stride_frames": 25,
    "source_dt_s": .02, "continuation_frames_required": 151,
    "hold_duration_s": .3, "measurement_dt_s": .005,
    "measurement_revision": "v2 checks every 5ms physics substep; v1 20ms measurements retained with identical physical thresholds",
    "velocity_groups": ["zero", "source_fd"],
    "position_error_limit_m": .02, "directed_axis_error_limit_deg": 20.,
    "penetration_limits_m": {"hand_object": .003, "hand_self": .003, "other": .003},
    "joint_limit_margin_rad": .001,
    "drop_rules": ["object COM below initial COM by more than 0.02m", "any solved object contact with non-hand support"],
    "initial_contact_rule": "at least one solver-active hand-object contact; count is not a grasp stability certificate",
    "control": "fixed wrist, hand target held at initial reference q, zero residual and no assistance/state correction",
    "pass_rule": "all initial and all sampled hold limits satisfied; no instability, drop or non-hand support",
    "selection_order": ["prefer source_fd pass for compatibility with existing reset/evaluator",
                        "lowest maximum penetration over all categories",
                        "lowest maximum hold position error", "lowest maximum hold shaft-angle error",
                        "lexical clip ID", "lowest source start frame"],
    "data_split": "all source-release clips share unknown-video-parent split_group; no validation/test claim",
    "failure_handling": "continue bounded hold on tracking failure, stop only on nonfinite state; retain failures",
    "interpretation": "0.3s hold screening does not establish feasibility of the following 3s motion",
}


def transform_source(poses, wrists, fixed):
    rotations = quat_matrix(wrists[:, 3:])
    local_position = np.einsum("tji,tj->ti", rotations, poses[:, :3] - wrists[:, :3])
    result = np.empty_like(poses, dtype=float)
    result[:, :3] = local_position @ quat_matrix(fixed[3:]).T + fixed[:3]
    for i in range(len(poses)):
        inverse = wrists[i, 3:].copy()
        inverse[1:] *= -1
        result[i, 3:] = mul_quat(fixed[3:], mul_quat(inverse, poses[i, 3:]))
    return result


def contact_snapshot(model, data, object_geoms, hand_geoms):
    depth = {"hand_object": 0., "hand_self": 0., "other": 0.}
    object_contacts = []
    active_fingers = set()
    active_count = 0
    other_support_count = 0
    object_body = model.body("object/cube").id
    if not np.allclose(model.body_ipos[object_body], 0., atol=1e-9):
        raise ValueError("current drop protocol uses the symmetric pen body origin as COM; adapt explicitly for nonzero COM")
    rt = data.xmat[object_body].reshape(3, 3).T
    for i in range(data.ncon):
        c = data.contact[i]
        # MuJoCo Warp host transfer fills geom[2], not deprecated geom1/geom2.
        g1, g2 = (int(g) for g in c.geom)
        is_object = g1 in object_geoms or g2 in object_geoms
        hand_count = int(g1 in hand_geoms) + int(g2 in hand_geoms)
        category = "hand_object" if is_object and hand_count else "hand_self" if hand_count == 2 else "other"
        depth[category] = max(depth[category], max(0., -float(c.dist)))
        if not is_object:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        active = bool(c.efc_address >= 0 and force[0] > 1e-6)
        sign = 1. if g2 in object_geoms else -1.
        normal_w = sign * np.asarray(c.frame).reshape(3, 3)[0]
        other = g1 if g2 in object_geoms else g2
        other_name = model.geom(other).name
        finger = re.search(r"right_finger([1-5])_", other_name)
        finger_id = int(finger.group(1)) if finger else None
        if active and category == "hand_object":
            active_count += 1
            if finger_id is not None:
                active_fingers.add(finger_id)
        if active and category != "hand_object":
            other_support_count += 1
        object_contacts.append({"other_geom": other_name, "category": category, "finger": finger_id,
                                "point_world_m": np.asarray(c.pos).tolist(),
                                "point_object_m": (rt @ (c.pos - data.xpos[object_body])).tolist(),
                                "normal_force_on_object_world": normal_w.tolist(),
                                "normal_force_on_object_object_axes": (rt @ normal_w).tolist(),
                                "normal_force_N": float(force[0]), "active": active,
                                "penetration_m": max(0., -float(c.dist))})
    return {"penetration_m": depth, "active_hand_object_contacts": active_count,
            "active_fingers": sorted(active_fingers), "nonhand_object_support_contacts": other_support_count,
            "object_contacts": object_contacts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=STUDY_ROOT / "outputs/a0_clip001_3s_seed0/model.mjb")
    parser.add_argument("--output", type=Path, default=STUDY_ROOT / "outputs/initial_grasp_scan")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite screening artifacts: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Freeze criteria on disk before reading any candidate's physics outcome.
    protocol_path = output / "protocol.json"
    protocol_path.write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    start_wall = time.perf_counter()
    sign_check = verify_contact_sign()
    (output / "contact_sign_check.json").write_text(json.dumps(sign_check, indent=2), encoding="utf-8")
    model_path = args.model.resolve(strict=True)
    model = mujoco.MjModel.from_binary_path(str(model_path))
    import wuji_description
    from dex_supermimic.tasks.pen_spin.config.wuji_hand2.hand2_constants import RIGHT_FINGER_JOINT_NAMES, RIGHT_FIXED_WRIST_POSE_W
    fixed = np.asarray(RIGHT_FIXED_WRIST_POSE_W)
    joint_ids = np.asarray([model.joint("robot/" + name).id for name in RIGHT_FINGER_JOINT_NAMES])
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    actuator_ids = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item() for jid in joint_ids])
    obj_joint = model.joint("object/cube_freejoint").id
    oq, ov = int(model.jnt_qposadr[obj_joint]), int(model.jnt_dofadr[obj_joint])
    object_body = model.body("object/cube").id
    object_geoms = set(np.flatnonzero(model.geom_bodyid == object_body).tolist())
    hand_geoms = {g for g in range(model.ngeom) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", model.geom(g).name)}
    lower, upper = model.jnt_range[joint_ids].T
    dt = PROTOCOL["source_dt_s"]
    measurement_dt = PROTOCOL["measurement_dt_s"]
    substeps = int(round(measurement_dt / model.opt.timestep))
    hold_intervals = int(round(PROTOCOL["hold_duration_s"] / measurement_dt))
    if not np.isclose(substeps * model.opt.timestep, measurement_dt):
        raise ValueError("reference/control/physics clock mismatch")
    root = args.clips_root.resolve()
    index_path = root / PROTOCOL["source_group"] / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    if index["frame_dt"] != dt or len(index["clips"]) != 10:
        raise ValueError("expected the frozen 10-clip single-axis source catalog at 50 Hz")
    records, source_arrays = [], {}
    model_digest = sha256(model_path)
    for entry in index["clips"]:
        source_path = index_path.parent / entry["file"]
        if sha256(source_path) != entry["sha256"]:
            raise ValueError(f"catalog hash mismatch: {source_path}")
        with np.load(source_path, allow_pickle=False) as loaded:
            arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
        source_arrays[entry["clip_id"]] = arrays
        q = arrays["reference_qpos"].astype(float)
        wrist = arrays["wrist_pose_w"].astype(float)
        pose = arrays["object_pose_w"].astype(float)
        wrist[:, 3:] /= np.linalg.norm(wrist[:, 3:], axis=-1, keepdims=True)
        pose[:, 3:] /= np.linalg.norm(pose[:, 3:], axis=-1, keepdims=True)
        transformed = transform_source(pose, wrist, fixed)
        last_start = len(q) - PROTOCOL["continuation_frames_required"]
        for frame in range(0, last_start + 1, PROTOCOL["candidate_stride_frames"]):
            window = wrist[frame:frame + PROTOCOL["continuation_frames_required"]]
            source_wrist = {"position_change_max_m": float(np.linalg.norm(window[:, :3] - window[0, :3], axis=-1).max()),
                            "angle_change_max_deg": float(np.rad2deg(rotation_error(window[:, 3:], window[0, 3:])).max())}
            for mode in PROTOCOL["velocity_groups"]:
                data = mujoco.MjData(model)
                if model.nkey:
                    mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
                else:
                    mujoco.mj_resetData(model, data)
                q0, pose0 = q[frame].copy(), transformed[frame].copy()
                data.qpos[qadr] = q0
                data.qpos[oq:oq + 7] = pose0
                if mode == "source_fd":
                    data.qvel[vadr] = (q[frame + 1] - q0) / dt
                    data.qvel[ov:ov + 6] = velocity_at_start(transformed[frame:frame + 2], dt)
                else:
                    data.qvel[:] = 0
                data.ctrl[actuator_ids] = q0
                data.xfrc_applied[:] = 0
                data.qfrc_applied[:] = 0
                mujoco.mj_forward(model, data)
                initial = contact_snapshot(model, data, object_geoms, hand_geoms)
                initial_q_ok = bool(np.all((q0 >= lower - .001) & (q0 <= upper + .001)))
                max_depth = dict(initial["penetration_m"])
                hold_rows = []
                failure_reasons = set()
                if not initial_q_ok:
                    failure_reasons.add("initial_joint_limits")
                if initial["active_hand_object_contacts"] < 1:
                    failure_reasons.add("initial_no_active_hand_object_contact")
                nonfinite = False
                for tick in range(hold_intervals + 1):
                    if tick:
                        for _ in range(substeps):
                            mujoco.mj_step(model, data)
                        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                            failure_reasons.add("nonfinite_physics")
                            nonfinite = True
                            break
                        mujoco.mj_forward(model, data)
                    c = contact_snapshot(model, data, object_geoms, hand_geoms)
                    for category, value in c["penetration_m"].items():
                        max_depth[category] = max(max_depth[category], value)
                        if value > PROTOCOL["penetration_limits_m"][category]:
                            failure_reasons.add(category + "_penetration")
                    current = np.r_[data.xpos[object_body], data.xquat[object_body]]
                    position_error = float(np.linalg.norm(current[:3] - pose0[:3]))
                    cosine = np.clip(np.dot(quat_matrix(current[3:])[:, 0], quat_matrix(pose0[3:])[:, 0]), -1, 1)
                    angle = float(np.rad2deg(np.arccos(cosine)))
                    if position_error > PROTOCOL["position_error_limit_m"]:
                        failure_reasons.add("hold_position_error")
                    if angle > PROTOCOL["directed_axis_error_limit_deg"]:
                        failure_reasons.add("hold_axis_error")
                    if current[2] < pose0[2] - .02:
                        failure_reasons.add("drop_height")
                    if c["nonhand_object_support_contacts"]:
                        failure_reasons.add("object_nonhand_support")
                    q_ok = bool(np.all((data.qpos[qadr] >= lower - .001) & (data.qpos[qadr] <= upper + .001)))
                    if not q_ok:
                        failure_reasons.add("hold_joint_limits")
                    hold_rows.append({"time_s": tick * measurement_dt, "position_error_m": position_error,
                                      "axis_error_deg": angle, "penetration_m": c["penetration_m"],
                                      "active_hand_object_contacts": c["active_hand_object_contacts"],
                                      "active_fingers": c["active_fingers"],
                                      "nonhand_object_support_contacts": c["nonhand_object_support_contacts"],
                                      "object_pose_w": current.tolist(), "joint_limits_ok": q_ok})
                record = {"candidate_id": f"{entry['clip_id']}_f{frame:04d}_{mode}",
                          "source_clip_id": f"{index['group_id']}/{entry['clip_id']}",
                          "source_file": str(source_path), "source_sha256": entry["sha256"],
                          "source_start_frame": frame, "source_stop_exclusive": frame + 151,
                          "velocity_group": mode, "initial_joint_pos_rad": q0.tolist(),
                          "initial_object_pose_w": pose0.tolist(), "initial_joint_limits_ok": initial_q_ok,
                          "source_wrist_motion_in_3s": source_wrist, "fixed_wrist_task_redesign": True,
                          "initial_contacts": initial, "hold_samples": hold_rows,
                          "max_penetration_m": max_depth,
                          "max_position_error_m": max(x["position_error_m"] for x in hold_rows),
                          "max_axis_error_deg": max(x["axis_error_deg"] for x in hold_rows),
                          "hold_censored": nonfinite, "failure_reasons": sorted(failure_reasons),
                          "passed": not failure_reasons and len(hold_rows) == hold_intervals + 1}
                records.append(record)
        print(f"screened {entry['clip_id']}: {sum(r['source_clip_id'].endswith(entry['clip_id']) for r in records)} trials", flush=True)
    passing = [r for r in records if r["passed"]]
    passing.sort(key=lambda r: (r["velocity_group"] != "source_fd", max(r["max_penetration_m"].values()),
                                r["max_position_error_m"], r["max_axis_error_deg"], r["source_clip_id"], r["source_start_frame"]))
    selected = passing[0] if passing else None
    report = {"schema": "supermimic_chord_initial_grasp_scan:1", "protocol_sha256": sha256(protocol_path),
              "script_sha256": sha256(__file__), "model_file": str(model_path), "model_sha256": model_digest,
              "mujoco_version": mujoco.__version__, "wuji_resource_revision": getattr(wuji_description, "REVISION", None),
              "source_catalog": str(index_path), "source_catalog_sha256": sha256(index_path),
              "candidate_trial_count": len(records), "candidate_state_count": len(records) // 2,
              "passing_count": len(passing), "passing_by_velocity_group": dict(Counter(r["velocity_group"] for r in passing)),
              "failure_reason_counts": dict(Counter(reason for r in records for reason in r["failure_reasons"])),
              "selected_candidate_id": selected["candidate_id"] if selected else None,
              "records": records, "wall_seconds": time.perf_counter() - start_wall,
              "split_group": "wuji:v2026.10.22:unknown_original_video_parent",
              "warning": "All outcomes are training-initialization screening; following 3s motion and Warp transfer remain unverified."}
    (output / "scan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if selected:
        export_selected(output, selected, source_arrays, root, model, object_body, fixed, list(RIGHT_FINGER_JOINT_NAMES))
    summary = {key: report[key] for key in ("candidate_trial_count", "candidate_state_count", "passing_count", "passing_by_velocity_group", "selected_candidate_id", "wall_seconds")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def export_selected(output, selected, source_arrays, source_root, model, object_body, fixed, joint_names):
    """Derived single-clip catalog is a lineage-preserving kinematic crop."""
    local_id = selected["source_clip_id"].split("/")[1]
    original = source_arrays[local_id]
    start, stop = selected["source_start_frame"], selected["source_stop_exclusive"]
    crop = {key: value[start:stop].copy() for key, value in original.items()}
    catalog = output / "selected_catalog"
    group = catalog / "selected_initial_grasp"
    group.mkdir(parents=True, exist_ok=False)
    clip_id = selected["candidate_id"]
    clip_path = group / f"{clip_id}.npz"
    np.savez_compressed(clip_path, **crop)
    index = {"schema": "wuji_pen_spin_clips:1", "group_id": group.name,
             "group_name": "prespecified training initial-grasp selection, source lineage in provenance.json",
             "frame_dt": .02, "clip_count": 1,
             "clips": [{"clip_id": clip_id, "file": clip_path.name, "sha256": sha256(clip_path)}]}
    (group / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    recipe = {"schema": "wuji_pen_spin_recipe:1", "clips": [f"{group.name}/{clip_id}"]}
    (output / "selected_recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    wrists, poses = crop["wrist_pose_w"].astype(float), crop["object_pose_w"].astype(float)
    wrists[:, 3:] /= np.linalg.norm(wrists[:, 3:], axis=-1, keepdims=True)
    poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=-1, keepdims=True)
    target_object = transform_source(poses, wrists, fixed)
    com = model.body_ipos[object_body]
    object_geoms = np.flatnonzero(model.geom_bodyid == object_body)
    radius = max(float(np.linalg.norm(model.geom_pos[i] - com) + model.geom_rbound[i]) for i in object_geoms)
    ir = quat_matrix(model.body_iquat[object_body])
    inertia = ir @ np.diag(model.body_inertia[object_body]) @ ir.T
    lineage = {"source_file": str(clip_path), "source_sha256": sha256(clip_path),
               "source_trajectory_id": f"{group.name}/{clip_id}", "source_frames": [0, 151],
               "original_source_file": selected["source_file"], "original_source_sha256": selected["source_sha256"],
               "original_source_trajectory_id": selected["source_clip_id"], "original_source_frames": [start, stop],
               "original_video_parent_id": None, "original_source_catalog": str(source_root / "01_single_axis/index.json"),
               "transformation": "pure 151-frame source crop; then remove wrist motion and rebase object reference to fixed wrist",
               "selection": "all candidates recorded under prespecified CPU 0.3s hold screening; selected for training initialization only",
               "selected_candidate_id": selected["candidate_id"], "screening_velocity_group": selected["velocity_group"],
               "reset_compatibility": "current source finite-difference reset" if selected["velocity_group"] == "source_fd" else "requires explicit zero-velocity initialization; existing source_fd reset not validated",
               "claim": "upstream kinematic crop, not a sim teacher, no ego reconstruction, no 3s feasibility claim",
               "contact_targets": "unknown; selection contacts are measurements, not reference labels"}
    (output / "provenance.json").write_text(json.dumps(lineage, indent=2), encoding="utf-8")
    meta = {"schema": "supermimic_chord_reference:1", "source_id": f"selected:{clip_id}:frames0-151",
            "source_kind": "upstream_kinematic", "split_group": "wuji:v2026.10.22:unknown_original_video_parent",
            "joint_names": joint_names, "units": "SI", "quaternion_order": "wxyz", "world_frame": "fixed_hand2_mount_world",
            "contact_normal": "force_on_object_inward", "object_radius_m": radius, "object_mass_kg": float(model.body_mass[object_body]),
            "object_inertia_kg_m2": inertia.tolist(), "orientation_metric": "directed_axis_x", "object_symmetry": "none",
            "orientation_task_equivalence": "ignore axial roll per upstream task; not a physical geometry symmetry",
            "provenance": lineage}
    arrays = {"time_s": np.arange(151) * .02, "joint_pos_rad": crop["reference_qpos"].astype(float),
              "object_pose_w": target_object, "wrist_pose_w": np.tile(fixed, (151, 1)), "phase": np.zeros(151, dtype=np.int32),
              "source_wrist_pose_w": wrists, "source_object_pose_w": poses, "source_link_pos_w": crop["link_pos_w"]}
    if "mediapipe_landmarks_w" in crop:
        arrays["source_mediapipe_landmarks_w"] = crop["mediapipe_landmarks_w"]
    Reference(arrays, meta).validate()
    write_bundle(output / "selected_reference.npz", arrays, meta)


if __name__ == "__main__":
    main()
