"""CPU pair-distance screen for a geometric IK candidate, without dynamics.

This diagnoses geometry compatibility under the assumed cached SI scale and
fixed-wrist redesign. It does not infer known contacts, forces, or task success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import mujoco
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def screen(candidate, model_path, context_path, output):
    if output.exists():
        raise FileExistsError(output)
    with np.load(candidate, allow_pickle=False) as z:
        t = np.array(z["time_s"])
        q = np.array(z["joint_pos_rad"])
        pose = np.array(z["mapped_object_pose_w"])
    if q.shape != (len(t), 20) or pose.shape != (len(t), 7):
        raise ValueError("Candidate dimensions")
    if not all(np.isfinite(a).all() for a in (t, q, pose)) or not np.all(np.diff(t) > 0):
        raise ValueError("Invalid finite/time contract")
    if np.max(np.abs(np.linalg.norm(pose[:, 3:], axis=1) - 1)) > 1e-6:
        raise ValueError("Expected normalized wxyz object poses")
    m = mujoco.MjModel.from_binary_path(str(model_path))
    # Use the actually frozen simulator geometry, not stale nominal defaults.
    with np.load(context_path, allow_pickle=False) as z:
        context = {k: np.array(z[k]) for k in ("geom_size", "geom_pos")}
    geometry_delta = {}
    for name, values in context.items():
        if values.shape[1:] != getattr(m, name).shape or not np.array_equal(values, np.broadcast_to(values[:1], values.shape)):
            raise ValueError("Expected identical fixed geometry across 128 worlds")
        geometry_delta[name] = float(np.max(np.abs(getattr(m, name) - values[0])))
        getattr(m, name)[:] = values[0]
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
    hand = [i for i, n in enumerate(names) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", n)]
    objects = [m.geom(n).id for n in ("object/cube", "object/A_bottom", "object/A_top")]
    if len(hand) != 26:
        raise ValueError("Hand collision geometry count changed")
    pairs = np.array([(h, o) for h in hand for o in objects], dtype=int)
    d = mujoco.MjData(m)
    qa = np.array([m.joint(f"robot/right_finger{f}_joint{j}").qposadr[0] for f in range(1, 6) for j in range(1, 5)])
    object_qa = int(m.joint("object/cube_freejoint").qposadr[0])
    distances, segments = [], []
    # Direct distance queries avoid broad-phase/rbound and constraint/solver
    # pipelines. Kinematics updates geom world transforms; time never advances.
    for joints, object_pose in zip(q, pose):
        d.qpos[qa] = joints
        d.qpos[object_qa:object_qa + 7] = object_pose
        expected_qpos, expected_qvel = d.qpos.copy(), d.qvel.copy()
        mujoco.mj_kinematics(m, d)
        rows, points = [], []
        for h, o in pairs:
            segment = np.full(6, np.nan)
            value = mujoco.mj_geomDistance(m, d, int(h), int(o), 1.0, segment)
            rows.append(value)
            points.append(segment.copy())
        if not np.array_equal(d.qpos, expected_qpos) or not np.array_equal(d.qvel, expected_qvel) or d.time != 0:
            raise RuntimeError("Geometry query mutated state or advanced time")
        distances.append(rows)
        segments.append(points)
    distances, segments = np.array(distances), np.array(segments)
    if not np.isfinite(distances).all() or np.any(distances >= 1):
        raise ValueError("Distance query unsupported or outside 1m query limit")
    minimum = distances.min(axis=1)
    # These are geometric candidates only. A nonpositive pair distance alone
    # says nothing about native contact filtering, friction, forces or dynamics.
    output.mkdir(parents=True)
    np.savez_compressed(output / "pair_distances.npz", time_s=t, geom_pairs=pairs,
                        pair_signed_distance_m=distances, pair_fromto_w_m=segments)
    report = {
        "schema": "geometric_contact_distance_screen:1", "status": "completed_geometry_only",
        "candidate_sha256": sha(candidate), "model_sha256": sha(model_path),
        "fixed_context_sha256": sha(context_path), "script_sha256": sha(__file__),
        "distances_sha256": sha(output / "pair_distances.npz"),
        "geometry_fields_applied": ["geom_size", "geom_pos"], "nominal_to_context_max_abs": geometry_delta,
        "frames": len(t), "hand_geometry_count": len(hand), "object_geometry_count": len(objects),
        "pair_count": len(pairs), "geom_pair_names": [[names[h], names[o]] for h, o in pairs],
        "nearest_pair_signed_distance_min_m": float(minimum.min()),
        "nearest_pair_signed_distance_max_m": float(minimum.max()),
        "nearest_pair_signed_distance_first_m": float(minimum[0]),
        "frames_any_nonpositive_pair_distance": int((minimum <= 0).sum()),
        "frames_nearest_pair_gap_over_3mm": int((minimum > .003).sum()),
        "frames_pair_depth_over_3mm": int((minimum < -.003).sum()),
        "negative_distance_depth_max_m": float(np.maximum(-distances, 0).max()),
        "dynamics_steps": 0, "physical_reference_qualified": False,
        "known_contact_labels_inferred": False, "phase_or_PD_targets_inferred": False,
        "scope": "Native CPU convex pair-distance diagnostic under assumed cached SI and fixed-wrist mapping. Pair component depths are not a union SDF or Warp rollout/contact-force certificate. 3mm counters are descriptive screens, not task acceptance.",
        "calls": ["mj_kinematics", "mj_geomDistance"],
        "state_integrity": "All151 qpos/qvel unchanged by queries; clock0, no forward, collision pipeline, solver or step",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "frames", "pair_count", "nearest_pair_signed_distance_min_m", "nearest_pair_signed_distance_max_m", "frames_any_nonpositive_pair_distance", "frames_nearest_pair_gap_over_3mm", "frames_pair_depth_over_3mm")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    screen(a.candidate.resolve(strict=True), a.model.resolve(strict=True), a.context.resolve(strict=True), a.output.resolve())
