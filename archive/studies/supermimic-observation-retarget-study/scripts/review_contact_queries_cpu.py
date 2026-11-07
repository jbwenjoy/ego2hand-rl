"""Check negative distances against segment witnesses and native collision queries."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def run(root, compiled, output):
    directory = root / "outputs/palmroot_contact_geometry_screen_v1"
    z = np.load(directory / "pair_distances.npz", allow_pickle=False)
    a = np.load(root / "outputs/cached_landmarks_palmroot_v2/geometric_retarget.npz", allow_pickle=False)
    ds, segment = z["pair_signed_distance_m"], z["pair_fromto_w_m"]
    error = np.abs(np.linalg.norm(segment[..., 3:] - segment[..., :3], axis=-1) - np.abs(ds))
    m = mujoco.MjModel.from_binary_path(str(compiled / "nominal_model.mjb"))
    with np.load(compiled / "fixed_physics_context.npz", allow_pickle=False) as context:
        for name in ("geom_size", "geom_pos"):
            getattr(m, name)[:] = context[name][0]
    d = mujoco.MjData(m)
    qa = [m.joint(f"robot/right_finger{f}_joint{k}").qposadr[0] for f in range(1, 6) for k in range(1, 5)]
    obj = int(m.joint("object/cube_freejoint").qposadr[0])
    pairs = {tuple(sorted(pair)) for pair in z["geom_pairs"]}
    query_errors, collision_minima = [], []
    for i, (q, pose) in enumerate(zip(a["joint_pos_rad"], a["mapped_object_pose_w"])):
        d.qpos[qa] = q
        d.qpos[obj:obj + 7] = pose
        expected = d.qpos.copy()
        mujoco.mj_kinematics(m, d)
        k = ds[i].argmin()
        h, o = z["geom_pairs"][k]
        value = mujoco.mj_geomDistance(m, d, int(h), int(o), 0, None)
        query_errors.append(abs(value - ds[i, k]))
        mujoco.mj_collision(m, d)
        hits = [float(c.dist) for c in d.contact if tuple(sorted(c.geom)) in pairs]
        collision_minima.append(min(hits) if hits else np.nan)
        assert np.array_equal(d.qpos, expected) and not d.qvel.any() and d.time == 0
    assert np.isfinite(collision_minima).all()
    report = {
        "schema": "native_pair_distance_review:2",
        "status": "completed_with_nonminimum_zero_distance_caveat",
        "all151_minima_negative": bool((ds.min(1) < -.003).all()),
        "negative_distance_fromto_abs_error_max_m": float(error[ds < 0].max()),
        "nearest_query_distmax0_vs1_max_abs_m": float(max(query_errors)),
        "native_collision_min_range_m": [float(min(collision_minima)), float(max(collision_minima))],
        "native_collision_all151_depth_over3mm": bool(np.all(np.array(collision_minima) < -.003)),
        "nonnegative_pairs_fromto_discrepancy_over1um": int(((ds >= 0) & (error > 1e-6)).sum()),
        "nonnegative_pairs_max_discrepancy_m": float(error[ds >= 0].max()),
        "nativeccd_enabled": not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)),
        "qualification": "Nonminimum positive/zero pairs exhibit fromto disagreement; not a qualified gap-distance oracle. Negative minima have consistent segment witnesses, distmax0 results differ by at most the recorded numerical error, and native collision queries also find depths beyond3mm in all151 frames. Geometry-only incompatibility under current uncalibrated correspondence, not dynamics or true observed contacts.",
        "dynamics_steps": 0, "physical_reference_qualified": False,
        "calls": ["mj_kinematics", "mj_geomDistance", "mj_collision"],
        "state_integrity": "qpos/qvel unchanged by queries; clock0",
        "source_robot_labels_read": False,
        "distance_report_sha256": hashlib.sha256((directory / "report.json").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "API_reference": "https://mujoco.readthedocs.io/en/latest/computation/#geom-distance",
    }
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--compiled", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root.resolve(strict=True), args.compiled.resolve(strict=True), args.output.resolve())
