"""Bounded landmarks-only IK with fixed-pose penetration regularization.

No contact attraction, dynamics, known contacts, phase or executed targets are
inferred. The object remains at its uncalibrated observation-derived mapping.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares

from contact_penetration_geometry import ContactGeometry, ALLOWANCE_M, NORMALIZATION_M
from retarget_palmroot_cpu import ALLOWED, BONES, hand_frames, directions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def forbid_dynamics():
    saved = {name: getattr(mujoco, name) for name in ("mj_step", "mj_forward")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Geometry-only refinement must not invoke dynamics")

    try:
        for name in saved:
            setattr(mujoco, name, forbidden)
        yield
    finally:
        for name, value in saved.items():
            setattr(mujoco, name, value)


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = {"schema": "fixed_object_penetration_IK:1", "max_nfev_per_frame": 80,
              "bone_direction_weight": 1., "penetration_weight": 1., "temporal_weight": .03,
              "nominal_weight": .001, "penetration_allowance_m": ALLOWANCE_M,
              "penetration_normalization_m": NORMALIZATION_M, "ftol_xtol_gtol": 1e-8,
              "soft_wall_seconds": 120, "initial_guess": "baseline geometric q atframe0, previousrefinedq thereafter",
              "object_pose_optimized": False, "contact_attraction": False,
              "known_contact_labels_inferred": False, "physical_reference_qualified": False,
              "scope": "Exploratory geometry-only regularization after observed5-15mm pairpenetration; fixedweights, no dose/hyperparameter search or physicalacceptance claims"}
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    manifest = json.loads((args.observations / "manifest.json").read_text(encoding="utf-8-sig"))
    source = args.observations / "observations.npz"
    if manifest.get("is_real_ego") is not False or sha(source) != manifest["observation_sha256"]:
        raise ValueError("Expected cached controlled observation fixture")
    with np.load(source, allow_pickle=False) as z:
        if set(z.files) != ALLOWED:
            raise ValueError("Input has forbidden/unknown fields")
        t, human, object_source = (z[k].copy() for k in ("time_s", "hand_landmarks_source_w", "object_pose_source_w"))
    with np.load(args.baseline, allow_pickle=False) as z:
        baseline_q = z["joint_pos_rad"].copy()
        baseline_object = z["mapped_object_pose_w"].copy()
        baseline_R, baseline_t = z["source_to_fixed_wrist_R"].copy(), z["source_to_fixed_wrist_translation_m"].copy()
        old_targets = z["morphology_normalized_target_points_w"].copy()
        if not np.array_equal(t, z["time_s"]):
            raise ValueError("Baseline/observation clock mismatch")
    if t.shape != (151,) or human.shape != (151, 21, 3) or baseline_q.shape != (151, 20):
        raise ValueError("This bounded recipe is only the fixed151frame controlled fixture")
    if not all(np.isfinite(a).all() for a in (t, human, object_source, baseline_q, baseline_object)):
        raise ValueError("Nonfinite input")
    g = ContactGeometry(args.model, args.context)
    robot = g.robot
    rotations = np.einsum("ij,tkj->tik", robot.palm_frame, hand_frames(human))
    translations = robot.neutral[0] - np.einsum("tij,tj->ti", rotations, human[:, 0])
    units = np.einsum("tij,tkj->tki", rotations, directions(human))
    if not np.allclose(rotations, baseline_R, rtol=0, atol=1e-12) or not np.allclose(translations, baseline_t, rtol=0, atol=1e-12):
        raise ValueError("Baseline palm mapping mismatch")
    from scipy.spatial.transform import Rotation
    source_R = Rotation.from_quat(object_source[:, [4, 5, 6, 3]]).as_matrix()
    mapped_R = Rotation.from_quat(baseline_object[:, [4, 5, 6, 3]]).as_matrix()
    if not np.allclose(baseline_object[:, :3], np.einsum("tij,tj->ti", rotations, object_source[:, :3]) + translations, rtol=0, atol=1e-12) or not np.allclose(mapped_R, rotations @ source_R, rtol=0, atol=1e-12):
        raise ValueError("Baseline object mapping mismatch")
    qs, fks, targets, angles, ds, initial_ds, optimizers = [], [], [], [], [], [], []
    previous = baseline_q[0].copy()
    identity = np.eye(20)

    def budget():
        if time.monotonic() - started > recipe["soft_wall_seconds"]:
            raise TimeoutError("Bounded CPU geometry recipe exceeded120s; no retry")

    with forbid_dynamics():
        for frame in range(151):
            budget()
            g.set_object_pose(baseline_object[frame])
            initial_ds.append(g.distance_and_jacobian(baseline_q[frame])[0].copy())
            desired = units[frame]

            def residual(q):
                budget()
                bone, _ = robot.bone_features(q)
                penetration, _ = g.penetration_residual(q)
                return np.r_[(bone - desired).ravel(), penetration, .03 * (q - previous), .001 * (q - robot.nominal)]

            def jacobian(q):
                budget()
                return np.vstack((robot.bone_features(q)[1], g.penetration_residual(q)[1], .03 * identity, .001 * identity))

            result = least_squares(residual, np.clip(previous, robot.lower, robot.upper), jac=jacobian,
                                   bounds=(robot.lower, robot.upper), max_nfev=80,
                                   ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q = result.x.copy()
            points, _ = robot.forward(q)
            bone, _ = robot.bone_features(q)
            angle = np.rad2deg(np.arccos(np.clip(np.sum(bone * desired, axis=-1), -1, 1)))
            target = robot.neutral.copy()
            for finger in range(5):
                first = 1 + finger * 4
                for segment in range(3):
                    index = finger * 3 + segment
                    target[first + segment + 1] = target[first + segment] + robot.lengths[index] * desired[index]
            qs.append(q); fks.append(points.copy()); targets.append(target); angles.append(angle)
            ds.append(g.distance_and_jacobian(q)[0].copy())
            optimizers.append({"frame": frame, "success": bool(result.success), "status": int(result.status),
                               "nfev": int(result.nfev), "cost": float(result.cost)})
            previous = q
    qs, fks, targets, angles, ds, initial_ds = map(np.asarray, (qs, fks, targets, angles, ds, initial_ds))
    if not np.array_equal(targets, old_targets):
        raise AssertionError("Candidate target definition changed")
    output_npz = args.output / "geometric_retarget.npz"
    np.savez_compressed(output_npz, time_s=t, joint_pos_rad=qs, fk_hand_points_w=fks,
                        morphology_normalized_target_points_w=targets, bone_angle_error_deg=angles,
                        source_to_fixed_wrist_R=rotations, source_to_fixed_wrist_translation_m=translations,
                        mapped_object_pose_w=baseline_object, joint_lower_rad=robot.lower, joint_upper_rad=robot.upper,
                        geometry_pair_indices=g.pairs, geometry_pair_signed_distance_m=ds,
                        baseline_pair_signed_distance_m=initial_ds)
    tips = [4, 8, 12, 16, 20]
    report = {"schema": "fixed_object_penetration_IK_result:1", "status": "completed_geometry_diagnostic",
              "frames": 151, "optimizer_success_count": sum(x["success"] for x in optimizers), "optimizer_per_frame": optimizers,
              "bone_angle_mean_deg": float(angles.mean()), "bone_angle_max_deg": float(angles.max()),
              "normalized_target_tip_error_mean_m": float(np.linalg.norm(fks[:, tips] - targets[:, tips], axis=-1).mean()),
              "hard_joint_excess_max_rad": float(np.maximum(np.maximum(robot.lower - qs, qs - robot.upper), 0).max()),
              "q_ref_fd_rate_abs_max_rad_s": float(np.abs(np.diff(qs, axis=0) / np.diff(t)[:, None]).max()),
              "q_change_from_geometric_baseline_abs_max_rad": float(np.abs(qs - baseline_q).max()),
              "baseline_pair_depth_max_m": float(np.maximum(-initial_ds, 0).max()),
              "refined_pair_depth_max_m": float(np.maximum(-ds, 0).max()),
              "frames_depth_over3mm": int((ds.min(axis=1) < -.003).sum()),
              "frames_depth_over_point5mm": int((ds.min(axis=1) < -.0005).sum()),
              "target_points_exact_same_as_palmroot_v2": True, "mapped_object_pose_exact_unchanged": True,
              "query_options": g.query_options, "negative_witness_error_max_m": g.negative_witness_max_error_m,
              "input_npz_sha256": sha(source), "baseline_npz_sha256": sha(args.baseline), "model_sha256": sha(args.model),
              "context_sha256": sha(args.context), "script_sha256": sha(Path(__file__)),
              "geometry_module_sha256": sha(Path(__file__).with_name("contact_penetration_geometry.py")),
              "output_npz_sha256": sha(output_npz), "recipe_sha256": sha(args.output / "recipe.json"),
              "wall_seconds_including_validation": time.monotonic() - started,
              "dynamics_steps": 0, "physical_reference_qualified": False, "source_robot_labels_read": False,
              "known_contacts_or_forces_or_phase_inferred": False,
              "scope": "Geometry-only tradeoff on exactly the v2target definition, with objectpose heldfixed. Soft nonpenetration does not imply grasp, force support, dynamic feasibility or calibrated ego geometry. qFD is reference derivative, not measured/executed velocity. Cached camera/scale/anatomical/palm-map assumptions remain unqualified."}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "optimizer_success_count", "bone_angle_mean_deg", "normalized_target_tip_error_mean_m", "baseline_pair_depth_max_m", "refined_pair_depth_max_m", "frames_depth_over3mm", "frames_depth_over_point5mm", "wall_seconds_including_validation")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("observations", "baseline", "model", "context", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    run(parser.parse_args())
