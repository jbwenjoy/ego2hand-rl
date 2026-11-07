"""Fixed-object geometric IK with native filtered hand/object/self overlaps.

Geometry candidate only. Ungenerated native contacts retain unknown gaps;
there is no contact attraction, force estimate, physics step or PD target.
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
from scipy.spatial.transform import Rotation

from native_filtered_contact_geometry import NativeFilteredContactGeometry, ALLOWANCE_M, NORMALIZATION_M
from retarget_palmroot_cpu import ALLOWED, hand_frames, directions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def geometry_only():
    saved = {n: getattr(mujoco, n) for n in ("mj_forward", "mj_step", "mj_geomDistance")}

    def forbidden(*args, **kwargs):
        raise AssertionError("Only FK/Jacobians/native filtered collision queries are allowed")

    try:
        for n in saved:
            setattr(mujoco, n, forbidden)
        yield
    finally:
        for n, function in saved.items():
            setattr(mujoco, n, function)


def depth_summary(distances, generated):
    active_frames = generated.any(axis=1)
    deepest = np.full(len(distances), np.nan)
    for i in np.flatnonzero(active_frames):
        deepest[i] = np.maximum(-distances[i, generated[i]], 0).max()
    return {"max_generated_depth_m": float(deepest[active_frames].max()) if active_frames.any() else None,
            "frames_with_generated_contact": int(active_frames.sum()),
            "frames_without_generated_contact_unknown_gap": int((~active_frames).sum()),
            "frames_generated_depth_over3mm": int((deepest[active_frames] > .003).sum()),
            "frames_generated_depth_over_allowance": int((deepest[active_frames] > ALLOWANCE_M).sum())}


def run(a):
    started = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    recipe = {"schema": "native_filtered_object_self_IK:1", "baseline_kind": "object_only_soft_penetration_refinement",
              "bone_direction_weight": 1., "hand_object_overlap_weight": 1., "hand_self_overlap_weight": 1.,
              "temporal_weight": .03, "nominal_weight": .001, "allowance_m": ALLOWANCE_M,
              "normalization_m": NORMALIZATION_M, "max_nfev_per_frame": 80, "ftol_xtol_gtol": 1e-8,
              "soft_wall_seconds": 120, "initial_guess": "baseline geometricq at0, previous refinedq thereafter",
              "object_pose_fixed": True, "contact_attraction": False, "source_robot_labels_read": False,
              "known_contacts_or_forces_inferred": False, "physical_reference_qualified": False,
              "nonsmoothness": "Selected deepest native contact uses first exact argmin; local unique/stable derivative checks do not prove ties/hinges globally smooth"}
    (a.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    manifest = json.loads((a.observations / "manifest.json").read_text(encoding="utf-8-sig"))
    source = a.observations / "observations.npz"
    assert manifest.get("is_real_ego") is False and sha(source) == manifest["observation_sha256"]
    with np.load(source, allow_pickle=False) as z:
        assert set(z.files) == ALLOWED
        t, human, object_source = (np.array(z[k], dtype=np.float64) for k in ("time_s", "hand_landmarks_source_w", "object_pose_source_w"))
    with np.load(a.baseline, allow_pickle=False) as z:
        baseline_q, mapped_pose = z["joint_pos_rad"].copy(), z["mapped_object_pose_w"].copy()
        expected_R, expected_translation = z["source_to_fixed_wrist_R"].copy(), z["source_to_fixed_wrist_translation_m"].copy()
        expected_targets = z["morphology_normalized_target_points_w"].copy()
        assert np.array_equal(t, z["time_s"])
    assert t.shape == (151,) and human.shape == (151, 21, 3) and baseline_q.shape == (151, 20)
    assert all(np.isfinite(v).all() for v in (t, human, object_source, baseline_q, mapped_pose))
    g = NativeFilteredContactGeometry(a.model, a.context)
    robot = g.robot
    rotations = np.einsum("ij,tkj->tik", robot.palm_frame, hand_frames(human))
    translations = robot.neutral[0] - np.einsum("tij,tj->ti", rotations, human[:, 0])
    units = np.einsum("tij,tkj->tki", rotations, directions(human))
    assert np.allclose(rotations, expected_R, rtol=0, atol=1e-12)
    assert np.allclose(translations, expected_translation, rtol=0, atol=1e-12)
    assert np.allclose(mapped_pose[:, :3], np.einsum("tij,tj->ti", rotations, object_source[:, :3]) + translations, rtol=0, atol=1e-12)
    assert np.allclose(Rotation.from_quat(mapped_pose[:, [4, 5, 6, 3]]).as_matrix(), rotations @ Rotation.from_quat(object_source[:, [4, 5, 6, 3]]).as_matrix(), rtol=0, atol=1e-12)
    qs, fks, targets, angles, ds, generated, baseline_ds, baseline_generated, counts, ties, optimizer = ([] for _ in range(11))
    previous, identity = baseline_q[0].copy(), np.eye(20)

    def budget():
        if time.monotonic() - started > 120:
            raise TimeoutError("Finite120s CPU geometric recipe exceeded; no retry")

    with geometry_only():
        for frame in range(151):
            budget()
            g.set_object_pose(mapped_pose[frame])
            before = g.evaluate(baseline_q[frame])
            baseline_ds.append(before.signed_distance_m.copy()); baseline_generated.append(before.generated_mask.copy())
            desired = units[frame]

            def residual(q):
                budget()
                bone, _ = robot.bone_features(q)
                try:
                    penetration, _ = g.penetration_residual(q)
                except Exception as error:
                    np.savez_compressed(a.output / "failure_snapshot.npz", frame=frame, solver_q=q,
                                        object_pose=mapped_pose[frame], completed_q=np.asarray(qs).reshape(-1, 20))
                    (a.output / "failure.json").write_text(json.dumps({"error": repr(error), "frame": frame, "dynamics_steps": 0, "complete_candidate": False}) + "\n")
                    raise
                return np.r_[(bone - desired).ravel(), penetration, .03 * (q - previous), .001 * (q - robot.nominal)]

            def jacobian(q):
                budget()
                return np.vstack((robot.bone_features(q)[1], g.penetration_residual(q)[1], .03 * identity, .001 * identity))

            result = least_squares(residual, np.clip(previous, robot.lower, robot.upper), jac=jacobian,
                                   bounds=(robot.lower, robot.upper), max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)
            q = result.x.copy()
            points, _ = robot.forward(q)
            bone, _ = robot.bone_features(q)
            target = robot.neutral.copy()
            for finger in range(5):
                first = 1 + finger * 4
                for segment in range(3):
                    k = finger * 3 + segment
                    target[first + segment + 1] = target[first + segment] + robot.lengths[k] * desired[k]
            current = g.evaluate(q)
            qs.append(q); fks.append(points.copy()); targets.append(target)
            angles.append(np.rad2deg(np.arccos(np.clip(np.sum(bone * desired, axis=-1), -1, 1))))
            ds.append(current.signed_distance_m.copy()); generated.append(current.generated_mask.copy())
            counts.append(current.contact_count.copy()); ties.append(current.tie_count.copy())
            row = {"frame": frame, "success": bool(result.success), "status": int(result.status), "nfev": int(result.nfev), "cost": float(result.cost)}
            optimizer.append(row)
            with (a.output / "optimizer.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            previous = q
    qs, fks, targets, angles, ds, generated, baseline_ds, baseline_generated, counts, ties = map(np.asarray, (qs, fks, targets, angles, ds, generated, baseline_ds, baseline_generated, counts, ties))
    assert np.array_equal(targets, expected_targets)
    output = a.output / "geometric_retarget.npz"
    np.savez_compressed(output, time_s=t, joint_pos_rad=qs, fk_hand_points_w=fks,
                        morphology_normalized_target_points_w=targets, bone_angle_error_deg=angles,
                        source_to_fixed_wrist_R=rotations, source_to_fixed_wrist_translation_m=translations,
                        mapped_object_pose_w=mapped_pose, joint_lower_rad=robot.lower, joint_upper_rad=robot.upper,
                        geometry_pair_indices=g.pairs, geometry_pair_kind=g.pair_kind,
                        selected_native_contact_distance_m=ds, generated_contact_mask=generated,
                        baseline_selected_native_contact_distance_m=baseline_ds, baseline_generated_contact_mask=baseline_generated,
                        native_contact_count_by_pair=counts, native_contact_tie_count_by_pair=ties)
    reports = {}
    for kind in ("hand_object", "hand_self"):
        mask = g.pair_kind == kind
        reports[kind] = {"before": depth_summary(baseline_ds[:, mask], baseline_generated[:, mask]),
                         "after": depth_summary(ds[:, mask], generated[:, mask])}
    report = {"schema": "native_filtered_object_self_IK_result:1", "status": "completed_geometry_diagnostic",
              "frames": 151, "optimizer_success_count": sum(x["success"] for x in optimizer),
              "optimizer_unconverged_frames": [x["frame"] for x in optimizer if not x["success"]],
              "native_contact_summaries": reports, "bone_angle_mean_deg": float(angles.mean()), "bone_angle_max_deg": float(angles.max()),
              "normalized_target_tip_error_mean_m": float(np.linalg.norm(fks[:, [4, 8, 12, 16, 20]] - targets[:, [4, 8, 12, 16, 20]], axis=-1).mean()),
              "hard_joint_excess_max_rad": float(np.maximum(np.maximum(robot.lower - qs, qs - robot.upper), 0).max()),
              "q_ref_fd_rate_abs_max_rad_s": float(np.abs(np.diff(qs, axis=0) / np.diff(t)[:, None]).max()),
              "q_change_from_input_baseline_abs_max_rad": float(np.abs(qs - baseline_q).max()),
              "native_contact_module_metadata": g.metadata, "query_options": g.query_options,
              "native_collision_query_count": g.collision_query_count, "generated_tied_pair_frames": int(((ties > 1) & generated).sum()),
              "target_points_exact_same_as_baseline": True, "mapped_object_pose_exact_unchanged": True,
              "input_npz_sha256": sha(source), "baseline_npz_sha256": sha(a.baseline), "model_sha256": sha(a.model), "context_sha256": sha(a.context),
              "script_sha256": sha(Path(__file__)), "module_sha256": sha(Path(__file__).with_name("native_filtered_contact_geometry.py")),
              "output_npz_sha256": sha(output), "recipe_sha256": sha(a.output / "recipe.json"),
              "wall_seconds_including_validation": time.monotonic() - started, "dynamics_steps": 0,
              "physical_reference_qualified": False, "known_contacts_or_force_or_phase_or_PDtargets_inferred": False,
              "scope": "Fixed same-target geometric tradeoff only. Native selected contact depths/filters are not an all-geometry gap oracle or grasp/force/dynamics certificate. Unknown gaps stay NaN; qFD is a geometric reference derivative. Camera/scale/palm/anatomical uncertainty and nonsmooth/tied contact derivatives remain."}
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "optimizer_success_count", "native_contact_summaries", "bone_angle_mean_deg", "normalized_target_tip_error_mean_m", "wall_seconds_including_validation")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("observations", "baseline", "model", "context", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    run(parser.parse_args())
