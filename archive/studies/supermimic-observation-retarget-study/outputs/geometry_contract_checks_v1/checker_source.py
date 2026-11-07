"""Input-boundary metamorphic checks and FK-only artifact plots; no dynamics."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import retarget_landmarks_cpu as ik


def check(input_dir, model, fitted, output):
    output.mkdir(parents=True, exist_ok=False)
    with np.load(input_dir / "observations.npz", allow_pickle=False) as z:
        original = {key: z[key].copy() for key in z.files}
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    with np.load(fitted / "geometric_retarget.npz", allow_pickle=False) as z:
        result = {key: z[key].copy() for key in z.files}
    checks = {}

    def fixture(name, arrays):
        folder = output / name
        folder.mkdir()
        np.savez_compressed(folder / "observations.npz", **arrays)
        meta = dict(manifest)
        meta["observation_sha256"] = ik.sha(folder / "observations.npz")
        (folder / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        return folder

    poisoned = {key: value.copy() for key, value in original.items()}
    poisoned["wrist_pose_source_w"][:] = np.nan
    folder = fixture("unused_cached_wrist_poison", poisoned)
    native_forward, native_step = mujoco.mj_forward, mujoco.mj_step

    def forbidden(*args, **kwargs):
        raise AssertionError("A dynamics API was called in a geometric-only test")

    log = io.StringIO()
    try:
        mujoco.mj_forward = forbidden
        mujoco.mj_step = forbidden
        with contextlib.redirect_stdout(log):
            ik.run(folder, model, output / "wrist_poison_fit")
    finally:
        mujoco.mj_forward, mujoco.mj_step = native_forward, native_step
    checks["no_python_mj_forward_or_mj_step_calls"] = True
    with np.load(output / "wrist_poison_fit/geometric_retarget.npz", allow_pickle=False) as z:
        checks["unused_cached_wrist_poison_leaves_all_outputs_exact"] = all(np.array_equal(result[key], z[key]) for key in result)

    for name, mutate in (("robot_label_rejected", lambda a: a.update(joint_pos_rad=np.full((len(a['time_s']), 20), np.nan))),
                         ("invalid_quaternion_rejected", lambda a: a['object_pose_source_w'].__setitem__((0, slice(3, 7)), 0.))):
        arrays = {key: value.copy() for key, value in original.items()}
        mutate(arrays)
        bad = fixture(name, arrays)
        rejected_output = output / (name + "_must_not_exist")
        try:
            ik.run(bad, model, rejected_output)
        except ValueError:
            checks[name] = not rejected_output.exists()
        else:
            checks[name] = False
    rotation, translation = result["source_to_fixed_wrist_R"], result["source_to_fixed_wrist_translation_m"]
    mapped = result["mapped_object_pose_w"]
    restored = np.einsum("tji,tj->ti", rotation, mapped[:, :3] - translation)
    position_error = float(np.abs(restored - original["object_pose_source_w"][:, :3]).max())
    original_r = Rotation.from_quat(original["object_pose_source_w"][:, [4, 5, 6, 3]]).as_matrix()
    mapped_r = Rotation.from_quat(mapped[:, [4, 5, 6, 3]]).as_matrix()
    restored_r = np.swapaxes(rotation, -1, -2) @ mapped_r
    rotation_error = float(np.abs(restored_r - original_r).max())
    checks["recorded_object_transform_roundtrip_position"] = position_error <= 1e-12
    checks["recorded_object_transform_roundtrip_rotation"] = rotation_error <= 1e-12
    checks["frame_rotation_proper_orthogonal"] = bool(np.allclose(np.swapaxes(rotation, -1, -2) @ rotation, np.eye(3), atol=1e-12)
                                                       and np.allclose(np.linalg.det(rotation), 1, atol=1e-12))
    q = result["joint_pos_rad"]
    checks["geometric_q_finite_and_within_hard_bounds"] = bool(np.isfinite(q).all() and np.all(q >= result['joint_lower_rad']) and np.all(q <= result['joint_upper_rad']))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(12, 4), constrained_layout=True)
    frame_ids = [0, len(q) // 2, len(q) - 1]
    all_points = np.r_[result['fk_hand_points_w'][frame_ids].reshape(-1, 3), result['morphology_normalized_target_points_w'][frame_ids].reshape(-1, 3)]
    center = (all_points.min(0) + all_points.max(0)) / 2
    radius = np.ptp(all_points, axis=0).max() * .55
    for column, frame in enumerate(frame_ids, 1):
        ax = fig.add_subplot(1, 3, column, projection="3d")
        for array, color, style, label in ((result['fk_hand_points_w'], '#1764ab', '-', 'Robot FK'),
                                            (result['morphology_normalized_target_points_w'], '#db761b', '--', 'Normalized target')):
            for finger in range(5):
                index = [0] + list(range(1 + finger * 4, 5 + finger * 4))
                points = array[frame, index]
                ax.plot(*points.T, color=color, linestyle=style, marker='o', markersize=2, label=label if finger == 0 else None)
        ax.set(xlim=(center[0] - radius, center[0] + radius), ylim=(center[1] - radius, center[1] + radius), zlim=(center[2] - radius, center[2] + radius),
               xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title=f"Frame{frame}, t={result['time_s'][frame]:.2f}s")
        ax.set_box_aspect((1, 1, 1));ax.view_init(elev=25, azim=-45)
        if column == 1:ax.legend(fontsize=7, loc='upper left')
    fig.suptitle('Cached-landmark geometric IK only | nominal MJB palm anchor\nDashed targets use robot bone lengths; no contact, dynamics or real-ego qualification')
    for extension in ('png', 'svg'):fig.savefig(output / ('geometry.' + extension), dpi=180)
    plt.close(fig)
    report = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "count": len(checks), "fit_report_sha256": ik.sha(fitted / "report.json"),
              "prototype_sha256": ik.sha(Path(ik.__file__)), "checker_sha256": ik.sha(Path(__file__)),
              "roundtrip_position_max_abs_m": position_error, "roundtrip_rotation_matrix_max_abs": rotation_error,
              "metamorphic_rerun_stdout": log.getvalue(), "gpu_used": False, "dynamics_steps": 0,
              "qualification": "Input/FK/coordinate artifact checks only. Geometry is not physical grounding, calibrated anatomy or ego reconstruction."}
    (output / 'checks.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({"status": report['status'], "checks": checks}))
    if report['status'] != 'passed':raise AssertionError('Geometry artifact contract failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--robot-model', type=Path, required=True)
    parser.add_argument('--fitted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.input_dir.resolve(strict=True), args.robot_model.resolve(strict=True), args.fitted.resolve(strict=True), args.output.resolve())
