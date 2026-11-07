"""Landmarks-only geometric IK on a cached controlled fixture, not physics.

Robot joint trajectory labels and the cached source wrist pose are never read.
The per-frame palm rebase is an explicit fixed-wrist task redesign. Object
scale/camera/handedness/confidence remain unqualified for a physical reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


ALLOWED = {"time_s", "hand_landmarks_source_w", "object_pose_source_w", "wrist_pose_source_w",
           "landmark_confidence_known", "object_confidence_known", "wrist_confidence_known"}
JOINT_NAMES = [f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]
BONES = np.array([(1 + finger * 4 + bone, 2 + finger * 4 + bone)
                  for finger in range(5) for bone in range(3)])
TEMPORAL_WEIGHT = .03
NOMINAL_WEIGHT = .001
MAX_NFEV = 80


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hand_frames(points):
    """Right-handed basis from palm landmarks; no measured wrist pose input."""
    x = points[..., 5, :] - points[..., 17, :]
    xnorm = np.linalg.norm(x, axis=-1, keepdims=True)
    if np.any(xnorm < 1e-6):
        raise ValueError("Degenerate palm width")
    x = x / xnorm
    y = points[..., 9, :] - points[..., 0, :]
    y = y - np.sum(y * x, axis=-1, keepdims=True) * x
    ynorm = np.linalg.norm(y, axis=-1, keepdims=True)
    if np.any(ynorm < 1e-6):
        raise ValueError("Degenerate palm length/basis")
    y = y / ynorm
    return np.stack((x, y, np.cross(x, y)), axis=-1)


def directions(points):
    vector = points[..., BONES[:, 1], :] - points[..., BONES[:, 0], :]
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    if np.any(length < 1e-6):
        raise ValueError("Degenerate observed bone")
    return vector / length


class RobotKinematics:
    """mj_kinematics/comPos/Jacobians only; never mj_forward or mj_step."""
    def __init__(self, path):
        self.model = m = mujoco.MjModel.from_binary_path(str(path))
        self.data = mujoco.MjData(m)
        ids = np.array([m.joint("robot/" + name).id for name in JOINT_NAMES])
        if not np.all(m.jnt_type[ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Expected twenty hinge finger joints")
        self.qa, self.va = m.jnt_qposadr[ids], m.jnt_dofadr[ids]
        self.lower, self.upper = m.jnt_range[ids].T.copy()
        self.nominal = np.clip(m.qpos0[self.qa], self.lower + 1e-6, self.upper - 1e-6)
        self.points = [("body", m.body("robot/right_palm_link").id)]
        for finger in range(1, 6):
            self.points.extend(("body", m.body(f"robot/right_finger{finger}_link{link}").id) for link in (2, 3, 4))
            self.points.append(("site", m.site(f"robot/right_finger{finger}_tip").id))
        self.last_q = None
        self.cached = None
        neutral, _ = self.forward(self.nominal)
        self.lengths = np.linalg.norm(neutral[BONES[:, 1]] - neutral[BONES[:, 0]], axis=-1)
        if np.any(self.lengths < 1e-6):
            raise ValueError("Degenerate robot bone mapping")
        self.neutral = neutral.copy()
        self.palm_frame = hand_frames(neutral)

    def forward(self, q):
        if self.last_q is not None and np.array_equal(q, self.last_q):
            return self.cached
        self.data.qpos[self.qa] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        points, jacobians = [], []
        for kind, index in self.points:
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            if kind == "body":
                points.append(self.data.xpos[index].copy())
                mujoco.mj_jacBody(self.model, self.data, jp, jr, index)
            else:
                points.append(self.data.site_xpos[index].copy())
                mujoco.mj_jacSite(self.model, self.data, jp, jr, index)
            jacobians.append(jp[:, self.va].copy())
        self.last_q = np.array(q, copy=True)
        self.cached = np.asarray(points), np.asarray(jacobians)
        return self.cached

    def bone_features(self, q):
        points, jac = self.forward(q)
        value = (points[BONES[:, 1]] - points[BONES[:, 0]]) / self.lengths[:, None]
        derivative = (jac[BONES[:, 1]] - jac[BONES[:, 0]]) / self.lengths[:, None, None]
        return value, derivative.reshape(-1, 20)


def run(input_dir, model_path, output):
    started = time.perf_counter()
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    if manifest.get("is_real_ego") is not False:
        raise ValueError("Prototype restricted to the explicitly cached controlled fixture")
    source = input_dir / "observations.npz"
    if sha(source) != manifest["observation_sha256"]:
        raise ValueError("Observation hash mismatch")
    with np.load(source, allow_pickle=False) as arrays:
        if set(arrays.files) != ALLOWED:
            raise ValueError("Observation allowlist differs; robot labels or unknown fields are forbidden")
        t = np.array(arrays["time_s"], dtype=np.float64)
        human = np.array(arrays["hand_landmarks_source_w"], dtype=np.float64)
        object_pose = np.array(arrays["object_pose_source_w"], dtype=np.float64)
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0) or human.shape != (len(t), 21, 3) or object_pose.shape != (len(t), 7):
        raise ValueError("Invalid observation/time dimensions")
    if not all(np.isfinite(a).all() for a in (t, human, object_pose)):
        raise ValueError("Nonfinite observations")
    object_quaternion_norm_error = float(np.abs(np.linalg.norm(object_pose[:, 3:], axis=-1) - 1).max())
    if object_quaternion_norm_error > 1e-4:
        raise ValueError("Object wxyz quaternions are not unit within1e-4; no silent large renormalization")
    output.mkdir(parents=True, exist_ok=False)
    robot = RobotKinematics(model_path)
    human_frames = hand_frames(human)
    rotations = np.einsum("ij,tkj->tik", robot.palm_frame, human_frames)
    unit_targets = np.einsum("tij,tkj->tki", rotations, directions(human))
    scaled_rotations = np.einsum("ij,tkj->tik", robot.palm_frame, hand_frames(human * 10))
    scaled_features = np.einsum("tij,tkj->tki", scaled_rotations, directions(human * 10))
    scale_error = float(np.abs(unit_targets - scaled_features).max())
    # Independent central differences at a synthetic interior pose, not a source label.
    probe = (robot.lower + robot.upper) / 2
    _, analytic = robot.bone_features(probe)
    finite_difference = np.empty_like(analytic)
    for joint in range(20):
        plus, minus = probe.copy(), probe.copy()
        plus[joint] += 1e-6
        minus[joint] -= 1e-6
        finite_difference[:, joint] = ((robot.bone_features(plus)[0] - robot.bone_features(minus)[0]) / 2e-6).ravel()
    jac_error = float(np.abs(analytic - finite_difference).max())
    if scale_error > 1e-10 or jac_error > 1e-6:
        raise RuntimeError("Kinematic feature/Jacobian contract failed")
    qs, fks, targets, angles, optimizer = [], [], [], [], []
    previous = robot.nominal.copy()
    identity = np.eye(20)
    for frame, desired in enumerate(unit_targets):
        def residual(q):
            features, _ = robot.bone_features(q)
            return np.r_[(features - desired).ravel(), TEMPORAL_WEIGHT * (q - previous), NOMINAL_WEIGHT * (q - robot.nominal)]

        def jacobian(q):
            return np.vstack((robot.bone_features(q)[1], TEMPORAL_WEIGHT * identity, NOMINAL_WEIGHT * identity))

        result = least_squares(residual, previous.copy(), jac=jacobian, bounds=(robot.lower, robot.upper),
                               max_nfev=MAX_NFEV, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        q = result.x.copy()
        points, _ = robot.forward(q)
        features, _ = robot.bone_features(q)
        angle = np.rad2deg(np.arccos(np.clip(np.sum(features * desired, axis=-1), -1, 1)))
        target = robot.neutral.copy()
        for finger in range(5):
            first = 1 + finger * 4
            for bone in range(3):
                index = finger * 3 + bone
                target[first + bone + 1] = target[first + bone] + robot.lengths[index] * desired[index]
        qs.append(q); fks.append(points.copy()); targets.append(target); angles.append(angle)
        optimizer.append({"frame": frame, "success": bool(result.success), "status": int(result.status), "nfev": int(result.nfev), "cost": float(result.cost)})
        previous = q
    qs, fks, targets, angles = map(np.asarray, (qs, fks, targets, angles))
    translations = robot.neutral[0] - np.einsum("tij,tj->ti", rotations, human[:, 0])
    mapped_position = np.einsum("tij,tj->ti", rotations, object_pose[:, :3]) + translations
    object_rotation = Rotation.from_quat(object_pose[:, [4, 5, 6, 3]]).as_matrix()
    mapped_quaternion = Rotation.from_matrix(rotations @ object_rotation).as_quat()[:, [3, 0, 1, 2]]
    np.savez_compressed(output / "geometric_retarget.npz", time_s=t, joint_pos_rad=qs,
                        fk_hand_points_w=fks, morphology_normalized_target_points_w=targets,
                        bone_angle_error_deg=angles, source_to_fixed_wrist_R=rotations,
                        source_to_fixed_wrist_translation_m=translations,
                        mapped_object_pose_w=np.c_[mapped_position, mapped_quaternion],
                        joint_lower_rad=robot.lower, joint_upper_rad=robot.upper)
    excess = np.maximum(np.maximum(robot.lower - qs, qs - robot.upper), 0)
    report = {"schema": "landmarks_only_geometric_retarget:1", "status": "completed_geometry_diagnostic",
              "input_npz_sha256": sha(source), "input_manifest_sha256": sha(input_dir / "manifest.json"),
              "robot_model_sha256": sha(model_path), "script_sha256": sha(__file__),
              "output_npz_sha256": sha(output / "geometric_retarget.npz"), "source_id": manifest["source_id"],
              "split_group": manifest["split_group"], "frames": len(t), "joint_names": JOINT_NAMES,
              "read_observation_fields": ["time_s", "hand_landmarks_source_w", "object_pose_source_w"],
              "input_object_quaternion_norm_max_error": object_quaternion_norm_error,
              "object_quaternion_rule": "Reject norm error>1e-4 before IK; SciPy may normalize admitted numerical roundoff",
              "source_robot_joint_labels_read": False, "source_cached_wrist_pose_read": False,
              "source_confidence_rule": "unknown; uniform weights for this controlled prototype only",
              "robot_landmark_mapping": "Palm origin plus link2/link3/link4/tip site paired with each observed ordinal4-point chain. Conventional MediaPipe thumb=CMC/MCP/IP/tip and other fingers=MCP/PIP/DIP/tip; actual source correspondence remains unqualified.",
              "right_hand_topology_assumption": True, "optimizer": {"max_nfev": MAX_NFEV, "temporal_weight": TEMPORAL_WEIGHT, "nominal_weight": NOMINAL_WEIGHT, "per_frame": optimizer},
              "optimizer_success_count": sum(row["success"] for row in optimizer),
              "hard_joint_excess_max_rad": float(excess.max()), "bone_angle_error_mean_deg": float(angles.mean()),
              "bone_angle_error_max_deg": float(angles.max()),
              "q_ref_rate_fd_abs_max_rad_s": float(np.abs(np.diff(qs, axis=0) / np.diff(t)[:, None]).max()),
              "q_ref_rate_fd_note": "Geometric-reference finite difference only, not measured joint velocity or an executed-target rate gate",
              "normalized_target_fingertip_error_mean_m": float(np.linalg.norm(fks[:, [4, 8, 12, 16, 20]] - targets[:, [4, 8, 12, 16, 20]], axis=-1).mean()),
              "feature_scale10_invariance_max_abs": scale_error, "analytic_jacobian_centralFD_max_abs": jac_error,
              "wall_seconds_excluding_imports": time.perf_counter() - started,
              "physical_reference_qualified": False, "dynamics_steps": 0, "physical_rollout_or_success": None,
              "task_redesign": "Per-frame landmark palm rotation/translation removed to fixed robot wrist; this changes wrist dynamics and object/gravity relation. R/t and mapped object pose retained, not hidden.",
              "fixed_anchor": "Palm body origin and palm basis at nominal MJB qpos0/default mocap; not an independently calibrated wrist landmark or the original physical reference wrist",
              "fixed_anchor_position_w_m": robot.neutral[0].tolist(),
              "fixed_anchor_basis_R": robot.palm_frame.tolist(),
              "hand_metric_scale": "Bone directions use robot segment lengths; object pose still assumes upstream cached SI, not independently calibrated",
              "not_inferred": ["phase labels", "known contacts", "confidence values", "PD targets", "initial physical velocities or forces"],
              "physical_adapter_blockers": ["camera/video lineage, metric object scale, right-handed landmark semantics and confidence unqualified", "fixed-wrist task redesign eligibility", "geometric correspondence and contact feasibility", "unassisted dynamic grounding with actual executed controls"],
              "claim": "Controlled landmarks-only kinematic IK prototype; neither real ego reconstruction nor physical retargeting success"}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "frames", "optimizer_success_count", "hard_joint_excess_max_rad", "bone_angle_error_mean_deg", "bone_angle_error_max_deg", "feature_scale10_invariance_max_abs", "analytic_jacobian_centralFD_max_abs", "wall_seconds_excluding_imports", "physical_reference_qualified")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--robot-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.input_dir.resolve(strict=True), args.robot_model.resolve(strict=True), args.output.resolve())
