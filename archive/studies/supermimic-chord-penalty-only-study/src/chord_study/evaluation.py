"""Independent full-clip metrics; physics claims require explicit rollout evidence."""
from dataclasses import dataclass, asdict
import numpy as np
from .geometry import quat_matrix, rotation_error
from .acceptance import evaluate_pilot_task


@dataclass(frozen=True)
class Thresholds:
    # Proposed screening values, frozen in every report; not measured performance.
    position_m: float = .02
    orientation_deg: float = 20.
    penetration_m: float = .003
    joint_limit_margin_rad: float = .001


def evaluate(reference, rollout, rollout_meta, thresholds=Thresholds()):
    ref_report = reference.validate()
    if not all(np.isfinite(v) and v > 0 for v in asdict(thresholds).values()):
        raise ValueError("screening thresholds must be finite and positive")
    a = reference.arrays
    required = {"time_s", "object_pose_w", "joint_pos_rad", "joint_target_rad", "joint_lower_rad", "joint_upper_rad",
                "object_assist_wrench_w", "object_pose_overwritten", "terminated", "penetration_m", "wrist_pose_w"}
    if required-set(rollout):
        raise ValueError(f"missing rollout evidence: {sorted(required-set(rollout))}")
    if rollout_meta.get("physics_engine") not in {"mujoco", "mujoco_warp"} or rollout_meta.get("object_free_dynamics") is not True:
        raise ValueError("independent physical rollout with a free object required")
    if rollout_meta.get("reference_source_id") != reference.meta["source_id"]:
        raise ValueError("rollout does not match reference source")
    if rollout_meta.get("world_frame") != reference.meta["world_frame"]:
        raise ValueError("rollout and reference world frames differ")
    for key in ("units","quaternion_order","joint_names"):
        if rollout_meta.get(key) != reference.meta[key]:
            raise ValueError(f"rollout and reference {key} differ")
    n, total = len(rollout["time_s"]), len(a["time_s"])
    if n < 1 or n > total:
        raise ValueError("invalid rollout length")
    shapes = {"time_s": (n,), "object_pose_w":(n,7), "joint_pos_rad":(n,20), "joint_target_rad":(n,20),
              "joint_lower_rad":(20,), "joint_upper_rad":(20,), "object_assist_wrench_w":(n,6),
              "object_pose_overwritten":(n,), "terminated":(n,), "penetration_m":(n,), "wrist_pose_w":(n,7)}
    for key, shape in shapes.items():
        value = np.asarray(rollout[key])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid rollout field {key}, expected {shape}")
    if not np.allclose(rollout["time_s"], a["time_s"][:n], rtol=0, atol=1e-5):
        raise ValueError("rollout and reference clocks are not aligned")
    if np.any(rollout["penetration_m"]<0):
        raise ValueError("penetration_m must be a non-negative penetration depth")
    for key in ("terminated","object_pose_overwritten"):
        if not np.isin(rollout[key],[0,1]).all():
            raise ValueError(f"{key} must be a binary evidence flag")
    pose = rollout["object_pose_w"]
    if not np.allclose(np.linalg.norm(pose[:,3:],axis=-1),1,atol=1e-4):
        raise ValueError("invalid rollout quaternion")
    pos = np.linalg.norm(pose[:,:3]-a["object_pose_w"][:n,:3], axis=-1)
    full_angle = np.rad2deg(rotation_error(pose[:,3:], a["object_pose_w"][:n,3:]))
    axis, ref_axis = quat_matrix(pose[:,3:])[:,:,0], quat_matrix(a["object_pose_w"][:n,3:])[:,:,0]
    # Task equivalence, not a claim about exact physical geometry symmetry.
    # x-axis polarity is retained; ignoring roll does not allow end swapping.
    axis_angle = np.rad2deg(np.arccos(np.clip((axis*ref_axis).sum(-1),-1,1)))
    angle = axis_angle if reference.meta["orientation_metric"] == "directed_axis_x" else full_angle
    lower, upper = rollout["joint_lower_rad"], rollout["joint_upper_rad"]
    if np.any(lower >= upper):
        raise ValueError("invalid joint limits")
    margin = thresholds.joint_limit_margin_rad
    joint_ok = all(np.all((rollout[k]>=lower-margin)&(rollout[k]<=upper+margin)) for k in ("joint_pos_rad","joint_target_rad"))
    state_violation=np.maximum(np.maximum(lower-rollout["joint_pos_rad"],rollout["joint_pos_rad"]-upper),0)
    target_violation=np.maximum(np.maximum(lower-rollout["joint_target_rad"],rollout["joint_target_rad"]-upper),0)
    unassisted = bool(np.all(np.abs(rollout["object_assist_wrench_w"])<1e-9) and not np.any(rollout["object_pose_overwritten"]))
    complete = n == total and not np.any(rollout["terminated"])
    penetration_ok = bool(np.max(rollout["penetration_m"])<=thresholds.penetration_m)
    tracking_ok = bool(np.all(pos<=thresholds.position_m) and np.all(angle<=thresholds.orientation_deg))
    wrist = rollout["wrist_pose_w"]
    if not np.allclose(np.linalg.norm(wrist[:,3:],axis=-1),1,atol=1e-4):
        raise ValueError("invalid rollout wrist quaternion")
    wrist_position = np.linalg.norm(wrist[:,:3]-a["wrist_pose_w"][0,:3],axis=-1).max()
    wrist_angle = np.rad2deg(rotation_error(wrist[:,3:],a["wrist_pose_w"][0,3:])).max()
    wrist_ok = bool(ref_report["fixed_wrist_admissible"] and wrist_position<=.005 and wrist_angle<=5.)
    def summary(v):
        return {"mean":float(v.mean()),"p95":float(np.quantile(v,.95)),"max":float(v.max())}
    dt = np.diff(rollout["time_s"])
    speed = np.diff(rollout["joint_target_rad"],axis=0)/dt[:,None] if n>1 else np.zeros((0,20))
    report = {"schema":"supermimic_chord_evaluation:1", "source_id":reference.meta["source_id"],
            "source_kind":reference.meta["source_kind"], "thresholds":asdict(thresholds),
            "threshold_status":"proposed_screening_values_not_final_acceptance", "frames":n,"expected_frames":total,
            "complete_clip":bool(complete),"reached_reference_end":bool(n==total),"unassisted":unassisted,"joint_limits_ok":bool(joint_ok),
            "joint_state_limits_ok":bool(np.all(state_violation<=margin)),"joint_target_limits_ok":bool(np.all(target_violation<=margin)),
            "joint_state_violation_max_rad":float(state_violation.max()),"joint_target_violation_max_rad":float(target_violation.max()),
            "rotation_progress_error_deg":None,"rotation_progress_note":"Requires a declared external rotation axis and phase convention; shaft angle alone does not measure turn count.",
            "penetration_ok":penetration_ok,"tracking_ok":tracking_ok,"fixed_wrist_ok":wrist_ok,
            "wrist_displacement_from_reference_max_m":float(wrist_position),"wrist_angle_from_reference_max_deg":float(wrist_angle),
            "screening_success":bool(complete and unassisted and joint_ok and penetration_ok and tracking_ok and wrist_ok),
            "task_success":None,"task_success_unavailable_reason":"Task-specific progress/drop/load acceptance must be defined and logged separately.",
            "position_error_m":summary(pos), "orientation_error_deg":summary(angle),
            "full_so3_error_deg_diagnostic":summary(full_angle),"orientation_metric":reference.meta["orientation_metric"],
            "target_speed_rms_rad_s":float(np.sqrt(np.mean(speed**2))) if speed.size else None,
            "note":"Metric screening does not establish task success, model-gradient reliability, or generalization."}
    report.update(evaluate_pilot_task(reference,rollout,rollout_meta,report))
    if report["task_success"] is not None:
        report["rotation_progress_note"] = "Frozen reference-only progress metric; observable shaft rotation excludes axial roll. See task_protocol_sha256."
    return report
