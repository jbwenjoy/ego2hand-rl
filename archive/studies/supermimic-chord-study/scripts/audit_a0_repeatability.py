"""Record differences between two uncensored A0 logs without selecting a winner."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compare(left, right):
    fields = ("time_s", "physics_time_s", "joint_pos_rad", "joint_vel_rad_s", "joint_target_rad",
              "object_pose_w", "object_lin_vel_w_m_s", "object_ang_vel_w_rad_s", "wrist_pose_w")
    report = {}
    with np.load(left / "rollout.npz", allow_pickle=False) as a, np.load(right / "rollout.npz", allow_pickle=False) as b:
        for field in fields:
            x, y = a[field], b[field]
            if x.shape != y.shape:
                report[field] = {"shape_mismatch": [list(x.shape), list(y.shape)]}
                continue
            delta = np.abs(x-y)
            changed = delta.reshape(len(x), -1).max(-1) > 0
            frame = int(np.flatnonzero(changed)[0]) if changed.any() else None
            report[field] = {"exact_equal": bool(np.array_equal(x, y)), "initial_max_abs_difference": float(delta[0].max()),
                             "first_difference_frame": frame, "max_abs_difference": float(delta.max()),
                             "first_frame_max_abs_difference": float(delta[frame].max()) if frame is not None else 0.}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cases = []
    for run in (args.left, args.right):
        metadata, evaluation = load(run / "rollout.json"), load(run / "evaluation.json")
        cases.append({"run": str(run.resolve()), "rollout_sha256": hashlib.sha256((run / "rollout.npz").read_bytes()).hexdigest(),
                      "reference_sha256": metadata["reference_bundle_sha256"], "model_sha256": metadata["model_sha256"],
                      "evaluator_sha256": metadata["script_sha256"], "frames": evaluation["frames"],
                      "censored": evaluation["rollout_censored"], "task_success": evaluation["task_success"],
                      "position_mean_m": evaluation["position_error_m"]["mean"], "drop_detected": evaluation["drop_detected"]})
    result = {"cases": cases, "field_comparison": compare(args.left, args.right),
              "interpretation": "Same recorded initial states and targets do not establish numerically identical later physics. Preserve both failures; do not select by outcome. Exact numerical cause is unproven.",
              "complete_clip_flag": "This evaluator also requires no latched termination for complete_clip=true; a false flag does not imply fewer recorded frames.",
              "next_validation": "Repeat A0 with the current evaluator and frozen imported control sources after the active GPU job releases."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"cases": cases, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
