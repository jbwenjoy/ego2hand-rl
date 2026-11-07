"""Read-only CPU diagnosis of all nine frozen replication evaluation logs.

Loads native MJB joint ranges; never creates MjData, steps physics or changes
the frozen criterion. Output to a new directory to independently reproduce.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mujoco
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def ranges(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), (np.flatnonzero(edges == -1)-1).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).resolve().parents[2] / "configs/replication-seeds123-evaluation-v1.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError("Preserve prior diagnosis; use --output NEW_DIRECTORY")
    plan = read(args.plan)
    assert len(plan["jobs"]) == 9
    cases, events, case_rows, matrices = [], [], [], []
    for index, job in enumerate(plan["jobs"]):
        directory = Path(job["output"])
        meta, evaluation = read(directory / "rollout.json"), read(directory / "evaluation.json")
        refmeta = read(directory / "reference.json")
        protocol = refmeta["task_acceptance"]
        margin = float(protocol["joint_limit_margin_rad"])
        with np.load(directory / "rollout.npz", allow_pickle=False) as data:
            a = {key: data[key].copy() for key in data.files}
        model = mujoco.MjModel.from_binary_path(str(directory / "model.mjb"))
        names = list(refmeta["joint_names"])
        ids = np.asarray([model.joint("robot/"+name).id for name in names])
        actuator_ids = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item() for jid in ids])
        bounds = model.jnt_range[ids].copy()
        assert np.array_equal(bounds[:, 0], a["joint_lower_rad"])
        assert np.array_equal(bounds[:, 1], a["joint_upper_rad"])
        assert len(a["time_s"]) == 151 and meta["rollout_censored"] is False
        assert np.array_equal(a["time_s"], np.arange(151)*.02)
        assert meta["checkpoint_sha256"] == job["checkpoint_sha256"] and meta["seed"] == 0
        q, qd, target, torque = [a[key] for key in ("joint_pos_rad", "joint_vel_rad_s", "joint_target_rad", "actuator_torque_Nm")]
        assert all(value.shape == (151, 20) and np.isfinite(value).all() for value in (q, qd, target, torque))
        lower_v = np.maximum(bounds[:, 0]-q, 0)
        upper_v = np.maximum(q-bounds[:, 1], 0)
        target_v = np.maximum(np.maximum(bounds[:, 0]-target, target-bounds[:, 1]), 0)
        # The logs record actuator force; with the released unit hinge gear it
        # is also the signed generalized joint torque. Refuse other layouts.
        gear = model.actuator_gear[actuator_ids]
        assert np.all(gear[:, 0] == 1) and np.all(gear[:, 1:] == 0)
        case_id = f"{job['method']}_seed{job['training_seed']}_{job['budget']}"

        def sample(frame, joint, side):
            sign = 1 if side == "upper" else -1
            bound = float(bounds[joint, 1 if side == "upper" else 0])
            violation = max(sign*(float(q[frame, joint])-bound), 0.)
            aid = actuator_ids[joint]
            ctrl = model.actuator_ctrlrange[aid]
            return {
                "frame": int(frame), "time_s": float(a["time_s"][frame]),
                "joint": names[joint], "side": side, "hard_bound_rad": bound,
                "q_actual_rad": float(q[frame, joint]), "qd_rad_s": float(qd[frame, joint]),
                "q_beyond_hard_bound_rad": violation,
                "q_beyond_frozen_margin_rad": max(violation-margin, 0.),
                "target_rad": float(target[frame, joint]),
                "target_margin_to_offending_hard_bound_rad": float(sign*(bound-target[frame, joint])),
                "target_min_margin_to_either_hard_bound_rad": float(min(target[frame, joint]-bounds[joint, 0], bounds[joint, 1]-target[frame, joint])),
                "actuator_ctrl_limited": bool(model.actuator_ctrllimited[aid]),
                "actuator_ctrlrange_rad": ctrl.tolist(),
                "actuator_torque_Nm": float(torque[frame, joint]),
                "velocity_outward_rad_s": float(sign*qd[frame, joint]),
                "actuator_torque_outward_Nm": float(sign*torque[frame, joint]),
                "target_interval_s": [float(a["time_s"][frame-1]), float(a["time_s"][frame])] if frame else None,
                "previous_q_actual_rad": float(q[frame-1, joint]) if frame else None,
                "previous_qd_rad_s": float(qd[frame-1, joint]) if frame else None,
            }

        all_v = np.stack([lower_v, upper_v], -1)
        peak_frame, peak_joint, peak_side = np.unravel_index(int(np.argmax(all_v)), all_v.shape)
        peak = sample(peak_frame, peak_joint, ["lower", "upper"][peak_side])
        case_events = []
        for joint in range(20):
            for side, violations in (("lower", lower_v[:, joint]), ("upper", upper_v[:, joint])):
                mask = violations > margin
                if not mask.any():
                    continue
                first, maximum = int(np.flatnonzero(mask)[0]), int(np.argmax(violations))
                item = {"joint": names[joint], "side": side, "frames_above_1mrad": int(mask.sum()),
                        "first": sample(first, joint, side), "peak": sample(maximum, joint, side),
                        "sampled_violation_runs": [{"first_frame": start, "last_frame": stop,
                              "first_time_s": float(a["time_s"][start]), "last_time_s": float(a["time_s"][stop]),
                              "frames": stop-start+1} for start, stop in ranges(mask)],
                        "native_joint_limited": bool(model.jnt_limited[ids[joint]]),
                        "native_joint_solref": model.jnt_solref[ids[joint]].tolist(),
                        "native_joint_solimp": model.jnt_solimp[ids[joint]].tolist()}
                case_events.append(item)
                event_row = {"case_id": case_id, "training_seed": job["training_seed"], "method": job["method"], "budget": job["budget"],
                             "joint": names[joint], "side": side, "frames_above_1mrad": int(mask.sum())}
                for when in ("first", "peak"):
                    for key in ("frame", "time_s", "q_actual_rad", "qd_rad_s", "target_rad", "q_beyond_hard_bound_rad",
                                "target_margin_to_offending_hard_bound_rad", "actuator_torque_Nm", "velocity_outward_rad_s", "actuator_torque_outward_Nm"):
                        event_row[when+"_"+key] = item[when][key]
                events.append(event_row)
        observed_ok = not case_events
        assert observed_ok == evaluation["task_joint_limits_ok"]
        assert np.isclose(max(lower_v.max(), upper_v.max()), evaluation["joint_state_violation_max_rad"], rtol=0, atol=1e-12)
        current = {"case_id": case_id, **job, "evaluation_seed": 0,
                   "task_success": evaluation["task_success"], "joint_state_limits_ok": observed_ok,
                   "frozen_margin_rad": margin, "first_violating_frame": min((event["first"]["frame"] for event in case_events), default=None),
                   "joint_sides_above_margin": len(case_events), "case_peak": peak,
                   "target_violation_max_rad": float(target_v.max()), "violations": case_events,
                   "inputs_sha256": {file: sha(directory / file) for file in ("rollout.npz", "rollout.json", "evaluation.json", "model.mjb", "reference.npz", "reference.json")},
                   "logged_bounds_exactly_match_native_MJB": True,
                   "native_model_timestep_s": float(model.opt.timestep)}
        cases.append(current)
        case_rows.append({"case_id": case_id, "method": job["method"], "training_seed": job["training_seed"],
                          "budget": job["budget"], "task_success": evaluation["task_success"], "joint_state_limits_ok": observed_ok,
                          "joint_sides_above_margin": len(case_events), "first_violating_frame": current["first_violating_frame"],
                          "peak_joint": peak["joint"], "peak_side": peak["side"], "peak_frame": peak["frame"],
                          "peak_time_s": peak["time_s"], "state_violation_max_mrad": 1000*peak["q_beyond_hard_bound_rad"],
                          "target_violation_max_rad": current["target_violation_max_rad"]})
        matrices.append((q, qd, target, torque, lower_v, upper_v, target_v))

    report = {"schema": "supermimic_chord_replication_joint_limit_diagnostics:1",
              "created_utc": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable,
              "script_sha256": sha(__file__), "plan": str(args.plan.resolve()), "plan_sha256": sha(args.plan),
              "mujoco_version": mujoco.__version__, "numpy_version": np.__version__,
              "scope": "All nine recorded evaluations, CPU read-only MJB range query; no MjData, physics step, new policy run or altered threshold",
              "threshold_semantics": "frozen actual joint state allowance is 0.001 rad outside MJB hard range; log maxima only, not substep/continuous extrema",
              "target_alignment": "target[k] was executed during [time[k-1],time[k]], not the next action; target[0] is initialization",
              "torque_semantics": "signed recorded actuator torque; verified unit hinge gear. Outward sign is positive toward the offending lower/upper side. Contact/constraint torques are not separately observed here.",
              "interpretation": "Bounded position targets do not guarantee bounded measured joint states. These logs describe dynamic overshoot in a soft-constraint simulation; they do not isolate its cause or certify hardware safety. Frozen failures remain failures.",
              "future_condition_boundary": "Any velocity-aware feasible controller is a separate new condition; do not insert it into current penalty-only attribution or retune these thresholds.",
              "cases": cases}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for filename, rows in (("case_peaks.csv", case_rows), ("joint_violations.csv", events)):
        with (output / filename).open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    stacked = np.asarray(matrices)
    np.savez_compressed(output / "all_case_joint_traces.npz", case_ids=np.asarray([case["case_id"] for case in cases]),
                        joint_names=np.asarray(names), time_s=np.arange(151)*.02,
                        q_actual_rad=stacked[:, 0], qd_rad_s=stacked[:, 1], target_rad=stacked[:, 2],
                        actuator_torque_Nm=stacked[:, 3], lower_violation_rad=stacked[:, 4],
                        upper_violation_rad=stacked[:, 5], target_violation_rad=stacked[:, 6])
    lines = ["Replication joint-limit diagnosis: all nine cases", "",
             "The frozen allowance remains 1 mrad. No physics or policy was rerun.",
             "case_peaks.csv has every case, including both fully passing cases.",
             "joint_violations.csv has every joint/side exceeding 1 mrad, with first and peak q/qd/target/torque.",
             "report.json preserves model/input hashes and contiguous sampled violation windows.",
             "all_case_joint_traces.npz preserves every sampled joint for all nine cases.", "",
             "target[k] belongs to the previous [k-1,k] control interval. Outward qd/torque use the offending-bound sign.",
             "A positive target margin is inside the hard range; tiny negative values are retained as recorded.",
             "All maxima are from 20 ms logs, not unobserved 5 ms or continuous extrema.",
             "Bounded targets and measured joint-state overshoot are distinct. This is not hardware safety evidence.",
             "A future velocity-aware feasible controller requires a separate condition; current penalty-only must stay unchanged.", "",
             "Reproduce independently:", f'"{sys.executable}" "{Path(__file__).resolve()}" --output NEW_DIRECTORY', ""]
    for row in case_rows:
        lines.append(f"{row['case_id']}: state max {row['state_violation_max_mrad']:.6f} mrad, "
                     f"{row['peak_joint']} {row['peak_side']} frame {row['peak_frame']} t={row['peak_time_s']:.2f}s; "
                     f"joint_gate={row['joint_state_limits_ok']}, task={row['task_success']}")
    (output / "README.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": len(cases), "joint_side_events": len(events), "case_peaks": case_rows}, indent=2))


if __name__ == "__main__":
    main()
