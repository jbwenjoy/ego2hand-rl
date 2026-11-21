"""Iterative learning control for the development clip using free-object simulation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.ego import hocap_carton_control as ctrl
from supermimic_research.strict_json import write_json
from evaluate_hocap_carton import errors, evaluate, rms


def clip_vectors(vectors, bound):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors * np.minimum(1, bound / np.maximum(norms, 1e-12))


def objective(trace, source, hand):
    pos, ang = errors(source, trace["world_from_carton_sim"])
    tip_error = np.linalg.norm(trace["sim_tips"] - hand[:, ctrl.TIPS], axis=2) * 1000
    joint = np.rad2deg(trace["interval_joint_excursion_rad"].max())
    score = (rms(pos) / 20 + rms(ang) / 8 + pos.max() / 40 + ang.max() / 15
             + np.mean(pos[-5:]) / 15 + max(0, rms(tip_error) / 20 - 1)
             + max(0, joint / 2 - 1))
    return dict(score=float(score), pos_rms_mm=rms(pos), rot_rms_deg=rms(ang),
                max_pos_mm=float(pos.max()), max_rot_deg=float(ang.max()),
                endpoint_mm=float(pos[-1]), finger_rms_mm=rms(tip_error), max_joint_deg=float(joint))


def candidate(wrist, actual, source, base, gain, last, shift_frames):
    # The pivots are the simulated carton centres. Only the wrist reference is actuated.
    t = np.arange(len(source))
    phase = np.interp(t, [0, 29, 39, last, 132, 140], [0, 0, 1, 1, 0, 0])
    index = np.clip(t + shift_frames, 0, len(t) - 1)
    dp = gaussian_filter1d(source[:, :3, 3] - actual[:, :3, 3], 2, axis=0)[index]
    rotation_error = Rotation.from_matrix(source[:, :3, :3]) * Rotation.from_matrix(actual[:, :3, :3]).inv()
    dr = gaussian_filter1d(rotation_error.as_rotvec(), 2, axis=0)[index]
    rot = Rotation.from_rotvec(dr * (gain * phase[:, None]))
    correction = rot.apply(wrist[:, :3, 3] - actual[:, :3, 3]) - (wrist[:, :3, 3] - actual[:, :3, 3])
    target = wrist.copy()
    relative = (rot * Rotation.from_matrix(wrist[:, :3, :3])) * Rotation.from_matrix(base[:, :3, :3]).inv()
    target[:, :3, :3] = (Rotation.from_rotvec(clip_vectors(relative.as_rotvec(), np.deg2rad(30))) *
                          Rotation.from_matrix(base[:, :3, :3])).as_matrix()
    delta = wrist[:, :3, 3] + gain * phase[:, None] * dp + correction - base[:, :3, 3]
    target[:, :3, 3] = base[:, :3, 3] + clip_vectors(delta, .045)
    return target


def carry_candidate(parent_path, anchor, start, stop, capture, margin, return_trace=False):
    report = json.loads(parent_path.read_bytes())
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(report["mjb"])))
    with np.load(verify_file_pin(report["trace"]), allow_pickle=False) as a:
        hand, source, actual = a["source_right_hand21"], a["world_from_carton_source"], a["world_from_carton_sim"]
        base, baseline_fingers, times = a["target_wrist"], a["target_finger_q"], a["source_nominal_time_s"]
        realised = a["sim_wrist"][anchor]
        retained = np.linalg.solve(actual[anchor], realised)
    # Carry with one physically attained robot grasp in the object's frame.
    # Approach and release blend with the initial human-derived hand commands.
    rigid = source @ retained
    alpha = np.interp(np.arange(len(base)), [0, start, 34, stop, 132, 140], [0, 0, 1, 1, 0, 0])
    alpha = capture * alpha ** 2 * (3 - 2 * alpha)
    wrist = base.copy()
    wrist[:, :3, 3] += alpha[:, None] * clip_vectors(rigid[:, :3, 3] - base[:, :3, 3], .065)
    original_r = Rotation.from_matrix(base[:, :3, :3])
    angular = (Rotation.from_matrix(rigid[:, :3, :3]) * original_r.inv()).as_rotvec()
    wrist[:, :3, :3] = (Rotation.from_rotvec(alpha[:, None] * clip_vectors(angular, np.deg2rad(45))) * original_r).as_matrix()
    targets = (1 - alpha[:, None]) * baseline_fingers + alpha[:, None] * baseline_fingers[anchor]
    lo, hi = model.actuator_ctrlrange.T
    targets = np.clip(targets, lo + margin, hi - margin)
    trace = ctrl.replay(model, wrist, targets, times, report["wrist"])
    if return_trace:
        return wrist, targets, trace
    return objective(trace, source, hand)


def refine(parent, source_report, protocol, out, rounds, transfer):
    start = perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    report = json.loads(parent.read_bytes())
    model_path = verify_file_pin(report["mjb"])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    with np.load(verify_file_pin(report["trace"]), allow_pickle=False) as archive:
        inherited = dict(archive)
    hand, source = inherited["source_right_hand21"], inherited["world_from_carton_source"]
    base = inherited["target_wrist"]
    wrist, best = base, inherited
    fingers = inherited["target_finger_q"]
    limits, times = report["wrist"], inherited["source_nominal_time_s"]
    winner = objective(best, source, hand)
    trials = [dict(stage=-1, parent=True, **winner)]
    selection_reruns = 0
    if transfer:
        settings = [(anchor, onset, stop, cap, margin) for anchor in (55, 68, 80)
                    for onset, stop in ((12, 104), (18, 115))
                    for cap, margin in ((.65,.025), (1.,.025), (1.,.07))]
        with ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(carry_candidate, parent, *setting) for setting in settings]
            scores = [future.result() for future in futures]
        for setting, quality in zip(settings, scores):
            trials.append(dict(stage="transfer", settings=list(setting), **quality))
        selected = min(range(len(scores)), key=lambda i: scores[i]["score"])
        if scores[selected]["score"] < winner["score"]:
            anchor, onset, stop, cap, margin = settings[selected]
            wrist, fingers, best = carry_candidate(parent, anchor, onset, stop, cap, margin, return_trace=True)
            winner = objective(best, source, hand)
            selection_reruns += 1
    for stage in range(rounds):
        previous_wrist = wrist
        previous_object = best["world_from_carton_sim"]
        improved = False
        for gain, end, lookahead in ((.3, 112, 0), (.6, 112, 0), (.9, 112, 0), (.6, 124, 2)):
            proposed = candidate(previous_wrist, previous_object, source, base, gain, end, lookahead)
            tr = ctrl.replay(model, proposed, fingers, times, limits)
            quality = objective(tr, source, hand)
            trials.append(dict(stage=stage, gain=gain, end=end, lead=lookahead, **quality))
            if quality["score"] < winner["score"] - 1e-5:
                wrist, best, winner = proposed, tr, quality
                improved = True
        if not improved:
            break
    replay_error = ctrl.audit_commands(mujoco.MjModel.from_binary_path(str(model_path)), best)
    tr = dict(inherited)
    tr.update(best)
    tr["target_wrist"] = wrist
    tr["target_finger_q"] = fingers
    for key in ("ik_tips", "ik_tip_targets", "pad_ik_cost", "ik_nfev"):
        tr.pop(key, None)
    np.savez_compressed(out / "trace.npz", **tr)
    write_json(out / "trials.json", trials)
    output = {k: report[k] for k in ["mjb", "wrist", "table", "nominal_fps", "source_report", "physics_dt_s"]}
    output.update(
        method="Source-conditioned wrist and finger feedforward refinement on the same development demonstration",
        source_sensor_timestamps_available=False, substeps_per_frame=int(np.diff(tr["frame_end_substep"])[0]),
        source_code=[file_pin(__file__), file_pin(ctrl.__file__)], parent=file_pin(parent),
        trace=file_pin(out / "trace.npz"), trials=file_pin(out / "trials.json"), best_development=winner,
        rollout_trials=len(trials)-1, selection_reruns=selection_reruns,
        new_simulator_steps_including_replay=len(tr["substep_finger_ctrl"])*(len(trials)+selection_reruns+1),
        independent_substep_action_replay_max_state_error=replay_error,
        external_object_wrench_or_reset_after_initialization=False,
        run_wall_seconds=perf_counter()-start, cost_scope="Includes search and command replay. Warm start cost separate.",
    )
    write_json(out/"report.json", output)
    evaluate(out/"report.json", source_report, protocol, out/"phase_evaluation.json")
    print(json.dumps(winner))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--transfer-grasp", action="store_true")
    args = parser.parse_args()
    refine(args.parent, args.source, args.protocol, args.output, args.rounds, args.transfer_grasp)
