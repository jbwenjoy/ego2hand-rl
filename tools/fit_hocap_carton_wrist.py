"""Fit a small set of wrist and release corrections by exact development-clip rollouts."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from evaluate_hocap_carton import evaluate
from refine_hocap_carton_wrist import objective
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.ego import hocap_carton_control as control
from supermimic_research.strict_json import write_json


def load_parent(parent):
    report = json.loads(parent.read_bytes())
    with np.load(verify_file_pin(report["trace"]), allow_pickle=False) as data:
        trace = dict(data)
    return report, trace, mujoco.MjModel.from_binary_path(str(verify_file_pin(report["mjb"])))


def build_targets(trace, parameters, local_pulse=False):
    if local_pulse:
        gain, width, lead, translation = parameters
        source = trace["world_from_carton_source"]
        sim = trace["world_from_carton_sim"]
        err = Rotation.from_matrix(source[:, :3, :3]) * Rotation.from_matrix(sim[:, :3, :3]).inv()
        peak = int(err.magnitude().argmax())
        scale = np.exp(-.5 * ((np.arange(len(source)) - peak + lead) / width)**2)
        scale[[0, -1]] = 0
        rotation = Rotation.from_rotvec(gain * scale[:, None] * err.as_rotvec()[peak])
        wrist = trace["target_wrist"].copy()
        pivot = sim[:, :3, 3]
        shift = source[peak, :3, 3] - sim[peak, :3, 3]
        wrist[:, :3, 3] = pivot + rotation.apply(wrist[:, :3, 3] - pivot) + translation * scale[:, None] * shift
        wrist[:, :3, :3] = (rotation * Rotation.from_matrix(wrist[:, :3, :3])).as_matrix()
        return wrist, trace["target_finger_q"]
    dx, dy, dz, yaw, onset, stop, release_lead = parameters
    wrist = trace["target_wrist"].copy()
    fingers = trace["target_finger_q"].copy()
    box = trace["world_from_carton_source"][:, :3, 3]
    t = np.arange(len(box))
    scale = np.interp(t, [0, onset, onset+12, stop, 140], [0, 0, 1, 1, 0])
    scale = scale**2 * (3 - 2*scale)
    rotation = Rotation.from_rotvec(np.deg2rad(yaw) * scale[:, None] * np.array([0., 0., 1.]))
    wrist[:, :3, 3] = (box + rotation.apply(wrist[:, :3, 3] - box)
                         + scale[:, None] * np.array([dx, dy, dz]) * .001)
    wrist[:, :3, :3] = (rotation * Rotation.from_matrix(wrist[:, :3, :3])).as_matrix()
    time_shift = t + np.interp(t, [0, 105, 114, 132, 140], [0, 0, release_lead, release_lead, 0])
    targets = np.column_stack([np.interp(time_shift, t, joint) for joint in fingers.T])
    return wrist, targets


def init_worker(parent, local_pulse):
    global inputs, pulse
    inputs = load_parent(parent)
    pulse = local_pulse


def run_candidate(params, materials, keep=False, local_pulse=False):
    report, trace, model = materials
    wrist, fingers = build_targets(trace, params, local_pulse)
    actual = control.replay(model, wrist, fingers, trace["source_nominal_time_s"], report["wrist"])
    metrics = objective(actual, trace["world_from_carton_source"], trace["source_right_hand21"])
    score = metrics["score"]
    # Rank with a continuous penalty on the previously fixed development limits.
    for key, limit in [("rot_rms_deg", 7.75), ("max_rot_deg", 15), ("max_joint_deg", 2),
                       ("endpoint_mm", 15), ("finger_rms_mm", 20), ("max_pos_mm", 40), ("pos_rms_mm", 20)]:
        score += 4 * max(0, metrics[key]/limit-1)
    if keep:
        actual.update(target_wrist=wrist, target_finger_q=fingers)
        return actual
    return dict(parameters=params.tolist(), rank=float(score), metrics=metrics)


def worker(params):
    return run_candidate(params, inputs, local_pulse=pulse)


def fit(parent, source_report, protocol, out, batches, count, local_pulse):
    start = perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261121)
    low = np.array([-12., -12., -8., -28., 13., 106., -4.])
    high = np.array([26., 22., 24., 6., 38., 134., 8.])
    if local_pulse:
        low = np.array([.05, 1.5, -2., -.15])
        high = np.array([.85, 8., 5., .65])
    first = qmc.scale(qmc.LatinHypercube(d=len(low), seed=420).random(count), low, high)
    if local_pulse:
        first[:3] = [[0,3,0,0], [.25,4,1,0], [.5,4,2,0]]
    else:
        first[:3] = [[0,0,0,0,24,120,0], [6,3,4,-10,25,123,1], [10,5,8,-15,25,123,3]]
    candidates = first
    rows = []
    with ProcessPoolExecutor(max_workers=6, initializer=init_worker, initargs=(parent,local_pulse)) as pool:
        for batch in range(batches):
            scored = list(pool.map(worker, candidates))
            for entry in scored:
                entry["batch"] = batch
            rows.extend(scored)
            write_json(out / "trials.json", rows)
            elite = sorted(rows, key=lambda row: row["rank"])[:max(4, count//6)]
            values = np.array([row["parameters"] for row in elite])
            stdev = np.maximum(values.std(0), (high-low)*.08/(batch+1)**.5)
            candidates = np.clip(rng.normal(values.mean(0), stdev, (count,len(low))), low, high)
            # Retain exploration around the best individually successful contact schedule.
            candidates[:count//2] = np.clip(rng.normal(values[0], stdev, (count//2,len(low))), low, high)
            print(batch, elite[0]["metrics"], flush=True)
    best = min(rows, key=lambda row: row["rank"])
    materials = load_parent(parent)
    report, trace, model = materials
    actual = run_candidate(np.array(best["parameters"]), materials, keep=True, local_pulse=local_pulse)
    error = control.audit_commands(model, actual)
    keep_keys = ["world_from_carton_source", "source_right_hand21", "source_nominal_time_s"]
    actual.update({k: trace[k] for k in keep_keys})
    np.savez_compressed(out/"trace.npz", **actual)
    info = {k:report[k] for k in ("mjb", "source_report", "wrist", "physics_dt_s", "nominal_fps", "table")}
    info.update(method="Native simulation wrist and release command fit on development clip", local_pulse=local_pulse,
                parent=file_pin(parent), trace=file_pin(out/"trace.npz"), trials=file_pin(out/"trials.json"),
                code=[file_pin(__file__), file_pin(control.__file__)], winner=best, trials_count=len(rows),
                independent_substep_action_replay_max_state_error=error, no_object_actuation=True)
    write_json(out/"report.json", info)
    evaluate(out/"report.json", source_report, protocol, out/"phase_evaluation.json")
    write_json(out/"cost.json", dict(wall_seconds=perf_counter()-start, cpu_workers=6, gpu_hours=0,
               native_steps=(len(rows)+3)*len(actual["substep_finger_ctrl"]),
               includes="All search rollouts, winner extraction, two command replays and qualification. Parent preparation separate."))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ["parent", "source", "protocol", "output"]:
        p.add_argument(key, type=Path)
    p.add_argument("--batches", type=int, default=5)
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--local-pulse", action="store_true")
    a=p.parse_args()
    fit(a.parent, a.source, a.protocol, a.output, a.batches, a.count, a.local_pulse)
