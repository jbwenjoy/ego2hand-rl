"""Prespecified absolute-target WM gradient check against matched Warp physics.

All branches restore mjSTATE_INTEGRATION, including warmstart, ctrl and mocap.
No native MuJoCo forward/step supplies physical truth. Native APIs only load
model constants and initialize an integration state before Warp simulation.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from chord_study.geometry import quat_matrix
from chord_study.reference import Reference
from chord_study.world_model import FrozenPenWorldModel
from controlled_pen_rollout import sha256, velocity_at_start

PLAN = {
    "schema": "supermimic_chord_wm_gradient_probe_plan:1",
    "phase_indices": [0, 20, 40, 60, 80, 100, 120, 140],
    "random_directions_per_state": 8, "direction_seed": 441,
    "epsilon_target_l2_rad": [.002, .001], "physics_steps_per_transition": 4,
    "state_signature": "mjSTATE_INTEGRATION", "snapshot_restore_absolute_tolerance": 0.,
    "baseline_replay_position_tolerance": 2e-5, "baseline_replay_velocity_tolerance": 2e-4,
    "baseline_replay_reward_tolerance": 2e-5,
    "reward_difference_floor": 1e-5, "repeat_noise_multiplier": 10.,
    "stable_nonflat_definition": "both epsilon central reward signals exceed max(1e-5,10*baseline repeat noise), signs agree, relative slope disagreement<=0.5",
    "epsilon_relative_slope_disagreement_max": .5,
    "bootstrap": "2000 resamples of whole states (8 clusters), seed=442; directions within state are not independent",
    "objective": "next-frame reference native object terms without dt: 2exp(-(position/.05)^2)+8exp(-(mean endpoint distance/.04)^2), endpoint radius=.1m",
    "intervention": "absolute executed target20, not raw policy action; no EMA or phase state is optimized",
    "wm_suggested_step": "normalize WM objective gradient, step epsilon, clip to hard target limits; record actual displacement",
    "random_step_comparator": "8 seeded positive random directions at each same epsilon, same snapshot; hard-limit violations reported separately",
    "negative_result_policy": "report flat, unstable, and stable nonflat separately; failed restore/replay invalidates physical gradient conclusions",
    "saturation_diagnostics": "record actuator ctrl clamp enable/ranges, effective target differences, baseline force saturation>=99%; do not interpret clipped-input flatness as contact-gradient failure",
}

FIELDS = ("body_mass", "body_inertia", "body_iquat", "geom_size", "geom_pos", "geom_friction", "actuator_gainprm", "actuator_biasprm")


def load_matched_model(model_path, context_path, training_model_path):
    model = mujoco.MjModel.from_binary_path(str(model_path))
    training = mujoco.MjModel.from_binary_path(str(training_model_path))
    if (model.nq, model.nv, model.nu, model.nbody, model.ngeom) != (training.nq, training.nv, training.nu, training.nbody, training.ngeom):
        raise ValueError("A0 and training model dimensions differ")
    for kind, count in ((mujoco.mjtObj.mjOBJ_BODY, model.nbody), (mujoco.mjtObj.mjOBJ_JOINT, model.njnt), (mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)):
        if any(mujoco.mj_id2name(model, kind, i) != mujoco.mj_id2name(training, kind, i) for i in range(count)):
            raise ValueError("A0/training model indexing differs")
    for key in ("timestep", "integrator", "cone", "solver", "iterations", "ls_iterations", "impratio", "disableflags", "enableflags", "gravity"):
        if not np.allclose(getattr(model.opt, key), getattr(training.opt, key), atol=1e-10):
            raise ValueError(f"training/A0 option differs: {key}")
    # These fixed passive/controller fields are absent from the per-world
    # context dump, so they must agree with the actual training host model.
    for key in ("body_ipos", "dof_damping", "dof_armature", "dof_frictionloss", "jnt_range", "actuator_forcerange", "actuator_gear", "geom_solref", "geom_solimp", "geom_contype", "geom_conaffinity"):
        if not np.allclose(getattr(model, key), getattr(training, key), rtol=1e-6, atol=1e-9):
            raise ValueError(f"unrecorded physical field differs: {key}")
    report, applied = {}, {}
    with np.load(context_path, allow_pickle=False) as context:
        for key in FIELDS:
            values = context[key]
            expected = getattr(model, key)
            if values.ndim != expected.ndim + 1 or values.shape[1:] != expected.shape or len(values) == 0 or not np.isfinite(values).all():
                raise ValueError(f"invalid fixed physical field {key}")
            if not np.allclose(values, values[0], rtol=0, atol=1e-6):
                raise ValueError(f"physical context is not fixed: {key}")
            report[key] = {"shape": list(values[0].shape), "max_nominal_difference": float(np.abs(expected - values[0]).max())}
            applied[key] = values[0].copy()
    # Preserve the nominal host model's structural compilation, as mjlab does.
    # Actual runtime fields are copied into Warp after put_model below. In
    # particular pen_dimensions reorders principal inertia axes at runtime.
    return model, report, applied


def state58(qpos, qvel, model, joint_ids, wrist):
    qpos, qvel = np.atleast_2d(qpos), np.atleast_2d(qvel)
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    obj = model.joint("object/cube_freejoint").id
    oq, ov = int(model.jnt_qposadr[obj]), int(model.jnt_dofadr[obj])
    pose = qpos[:, oq:oq + 7]
    rotation = quat_matrix(pose[:, 3:])
    inverse = quat_matrix(wrist[3:]).T
    position = (pose[:, :3] - wrist[:3]) @ inverse.T
    rel_rotation = inverse[None] @ rotation
    lin = qvel[:, ov:ov + 3] @ inverse.T
    angular_w = np.einsum("nij,nj->ni", rotation, qvel[:, ov + 3:ov + 6])
    angular = angular_w @ inverse.T
    return np.concatenate((qpos[:, qadr], qvel[:, vadr], position, rel_rotation.reshape(-1, 9), lin, angular), -1).astype(np.float32)


def object_goal(reference_pose, wrist):
    inverse = quat_matrix(wrist[3:]).T
    p = inverse @ (reference_pose[:3] - wrist[:3])
    r = inverse @ quat_matrix(reference_pose[3:])
    return np.r_[p, r.reshape(-1)]


def object_reward(x, goal):
    """Torch, supports either float32 autograd or float64 physical scoring."""
    position, rot = x[..., 40:43], x[..., 43:52].reshape(*x.shape[:-1], 3, 3)
    target_position = goal[..., :3]
    target_rot = goal[..., 3:].reshape(*goal.shape[:-1], 3, 3)
    offset, target_offset = .1 * rot[..., :, 0], .1 * target_rot[..., :, 0]
    point_error = (torch.linalg.vector_norm(position + offset - target_position - target_offset, dim=-1)
                   + torch.linalg.vector_norm(position - offset - target_position + target_offset, dim=-1)) * .5
    return 2 * torch.exp(-((position - target_position).norm(dim=-1) / .05).square()) + 8 * torch.exp(-(point_error / .04).square())


def cluster_interval(values, eligible=None, *, seed=442):
    values = np.asarray(values)
    eligible = np.isfinite(values) if eligible is None else np.asarray(eligible) & np.isfinite(values)
    def value(ids):
        selected = values[ids][eligible[ids]]
        return float(selected.mean()) if len(selected) else np.nan
    estimate = value(np.arange(len(values)))
    rng = np.random.default_rng(seed)
    draws = np.asarray([value(rng.integers(0, len(values), len(values))) for _ in range(2000)])
    draws = draws[np.isfinite(draws)]
    return {"estimate": estimate if np.isfinite(estimate) else None,
            "ci95_cluster_bootstrap": np.quantile(draws, [.025, .975]).tolist() if len(draws) else None,
            "state_clusters": len(values), "eligible_observations": int(eligible.sum()), "valid_bootstrap_draws": len(draws)}


def cluster_correlation(x, y, eligible, seed=442):
    x, y, eligible = np.asarray(x), np.asarray(y), np.asarray(eligible)
    def one(ids):
        mask = eligible[ids]
        a, b = x[ids][mask], y[ids][mask]
        if len(a) < 3 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    estimate = one(np.arange(len(x)))
    rng = np.random.default_rng(seed)
    draws = np.asarray([one(rng.integers(0, len(x), len(x))) for _ in range(2000)])
    draws = draws[np.isfinite(draws)]
    return {"estimate": estimate if np.isfinite(estimate) else None,
            "ci95_cluster_bootstrap": np.quantile(draws, [.025, .975]).tolist() if len(draws) else None,
            "state_clusters": len(x), "eligible_observations": int(eligible.sum()), "valid_bootstrap_draws": len(draws)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu-check-only", action="store_true")
    p.add_argument("--serial", action="store_true", help="restore/step every branch sequentially in the same Warp world; three baseline repetitions")
    args = p.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    owner = {"pid": os.getpid(), "start_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "device": args.device, "argv": sys.argv, "gpu_exclusive_owner": "supermimic_audit world-model-gradient probe"}
    (output / "owner.json").write_text(json.dumps(owner, indent=2), encoding="utf-8")
    began = time.perf_counter()
    stages = []
    def stage(name):
        stages.append({"stage": name, "wall_seconds": time.perf_counter() - began})
        (output / "stages.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
        print(json.dumps(stages[-1]), flush=True)
    stage("cpu_input_validation")
    plan = {**PLAN, "branch_execution": "single_world_serial" if args.serial else "cross_world_batch",
            "baseline_repetitions": 3 if args.serial else 2,
            "checkpoint_sha256": sha256(args.checkpoint), "context_sha256": sha256(args.context),
            "reference_sha256": sha256(args.reference), "model_sha256": sha256(args.model), "script_sha256": sha256(__file__)}
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    reference = Reference.load(args.reference)
    reference.validate()
    model, context_report, physical = load_matched_model(args.model, args.context, args.context.parent / "nominal_model.mjb")
    wm = FrozenPenWorldModel(args.checkpoint, args.context, device="cpu")
    if list(wm.joint_names) != reference.meta["joint_names"]:
        raise ValueError("checkpoint/reference joint ordering differs")
    (output / "context_check.json").write_text(json.dumps({"fields": context_report, "world_model": wm.metadata}, indent=2), encoding="utf-8")
    stage("cpu_interface_loaded")
    if args.cpu_check_only:
        x = torch.zeros(1, 58)
        x[:, 43:52] = torch.eye(3).flatten()
        u = torch.tensor(reference.arrays["joint_pos_rad"][:1], dtype=torch.float32, requires_grad=True)
        pred, collision = wm(x, u)
        gradient = torch.autograd.grad(pred[:, 40:43].sum(), u)[0]
        result = {"cpu_interface_passed": bool(torch.isfinite(pred).all() and torch.isfinite(gradient).all()),
                  "prediction_shape": list(pred.shape), "contact_shape": list(collision.shape),
                  "target_gradient_l2": float(gradient.norm()), "physics_executed": False, "world_model": wm.metadata}
        (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2)); return

    import warp as wp
    import mujoco_warp as mjwarp
    wp.set_device(args.device)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("same-backend physics probe requires Warp CUDA")
    started = time.perf_counter()
    joint_ids = np.asarray([model.joint("robot/" + n).id for n in wm.joint_names])
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    actuator_ids = np.asarray([np.flatnonzero(model.actuator_trnid[:, 0] == jid).item() for jid in joint_ids])
    object_joint = model.joint("object/cube_freejoint").id
    oq, ov = int(model.jnt_qposadr[object_joint]), int(model.jnt_dofadr[object_joint])
    host = mujoco.MjData(model)
    if model.nkey: mujoco.mj_resetDataKeyframe(model, host, model.key("init_state").id)
    else: mujoco.mj_resetData(model, host)
    a = reference.arrays
    wrist = a["wrist_pose_w"][0]
    host.qpos[qadr] = a["joint_pos_rad"][0]
    host.qvel[vadr] = (a["joint_pos_rad"][1] - a["joint_pos_rad"][0]) / .02
    host.qpos[oq:oq + 7] = a["object_pose_w"][0]
    host.qvel[ov:ov + 6] = velocity_at_start(a["object_pose_w"][:2], .02)
    host.ctrl[actuator_ids] = a["joint_pos_rad"][0]
    host.xfrc_applied[:] = 0; host.qfrc_applied[:] = 0
    wp_model = mjwarp.put_model(model)
    stage("warp_nominal_model_created")
    for key, value in physical.items():
        destination = getattr(wp_model, key)
        copied = wp.array(value[None], dtype=destination.dtype, device=args.device)
        wp.copy(destination, copied)
        got = destination.numpy()
        if not np.allclose(got.reshape((-1,) + value.shape)[0], value, rtol=0, atol=1e-7):
            raise ValueError(f"Warp/context field mismatch after put_model: {key}")
    data = mjwarp.put_data(model, host, nworld=1, nconmax=192, njmax=4096)
    mjwarp.set_const_0(wp_model, data)
    stage("warp_actual_context_copied_and_constants_recomputed")
    mjwarp.forward(wp_model, data)
    stage("warp_initial_forward_complete")
    signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state_size = mujoco.mj_stateSize(model, signature)
    state_buffer = wp.zeros((1, state_size), dtype=wp.float32, device=args.device)
    trajectory_snapshots, trajectory_x = [], []
    for frame in range(len(a["time_s"])):
        if frame:
            ctrl = np.zeros((1, model.nu), dtype=np.float32)
            ctrl[:, actuator_ids] = a["joint_pos_rad"][frame - 1]
            wp.copy(data.ctrl, wp.array(ctrl, dtype=wp.float32, device=args.device))
            for _ in range(4): mjwarp.step(wp_model, data)
            mjwarp.forward(wp_model, data)
        mjwarp.get_state(wp_model, data, state_buffer, signature)
        trajectory_snapshots.append(state_buffer.numpy()[0].copy())
        trajectory_x.append(state58(data.qpos.numpy(), data.qvel.numpy(), model, joint_ids, wrist)[0])
    trajectory_snapshots, trajectory_x = np.asarray(trajectory_snapshots), np.asarray(trajectory_x)
    stage("a0_trajectory_and_integration_snapshots_saved")
    phases = np.asarray(PLAN["phase_indices"])
    states, target = trajectory_x[phases], a["joint_pos_rad"][phases].astype(np.float32)
    goals = np.asarray([object_goal(a["object_pose_w"][i + 1], wrist) for i in phases], dtype=np.float32)
    target_tensor = torch.tensor(target, requires_grad=True)
    prediction, _ = wm(torch.from_numpy(states), target_tensor)
    predicted_reward = object_reward(prediction, torch.from_numpy(goals))
    wm_gradient = torch.autograd.grad(predicted_reward.sum(), target_tensor)[0].detach().numpy()
    if not np.isfinite(wm_gradient).all(): raise FloatingPointError("nonfinite world-model objective gradient")
    rng = np.random.default_rng(PLAN["direction_seed"])
    directions = rng.standard_normal((8, 8, 20))
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    epsilons = np.asarray(PLAN["epsilon_target_l2_rad"])
    lower, upper = model.jnt_range[joint_ids].T
    ctrl_limited = model.actuator_ctrllimited[actuator_ids].astype(bool)
    ctrl_ranges = model.actuator_ctrlrange[actuator_ids].copy()
    clamp_enabled = not bool(int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL))
    entries, branch_target = [], []
    for s in range(8):
        for repeat in range(3 if args.serial else 2):
            entries.append((s, "baseline", repeat, 0, 0)); branch_target.append(target[s])
        for e, eps in enumerate(epsilons):
            for direction in range(8):
                for sign in (-1, 1):
                    entries.append((s, "direction", e, direction, sign))
                    branch_target.append(target[s] + sign * eps * directions[s, direction])
            unit = wm_gradient[s] / max(np.linalg.norm(wm_gradient[s]), 1e-12)
            entries.append((s, "gradient", e, 0, 1))
            branch_target.append(np.clip(target[s] + eps * unit, lower, upper))
    branch_target = np.asarray(branch_target, dtype=np.float32)
    branch_source = np.asarray([entry[0] for entry in entries])
    branch_states = trajectory_snapshots[phases][branch_source]
    controls = np.zeros((len(entries), model.nu), dtype=np.float32)
    controls[:, actuator_ids] = branch_target
    if args.serial:
        actual_x_rows, force_rows, restore_errors = [], [], []
        restored = wp.zeros((1, state_size), dtype=wp.float32, device=args.device)
        for j, entry in enumerate(entries):
            saved = wp.array(branch_states[j:j + 1], dtype=wp.float32, device=args.device)
            mjwarp.set_state(wp_model, data, saved, signature)
            mjwarp.get_state(wp_model, data, restored, signature)
            error = float(np.abs(restored.numpy() - branch_states[j:j + 1]).max())
            restore_errors.append(error)
            if error != 0: raise RuntimeError(f"integration state restore is not exact in case {j}: {error}")
            wp.copy(data.ctrl, wp.array(controls[j:j + 1], dtype=wp.float32, device=args.device))
            for _ in range(4): mjwarp.step(wp_model, data)
            mjwarp.forward(wp_model, data)
            actual_x_rows.append(state58(data.qpos.numpy(), data.qvel.numpy(), model, joint_ids, wrist)[0])
            force_rows.append(data.actuator_force.numpy()[0, actuator_ids].copy())
            if j % 37 == 36: stage(f"serial_state_{entry[0]}_complete")
        restore_error = max(restore_errors)
        actual_x, actuator_force = np.asarray(actual_x_rows), np.asarray(force_rows)
        stage("all_serial_restores_exact")
    else:
        branches = mjwarp.put_data(model, host, nworld=len(entries), nconmax=192, njmax=4096)
        saved = wp.array(branch_states, dtype=wp.float32, device=args.device)
        mjwarp.set_state(wp_model, branches, saved, signature)
        restored = wp.zeros((len(entries), state_size), dtype=wp.float32, device=args.device)
        mjwarp.get_state(wp_model, branches, restored, signature)
        restore_error = float(np.abs(restored.numpy() - branch_states).max())
        if restore_error != 0: raise RuntimeError(f"integration state restore is not exact: {restore_error}")
        stage("branch_integration_restore_exact")
        wp.copy(branches.ctrl, wp.array(controls, dtype=wp.float32, device=args.device))
        for _ in range(4): mjwarp.step(wp_model, branches)
        mjwarp.forward(wp_model, branches)
        actual_x = state58(branches.qpos.numpy(), branches.qvel.numpy(), model, joint_ids, wrist)
        actuator_force = branches.actuator_force.numpy()[:, actuator_ids]
    stage("all_branch_physics_complete")
    force_limit = model.actuator_forcerange[actuator_ids, 1]
    effective_target = branch_target.copy()
    if clamp_enabled:
        effective_target[:, ctrl_limited] = np.clip(effective_target[:, ctrl_limited], ctrl_ranges[ctrl_limited, 0], ctrl_ranges[ctrl_limited, 1])
    rewards = object_reward(torch.from_numpy(actual_x).double(), torch.from_numpy(goals[branch_source]).double()).numpy()
    baseline = np.zeros((8, 3 if args.serial else 2)); central_reward = np.zeros((8, 2, 8, 2)); suggested = np.zeros((8, 2))
    baseline_state_error = []
    bounds_ok = np.zeros((8, 2, 8, 2), dtype=bool)
    clipped_input_pairs = np.zeros((8, 2, 8), dtype=bool)
    baseline_force_ratio = np.zeros((8, 20))
    direction_effective = np.zeros((8, 2, 8, 2, 20))
    for j, (s, kind, e, d, sign) in enumerate(entries):
        if kind == "baseline":
            baseline[s, e] = rewards[j]
            baseline_state_error.append(actual_x[j] - trajectory_x[phases[s] + 1])
            baseline_force_ratio[s] = np.abs(actuator_force[j]) / np.maximum(force_limit, 1e-12)
        elif kind == "direction":
            central_reward[s, e, d, int(sign > 0)] = rewards[j]
            ctrl_ok = np.all(effective_target[j] == branch_target[j])
            bounds_ok[s, e, d, int(sign > 0)] = np.all((branch_target[j] >= lower) & (branch_target[j] <= upper)) and ctrl_ok
            direction_effective[s, e, d, int(sign > 0)] = effective_target[j]
        else: suggested[s, e] = rewards[j]
    baseline_state_error = np.asarray(baseline_state_error)
    clipped_input_pairs = np.linalg.norm(direction_effective[..., 1, :] - direction_effective[..., 0, :], axis=-1) < 1e-10
    pos_error = float(np.abs(baseline_state_error[:, np.r_[0:20, 40:52]]).max())
    vel_error = float(np.abs(baseline_state_error[:, np.r_[20:40, 52:58]]).max())
    expected_reward = object_reward(torch.from_numpy(trajectory_x[phases + 1]).double(), torch.from_numpy(goals).double()).numpy()
    reward_error = float(np.abs(baseline - expected_reward[:, None]).max())
    replay_ok = pos_error <= PLAN["baseline_replay_position_tolerance"] and vel_error <= PLAN["baseline_replay_velocity_tolerance"] and reward_error <= PLAN["baseline_replay_reward_tolerance"]
    noise = np.maximum(np.ptp(baseline, axis=1), np.abs(baseline - expected_reward[:, None]).max(-1))
    signal = central_reward[..., 1] - central_reward[..., 0]
    physical_slope = signal / (2 * epsilons[None, :, None])
    wm_slope = np.einsum("sdj,sj->sd", directions, wm_gradient)
    nonflat = np.all(np.abs(signal) > np.maximum(PLAN["reward_difference_floor"], 10 * noise[:, None, None]), axis=1)
    epsilon_sign_agrees = physical_slope[:, 0] * physical_slope[:, 1] > 0
    disagreement = np.abs(physical_slope[:, 0] - physical_slope[:, 1]) / np.maximum(np.abs(physical_slope).max(1), 1e-12)
    stable = nonflat & epsilon_sign_agrees & (disagreement <= PLAN["epsilon_relative_slope_disagreement_max"])
    eligible = stable & bounds_ok.all(axis=(1, 3))
    sign_match = (np.sign(wm_slope) == np.sign(physical_slope[:, 0])).astype(float)
    baseline_mean = baseline.mean(-1)
    suggested_delta = suggested - baseline_mean[:, None]
    random_delta = central_reward[..., 1] - baseline_mean[:, None, None]
    random_valid = bounds_ok[..., 1]
    random_count = random_valid.sum(-1)
    random_mean = (random_delta * random_valid).sum(-1) / np.maximum(random_count, 1)
    random_mean[random_count == 0] = np.nan
    np.savez_compressed(output / "probe.npz", trajectory_integration_snapshots=trajectory_snapshots,
                        trajectory_state58=trajectory_x, baseline_state_error=baseline_state_error,
                        phase_integration_snapshots=trajectory_snapshots[phases], state58=states, targets=target, goals=goals,
                        random_directions=directions, epsilons=epsilons, wm_gradient=wm_gradient,
                        wm_directional_derivative=wm_slope, physical_directional_derivative=physical_slope,
                        central_rewards=central_reward, baseline_rewards=baseline, suggested_rewards=suggested,
                        branch_targets=branch_target, branch_source_state=branch_source, branch_next_state58=actual_x,
                        effective_actuator_targets=effective_target, branch_actuator_force=actuator_force,
                        baseline_force_ratio=baseline_force_ratio, clipped_input_pairs=clipped_input_pairs,
                        nonflat=nonflat, stable_nonflat=stable, eligible_stable_nonflat=eligible, direction_targets_in_bounds=bounds_ok)
    summary = {"schema": "supermimic_chord_wm_gradient_probe:1", "world_model": wm.metadata,
               "physics_engine": "mujoco_warp", "native_physics_truth_used": False,
               "phase_indices": phases.tolist(), "states": 8, "directions_per_state": 8,
               "total_direction_pairs": 64, "branch_cases": len(entries), "branch_worlds": 1 if args.serial else len(entries),
               "branch_execution": "single_world_serial" if args.serial else "cross_world_batch", "baseline_repetitions": baseline.shape[1],
               "state_signature_int": signature,
               "snapshot_size": state_size, "snapshot_restore_max_abs_error": restore_error,
               "baseline_replay_ok": bool(replay_ok), "baseline_replay_position_max_abs": pos_error,
               "baseline_replay_velocity_max_abs": vel_error, "baseline_replay_reward_max_abs": reward_error,
               "baseline_repeat_noise_max": float(noise.max()), "gradient_conclusions_valid": bool(replay_ok),
               "flat_or_below_noise_pairs": int((~nonflat).sum()), "nonflat_pairs": int(nonflat.sum()),
               "epsilon_stable_nonflat_pairs": int(stable.sum()), "eligible_stable_nonflat_pairs": int(eligible.sum()),
               "bound_violating_direction_epsilon_sign_branches": int((~bounds_ok).sum()),
               "actuator_ctrl_clamp_enabled": clamp_enabled, "actuator_ctrl_limited": ctrl_limited.tolist(),
               "actuator_ctrl_ranges": ctrl_ranges.tolist(), "effective_input_flat_pairs_by_epsilon": clipped_input_pairs.sum(axis=(0, 2)).tolist(),
               "baseline_force_saturation_ge99pct_joint_counts": (baseline_force_ratio >= .99).sum(-1).tolist(),
               "baseline_target_outside_joint_limit": np.any((target < lower) | (target > upper), axis=-1).tolist(),
               "stable_nonflat_sign_agreement": cluster_interval(sign_match, eligible),
               "stable_nonflat_pearson_correlation": cluster_correlation(wm_slope, physical_slope[:, 0], eligible),
               "all_nonflat_sign_agreement": cluster_interval(sign_match, nonflat),
               "all_nonflat_pearson_correlation": cluster_correlation(wm_slope, physical_slope[:, 0], nonflat),
               "suggested_step_improvement": {str(eps): {"actual_delta": cluster_interval(suggested_delta[:, i]),
                    "paired_delta_vs_mean_in_bounds_random": cluster_interval(suggested_delta[:, i] - random_mean[:, i]),
                    "random_in_bounds_directions_per_state": random_count[:, i].tolist(),
                    "positive_above_noise_states": int((suggested_delta[:, i] > np.maximum(1e-5, 10 * noise)).sum())}
                    for i, eps in enumerate(epsilons)},
               "absolute_target_intervention_not_policy_gradient": True,
               "limits": "8 clustered A0 states, one checkpoint/seed, short one-step objective; not task success or multi-step policy improvement",
               "plan_sha256": sha256(output / "plan.json"), "raw_probe_sha256": sha256(output / "probe.npz"),
               "wall_seconds": time.perf_counter() - started}
    (output / "branch_index.json").write_text(json.dumps(entries), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    stage("complete")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
