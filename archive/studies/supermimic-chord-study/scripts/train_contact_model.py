"""CPU-only real hand-object contact surrogate collection, fit and gradient audit.

Targets are MuJoCo solver-active hand/object contacts, not self-collision labels.
Independent reset-seed trajectories stay in one split. A nominal one-step
teacher contact fingerprint is frozen before collection; its negative support
MSE is used only as a diagnostic objective, never claimed as CHORD task success.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

# This subprocess must not take the GPU reserved for the actual train agent.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from chord_study.contact_model import (CONTACT_GROUPS, STATE_LAYOUT, ContactSurrogate, calibration_metrics,
                                       frozen_support_objective, grouped_split, pack_palm_state,
                                       qualify_local_proxy, secant_response_loss)
from chord_study.contacts import sample_basis, wrench_support


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_logger():
    path = ROOT / "scripts" / "controlled_pen_rollout.py"
    spec = importlib.util.spec_from_file_location("readonly_contact_logger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeSampler:
    def __init__(self, source, support_dim, basis_seed, selected_reference=None):
        self.source, self.logger = source, load_logger()
        self.model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
        self.data = mujoco.MjData(self.model)
        self.meta = json.loads((source / "rollout_a0.json").read_text(encoding="utf-8"))
        with np.load(source / "rollout_a0.npz", allow_pickle=False) as raw:
            self.logged = {key: raw[key].copy() for key in raw.files}
        with np.load(source / "reference.npz", allow_pickle=False) as raw:
            self.reference = {key: raw[key].copy() for key in raw.files}
        self.selected_reference = selected_reference
        if selected_reference is not None:
            with np.load(selected_reference, allow_pickle=False) as raw:
                self.reference = {key: raw[key].copy() for key in raw.files}
            self.logged["joint_pos_rad"][0] = self.reference["joint_pos_rad"][0]
            self.logged["joint_vel_rad_s"][0] = (self.reference["joint_pos_rad"][1]-self.reference["joint_pos_rad"][0])/.02
            self.logged["object_pose_w"][0] = self.reference["object_pose_w"][0]
        m = self.model
        self.object_body = m.body("object/cube").id
        self.palm_body = m.body("robot/right_palm_link").id
        object_joint = m.joint("object/cube_freejoint").id
        self.object_qadr, self.object_vadr = int(m.jnt_qposadr[object_joint]), int(m.jnt_dofadr[object_joint])
        self.joint_ids = np.asarray([m.joint("robot/"+name).id for name in self.meta["joint_names"]])
        self.qadr, self.vadr = m.jnt_qposadr[self.joint_ids], m.jnt_dofadr[self.joint_ids]
        self.actuators = np.asarray([np.flatnonzero(m.actuator_trnid[:, 0] == joint)[0] for joint in self.joint_ids])
        self.object_geoms = set(np.flatnonzero(m.geom_bodyid == self.object_body).tolist())
        self.robot_geoms = {index for index in range(m.ngeom) if m.geom(index).name.startswith("robot/")}
        self.com_offset = m.body_ipos[self.object_body].copy()
        self.radius = max(float(np.linalg.norm(m.geom_pos[index]-self.com_offset)+m.geom_rbound[index]) for index in self.object_geoms)
        self.basis = sample_basis(support_dim, seed=basis_seed, dtype=torch.float32, device="cpu")
        self.dt = float(self.meta["control_hz"]**-1)
        self.decimation = int(round(self.dt/m.opt.timestep))
        if not np.isclose(self.decimation*m.opt.timestep, self.dt) or m.na != 0:
            raise ValueError("requires exact control clock and no hidden dynamic actuator activation")
        self.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.snapshot_size = mujoco.mj_stateSize(m, self.state_spec)
        self.context, self.context_names, self.context_metadata = self.make_context()

    def make_context(self):
        m = self.model
        fields = [("object_mass_kg", m.body_mass[self.object_body]), ("object_radius_m", self.radius),
                  ("physics_dt_s", m.opt.timestep), ("control_dt_s", self.dt), ("support_mu", .1)]
        for name, values in (("object_inertia_principal", m.body_inertia[self.object_body]),
                             ("object_inertial_quat_wxyz", m.body_iquat[self.object_body]),
                             ("object_com_offset", self.com_offset), ("gravity_w", m.opt.gravity)):
            fields += [(f"{name}_{index}", value) for index, value in enumerate(values)]
        for index, actuator in enumerate(self.actuators):
            fields += [(f"joint{index}_kp", m.actuator_gainprm[actuator, 0]),
                       (f"joint{index}_kd_bias", m.actuator_biasprm[actuator, 2]),
                       (f"joint{index}_force_lower", m.actuator_forcerange[actuator, 0]),
                       (f"joint{index}_force_upper", m.actuator_forcerange[actuator, 1])]
        geoms = sorted(index for index in self.robot_geoms | self.object_geoms if m.geom_contype[index] or m.geom_conaffinity[index])
        for geom in geoms:
            fields += [(f"{m.geom(geom).name}_friction{index}", value) for index, value in enumerate(m.geom_friction[geom])]
        return (np.asarray([value for _, value in fields], dtype=np.float32), [name for name, _ in fields],
                {"scope": "fixed model/context only; no domain-randomization generalization claim",
                 "model_sha256": digest(self.source / "model.mjb"), "joint_names": self.meta["joint_names"],
                 "friction_geom_names": [m.geom(index).name for index in geoms], "mujoco_version": mujoco.__version__})

    def reset(self, seed=None):
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("init_state").id)
        d.qpos[self.qadr] = self.logged["joint_pos_rad"][0]
        d.qvel[self.vadr] = self.logged["joint_vel_rad_s"][0]
        d.qpos[self.object_qadr:self.object_qadr+7] = self.logged["object_pose_w"][0]
        d.qvel[self.object_vadr:self.object_vadr+6] = self.logger.velocity_at_start(self.reference["object_pose_w"], self.dt)
        if seed is not None:
            rng = np.random.default_rng(seed)
            d.qpos[self.qadr] += rng.normal(0, .025, 20)
            d.qvel[self.vadr] += rng.normal(0, .05, 20)
            d.qpos[self.object_qadr:self.object_qadr+3] += rng.normal(0, .0015, 3)
            mujoco.mju_quatIntegrate(d.qpos[self.object_qadr+3:self.object_qadr+7], rng.normal(0, .025, 3), 1.)
            d.qvel[self.object_vadr:self.object_vadr+3] += rng.normal(0, .01, 3)
            d.qvel[self.object_vadr+3:self.object_vadr+6] += rng.normal(0, .05, 3)
        d.ctrl[self.actuators] = self.reference["joint_pos_rad"][0]
        d.qfrc_applied[:] = 0
        d.xfrc_applied[:] = 0
        mujoco.mj_forward(m, d)

    def state(self):
        d = self.data
        return pack_palm_state(d.qpos[self.qadr], d.qvel[self.vadr], d.xpos[self.object_body],
                               d.xmat[self.object_body].reshape(3, 3),
                               d.qvel[self.object_vadr:self.object_vadr+3], d.qvel[self.object_vadr+3:self.object_vadr+6],
                               d.xpos[self.palm_body], d.xmat[self.palm_body].reshape(3, 3))

    def snapshot(self):
        output = np.zeros(self.snapshot_size)
        mujoco.mj_getState(self.model, self.data, output, self.state_spec)
        return output

    def contacts(self):
        c = self.logger.capture_contacts(self.model, self.data, self.object_body, self.object_geoms, self.robot_geoms)
        support = wrench_support(torch.as_tensor(c["pos_o_m"]-self.com_offset, dtype=torch.float32),
                                 torch.as_tensor(c["normal_o"], dtype=torch.float32), torch.as_tensor(c["active"]),
                                 self.basis, self.radius).detach().numpy()
        groups = np.zeros(len(CONTACT_GROUPS), dtype=np.float32)
        for pair, active in zip(c["geom_pair"], c["active"], strict=True):
            if not active:
                continue
            robot = next(int(geom) for geom in pair if int(geom) in self.robot_geoms)
            name = self.model.geom(robot).name
            matched = False
            for finger in range(1, 6):
                if f"right_finger{finger}_" in name:
                    groups[finger-1], matched = 1, True
                    break
            if not matched:
                if "palm" not in name:
                    raise ValueError(f"active hand/object contact has unclassified geometry {name}")
                groups[5] = 1
        return support, groups, c

    def step(self, executed_target, *, snapshot=None):
        if snapshot is not None:
            mujoco.mj_setState(self.model, self.data, snapshot, self.state_spec)
            mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[self.actuators] = executed_target
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("nonfinite native simulation state")
        return self.contacts()


def collect(sampler, args, output):
    started = time.perf_counter()
    sampler.reset()
    teacher_initial = sampler.snapshot()
    teacher_action = sampler.reference["joint_pos_rad"][0].copy()
    teacher_support, teacher_contacts, teacher_geometry = sampler.step(teacher_action)
    np.savez_compressed(output / "frozen_teacher.npz", initial_snapshot=teacher_initial, executed_target=teacher_action,
                        support=teacher_support, contact_active=teacher_contacts, support_basis=sampler.basis.numpy(),
                        contact_pos_com_o_m=teacher_geometry["pos_o_m"]-sampler.com_offset,
                        contact_normal_o=teacher_geometry["normal_o"], geometric_contact_active=teacher_geometry["active"],
                        contact_force_w_N=teacher_geometry["force_w_N"], contact_geom_pair=teacher_geometry["geom_pair"])
    rows = {name: [] for name in ("state", "executed_target", "context", "next_support", "next_contact_active",
                                  "group_seed", "step", "integration_snapshot", "raw_action", "ema_before", "reference_target",
                                  "next_object_pose_w", "contact_count", "penetration_m")}
    maxcontacts, geometries = 0, []
    for trajectory in range(args.trajectories):
        seed = args.seed+trajectory
        rng = np.random.default_rng(seed+190039)
        sampler.reset(seed)
        ema = np.zeros(20)
        for step in range(args.steps_per_trajectory):
            snapshot, state = sampler.snapshot(), sampler.state()
            reference = sampler.reference["joint_pos_rad"][step].copy()
            raw = rng.normal(0, .3, 20)
            before = ema.copy()
            ema = .5*np.clip(.5*raw, -.6, .6)+.5*ema
            target = reference+ema
            support, active, contacts = sampler.step(target)
            geometries.append(contacts)
            row = {"state": state, "executed_target": target, "context": sampler.context,
                   "next_support": support, "next_contact_active": active, "group_seed": seed, "step": step,
                   "integration_snapshot": snapshot, "raw_action": raw, "ema_before": before, "reference_target": reference,
                   "next_object_pose_w": np.r_[sampler.data.xpos[sampler.object_body], sampler.data.xquat[sampler.object_body]],
                   "contact_count": int(contacts["active"].sum()), "penetration_m": contacts["penetration_m"]}
            maxcontacts = max(maxcontacts, len(contacts["active"]))
            for name in rows:
                rows[name].append(row[name])
        if (trajectory+1) % 12 == 0:
            print(json.dumps({"phase": "collect", "trajectories": trajectory+1, "samples": len(rows["state"])}), flush=True)
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    # Keep the independently solved geometry, including inactive close contacts,
    # so a reader can recompute every label without trusting a fitted network.
    for name, tail, dtype in (("pos_o_m", (3,), float), ("normal_o", (3,), float),
                              ("force_w_N", (3,), float), ("geom_pair", (2,), np.int32), ("active", (), bool)):
        values = np.zeros((len(geometries), maxcontacts)+tail, dtype=dtype)
        for index, geometry in enumerate(geometries):
            values[index, :len(geometry[name])] = geometry[name]
        if name == "pos_o_m":
            values -= sampler.com_offset
            arrays["next_contact_pos_com_o_m"] = values
        else:
            arrays["next_geometric_contact_"+name] = values
    arrays["support_basis"] = sampler.basis.numpy()
    with torch.no_grad():
        recomputed = wrench_support(torch.as_tensor(arrays["next_contact_pos_com_o_m"], dtype=torch.float32),
                                    torch.as_tensor(arrays["next_geometric_contact_normal_o"], dtype=torch.float32),
                                    torch.as_tensor(arrays["next_geometric_contact_active"]), sampler.basis, sampler.radius).numpy()
    support_recompute_error = float(np.abs(recomputed-arrays["next_support"]).max())
    if support_recompute_error > 1e-6:
        raise ValueError("saved raw contact geometry cannot reproduce support labels")
    np.savez_compressed(output / "transitions.npz", **arrays)
    return arrays, teacher_support, {"seconds": time.perf_counter()-started, "samples": len(arrays["state"]),
                                     "max_geometric_contacts": maxcontacts,
                                     "support_recomputed_from_saved_geometry_max_abs_error": support_recompute_error,
                                     "positive_contact_fraction": float((arrays["next_contact_active"].sum(-1)>0).mean()),
                                     "max_penetration_m": float(arrays["penetration_m"].max())}


def fit(arrays, args, output):
    started = time.perf_counter()
    split = grouped_split(arrays["group_seed"], validation_groups=args.validation_groups, test_groups=args.test_groups)
    tensors = {key: torch.as_tensor(arrays[key], dtype=torch.float32) for key in ("state", "executed_target", "context", "next_support", "next_contact_active")}
    indices = {key: torch.as_tensor(np.flatnonzero(mask)) for key, mask in split.items()}
    model = ContactSurrogate(tensors["context"].shape[-1], args.support_dim, hidden_dim=args.hidden_dim).cpu()
    model.fit_normalizer(*(tensors[key][indices["train"]] for key in ("state", "executed_target", "context")))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best, best_state = [], float("inf"), None

    def loss_for(which):
        predicted, logits = model(*(tensors[key][which] for key in ("state", "executed_target", "context")))
        support_loss = F.mse_loss(predicted, tensors["next_support"][which])
        contact_loss = F.binary_cross_entropy_with_logits(logits, tensors["next_contact_active"][which])
        return support_loss+contact_loss, support_loss, contact_loss

    train_indices = indices["train"]
    for update in range(args.updates):
        batch = train_indices[torch.randint(len(train_indices), (min(128, len(train_indices)),))]
        loss, _, _ = loss_for(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if update % 50 == 0 or update == args.updates-1:
            with torch.no_grad():
                val_loss, support_loss, contact_loss = loss_for(indices["validation"])
            value = float(val_loss)
            history.append({"update": update, "validation_total": value,
                            "validation_support_mse": float(support_loss), "validation_contact_bce": float(contact_loss)})
            if value < best:
                best, best_state = value, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions, logits = model(tensors["state"], tensors["executed_target"], tensors["context"])
    val = indices["validation"]
    temperatures = torch.logspace(-1, 1, 81)
    losses = [F.binary_cross_entropy_with_logits(logits[val]/temperature, tensors["next_contact_active"][val]).item() for temperature in temperatures]
    temperature = float(temperatures[int(np.argmin(losses))])
    mean_support = tensors["next_support"][train_indices].mean(0)
    metrics = {}
    for name, index in indices.items():
        truth = tensors["next_support"][index]
        no_contacts = tensors["next_contact_active"][index].sum(-1) == 0
        metrics[name] = {"samples": len(index), "reset_seeds": np.unique(arrays["group_seed"][split[name]]).tolist(),
                         "support_rmse": float(torch.sqrt(F.mse_loss(predictions[index], truth))),
                         "constant_train_mean_support_rmse": float(torch.sqrt((truth-mean_support).square().mean())),
                         "zero_support_rmse": float(torch.sqrt(truth.square().mean())),
                         "known_no_contact_samples": int(no_contacts.sum()),
                         "support_rmse_known_no_contact": float(torch.sqrt(predictions[index][no_contacts].square().mean())) if no_contacts.any() else None,
                         "support_max_known_no_contact": float(predictions[index][no_contacts].max()) if no_contacts.any() else None,
                         "contact_uncalibrated": calibration_metrics(logits[index], arrays["next_contact_active"][split[name]]),
                         "contact_calibrated": calibration_metrics(logits[index], arrays["next_contact_active"][split[name]], temperature=temperature)}
    torch.save({"model": model.state_dict(), "context_dim": model.context_dim, "support_dim": model.support_dim,
                "hidden_dim": model.hidden_dim, "temperature": temperature, "training_seed": args.seed,
                "target_claim": "next actual hand/object support and 6 contact groups; not self-collision"}, output / "contact_model.pt")
    np.savez_compressed(output / "holdout_predictions.npz", support_prediction=predictions.numpy(), active_logits=logits.numpy(),
                        **{f"split_{name}": mask for name, mask in split.items()})
    return model, split, {"seconds": time.perf_counter()-started, "updates": args.updates,
                          "best_validation_total_loss": best, "temperature_from_validation_only": temperature,
                          "history": history, "metrics": metrics}


def probe(model, sampler, arrays, split, teacher, args, output):
    started = time.perf_counter()
    candidates = np.flatnonzero(split["test"])
    # Deterministic coverage of the entire held-out reset-seed/step range.
    chosen = candidates[np.linspace(0, len(candidates)-1, min(args.probe_states, len(candidates)), dtype=int)]
    target = torch.tensor(teacher)
    rng = np.random.default_rng(args.seed+8821)
    rows, all_predicted, all_actual, half_actual = [], [], [], []
    state_records = []

    def actual_score(snapshot, action):
        support, _, _ = sampler.step(action, snapshot=snapshot)
        return -float(np.mean((support-teacher)**2)), support

    for row_index in chosen:
        state = torch.tensor(arrays["state"][row_index:row_index+1], dtype=torch.float32)
        action = torch.tensor(arrays["executed_target"][row_index:row_index+1], dtype=torch.float32, requires_grad=True)
        context = torch.tensor(arrays["context"][row_index:row_index+1], dtype=torch.float32)
        snapshot = arrays["integration_snapshot"][row_index]
        support, _ = model(state, action, context)
        score = frozen_support_objective(support, target).sum()
        gradient = torch.autograd.grad(score, action)[0].detach().numpy()[0].astype(float)
        base_action = arrays["executed_target"][row_index]
        baseline, support_baseline = actual_score(snapshot, base_action)
        directions = rng.normal(size=(args.probe_directions, 20))
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        for direction in directions:
            plus, _ = actual_score(snapshot, base_action+args.fd_epsilon*direction)
            minus, _ = actual_score(snapshot, base_action-args.fd_epsilon*direction)
            half_plus, _ = actual_score(snapshot, base_action+.5*args.fd_epsilon*direction)
            half_minus, _ = actual_score(snapshot, base_action-.5*args.fd_epsilon*direction)
            predicted = float(gradient@direction)
            derivative, derivative_half = (plus-minus)/(2*args.fd_epsilon), (half_plus-half_minus)/args.fd_epsilon
            all_predicted.append(predicted)
            all_actual.append(derivative)
            half_actual.append(derivative_half)
            rows.append({"sample": int(row_index), "reset_seed": int(arrays["group_seed"][row_index]),
                         "step": int(arrays["step"][row_index]), "predicted_directional_derivative": predicted,
                         "physics_central_difference": derivative, "physics_half_epsilon_difference": derivative_half})
        norm = np.linalg.norm(gradient)
        proposed = base_action+args.improvement_step*gradient/max(norm, 1e-12)
        improved, _ = actual_score(snapshot, proposed)
        random_score, _ = actual_score(snapshot, base_action+args.improvement_step*directions[0])
        with torch.no_grad():
            proposed_support, _ = model(state, torch.tensor(proposed[None], dtype=torch.float32), context)
            predicted_delta = float(frozen_support_objective(proposed_support, target).sum()-score.detach())
        state_records.append({"sample": int(row_index), "reset_seed": int(arrays["group_seed"][row_index]),
                              "step": int(arrays["step"][row_index]), "base_physics_objective": baseline,
                              "proposed_physics_objective": improved, "actual_improvement": improved-baseline,
                              "predicted_improvement": predicted_delta, "random_direction_improvement": random_score-baseline,
                              "model_action_gradient_norm": float(norm),
                              "restore_baseline_support_max_error": float(np.abs(support_baseline-arrays["next_support"][row_index]).max())})
    predicted, actual, half = np.asarray(all_predicted), np.asarray(all_actual), np.asarray(half_actual)
    meaningful = np.abs(actual) > 1e-4
    relative_halving_error = np.abs(actual-half)/np.maximum(np.maximum(np.abs(actual), np.abs(half)), 1e-4)
    stable = meaningful & (np.sign(actual) == np.sign(half)) & (relative_halving_error <= .25)
    correlation = float(np.corrcoef(predicted, actual)[0, 1]) if np.std(predicted) > 1e-12 and np.std(actual) > 1e-12 else None
    stable_correlation = float(np.corrcoef(predicted[stable], actual[stable])[0, 1]) if stable.sum() > 2 and np.std(predicted[stable]) > 1e-12 and np.std(actual[stable]) > 1e-12 else None
    np.savez_compressed(output / "gradient_probe.npz", predicted_directional_derivative=predicted,
                        physics_central_difference=actual, physics_half_epsilon_difference=half,
                        test_sample_indices=chosen, frozen_target_support=teacher)
    # Resample whole held-out reset seeds, never pretend the directions or
    # intervention branches are independent demonstrations.
    row_groups = np.asarray([row["reset_seed"] for row in rows])
    state_groups = np.asarray([row["reset_seed"] for row in state_records])
    group_ids = np.unique(state_groups)
    bootstrap = np.random.default_rng(args.seed+9107)
    agreement_draws, correlation_draws, improvement_draws = [], [], []
    for _ in range(500):
        picked = bootstrap.choice(group_ids, size=len(group_ids), replace=True)
        directional_index = np.concatenate([np.flatnonzero((row_groups == group) & stable) for group in picked])
        state_index = np.concatenate([np.flatnonzero(state_groups == group) for group in picked])
        if len(directional_index):
            agreement_draws.append(float(np.mean(predicted[directional_index]*actual[directional_index] > 0)))
            if np.std(predicted[directional_index]) > 1e-12 and np.std(actual[directional_index]) > 1e-12:
                correlation_draws.append(float(np.corrcoef(predicted[directional_index], actual[directional_index])[0, 1]))
        improvement_draws.append(float(np.mean([state_records[index]["actual_improvement"] > 1e-7 for index in state_index])))
    uncertainty = {"resampling_unit": "whole reset_seed, not directions", "heldout_reset_seed_count": len(group_ids), "bootstrap_draws": 500,
                   "limitation": "few reset seeds from one source crop; intervals are local descriptive uncertainty only",
                   "stable_sign_95_percentile_interval": np.quantile(agreement_draws, [.025, .975]).tolist() if agreement_draws else None,
                   "stable_correlation_95_percentile_interval": np.quantile(correlation_draws, [.025, .975]).tolist() if correlation_draws else None,
                   "improved_state_fraction_95_percentile_interval": np.quantile(improvement_draws, [.025, .975]).tolist()}
    return {"seconds": time.perf_counter()-started, "objective": "negative MSE to pre-collected fixed nominal one-step teacher support; diagnostic only",
            "fd_epsilon_rad": args.fd_epsilon, "actual_target_step_l2_rad": args.improvement_step,
            "states": len(chosen), "directions_per_state": args.probe_directions, "directional_comparisons": len(rows),
            "pearson_directional_derivatives_all": correlation, "meaningful_physics_derivatives": int(meaningful.sum()),
            "sign_agreement_nonflat": float(np.mean(predicted[meaningful]*actual[meaningful] > 0)) if meaningful.any() else None,
            "epsilon_halving_physics_sign_agreement_nonflat": float(np.mean(np.sign(actual[meaningful]) == np.sign(half[meaningful]))) if meaningful.any() else None,
            "stable_direction_definition": "abs(physics derivative)>1e-4, epsilon/2 sign agrees and relative magnitude change<=.25",
            "epsilon_halving_relative_magnitude_tolerance": .25,
            "flat_physics_direction_count": int((~meaningful).sum()),
            "mean_abs_model_derivative_on_flat_physics": float(np.abs(predicted[~meaningful]).mean()) if (~meaningful).any() else None,
            "sign_agreement_epsilon_stable": float(np.mean(predicted[stable]*actual[stable] > 0)) if stable.any() else None,
            "epsilon_stable_nonflat_direction_count": int(stable.sum()),
            "pearson_epsilon_stable": stable_correlation,
            "mean_actual_improvement": float(np.mean([row["actual_improvement"] for row in state_records])),
            "improved_fraction": float(np.mean([row["actual_improvement"] > 1e-7 for row in state_records])),
            "mean_random_improvement": float(np.mean([row["random_direction_improvement"] for row in state_records])),
            "cluster_uncertainty": uncertainty, "per_state": state_records, "directional_rows": rows}


def collect_paired(sampler, args, output, deadline):
    """Intervene on actual held targets, restoring the full same integration state."""
    started = time.perf_counter()
    sampler.reset()
    teacher_snapshot = sampler.snapshot()
    teacher_action = sampler.reference["joint_pos_rad"][0].copy()
    teacher, teacher_groups, teacher_geometry = sampler.step(teacher_action)
    np.savez_compressed(output / "frozen_teacher.npz", initial_snapshot=teacher_snapshot, executed_target=teacher_action,
                        support=teacher, contact_active=teacher_groups, support_basis=sampler.basis.numpy(),
                        contact_pos_com_o_m=teacher_geometry["pos_o_m"]-sampler.com_offset,
                        contact_normal_o=teacher_geometry["normal_o"], geometric_contact_active=teacher_geometry["active"],
                        contact_force_w_N=teacher_geometry["force_w_N"], contact_geom_pair=teacher_geometry["geom_pair"])
    names = ("state", "executed_target", "context", "next_support", "next_contact_active", "group_seed", "step",
             "integration_snapshot", "raw_action", "ema_before", "reference_target", "next_object_pose_w",
             "contact_count", "penetration_m", "is_base", "source_state_id")
    rows = {name: [] for name in names}
    geometry_rows, pair_plus, pair_minus, pair_base, pair_direction, pair_epsilon = [], [], [], [], [], []
    lifted_seeds = []

    def append(snapshot, state, target, reference, before, seed, step, is_base, state_id):
        support, active, geometry = sampler.step(target, snapshot=snapshot)
        row = {"state": state, "executed_target": target, "context": sampler.context,
               "next_support": support, "next_contact_active": active, "group_seed": seed, "step": step,
               "integration_snapshot": snapshot,
               "raw_action": 4*(target-reference-.5*before), "ema_before": before, "reference_target": reference,
               "next_object_pose_w": np.r_[sampler.data.xpos[sampler.object_body], sampler.data.xquat[sampler.object_body]],
               "contact_count": int(geometry["active"].sum()), "penetration_m": geometry["penetration_m"],
               "is_base": is_base, "source_state_id": state_id}
        for key in names:
            rows[key].append(row[key])
        geometry_rows.append(geometry)
        return len(rows["state"])-1

    for trajectory in range(32):
        if time.perf_counter() > deadline-15:
            raise TimeoutError("preregistered CPU collection budget exhausted")
        seed = args.seed+trajectory
        rng = np.random.default_rng(seed+190039)
        sampler.reset(seed)
        if trajectory % 4 == 0:
            sampler.data.qpos[sampler.object_qadr+2] += .08
            mujoco.mj_forward(sampler.model, sampler.data)
            lifted_seeds.append(seed)
        ema = np.zeros(20)
        for step in range(4):
            state_id = trajectory*4+step
            snapshot, state = sampler.snapshot(), sampler.state()
            reference = sampler.reference["joint_pos_rad"][step].copy()
            before = ema.copy()
            ema = .5*np.clip(.5*rng.normal(0, .3, 20), -.6, .6)+.5*ema
            target = reference+ema
            base_index = append(snapshot, state, target, reference, before, seed, step, True, state_id)
            continuation = sampler.snapshot()
            directions = rng.normal(size=(args.pair_directions, 20))
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            for direction in directions:
                for epsilon in args.pair_epsilons:
                    plus = append(snapshot, state, target+epsilon*direction, reference, before, seed, step, False, state_id)
                    minus = append(snapshot, state, target-epsilon*direction, reference, before, seed, step, False, state_id)
                    pair_plus.append(plus)
                    pair_minus.append(minus)
                    pair_base.append(base_index)
                    pair_direction.append(direction)
                    pair_epsilon.append(epsilon)
            mujoco.mj_setState(sampler.model, sampler.data, continuation, sampler.state_spec)
            mujoco.mj_forward(sampler.model, sampler.data)
        if (trajectory+1) % 8 == 0:
            print(json.dumps({"phase": "paired_collect", "trajectories": trajectory+1, "transitions": len(rows["state"])}), flush=True)
    arrays = {name: np.asarray(values) for name, values in rows.items()}
    arrays.update(pair_plus_index=np.asarray(pair_plus), pair_minus_index=np.asarray(pair_minus),
                  pair_base_index=np.asarray(pair_base), pair_direction=np.asarray(pair_direction),
                  pair_epsilon=np.asarray(pair_epsilon), support_basis=sampler.basis.numpy())
    maxcontacts = max(len(geometry["active"]) for geometry in geometry_rows)
    for name, tail, dtype in (("pos_o_m", (3,), float), ("normal_o", (3,), float),
                              ("force_w_N", (3,), float), ("geom_pair", (2,), np.int32), ("active", (), bool)):
        values = np.zeros((len(geometry_rows), maxcontacts)+tail, dtype=dtype)
        for index, geometry in enumerate(geometry_rows):
            values[index, :len(geometry[name])] = geometry[name]
        if name == "pos_o_m":
            values -= sampler.com_offset
            arrays["next_contact_pos_com_o_m"] = values
        else:
            arrays["next_geometric_contact_"+name] = values
    # Chunking bounds peak memory without changing the independently saved labels.
    maximum_error = 0.
    for start in range(0, len(geometry_rows), 512):
        sl = slice(start, start+512)
        recomputed = wrench_support(torch.tensor(arrays["next_contact_pos_com_o_m"][sl], dtype=torch.float32),
                                    torch.tensor(arrays["next_geometric_contact_normal_o"][sl], dtype=torch.float32),
                                    torch.tensor(arrays["next_geometric_contact_active"][sl]), sampler.basis, sampler.radius).numpy()
        maximum_error = max(maximum_error, float(np.abs(recomputed-arrays["next_support"][sl]).max()))
    if maximum_error > 1e-6:
        raise ValueError("raw contact geometry does not reproduce paired support labels")
    np.savez_compressed(output / "paired_transitions.npz", **arrays)
    plus, minus = arrays["pair_plus_index"], arrays["pair_minus_index"]
    assert np.array_equal(arrays["group_seed"][plus], arrays["group_seed"][minus])
    assert np.array_equal(arrays["source_state_id"][plus], arrays["source_state_id"][minus])
    derivatives = (arrays["next_support"][plus]-arrays["next_support"][minus])/(2*arrays["pair_epsilon"][:, None])
    return arrays, teacher, {"seconds": time.perf_counter()-started, "transitions": len(arrays["state"]),
                             "source_states": int(arrays["is_base"].sum()), "pairs": len(plus),
                             "lifted_contact_free_diagnostic_seeds": lifted_seeds,
                             "positive_contact_fraction": float((arrays["next_contact_active"].sum(-1)>0).mean()),
                             "max_penetration_m": float(arrays["penetration_m"].max()),
                             "support_from_geometry_max_abs_error": maximum_error,
                             "support_secant_component_abs_quantiles": np.quantile(np.abs(derivatives), [.5, .75, .95, .99, 1]).tolist()}


def fit_paired(arrays, args, output, *, use_secants, deadline):
    started = time.perf_counter()
    output.mkdir()
    split = grouped_split(arrays["group_seed"], validation_groups=4, test_groups=4)
    indices = {name: torch.tensor(np.flatnonzero(mask)) for name, mask in split.items()}
    tensors = {key: torch.as_tensor(arrays[key], dtype=torch.float32) for key in
               ("state", "executed_target", "context", "next_support", "next_contact_active")}
    plus, minus = torch.tensor(arrays["pair_plus_index"]), torch.tensor(arrays["pair_minus_index"])
    epsilon = torch.tensor(arrays["pair_epsilon"], dtype=torch.float32)
    pair_split = {name: torch.tensor(np.flatnonzero(mask[arrays["pair_base_index"]])) for name, mask in split.items()}
    train_pairs = pair_split["train"]
    derivative = (tensors["next_support"][plus[train_pairs]]-tensors["next_support"][minus[train_pairs]])/(2*epsilon[train_pairs, None])
    # Robust, training-only per-basis scaling: flat directions remain supervised
    # as zero, while rare mode-switch secants cannot set the entire loss scale.
    scale = torch.quantile(derivative.abs(), .75, dim=0).clamp_min(.02)
    torch.manual_seed(args.seed+127)
    model = ContactSurrogate(tensors["context"].shape[-1], args.support_dim, args.hidden_dim).cpu()
    model.fit_normalizer(*(tensors[key][indices["train"]] for key in ("state", "executed_target", "context")))
    generator = torch.Generator().manual_seed(args.seed+321)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history, best, best_state = [], float("inf"), None

    def evaluate_loss(point_index, pair_index):
        support, logits = model(*(tensors[key][point_index] for key in ("state", "executed_target", "context")))
        point_mse = F.mse_loss(support, tensors["next_support"][point_index])
        bce = F.binary_cross_entropy_with_logits(logits, tensors["next_contact_active"][point_index])
        both = torch.cat((plus[pair_index], minus[pair_index]))
        pair_support, _ = model(*(tensors[key][both] for key in ("state", "executed_target", "context")))
        count = len(pair_index)
        response = secant_response_loss(pair_support[:count], pair_support[count:],
                                       tensors["next_support"][plus[pair_index]], tensors["next_support"][minus[pair_index]],
                                       epsilon[pair_index], scale)
        return point_mse, bce, response

    updates_run = 0
    for update in range(args.paired_updates):
        if time.perf_counter() > deadline:
            break
        selected = indices["train"][torch.randint(len(indices["train"]), (128,), generator=generator)]
        paired = train_pairs[torch.randint(len(train_pairs), (64,), generator=generator)]
        mse, bce, response = evaluate_loss(selected, paired)
        loss = mse+.25*bce+(args.secant_weight if use_secants else 0.)*response
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        updates_run += 1
        if update % 100 == 0 or update == args.paired_updates-1:
            with torch.no_grad():
                vm, vb, vr = evaluate_loss(indices["validation"], pair_split["validation"])
                # Identical point-validation rule avoids penalizing the MSE
                # baseline using a response term it was never trained to fit.
                value = float(vm+.25*vb+(args.secant_weight*vr if args.paired_validation_selection == "common-response" else 0))
            history.append({"update": update, "validation_common_selection": value, "support_mse": float(vm),
                            "contact_bce": float(vb), "normalized_secant_loss": float(vr)})
            if value < best:
                best, best_state = value, copy.deepcopy(model.state_dict())
    if best_state is None:
        raise TimeoutError("CPU fit budget ended before the first validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted, logits = model(tensors["state"], tensors["executed_target"], tensors["context"])
    temperatures = torch.logspace(-1, 1, 81)
    val = indices["validation"]
    costs = [float(F.binary_cross_entropy_with_logits(logits[val]/value, tensors["next_contact_active"][val])) for value in temperatures]
    temperature = float(temperatures[int(np.argmin(costs))])
    average = tensors["next_support"][indices["train"]].mean(0)
    metrics = {}
    for name, index in indices.items():
        truth = tensors["next_support"][index]
        absent = tensors["next_contact_active"][index].sum(-1) == 0
        pair_index = pair_split[name]
        with torch.no_grad():
            _, _, response = evaluate_loss(index, pair_index)
        metrics[name] = {"samples": len(index), "source_states": int(arrays["is_base"][split[name]].sum()),
                         "reset_seeds": np.unique(arrays["group_seed"][split[name]]).tolist(),
                         "support_rmse": float(torch.sqrt(F.mse_loss(predicted[index], truth))),
                         "constant_train_mean_support_rmse": float(torch.sqrt((truth-average).square().mean())),
                         "known_no_contact_samples": int(absent.sum()),
                         "support_rmse_known_no_contact": float(torch.sqrt(predicted[index][absent].square().mean())) if absent.any() else None,
                         "support_max_known_no_contact": float(predicted[index][absent].max()) if absent.any() else None,
                         "normalized_secant_loss": float(response),
                         "contact_uncalibrated": calibration_metrics(logits[index], arrays["next_contact_active"][split[name]]),
                         "contact_calibrated": calibration_metrics(logits[index], arrays["next_contact_active"][split[name]], temperature=temperature)}
    torch.save({"model": model.state_dict(), "context_dim": model.context_dim, "support_dim": model.support_dim,
                "hidden_dim": model.hidden_dim, "temperature": temperature, "secant_supervision": use_secants,
                "derivative_scale_training_only": scale}, output / "contact_model.pt")
    np.savez_compressed(output / "predictions.npz", support=predicted.numpy(), logits=logits.numpy(),
                        **{f"split_{name}": mask for name, mask in split.items()})
    return model, split, {"seconds": time.perf_counter()-started, "updates_run": updates_run, "updates_budget": args.paired_updates,
                          "same_initialization_seed": args.seed+127, "same_batch_order_seed": args.seed+321,
                          "best_common_validation_score": best, "temperature_validation_only": temperature,
                          "secant_weight": args.secant_weight if use_secants else 0., "history": history, "metrics": metrics}


def run_paired_experiment(args):
    started = time.perf_counter()
    deadline = started+args.max_cpu_seconds
    args.output.mkdir(parents=True)
    protocol = {
        "version": 4, "written_before_collection": True, "device": "CPU only",
        "source_initial_grasp": str(args.selected_reference.resolve()),
        "source_reference_sha256": digest(args.selected_reference),
        "reset_seed_groups": {"train": 24, "validation": 4, "test": 4},
        "states_per_reset_seed": 4, "pair_directions_per_state": args.pair_directions,
        "paired_epsilons_rad": args.pair_epsilons, "paired_transition_count": 32*4*(1+2*args.pair_directions*len(args.pair_epsilons)),
        "contact_free_stress": "every fourth reset seed lifts object 8cm; diagnostic only, no stable-grasp success claim",
        "same_data_same_initialization_same_minibatches_same_updates": True,
        "updates_per_model": args.paired_updates, "secant_weight": args.secant_weight,
        "loss": "support MSE + .25 contact BCE; intervention model adds weighted robust normalized support secant response",
        "validation_selection_common": "support MSE + .25 BCE" if args.paired_validation_selection == "point" else "support MSE + .25 BCE + secant_weight*normalized secant loss",
        "protocol_amendment": "point-validation replicate uses a fresh reset-seed range; prior common-response v4 results are retained and qualification thresholds unchanged" if args.paired_validation_selection == "point" else None,
        "stable_gradient_direction_definition": "abs derivative>1e-4, epsilon-halving sign agrees and relative magnitude change<=25%",
        "frozen_objective": "negative support MSE to independent nominal one-step selected-grasp teacher collected before interventions",
        "qualification": {"stable_nonflat_direction_count_min": 16, "stable_nonflat_sign_min": .75,
                          "stable_nonflat_correlation_min": .5, "test_states_min": 8,
                          "actual_improvement_fraction_strictly_greater_than": .5,
                          "known_no_contact_support_rmse_max": .10, "known_no_contact_support_max": .25},
        "qualification_limit": "local candidate status only, no actor use unless qualified; no CHORD/ego task success inference",
        "max_cpu_wall_seconds": args.max_cpu_seconds,
    }
    (args.output / "preregistered_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    # Freeze executable evidence because this sidecar has no Git history yet.
    (args.output / "train_contact_model_source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "contact_model_source.py").write_bytes((ROOT / "src" / "chord_study" / "contact_model.py").read_bytes())
    torch.set_num_threads(args.cpu_threads)
    sampler = NativeSampler(args.source.resolve(), args.support_dim, args.basis_seed, args.selected_reference.resolve())
    arrays, teacher, collection = collect_paired(sampler, args, args.output, deadline)
    models = {}
    for name, use_secants in (("mse_baseline", False), ("secant_supervised", True)):
        # Split the available time evenly; both still have the same update cap.
        remaining_models = 2 if not models else 1
        fit_deadline = time.perf_counter()+max(1., (deadline-time.perf_counter()-5)/remaining_models)
        model, split, fit_report = fit_paired(arrays, args, args.output / name, use_secants=use_secants, deadline=fit_deadline)
        probe_args = copy.copy(args)
        probe_args.probe_states = 16
        test_bases = {"test": split["test"] & arrays["is_base"]}
        physical_probe = probe(model, sampler, arrays, test_bases, teacher, probe_args, args.output / name)
        qualification = qualify_local_proxy(fit_report["metrics"]["test"], physical_probe)
        models[name] = {"fit": fit_report, "physical_gradient_probe": physical_probe, "qualification": qualification}
        print(json.dumps({"phase": name, "qualified": qualification["qualified_as_local_candidate"],
                          "sign": physical_probe["sign_agreement_epsilon_stable"], "correlation": physical_probe["pearson_epsilon_stable"],
                          "improved_fraction": physical_probe["improved_fraction"]}), flush=True)
    report = {"version": 4, "scope": "paired causal hand/object contact surrogate diagnostic, not a static classification success",
              "protocol": protocol, "protocol_sha256": digest(args.output / "preregistered_protocol.json"),
              "python": sys.executable, "mujoco": mujoco.__version__, "torch": torch.__version__, "device": "cpu",
              "state_layout": STATE_LAYOUT, "contact_groups": CONTACT_GROUPS,
              "context_names": sampler.context_names, "context_values": sampler.context.tolist(),
              "context_metadata": sampler.context_metadata, "collection": collection, "models": models,
              "dataset_sha256": digest(args.output / "paired_transitions.npz"), "teacher_sha256": digest(args.output / "frozen_teacher.npz"),
              "code_sha256": {"script": digest(__file__), "model": digest(ROOT / "src" / "chord_study" / "contact_model.py")},
              "total_seconds": time.perf_counter()-started,
              "limitations": ["one selected source crop, fixed context and short local interventions",
                              "independent reset-seed holdout does not establish unseen-original-demo generalization",
                              "multi-epsilon secants can straddle a contact switch and need not define a smooth Jacobian",
                              "no actor integration or object-motion success follows from prediction calibration",
                              "lifting object is a deliberate contact-free diagnostic, not a successful manipulation task"]}
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seconds": report["total_seconds"], "transitions": collection["transitions"],
                      "qualified": {name: result["qualification"]["qualified_as_local_candidate"] for name, result in models.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "a0_clip001_3s_seed0")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "contact_model_cpu_v1")
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--trajectories", type=int, default=48)
    parser.add_argument("--steps-per-trajectory", type=int, default=12)
    parser.add_argument("--validation-groups", type=int, default=6)
    parser.add_argument("--test-groups", type=int, default=6)
    parser.add_argument("--support-dim", type=int, default=64)
    parser.add_argument("--basis-seed", type=int, default=11)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1200)
    parser.add_argument("--probe-states", type=int, default=8)
    parser.add_argument("--probe-directions", type=int, default=8)
    parser.add_argument("--fd-epsilon", type=float, default=.003)
    parser.add_argument("--improvement-step", type=float, default=.02)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--paired-v4", action="store_true")
    parser.add_argument("--selected-reference", type=Path, default=ROOT / "outputs" / "initial_grasp_scan" / "substeps_v2" / "selected_reference.npz")
    parser.add_argument("--pair-directions", type=int, default=12)
    parser.add_argument("--pair-epsilons", type=float, nargs="+", default=[.002, .006])
    parser.add_argument("--paired-updates", type=int, default=3000)
    parser.add_argument("--secant-weight", type=float, default=.25)
    parser.add_argument("--max-cpu-seconds", type=float, default=120)
    parser.add_argument("--paired-validation-selection", choices=("point", "common-response"), default="point")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("choose a fresh experiment output directory")
    if args.steps_per_trajectory < 1 or args.fd_epsilon <= 0 or args.improvement_step <= 0:
        raise ValueError("positive step count and perturbations required")
    if args.paired_v4:
        run_paired_experiment(args)
        return
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    sampler = NativeSampler(args.source.resolve(), args.support_dim, args.basis_seed)
    if args.steps_per_trajectory >= len(sampler.reference["joint_pos_rad"]):
        raise ValueError("collection reference is too short")
    arrays, teacher, collection = collect(sampler, args, args.output)
    model, split, fit_result = fit(arrays, args, args.output)
    gradient = probe(model, sampler, arrays, split, teacher, args, args.output)
    report = {
        "scope": "CPU real hand-object supervised proxy and independent local physical-gradient diagnostic",
        "status": "pipeline_executed; assess calibration and physical-gradient results independently",
        "python": sys.executable, "torch": torch.__version__, "mujoco": mujoco.__version__, "device": "cpu",
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "state_layout": STATE_LAYOUT, "contact_groups": CONTACT_GROUPS,
        "context_names": sampler.context_names, "fixed_context_values": sampler.context.tolist(),
        "context_metadata": sampler.context_metadata,
        "control_contract": {"alpha": .5, "residual": ".5*clip(.5*raw_action,-.6,.6)+.5*previous_residual",
                             "target": "source_qref+residual, actual held target is model input; no actuator dynamic state",
                             "recorded_memory": ["raw_action", "ema_before", "reference_target", "executed_target"],
                             "object_assistance": "none", "state_resets": "only at start of each explicitly grouped short trajectory"},
        "teacher": {"file": "frozen_teacher.npz", "sha256": digest(args.output / "frozen_teacher.npz"),
                    "source": "independent nominal one-step native-physics transition collected before perturbation data",
                    "claim": "fixed diagnostic contact target, not a successful manipulation expert or an ego demonstration",
                    "support_nonzero_directions": int((teacher > 1e-3).sum())},
        "split_policy": "whole independent reset_seed trajectories; all seeds derive from ONE original clip, no cross-original-demo claim",
        "collection": collection, "fit": fit_result, "physical_gradient_probe": gradient,
        "actor_integration_decision": "not supported by this diagnostic run; inspect directional agreement, ghost support and actual improvements before any actor use",
        "dataset_sha256": digest(args.output / "transitions.npz"),
        "code_sha256": {"script": digest(__file__), "contact_model": digest(ROOT / "src" / "chord_study" / "contact_model.py")},
        "total_seconds": time.perf_counter()-started,
        "limitations": ["single fixed physical context and small initial-state neighborhood",
                        "held-out reset seeds are not unseen original demonstrations or objects",
                        "endpoint contacts may switch discontinuously; epsilon-halving stability is reported",
                        "a support head can predict physically nonexistent contacts; no actor integration is authorized by low MSE alone",
                        "fixed contact-match diagnostic does not establish object-motion or CHORD success",
                        "native CPU physics only; do not assume Warp transition or gradient parity"],
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "samples": collection["samples"], "total_seconds": report["total_seconds"],
                      "test_support_rmse": fit_result["metrics"]["test"]["support_rmse"],
                      "derivative_sign_agreement": gradient["sign_agreement_nonflat"],
                      "derivative_pearson": gradient["pearson_directional_derivatives_all"],
                      "actual_mean_improvement": gradient["mean_actual_improvement"]}), flush=True)


if __name__ == "__main__":
    main()
