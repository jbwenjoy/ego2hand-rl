"""CPU validation of an interpretable state-only contact geometry proxy.

Calibrate only SDF bandwidth on whole reset-seed validation groups. Report
static contact semantics, algebraic derivatives and a separate frozen-WM /
native-physics action diagnostic. The last is cross-backend evidence, never a
same-backend physical-gradient qualification or automatic actor acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mujoco
import numpy as np
import torch

from chord_study.contact_geometry import ContactGeometryProxy
from chord_study.contact_model import grouped_split, frozen_support_objective
from chord_study.world_model import FrozenPenWorldModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_state64(sampler):
    d = sampler.data
    inv = d.xmat[sampler.palm_body].reshape(3, 3).T
    rotation = d.xmat[sampler.object_body].reshape(3, 3)
    return np.r_[d.qpos[sampler.qadr], d.qvel[sampler.vadr], inv@(d.xpos[sampler.object_body]-d.xpos[sampler.palm_body]),
                 (inv@rotation).reshape(-1), inv@d.qvel[sampler.object_vadr:sampler.object_vadr+3],
                 inv@rotation@d.qvel[sampler.object_vadr+3:sampler.object_vadr+6]]


def classify_scores(scores, labels):
    scores, labels = np.asarray(scores), np.asarray(labels).astype(bool)
    predicted = scores >= .5
    tp, fp, fn = int((predicted & labels).sum()), int((predicted & ~labels).sum()), int((~predicted & labels).sum())
    absent = ~labels.any(-1)
    return {"micro_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "accuracy": float(np.mean(predicted == labels)), "soft_strength_brier": float(np.mean((scores-labels)**2)),
            "any_contact_presence_recall": float(predicted.any(-1)[labels.any(-1)].mean()) if labels.any() else None,
            "no_contact_samples": int(absent.sum()),
            "no_contact_any_false_presence_fraction": float(predicted.any(-1)[absent].mean()) if absent.any() else None,
            "per_group_positive_rate": labels.mean(0).tolist(),
            "semantics": "distance-based soft strength; not a measured force or calibrated posterior"}


def score_static(support, presence, truth_support, labels):
    absent = ~labels.astype(bool).any(-1)
    return {"support_rmse": float(np.sqrt(np.mean((support-truth_support)**2))),
            "support_constant_mean_rmse_placeholder": None,
            "no_contact_support_rmse": float(np.sqrt(np.mean(support[absent]**2))) if absent.any() else None,
            "no_contact_support_max": float(support[absent].max()) if absent.any() else None,
            "presence": classify_scores(presence, labels)}


def array_forward(proxy, states, bandwidth, *, batch=16):
    support, presence = [], []
    with torch.no_grad():
        for start in range(0, len(states), batch):
            result = proxy(torch.tensor(states[start:start+batch], dtype=torch.float64), bandwidth)
            support.append(result["support"].numpy())
            presence.append(result["presence"].numpy())
    return np.concatenate(support), np.concatenate(presence)


def geometry_derivatives(proxy, states, teacher, bandwidth, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for state in states[np.linspace(0, len(states)-1, min(8, len(states)), dtype=int)]:
        value = torch.tensor(state[None], dtype=torch.float64, requires_grad=True)
        score = frozen_support_objective(proxy(value, bandwidth)["support"], teacher).sum()
        gradient = torch.autograd.grad(score, value)[0].detach().numpy()[0]
        for _ in range(4):
            direction = np.zeros(58)
            direction[:20] = rng.normal(size=20)
            direction[:20] /= np.linalg.norm(direction[:20])
            direction[40:43] = rng.normal(size=3)*.01
            epsilon = 1e-5
            with torch.no_grad():
                plus = frozen_support_objective(proxy(torch.tensor((state+epsilon*direction)[None]), bandwidth)["support"], teacher).item()
                minus = frozen_support_objective(proxy(torch.tensor((state-epsilon*direction)[None]), bandwidth)["support"], teacher).item()
            automatic = float(gradient@direction)
            finite = (plus-minus)/(2*epsilon)
            rows.append({"autograd": automatic, "central_difference": finite, "absolute_error": abs(automatic-finite),
                         "within_tolerance": abs(automatic-finite) <= 2e-4+.01*max(abs(automatic), abs(finite))})
    return {"scope": "proxy algebra only; not a physical action-gradient validation", "comparisons": len(rows),
            "fraction_within_tolerance": float(np.mean([row["within_tolerance"] for row in rows])),
            "max_absolute_error": max(row["absolute_error"] for row in rows), "rows": rows}


def action_chain(proxy, wm, sampler, data, indices, teacher, bandwidth, seed):
    rng = np.random.default_rng(seed)
    rows, state_rows = [], []

    def physical(snapshot, action):
        support, _, _ = sampler.step(action, snapshot=snapshot)
        return -float(np.mean((support-teacher.numpy())**2)), native_state64(sampler)

    for index in indices:
        snapshot = data["integration_snapshot"][index]
        executed = data["executed_target"][index]
        mujoco.mj_setState(sampler.model, sampler.data, snapshot, sampler.state_spec)
        mujoco.mj_forward(sampler.model, sampler.data)
        initial = native_state64(sampler)
        baseline_score, next_true = physical(snapshot, executed)
        action = torch.tensor(executed[None], dtype=torch.float32, requires_grad=True)
        predicted, self_collision_logits = wm(torch.tensor(initial[None], dtype=torch.float32), action)
        wm_score = frozen_support_objective(proxy(predicted.double(), bandwidth)["support"], teacher).sum()
        wm_gradient = torch.autograd.grad(wm_score, action)[0].detach().numpy()[0].astype(float)
        true_input = torch.tensor(next_true[None], dtype=torch.float64, requires_grad=True)
        geometric_score = frozen_support_objective(proxy(true_input, bandwidth)["support"], teacher).sum()
        geometric_gradient = torch.autograd.grad(geometric_score, true_input)[0].detach().numpy()[0]
        directions = rng.normal(size=(8, 20))
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        for direction in directions:
            epsilon = .003
            plus, next_plus = physical(snapshot, executed+epsilon*direction)
            minus, next_minus = physical(snapshot, executed-epsilon*direction)
            half_plus, _ = physical(snapshot, executed+.5*epsilon*direction)
            half_minus, _ = physical(snapshot, executed-.5*epsilon*direction)
            actual, half = (plus-minus)/(2*epsilon), (half_plus-half_minus)/epsilon
            rows.append({"sample": int(index), "reset_seed": int(data["group_seed"][index]),
                         "wm_geometry_derivative": float(wm_gradient@direction),
                         "oracle_transition_geometry_derivative": float(geometric_gradient@((next_plus-next_minus)/(2*epsilon))),
                         "native_actual_contact_derivative": actual, "native_half_epsilon_derivative": half})
        norm = np.linalg.norm(wm_gradient)
        candidate = executed+.02*wm_gradient/max(norm, 1e-12)
        candidate_score, _ = physical(snapshot, candidate)
        pred_state = predicted.detach().numpy()[0]
        state_rows.append({"sample": int(index), "reset_seed": int(data["group_seed"][index]),
                           "actual_base_objective": baseline_score, "actual_candidate_objective": candidate_score,
                           "actual_improvement": candidate_score-baseline_score, "wm_action_gradient_norm": float(norm),
                           "wm_joint_rmse_rad": float(np.sqrt(np.mean((pred_state[:20]-next_true[:20])**2))),
                           "wm_object_position_error_m": float(np.linalg.norm(pred_state[40:43]-next_true[40:43])),
                           "ignored_self_collision_logit_dimension": int(self_collision_logits.shape[-1])})
    actual = np.asarray([row["native_actual_contact_derivative"] for row in rows])
    half = np.asarray([row["native_half_epsilon_derivative"] for row in rows])
    stable = (np.abs(actual) > 1e-4) & (np.sign(actual) == np.sign(half))
    stable &= np.abs(actual-half)/np.maximum(np.maximum(np.abs(actual), np.abs(half)), 1e-4) <= .25
    metrics = {}
    for label, key in (("wm_plus_geometry", "wm_geometry_derivative"), ("oracle_transition_plus_geometry", "oracle_transition_geometry_derivative")):
        predicted = np.asarray([row[key] for row in rows])
        correlation = float(np.corrcoef(predicted[stable], actual[stable])[0, 1]) if stable.sum() > 2 and np.std(predicted[stable]) > 1e-12 else None
        metrics[label] = {"stable_nonflat_sign_agreement": float(np.mean(predicted[stable]*actual[stable] > 0)) if stable.any() else None,
                          "stable_nonflat_pearson": correlation}
    return {"scope": "WM trained in Warp, compared with native MuJoCo interventions: cross-backend/OOD diagnostic only",
            "same_backend_physical_gradient_qualification": False,
            "oracle_scope": "offline isolation of geometry semantics using true transition differences; not an inference input or deployable WM",
            "states": len(indices), "directions": len(rows), "stable_nonflat_directions": int(stable.sum()),
            "metrics": metrics, "wm_recommended_step_l2_rad": .02,
            "actual_improved_state_fraction": float(np.mean([row["actual_improvement"] > 1e-7 for row in state_rows])),
            "mean_actual_improvement": float(np.mean([row["actual_improvement"] for row in state_rows])),
            "state_rows": state_rows, "directional_rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs" / "contact_model_cpu_v4_pointval" / "paired_transitions.npz")
    parser.add_argument("--teacher", type=Path, default=ROOT / "outputs" / "contact_model_cpu_v4_pointval" / "frozen_teacher.npz")
    parser.add_argument("--source", type=Path, default=ROOT / "outputs" / "a0_clip001_3s_seed0")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "outputs" / "a1_pilot_clip007_f0200_2000iter_seed0" / "compiled" / "train" / "model_300.pt")
    parser.add_argument("--context", type=Path, default=ROOT / "outputs" / "a1_pilot_clip007_f0200_2000iter_seed0" / "compiled" / "fixed_physics_context.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "contact_geometry_cpu_v1")
    parser.add_argument("--points-per-geom", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("choose a fresh geometry diagnostic output directory")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(2)
    with np.load(args.dataset, allow_pickle=False) as raw:
        data = {name: raw[name].copy() for name in raw.files}
    with np.load(args.teacher, allow_pickle=False) as raw:
        teacher = torch.tensor(raw["support"].copy(), dtype=torch.float64)
    spec = importlib.util.spec_from_file_location("readonly_native_sampler", ROOT / "scripts" / "train_contact_model.py")
    collection_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collection_module)
    sampler = collection_module.NativeSampler(args.source.resolve(), len(teacher), 11)
    sampler.basis = torch.tensor(data["support_basis"], dtype=torch.float32)
    proxy = ContactGeometryProxy(sampler.model, sampler.meta["joint_names"], data["support_basis"], sampler.radius,
                                  points_per_geom=args.points_per_geom, dtype=torch.float64).cpu()
    bases = np.flatnonzero(data["is_base"])
    split = grouped_split(data["group_seed"][bases], validation_groups=4, test_groups=4)
    protocol = {"scope": "static state-only geometric proxy, followed by separately labeled action-chain diagnostics",
                "split_unit": "whole reset seed; one source crop only", "points_per_geom": args.points_per_geom,
                "bandwidth_grid_m": [.0001, .00025, .0005, .001, .002, .004],
                "bandwidth_selection": "validation support MSE + .25*per-group soft-strength Brier; test untouched",
                "contact_presence_threshold": .5, "max_cpu_wall_budget_s": 120,
                "actor_qualification": "not established by static calibration or algebraic gradients; same-backend WM intervention gate still required"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2)+"\n", encoding="utf-8")
    states, label_support, label_presence = [], [], []
    fk_position_error, fk_rotation_error, restored_support_error = 0., 0., 0.
    for index in bases:
        support, active, _ = sampler.step(data["executed_target"][index], snapshot=data["integration_snapshot"][index])
        states.append(native_state64(sampler))
        label_support.append(support)
        label_presence.append(active)
        restored_support_error = max(restored_support_error, float(np.abs(support-data["next_support"][index]).max()))
        if len(states) % 8 == 0:
            with torch.no_grad():
                p, r = proxy.fk(torch.tensor(states[-1][None, :20], dtype=torch.float64))
            inv = sampler.data.xmat[sampler.palm_body].reshape(3, 3).T
            truth_p = (sampler.data.xpos[proxy.fk.body_ids]-sampler.data.xpos[sampler.palm_body])@inv.T
            truth_r = inv@sampler.data.xmat[proxy.fk.body_ids].reshape(-1, 3, 3)
            fk_position_error = max(fk_position_error, float(np.abs(p.numpy()[0]-truth_p).max()))
            fk_rotation_error = max(fk_rotation_error, float(np.abs(r.numpy()[0]-truth_r).max()))
    states, label_support, label_presence = np.asarray(states), np.asarray(label_support), np.asarray(label_presence)
    calibration = []
    for bandwidth in protocol["bandwidth_grid_m"]:
        if time.perf_counter()-started > 85:
            raise TimeoutError("calibration exhausted bounded CPU budget")
        support, presence = array_forward(proxy, states[split["validation"]], bandwidth)
        mse = float(np.mean((support-label_support[split["validation"]])**2))
        brier = float(np.mean((presence-label_presence[split["validation"]])**2))
        calibration.append({"bandwidth_m": bandwidth, "validation_support_mse": mse, "validation_strength_brier": brier,
                            "selection_score": mse+.25*brier})
    bandwidth = min(calibration, key=lambda row: row["selection_score"])["bandwidth_m"]
    support, presence = array_forward(proxy, states, bandwidth)
    static = {name: score_static(support[mask], presence[mask], label_support[mask], label_presence[mask]) for name, mask in split.items()}
    constant = label_support[split["train"]].mean(0)
    for name, mask in split.items():
        static[name]["source_states"] = int(mask.sum())
        static[name]["reset_seeds"] = np.unique(data["group_seed"][bases][mask]).tolist()
        static[name]["support_constant_train_mean_rmse"] = float(np.sqrt(np.mean((label_support[mask]-constant)**2)))
        static[name].pop("support_constant_mean_rmse_placeholder")
    algebra = geometry_derivatives(proxy, states[split["test"]], teacher, bandwidth, 70217)
    wm = FrozenPenWorldModel(args.checkpoint, args.context, device="cpu")
    if wm.joint_names != sampler.meta["joint_names"]:
        raise ValueError("WM and collision FK use different joint ordering")
    heldout = bases[split["test"]]
    action_indices = heldout[np.linspace(0, len(heldout)-1, min(8, len(heldout)), dtype=int)]
    chain = action_chain(proxy, wm, sampler, data, action_indices, teacher, bandwidth, 31771)
    np.savez_compressed(args.output / "static_predictions.npz", state58=states, support_prediction=support,
                        presence_strength=presence, actual_support=label_support, actual_contact_active=label_presence,
                        original_dataset_indices=bases, **{f"split_{name}": mask for name, mask in split.items()})
    torch.save({"state_dict": proxy.state_dict(), "bandwidth_m": bandwidth, "metadata": proxy.metadata}, args.output / "geometry_proxy.pt")
    report = {"scope": protocol["scope"], "device": "cpu", "mujoco": mujoco.__version__, "torch": torch.__version__,
              "protocol": protocol, "asset_metadata": proxy.metadata, "model_sha256": digest(args.source / "model.mjb"),
              "dataset_sha256": digest(args.dataset), "teacher_sha256": digest(args.teacher),
              "calibration": calibration, "selected_bandwidth_m": bandwidth, "static_state_metrics": static,
              "native_fk_parity": {"max_position_error_m": fk_position_error, "max_rotation_component_error": fk_rotation_error,
                                   "all_palm_finger_bodies_checked_every_8th_state": True},
              "native_restore_support_max_error": restored_support_error, "geometry_algebraic_derivatives": algebra,
              "frozen_wm": wm.metadata, "action_chain_diagnostic": chain,
              "actor_eligible": False, "same_backend_action_gradient_validated": False,
              "limitations": ["fixed random surface sampling can miss narrow contacts or create proximity ghosts",
                              "compiled mesh convex hull, not appearance mesh, is sampled; sampling remains an approximation",
                              "analytic proximity cannot identify solver-active force, load or frictional slip by itself",
                              "one source crop and few held-out reset seeds; no ego or cross-object claim",
                              "WM was trained on a different reference in Warp; native intervention results mix backend and coverage errors",
                              "geometric differentiability does not establish useful dynamics gradients"],
              "total_seconds": time.perf_counter()-started}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    (args.output / "contact_geometry_source.py").write_bytes((ROOT / "src" / "chord_study" / "contact_geometry.py").read_bytes())
    (args.output / "validate_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"output": str(args.output), "seconds": report["total_seconds"], "bandwidth_m": bandwidth,
                      "test_support_rmse": static["test"]["support_rmse"], "fk_error_m": fk_position_error,
                      "algebra_fraction_pass": algebra["fraction_within_tolerance"], "chain": chain["metrics"],
                      "actor_eligible": False}), flush=True)


if __name__ == "__main__":
    main()
