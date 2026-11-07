"""Fixed-final phone model evaluation. No optimizer, native model or physics.

Protocol/source must be pinned before first TEST array read. Full partial rows
remain saved; only genuine H targets are scored. No checkpoint selection.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent
TRAINER = PILOT / "phone_world_model_train_v1/train_phone_wm_v2.py"
TRAINER_SHA = "1234ea0491edcd23e3d15d32959a39bbac2f851b2514b17b5e2c41d87c5fd9b9"
SCHEMA = "phone_fixed_final5000_evaluation_protocol:1"
H = .02
HORIZONS = [1, 2, 5, 9, 10]
METHODS = ["WM_final5000", "TRAIN_mean", "constant_joint_COM_world_angular_velocity", "continuous_ballistic_COM_constant_joint_angular_velocity"]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path):
    p = Path(path).resolve(strict=True); b = p.read_bytes()
    return {"path": str(p), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def checked(record):
    p = Path(record["path"])
    require(p.is_absolute(), "Absolute input pin")
    a = pin(p)
    require(a["bytes"] == record["bytes"] and a["sha256"] == record["sha256"], "Input/source changed")
    return p.resolve(strict=True)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def preflight(protocol_path):
    p = read(protocol_path)
    require(p["schema"] == SCHEMA and p["primary_checkpoint_iteration"] == 5000
        and p["recursive_horizons"] == HORIZONS and p["methods"] == METHODS
        and p["select_checkpoint_by_TEST"] is False and p["new_training_AD_native_GPU"] is False, "Frozen evaluation scope")
    for r in [*p["inputs"].values(), *p["sources"].values()]:
        checked(r)
    require(checked(p["sources"]["evaluator"]).samefile(__file__) and pin(TRAINER)["sha256"] == TRAINER_SHA,
        "Frozen evaluator/actual trainer")
    terminal = read(checked(p["inputs"]["training_terminal"]))
    train = read(checked(p["inputs"]["training_report"]))
    tp = read(checked(p["inputs"]["training_protocol"]))
    require(terminal["returncode"] == 0 and terminal["protocol"] == train["protocol"] == p["inputs"]["training_protocol"], "Training terminal0/protocol identity")
    for k in ("stdout", "stderr", "runner"):
        checked(terminal[k])
    require(train["status"] == "COMPLETED_FIXED5000_NOT_POLICY_QUALIFICATION" and train["counts"] == {
        "partition_NPZ_reads": 2, "TEST_NPZ_reads": 0, "model_constructors": 1,
        "training_FW": 5000, "backward": 5000, "optimizer_steps": 5000,
        "validation_FW": 3, "checkpoint_readback_FW": 3, "own_checkpoint_loads": 3,
        "physics_Env_GPU_old_checkpoint": 0}, "Exact terminal training budget/no TEST use")
    require([v["iteration"] for v in train["checkpoints"]] == [250, 1000, 5000], "All fixed checkpoints retained")
    for health in train["checkpoints"]:
        checked(health["checkpoint"])
        require(health["status"] == "HEALTHY_SAVED_READBACK_NOT_DYNAMICS_QUALIFICATION"
            and health["exact_save_load_predictions"] is True, "Saved checkpoint health")
    require(train["checkpoints"][-1]["checkpoint"] == p["inputs"]["final_checkpoint"], "Fixed final5000, never best-TEST")
    acq = read(checked(p["inputs"]["acquisition_report"]))
    audit = read(checked(p["inputs"]["dataset_audit"]))
    require(acq["schema"] == "phone_episode_acquisition_result:1" and acq["partition_qualified"] is True
        and acq["cross_partition_alias_groups"] == [], "Frozen qualified family split")
    require(audit["dataset_report"] == p["inputs"]["acquisition_report"]
        and audit["status"] == "PASS_SAVED_ALL_EPISODE_FAMILY_H20MS_STATE_CONTACT_SPLIT_RECEIPTS"
        and audit["engineering_qualified"] is True and all(v is True for v in audit["checks"].values()), "Positive exact dataset review")
    require(tp["inputs"]["dataset_report"] == p["inputs"]["acquisition_report"]
        and acq["family_plan"] == p["inputs"]["family_plan"], "Same trained/acquired data")
    require(acq["outputs"]["test"] == p["inputs"]["TEST_arrays"]
        and acq["sample_identities"] == p["inputs"]["sample_identities"], "Original TEST/identity pins")
    out = Path(p["output"])
    require(out.is_absolute() and out.parent.resolve() == HERE.resolve() and not out.exists(), "Fresh evaluation output")
    return p, train, tp, acq, out


def so3_flags(states):
    x = np.asarray(states, dtype=np.float64)
    R = x[..., 43:52].reshape(*x.shape[:-1], 3, 3)
    finite = np.isfinite(x).all(-1)
    safe = np.where(finite[..., None, None], R, np.eye(3))
    defect = np.maximum(np.max(abs(np.swapaxes(safe, -1, -2) @ safe - np.eye(3)), axis=(-2, -1)), abs(np.linalg.det(safe)-1))
    return finite, finite & (defect <= 1e-5), defect


def full_rotation_error(pred, target):
    """Near-SO3 raw observer: atan2(||vee(Q-Q.T)/2||,(trQ-1)/2).

    Both full body rotations retained; no roll-blind shaft axis or projection.
    Inputs' raw SO3 defects are recorded separately (F32 arithmetic tolerance).
    """
    R = pred[:, 43:52].reshape(-1, 3, 3) @ target[:, 43:52].reshape(-1, 3, 3).transpose(0, 2, 1)
    v = np.stack((R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]), -1)/2
    return np.arctan2(np.linalg.norm(v, axis=-1), (np.trace(R, axis1=1, axis2=2)-1)/2)


def stable_sigmoid(logits):
    a = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0., -a))


def decode_rows_torch(x, d, adapter, torch):
    """Fixed per-row pure decoder policy; a bad row is recorded, never repaired.

    This does not re-run the neural model. Its FP32 source operations are the
    frozen decoder, with every call's reason retained. No dtype/mode fallback.
    """
    out = np.full((len(x), 58), np.nan); reasons = []
    for i in range(len(x)):
        try:
            y = adapter.decode_torch(x[i:i+1], d[i:i+1], state_schema=adapter.STATE_SCHEMA, delta_schema=adapter.DELTA_SCHEMA)
            out[i] = y.numpy()[0]; reasons.append(None)
        except ValueError as e:
            reasons.append(str(e))
    return out, reasons


def summarize(pred, target, *, delta=None, true_delta=None, coverage=None, true_contact=None, force=None):
    n = len(target)
    finite, proper, defect = so3_flags(pred)
    result = {"target_rows": n, "finite_predictions": int(finite.sum()), "near_SO3_predictions": int(proper.sum()),
        "raw_SO3_max_defect": float(defect.max()) if n else None}
    if not n or not bool(np.all(proper)):
        result["error_metrics"] = None; return result
    e = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    angle = full_rotation_error(pred, target)
    metrics = {"q_component_RMS_mrad": float(np.sqrt(np.mean(e[:, :20]**2))*1000),
        "qd_component_RMS_rad_s": float(np.sqrt(np.mean(e[:, 20:40]**2))),
        "COM_vector_RMS_mm": float(np.sqrt(np.mean(np.sum(e[:, 40:43]**2, -1)))*1000),
        "COM_max_mm": float(np.linalg.norm(e[:, 40:43], axis=-1).max()*1000),
        "full_SO3_geodesic_RMS_deg": float(np.sqrt(np.mean(angle**2))*180/np.pi),
        "full_SO3_geodesic_max_deg": float(angle.max()*180/np.pi),
        "vCOM_vector_RMS_m_s": float(np.sqrt(np.mean(np.sum(e[:, 52:55]**2, -1)))),
        "omega_world_vector_RMS_rad_s": float(np.sqrt(np.mean(np.sum(e[:, 55:58]**2, -1)))),
        "state58_RMSE_per_feature_raw_units": np.sqrt(np.mean(e**2, 0)).tolist()}
    if delta is not None:
        metrics["physical_delta52_RMSE_per_feature"] = np.sqrt(np.mean((delta-true_delta)**2, 0)).tolist()
    if coverage is not None:
        bad = ~np.isfinite(np.column_stack((coverage, force))).all(-1)
        metrics["nonfinite_contact_prediction_rows"] = int(bad.sum())
        if not np.any(bad):
            metrics["pad_coverage_Brier_mean"] = float(np.mean((coverage-true_contact[:, :2])**2))
            metrics["pad_coverage_Brier_per_pad"] = np.mean((coverage-true_contact[:, :2])**2, 0).tolist()
            metrics["force5_RMSE_N"] = np.sqrt(np.mean((force-true_contact[:, 2:7])**2, 0)).tolist()
            metrics["force5_MAE_N"] = np.mean(abs(force-true_contact[:, 2:7]), 0).tolist()
            metrics["log1p_force5_RMSE"] = np.sqrt(np.mean((np.log1p(force)-np.log1p(true_contact[:, 2:7]))**2, 0)).tolist()
    result["error_metrics"] = metrics
    return result


def constant_velocity_delta(x):
    out = np.zeros((*x.shape[:-1], 52), dtype=np.float64)
    out[..., :20] = x[..., 20:40]*H
    out[..., 40:43] = x[..., 52:55]*H
    out[..., 43:46] = x[..., 55:58]*H
    return out


def groups(family, dt, branch):
    yield "ALL", np.ones(len(family), dtype=bool)
    for value in sorted(set(family.tolist())):
        yield f"family_{value}", family == value
    for value in sorted(set(dt.tolist())):
        yield f"dt_{value:g}", dt == value
    for value in sorted(set(branch.tolist())):
        yield str(value), branch == value


def occupancy_groups(contact):
    count = np.sum(contact[:, :2] > 0, axis=-1)
    # Per-pad positive coverage is not proof the two pads were simultaneous.
    for n, name in ((0, "no_pad_positive_coverage"), (1, "one_pad_positive_coverage"),
                    (2, "both_pads_positive_coverage_not_simultaneity")):
        yield name, count == n


def save_csv(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    with Path(path).open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def run(protocol_path):
    p, training, tp, acq, out = preflight(protocol_path)
    require(p["CPU_inference_authorized"] is True, "Explicit fixed evaluation authorization")
    out.mkdir(); write(out/"protocol.json", p)
    counts = {"trusted_final_checkpoint_CPU_loads": 0, "model_constructors": 0, "TEST_NPZ_reads": 0,
        "one_step_FW": 0, "recursive_FW": 0, "predicted_rows": 0, "AD_optimizer_training_Native_Env_GPU": 0}
    phase = "setup"
    started = time.perf_counter()
    try:
        import torch
        torch.set_num_threads(4); torch.set_default_dtype(torch.float32); torch.use_deterministic_algorithms(True)
        torch.manual_seed(1729)
        require(not torch.cuda.is_initialized(), "No CUDA context")
        m = module(TRAINER, "fixed_phone_actual_trainer")
        payload = torch.load(checked(p["inputs"]["final_checkpoint"]), map_location="cpu", weights_only=False)
        counts["trusted_final_checkpoint_CPU_loads"] += 1
        require(payload["schema"] == m.ARTIFACT_SCHEMA and payload["iteration"] == 5000
            and payload["protocol_pin"] == p["inputs"]["training_protocol"]
            and payload["dataset_input_pins"] == tp["inputs"] and payload["recipe"] == tp["recipe"]
            and payload["normalizers_fit_partition"] == "train_only", "Actual final checkpoint provenance")
        require(all(float(s["step"]) == 5000 and all(bool(torch.isfinite(s[k]).all()) for k in ("exp_avg", "exp_avg_sq"))
            for s in payload["optimizer_state"]["state"].values()), "Saved final Adam5000 finite; no optimizer construction")
        model = m.make_model(payload["normalizers_F64"], torch); counts["model_constructors"] += 1
        model.load_state_dict(payload["model_state"], strict=True); model.eval()
        for v in model.parameters():
            v.requires_grad_(False)
        model_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rng_before = m.rng_state(torch)
        adapter = m._adapter(); codec = adapter._codec()
        plan = read(checked(p["inputs"]["family_plan"]))
        families = {f["family_id"]: f for f in plan["families"] if f["split"] == "test"}
        require(len(families) == 6 and payload["TRAIN_family_ids"] == [f["family_id"] for f in plan["families"] if f["split"] == "train"], "Unchanged prior family split")
        phase = "first_TEST_array_read"
        with np.load(checked(p["inputs"]["TEST_arrays"]), allow_pickle=False) as z:
            require(set(z.files) == m.ARRAY_KEYS, "Original raw14-key TEST ABI")
            raw = {k: z[k].copy() for k in z.files}
        counts["TEST_NPZ_reads"] += 1
        raw_before = {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in raw.items()}
        eligible = raw["training_eligible_bool"]
        require(eligible.dtype == np.bool_ and len(eligible) == 120 and int(eligible.sum()) == 114
            and int((~raw["complete_H_bool"]).sum()) == 6, "Predeclared114 complete/6partial TEST rows")
        require(set(raw["family_id_i64"].tolist()) == set(families), "All6 fixed TEST families")
        np.savez_compressed(out/"all120_TEST_rows_masks_and_identity.npz", **raw)
        np.savez_compressed(out/"TRAIN_only_normalizers_as_used.npz", **payload["normalizers_F64"])
        idx = np.flatnonzero(eligible)
        x = raw["x58_F64"][idx]; target = raw["y58_F64"][idx]; u = raw["u20_F64"][idx]
        h = raw["native_dt_s_F64"][idx]; n = raw["n_substeps_i64"][idx]
        require(bool(np.isfinite(np.column_stack((x, target, u, raw["target_delta52_F64"][idx], raw["contact_summary_F64"][idx]))).all()), "Finite eligible rows; no extra data filter")
        x80 = torch.from_numpy(np.column_stack((x, u, h, n)).astype(np.float32))
        phase = "one_step"
        with torch.inference_mode():
            prediction = model(x80); counts["one_step_FW"] += 1; counts["predicted_rows"] += len(x)
            d_t = model.physical_delta(prediction)
            y_array, decode_reasons = decode_rows_torch(torch.from_numpy(x.astype(np.float32)), d_t, adapter, torch)
        write(out/"one_step_decoder_reasons.json", decode_reasons)
        mean_delta = payload["normalizers_F64"]["physical_world_delta52_mean"]
        mean_cov = payload["normalizers_F64"]["coverage2_train_mean"]
        mean_force = np.expm1(payload["normalizers_F64"]["logforce5_mean"])
        gravities = [np.asarray(read(checked(p["inputs"][key]))["candidate"]["static"]["support_operands"]["gravity_world"], dtype=np.float64)
            for key in ("gravity_source_exact", "gravity_source_side")]
        require(all(g.shape == (3,) and np.array_equal(g, np.array([0., 0., -9.81])) for g in gravities), "Pinned common native world gravity")
        gravity = gravities[0]
        d_cv = constant_velocity_delta(x)
        d_ballistic = d_cv.copy(); d_ballistic[:, 40:43] += .5*gravity*H*H; d_ballistic[:, 46:49] = gravity*H
        result = {
            METHODS[0]: {"state": y_array, "delta": d_t.numpy().astype(np.float64),
                "coverage": stable_sigmoid(prediction["coverage_logits2"].numpy()),
                "force": np.expm1(prediction["logforce5"].numpy().astype(np.float64))},
            METHODS[1]: {"state": adapter.decode_numpy(x, np.broadcast_to(mean_delta, (len(x), 52)), state_schema=adapter.STATE_SCHEMA, delta_schema=adapter.DELTA_SCHEMA),
                "delta": np.broadcast_to(mean_delta, (len(x), 52)).copy(), "coverage": np.broadcast_to(mean_cov, (len(x), 2)).copy(), "force": np.broadcast_to(mean_force, (len(x), 5)).copy()},
            METHODS[2]: {"state": adapter.decode_numpy(x, d_cv, state_schema=adapter.STATE_SCHEMA, delta_schema=adapter.DELTA_SCHEMA), "delta": d_cv, "coverage": None, "force": None},
            METHODS[3]: {"state": adapter.decode_numpy(x, d_ballistic, state_schema=adapter.STATE_SCHEMA, delta_schema=adapter.DELTA_SCHEMA), "delta": d_ballistic, "coverage": None, "force": None}}
        branch = np.array([families[int(f)]["source_branch"] for f in raw["family_id_i64"][idx]])
        one_rows = []
        for method, values in result.items():
            np.savez_compressed(out/(method+"_one_step.npz"), prediction58=values["state"], delta52=values["delta"],
                row_id=raw["row_id_i64"][idx], target58=target,
                coverage2=np.full((len(x), 2), np.nan) if values["coverage"] is None else values["coverage"],
                force5_N=np.full((len(x), 5), np.nan) if values["force"] is None else values["force"])
            for label, mask in [*groups(raw["family_id_i64"][idx], h, branch), *occupancy_groups(raw["contact_summary_F64"][idx])]:
                one_rows.append({"method": method, "cohort": label, **summarize(values["state"][mask], target[mask],
                    delta=values["delta"][mask], true_delta=raw["target_delta52_F64"][idx][mask],
                    coverage=None if values["coverage"] is None else values["coverage"][mask],
                    force=None if values["force"] is None else values["force"][mask], true_contact=raw["contact_summary_F64"][idx][mask])})
        write(out/"one_step_statistics.json", one_rows); save_csv(out/"one_step_statistics.csv", one_rows)
        # Every episode starts from its actual source state; subsequent inputs
        # are predictions. Real planned controls are held even if target missing.
        phase = "recursive"
        episodes = sorted(set(raw["episode_id_i64"].tolist()))
        require(len(episodes) == 12, "All12 TEST episodes")
        e_info = []; truth = np.full((12, 10, 58), np.nan); target_mask = np.zeros((12, 10), dtype=bool)
        for ei, ep in enumerate(episodes):
            rr = np.flatnonzero(raw["episode_id_i64"] == ep)
            rr = rr[np.argsort(raw["interval_index_i64"][rr])]
            require(raw["interval_index_i64"][rr[0]] == 0, "Real episode start, no synthetic fill")
            fid = int(raw["family_id_i64"][rr[0]]); family = families[fid]
            commands = np.asarray(family["held_ctrl10x20_F64"], dtype=np.float64)
            for row in rr:
                j = int(raw["interval_index_i64"][row])
                require(np.array_equal(raw["u20_F64"][row].view(np.uint8), commands[j].view(np.uint8)), "Original planned/actual action bytes")
                if raw["complete_H_bool"][row] and raw["training_eligible_bool"][row]:
                    truth[ei, j] = raw["y58_F64"][row]; target_mask[ei, j] = True
            e_info.append({"episode_id": int(ep), "family_id": fid, "dt_s": float(raw["native_dt_s_F64"][rr[0]]),
                "n": int(raw["n_substeps_i64"][rr[0]]), "branch": family["source_branch"],
                "x0": raw["x58_F64"][rr[0]], "commands": commands})
        qrange = np.asarray(next(iter(acq["model_audits"].values()))["joint_range20x2"], dtype=np.float64)
        recursive_rows = []
        for method in METHODS:
            current = np.array([e["x0"] for e in e_info])
            pred = np.full((12, 10, 58), np.nan); completed = np.zeros((12, 10), dtype=bool)
            attempted = np.zeros((12, 10), dtype=bool); domain = np.ones((12, 10), dtype=bool)
            joint_out = np.zeros((12, 10), dtype=bool); active = np.ones(12, dtype=bool)
            decoder_reasons = [[None]*10 for _ in range(12)]
            for j in range(10):
                finite, proper, _ = so3_flags(current)
                active &= finite & proper
                rows = np.flatnonzero(active)
                if not len(rows):
                    break
                attempted[rows, j] = True
                if method == METHODS[0]:
                    feature = np.column_stack((current[rows], np.array([e_info[i]["commands"][j] for i in rows]),
                        [e_info[i]["dt_s"] for i in rows], [e_info[i]["n"] for i in rows])).astype(np.float32)
                    with torch.inference_mode():
                        pr = model(torch.from_numpy(feature)); counts["recursive_FW"] += 1; counts["predicted_rows"] += len(rows)
                        dd = model.physical_delta(pr)
                        yy, reason = decode_rows_torch(torch.from_numpy(current[rows].astype(np.float32)), dd, adapter, torch)
                    for ri, ei in enumerate(rows):
                        decoder_reasons[ei][j] = reason[ri]
                else:
                    dd = np.broadcast_to(mean_delta, (len(rows), 52)) if method == METHODS[1] else constant_velocity_delta(current[rows])
                    if method == METHODS[3]:
                        dd[:, 40:43] += .5*gravity*H*H; dd[:, 46:49] = gravity*H
                    yy = adapter.decode_numpy(current[rows], dd, state_schema=adapter.STATE_SCHEMA, delta_schema=adapter.DELTA_SCHEMA)
                pred[rows, j] = yy; completed[rows, j] = np.isfinite(yy).all(-1)
                good_finite, good_rotation, _ = so3_flags(yy)
                domain[rows, j] = good_rotation
                joint_out[rows, j] = np.any((yy[:, :20] < qrange[:, 0]) | (yy[:, :20] > qrange[:, 1]), -1)
                current[rows] = yy; active[rows] = good_finite & good_rotation
            np.savez_compressed(out/(method+"_recursive.npz"), prediction58=pred, target58=truth,
                target_available_mask=target_mask, attempted_mask=attempted, completed_mask=completed,
                raw_SO3_domain_mask=domain, joint_outside_true_hard_range=joint_out,
                episode_ids=np.array(episodes, dtype=np.int64), horizons=np.arange(1, 11, dtype=np.int64))
            write(out/(method+"_recursive_decode_reasons.json"), decoder_reasons)
            ef = np.array([e["family_id"] for e in e_info]); ed = np.array([e["dt_s"] for e in e_info]); eb = np.array([e["branch"] for e in e_info])
            for horizon in HORIZONS:
                j = horizon-1
                for label, gm in groups(ef, ed, eb):
                    paired = gm & target_mask[:, j]
                    metrics = summarize(pred[paired, j], truth[paired, j])
                    recursive_rows.append({"method": method, "horizon": horizon, "cohort": label,
                        "episodes": int(gm.sum()), "target_available": int(paired.sum()), "missing_target": int((gm & ~target_mask[:, j]).sum()),
                        "attempted_predictions": int(attempted[gm, j].sum()), "completed_predictions": int(completed[gm, j].sum()),
                        "nonfinite_prediction_count": int((attempted[gm, j] & ~np.isfinite(pred[gm, j]).all(-1)).sum()),
                        "joint_outside_true_range_count": int(joint_out[gm, j].sum()),
                        "SO3_domain_failure_count": int((completed[gm, j] & ~domain[gm, j]).sum()), **metrics})
        write(out/"recursive_statistics.json", recursive_rows); save_csv(out/"recursive_statistics.csv", recursive_rows)
        # Saved curves only; no earlier checkpoint inference and no selection.
        updates = [json.loads(line) for line in checked(p["inputs"]["training_log"]).read_text().splitlines()]
        require([v["iteration"] for v in updates] == list(range(1, 5001)), "All5000 training rows retained")
        curve = {"training_rows": updates, "validation_checkpoints": [{"iteration": c["iteration"], "validation": c["validation"]} for c in training["checkpoints"]],
            "checkpoint_selection_by_TEST": False, "fixed_primary_iteration": 5000}
        write(out/"all_training_and_saved_validation_curves.json", curve)
        require(m.raw_equal(model_before, model.state_dict(), torch) and m.raw_equal(rng_before, m.rng_state(torch), torch)
            and all(v.grad is None and not v.requires_grad for v in model.parameters()) and not torch.cuda.is_initialized(), "Weights/norm/RNG unchanged; no parameter AD")
        require(raw_before == {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in raw.items()}, "TEST raw arrays unchanged")
        outputs = {f.name: pin(f) for f in out.iterdir() if f.is_file()}
        report = {"schema": "phone_fixed_final5000_evaluation:1", "status": "COMPLETED_FIXED_MODEL_TEST_AND_RECURSIVE_NOT_POLICY_QUALIFICATION",
            "protocol": pin(protocol_path), "checkpoint": p["inputs"]["final_checkpoint"], "counts": counts,
            "TEST_rows": 120, "eligible_complete_rows": 114, "partial_rows_preserved": 6,
            "TEST_families": 6, "TEST_episodes": 12, "methods": METHODS, "recursive_horizons": HORIZONS,
            "invariants_weights_norm_inputs_RNG_unchanged": True, "outputs": outputs,
            "one_step_ALL": [v for v in one_rows if v["cohort"] == "ALL"],
            "recursive_ALL": [v for v in recursive_rows if v["cohort"] == "ALL"],
            "saved_validation_curve": curve["validation_checkpoints"], "elapsed_s": time.perf_counter()-started,
            "limitations": ["Six local simulated heldout families; bothdt belong to eachfamily. Same two mapped poses/assumed physical model, not independent ego/real contact truth.",
                "State58 omits hidden solver/history context. Recursive errors do not qualify a Markov model or Actor/CHORD benefit.",
                "Full Rbody near-SO3 atan2 observer, no projection or roll-blind axis. Domain defects/joint violations remain reported, not repaired.",
                "Mean baseline pools TRAIN dt; constantvelocity assumes no acceleration/gravity/contact and makes no contact prediction.",
                "Ballistic COM uses pinned world gravity, constant joint/COM angular velocities and no contact: not full torque-free rigid-body dynamics.",
                "Occupancy cohorts use raw per-pad coverage>0; both pads seen in an interval does not establish simultaneous contact.",
                "WM arithmetic FP32; pure kinematic/mean baselines F64. Force probabilities/physical force descriptors are postprocessed inF64.",
                "Contact labels summarize solver-PRE; state is POST20ms. Missing/partial H targets are never synthesized.",
                "Primary final5000 fixed before TEST; saved validation deterioration retained, no best-TEST checkpoint selection."]}
        write(out/"report.json", report)
        print(json.dumps({"status": report["status"], "report": pin(out/"report.json"), "counts": counts, "one_step_ALL": report["one_step_ALL"]}), flush=True)
    except BaseException as e:
        write(out/"failure.json", {"phase": phase, "error": repr(e), "counts": counts, "automatic_retry": False})
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.run:
        run(args.protocol.resolve(strict=True))
    else:
        _, _, _, _, out = preflight(args.protocol.resolve(strict=True))
        print(json.dumps({"status": "METADATA_VALID_NO_TEST_ARRAY_OR_MODEL", "output": str(out)}))


if __name__ == "__main__":
    main()
