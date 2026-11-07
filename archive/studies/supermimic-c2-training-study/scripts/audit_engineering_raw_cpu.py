"""Independent raw CPU audit of the three completed seed0 engineering runs."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(x):
    if isinstance(x, torch.Tensor):
        return bool(torch.isfinite(x).all())
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or np.isfinite(x)


def main():
    out = ROOT / "outputs/root_engineering_raw_review_v1"
    out.mkdir(parents=True, exist_ok=False)
    reports, initial, contexts, estimates = [], [], [], {}
    for name, arm in (("a1l", "A1L"), ("p_only", "P_ONLY"), ("a2", "A2")):
        run = ROOT / f"outputs/c2_{name}_10iter_seed0_engineering_v1"
        attempt = run / "compiled"
        cfg = read(run / "input_config.json")
        result = read(attempt / "result.json")
        checkpoint = attempt / "train/model_10.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        controller = saved["c2_controller"]
        checks = {}

        def check(key, value):
            checks[key] = bool(value)
            assert value, (arm, key)

        check("seed0_arm_role", cfg["seed"] == controller["training_seed"] == 0 and cfg["method"] == controller["trained_method"] == arm and cfg["run_role"] == "engineering_smoke")
        check("actual_updates_and_samples", result["status"] == "passed" and result["actual_optimizer_steps"] == {"wm": 20, "fail_head": 20, "actor": 8, "critic": 32} and saved["sim_steps"] == result["collected_sim_transitions"] == 11392)
        check("format103_fromscratch", saved["format_version"] == controller["format_version"] == 103 and controller["exact_resume_supported"] is False and controller["margin_rad"] == .020 and not ({"checkpoint", "model_checkpoint"} & set(cfg["training"])))
        check("finite_modules_optimizers_norm", finite(saved["modules"]) and finite(saved["optimizers"]) and finite(saved["torque_norm"]))
        check("completed_SHA_index", sha(checkpoint) == read(checkpoint.with_suffix(".meta.json"))["checkpoint_sha256"] and read(checkpoint.with_suffix(".meta.json")) in [json.loads(x) for x in (attempt / "checkpoint_index.jsonl").read_text().splitlines()])
        bundle = run / "source/c2_controller"
        contract = read(bundle / "controller_contract.json")
        check("extension_archive_hashes", all(sha(bundle / pin["filename"]) == pin["sha256"] for pin in contract["files"].values()))
        check("current_source_matches_checkpoint", controller["adapter_sha256"] == sha(ROOT / "src/c2_training_adapter.py") and controller["pure_projection_sha256"] == sha(ROOT / "src/target_inset.py") and controller["entry_sha256"] == sha(ROOT / "scripts/train_controlled_pen.py"))
        marker = controller["latest_actualu_evidence"]
        logs = [json.loads(x) for x in (attempt / "c2_actualu_evidence.jsonl").read_text().splitlines()][:marker["collect_call"]]
        check("all11_collect_calls_prefix", marker["collect_call"] == 11 and logs[-1] == marker and [x["collect_call"] for x in logs] == list(range(1, 12)))
        check("allbatch_finite_inset_and_five_errors0", all(x["all_finite"] and x["within_inset_domain"] and x["all_batch_exact"] and len(x["max_abs_errors"]) == 5 and all(v == 0 for v in x["max_abs_errors"].values()) for x in logs))
        check("allbatch_coverage", sum(x["steps"] * x["worlds"] for x in logs) == 11392 and sum(x["total_components"] for x in logs) == 11392 * 20)
        lower = np.array(controller["target_lower_rad"], dtype=np.float32)
        upper = np.array(controller["target_upper_rad"], dtype=np.float32)
        for call in (1, 2, 3):
            with np.load(attempt / f"c2_actualu_trace_{call:05d}.npz", allow_pickle=False) as z:
                expected = np.clip(z["proposed"], lower + np.float32(.020), upper - np.float32(.020))
                check(f"raw_numpy_project_cache_actual_replay_call{call}", all(np.array_equal(expected, z[k]) for k in ("cached", "actual", "replay_u")))
                check(f"raw_residual_preserved_call{call}", all(np.array_equal(z["residual"], z[k]) for k in ("residual_before_filter", "replay_c")))
        metrics = [json.loads(x) for x in (attempt / "metrics.jsonl").read_text().splitlines()]
        prior = [x for x in metrics if any(k.startswith("prior/") for k in x)]
        check("prior_only_policy_updates", [x["iteration"] for x in prior] == ([] if arm == "A1L" else list(range(2, 10))))
        if arm == "P_ONLY":
            check("P_direct_value_nonzero_support0", all(abs(x["prior/geometry_intent_bonus_unscaled"] + .1 * x["prior/geometry_intent_penetration_prior"]) < 1e-6 and x["prior/geometry_intent_support_coefficient"] == 0 for x in prior) and any(x["prior/geometry_intent_penetration_prior"] > 0 for x in prior))
        if arm == "A2":
            check("A2_prior_exercised_nonzero", any(abs(x["prior/geometry_intent_bonus_unscaled"]) > 1e-6 for x in prior))
        check("native_MJB_identity", sha(attempt / "nominal_model.mjb") == "24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd")
        initial.append(read(attempt / "initial_parameters.json"))
        with np.load(attempt / "fixed_physics_context.npz", allow_pickle=False) as z:
            contexts.append({k: z[k].copy() for k in z.files})
        phase = {}
        for key in ("collect", "wm", "policy"):
            field = "profile/seconds/" + key
            deltas = [metrics[i][field] - metrics[i - 1][field] for i in range(4, 10)]
            phase[key] = {"mean": float(np.mean(deltas)), "min": float(min(deltas)), "max": float(max(deltas)), "iteration_indices": list(range(4, 10))}
        work10 = 10 * (phase["collect"]["mean"] + phase["wm"]["mean"]) + 8 * phase["policy"]["mean"]
        fixed = max(0, result["attempt_wall_seconds"] - work10)
        point = fixed + 2000 * (phase["collect"]["mean"] + phase["wm"]["mean"]) + 1800 * phase["policy"]["mean"]
        estimates[arm] = {"phase_seconds_per_iteration": phase, "estimated_fixed_seconds": fixed, "point_seconds": point, "low_seconds": .9 * point, "high_seconds": 1.3 * point}
        reports.append({"method": arm, "checks": checks, "status": "passed", "checkpoint_sha256": sha(checkpoint), "clamped_components": sum(x["clamped_components"] for x in logs), "peak_Torch_GiB": result["torch_peak_allocated_gib"]})
    assert initial[0] == initial[1] == initial[2]
    assert all(set(x) == set(contexts[0]) and all(np.array_equal(x[k], contexts[0][k]) for k in x) for x in contexts)
    assert not torch.cuda.is_initialized()
    report = {"status": "passed", "runs": reports, "initial_parameter_fingerprints_all3_equal": True, "all9_physics_arrays_all3_equal": True, "cuda_initialized": False, "runtime_estimates": estimates, "queue3seeds_point_seconds": 3 * sum(x["point_seconds"] for x in estimates.values()), "ETA_scope": "Six steady iteration timing deltas4..9, extrapolated to fixed2000/1800 budgets plus measured setup/compile overhead. Engineering range0.9..1.3 is descriptive, not a confidence interval; firstformal100 timing must update the estimate. No task-quality selection or equal-compute claim.", "scope": "Raw control/checkpoint/budget/source and imagination-prior execution health, not physical task success, WM accuracy or real ego qualification.", "script_sha256": sha(__file__)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "checks_each": [len(x["checks"]) for x in reports], "point_seconds_by_method": {k: v["point_seconds"] for k, v in estimates.items()}, "nine_run_point_seconds": report["queue3seeds_point_seconds"]}))


if __name__ == "__main__":
    main()
