"""Reproduce six-run training integrity and cost accounting using CPU only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refuse to replace a prior cost audit"
    torch.set_num_threads(2)
    plan, queue = read(args.manifest), read(args.queue / "state.json")
    assert queue["status"] == "passed" and len(queue["jobs"]) == 6
    assert sha(args.manifest) == queue["manifest_sha256"] == sha(args.queue / "manifest.json")
    expected_updates = {"wm": 4000, "fail_head": 4000, "actor": 1800, "critic": 7200}
    rows, native_inventory, reference, physical_base = [], None, None, None
    for job, planned in zip(queue["jobs"], plan["jobs"]):
        run = Path(job["output"])
        summary, config = read(run / "summary.json"), read(run / "input_config.json")
        attempt = summary["attempts"][0]
        assert job["status"] == summary["status"] == attempt["status"] == "passed"
        assert len(summary["attempts"]) == 1 and attempt["final_iteration"] == 2000
        assert attempt["actual_optimizer_steps"] == expected_updates and attempt["collected_sim_transitions"] == 648192
        assert config["method"] == job["method"] == planned["method"] and config["seed"] == job["seed"] == planned["seed"]
        assert sha(run / "input_config.json") == sha(job["config"]) == planned["config_sha256"]
        inventory = read(run / "source/inventory.json")
        assert sha(run / "source/train_controlled_pen.py") == inventory["script_sha256"] == plan["trainer_sha256"] == sha(plan["trainer"])
        assert sha(run / "source/dex_source.zip") == inventory["dex_source_zip_sha256"]
        assert sha(run / "source/paired_protocol.json") == plan["paired_protocol_sha256"]
        if native_inventory is None:
            native_inventory = inventory["files"]
            with zipfile.ZipFile(run / "source/dex_source.zip") as archive:
                assert all(hashlib.sha256(archive.read(k)).hexdigest() == v for k, v in native_inventory.items())
            assert all(sha(Path(inventory["dex_root"]) / p) == v for p, v in native_inventory.items())
        assert native_inventory == inventory["files"]
        audits = {}
        for label, iteration in (("first100", 100), ("final2000", 2000)):
            report = read(Path(job["logs"]) / f"{label}_cpu_audit/report.json")
            assert report["status"] == "passed" and all(report["checks"].values()) and report["iteration"] == iteration
            assert report["audit_script_sha256"] == plan["audit_script_sha256"]
            audits[label] = {"checks": len(report["checks"]), "status": "passed"}
        index = [json.loads(line) for line in (run / "compiled/checkpoint_index.jsonl").read_text().splitlines()]
        assert len(index) == 21
        for entry in index:
            path = Path(entry["checkpoint"])
            assert sha(path) == entry["checkpoint_sha256"] and read(path.with_suffix(".meta.json")) == entry
        final = Path(attempt["checkpoint"])
        assert sha(final) == attempt["checkpoint_sha256"] == job["final_checkpoint_sha256"]
        saved = torch.load(final, map_location="cpu", weights_only=False)
        assert finite(saved) and saved["iteration"] == 2000 and saved["sim_steps"] == 648192
        assert all(v > 0 and math.isfinite(v) for v in attempt["parameter_delta_l2"].values())
        assert attempt["self_collision_head_weight_delta_l2"] > 0
        metrics = [json.loads(line) for line in (run / "compiled/metrics.jsonl").read_text().splitlines()]
        assert len(metrics) == 200 and all(finite(row) for row in metrics)
        if reference is None:
            reference = (attempt["reference_signature"], summary["reference"]["derived_sha256"], attempt["nominal_model_sha256"])
        assert reference == (attempt["reference_signature"], summary["reference"]["derived_sha256"], attempt["nominal_model_sha256"])
        with np.load(run / "compiled/fixed_physics_context.npz") as data:
            values = {k: data[k].copy() for k in data.files}
        if physical_base is None:
            physical_base = values
        assert values.keys() == physical_base.keys() and all(np.array_equal(values[k], physical_base[k]) for k in values)
        contact = None
        if job["method"] == "A2":
            spec = read(run / "source/contact_objective/frozen_inputs.json")
            assert spec["expected_hashes"] == config["contact_objective"]["expected_hashes"]
            with zipfile.ZipFile(run / "source/contact_objective/study_contact_source.zip") as archive:
                for name, key in (("contact_geometry.py", "geometry_module_sha256"), ("contact_objective.py", "objective_module_sha256")):
                    assert hashlib.sha256(archive.read("chord_study/" + name)).hexdigest() == spec["expected_hashes"][key]
            objective = attempt["contact_objective_metadata"]
            assert objective["sampled_hand_surface_points"] == 1344 and objective["basis"]["training_directions"] == 64
            assert objective["qualified_contact_predictor"] is False and objective["dt_applied_inside_module"] is False
            contact = {"geometry_sha256": spec["expected_hashes"]["geometry_module_sha256"],
                       "objective_sha256": spec["expected_hashes"]["objective_module_sha256"],
                       "production_execution": "eager; unused compiled prior precision qualification failed"}
        observation = job["actual_process_OS_observation"]
        assert observation["status"] == "observed_exact_PID_and_identity"
        os_start = instant(observation["creation_date_utc"])
        eligible = [entry for entry in index if (instant(entry["completed_utc"]) - os_start).total_seconds() <= 1200]
        strict = max(eligible, key=lambda entry: instant(entry["completed_utc"]))
        runtime = read(run / "runtime.json")
        rows.append({"position": job["position"], "seed": job["seed"], "method": job["method"], "run": str(run),
                     "health": audits, "source_files": len(inventory["files"]), "native_zip_sha256": inventory["dex_source_zip_sha256"],
                     "trainer_sha256": inventory["script_sha256"], "contact": contact, "actual_updates": attempt["actual_optimizer_steps"],
                     "real_transitions": attempt["collected_sim_transitions"], "parameter_delta_l2": attempt["parameter_delta_l2"],
                     "self_collision_head_weight_delta_l2": attempt["self_collision_head_weight_delta_l2"],
                     "native_compiled_unique_graphs": attempt["compiled_counters"]["stats"]["unique_graphs"],
                     "torch_peak_allocated_gib": attempt["torch_peak_allocated_gib"],
                     "training_STARTED_wall_seconds": summary["total_wall_seconds_including_failed_attempts"],
                     "OS_creation_to_runtime_finished_seconds": (instant(runtime["finished_utc"]) - os_start).total_seconds(),
                     "reserved_wall_seconds_including_CPU_health": job["reserved_wall_seconds_including_CPU_health"],
                     "learn_seconds": attempt["learn_wall_seconds"], "phase_seconds": attempt["phase_seconds"],
                     "cache_context": summary["cache_context"], "verified_checkpoint_count": len(index),
                     "strict_OS1200_checkpoint": {"iteration": strict["iteration"], "path": strict["checkpoint"],
                                                 "sha256": strict["checkpoint_sha256"],
                                                 "OS_elapsed_seconds": (instant(strict["completed_utc"]) - os_start).total_seconds()},
                     "final_sha256": sha(final), "finite_metric_rows": len(metrics)})
    total = (instant(queue["finished_utc"]) - instant(queue["started_utc"])).total_seconds()
    training_total = sum(row["training_STARTED_wall_seconds"] for row in rows)
    methods = {}
    for method in ("A1L", "A2"):
        selected = [row for row in rows if row["method"] == method]
        times = [row["training_STARTED_wall_seconds"] for row in selected]
        methods[method] = {"n": len(times), "training_seconds_total": sum(times), "mean_seconds": statistics.mean(times),
                           "sample_std_seconds": statistics.stdev(times), "min_max_seconds": [min(times), max(times)],
                           "torch_peak_allocated_gib_range": [min(row["torch_peak_allocated_gib"] for row in selected), max(row["torch_peak_allocated_gib"] for row in selected)]}
    paired = [{"seed": seed, "A2_to_A1L_training_wall_ratio": next(r for r in rows if r["seed"] == seed and r["method"] == "A2")["training_STARTED_wall_seconds"] / next(r for r in rows if r["seed"] == seed and r["method"] == "A1L")["training_STARTED_wall_seconds"]} for seed in (1,2,3)]
    report = {"status": "passed", "created_utc": datetime.now(timezone.utc).isoformat(), "audit_script_sha256": sha(__file__),
              "scope": "Six replication training runs and occupied queue slot only; excludes pilot, engineering diagnostics, evaluation/render and idle research time",
              "manifest_sha256": sha(args.manifest), "queue_total_wall_seconds": total, "queue_reserved_GPU_slot_hours": total/3600,
              "training_STARTED_wall_seconds_sum": training_total, "queue_minus_training_seconds": total-training_total,
              "sum_per_run_reserved_including_CPU_health_seconds": sum(row["reserved_wall_seconds_including_CPU_health"] for row in rows),
              "runs": rows, "by_method_descriptive_costs": methods, "paired_cost_ratios": paired,
              "ratio_of_method_mean_training_wall": methods["A2"]["mean_seconds"]/methods["A1L"]["mean_seconds"],
              "aggregate_real_transitions": sum(r["real_transitions"] for r in rows),
              "aggregate_updates": {name: sum(row["actual_updates"][name] for row in rows) for name in expected_updates},
              "verified_checkpoint_files": sum(r["verified_checkpoint_count"] for r in rows),
              "memory_scope": "Torch allocator peak only; full-run global GPU utilization/reserved memory was not sampled, so neither peak device memory nor active GPU compute hours can be recovered",
              "interpretation": "Equal counted updates/transitions and frozen settings, not identical on-policy samples or equal compute. Source/finite health and positive parameter deltas are not physical task results.",
              "CUDA_initialized_by_audit": torch.cuda.is_initialized()}
    assert not report["CUDA_initialized_by_audit"]
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    (args.output / "audit_replication_queue_cpu.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"status": report["status"], "report": str(args.output / "report.json"), "queue_minutes": total/60,
                      "training_minutes": training_total/60, "overhead_seconds": total-training_total, "by_method": methods,
                      "ratio_of_mean_costs": report["ratio_of_method_mean_training_wall"], "checkpoints_verified": report["verified_checkpoint_files"],
                      "secondary_iterations": [(row["seed"],row["method"],row["strict_OS1200_checkpoint"]["iteration"]) for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
