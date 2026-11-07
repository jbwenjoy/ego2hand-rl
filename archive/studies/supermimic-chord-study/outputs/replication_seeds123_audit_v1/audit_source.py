"""Independent CPU replication audit. Writes only its own new output directory.

Reuse the frozen health-audit implementation, then independently check archives,
queue identity, final serialization, prior logs and exact OS-anchored selection.
No training, physics, GPU calls or evaluation-quality selection.
"""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
QUEUE = ROOT / "outputs/replication_seeds123_queue_v1"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n", encoding="utf-8")


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def timestamp(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def archives(path, hashes):
    with zipfile.ZipFile(path) as archive:
        mismatches = [name for name, digest in hashes.items()
                      if hashlib.sha256(archive.read(name)).hexdigest() != digest]
    return {"entries": len(hashes), "sha256": sha(path), "entry_mismatches": mismatches}


def save_health(auditor, run, iteration, output):
    output.mkdir()
    result, metrics = auditor.audit(run, iteration)
    write(output / "report.json", result)
    (output / "metrics_snapshot.jsonl").write_bytes(metrics)
    return result


def main():
    torch.set_num_threads(2)
    manifest, state = read(QUEUE / "manifest.json"), read(QUEUE / "state.json")
    audit_path = ROOT / "scripts/audit_training_checkpoint.py"
    spec = importlib.util.spec_from_file_location("existing_cpu_checkpoint_audit", audit_path)
    auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)
    global_checks = {
        "queue_status_passed": state["status"] == "passed",
        "manifest_sha_matches_queue_state": sha(QUEUE / "manifest.json") == state["manifest_sha256"],
        "audit_script_matches_manifest": sha(audit_path) == manifest["audit_script_sha256"],
        "trainer_matches_manifest": sha(Path(manifest["trainer"])) == manifest["trainer_sha256"],
        "paired_protocol_matches_manifest": sha(Path(manifest["paired_protocol_file"])) == manifest["paired_protocol_sha256"],
        "source_inventory_matches_manifest": sha(Path(manifest["source_inventory"])) == manifest["source_inventory_sha256"],
        "queue_source_matches_manifest": sha(QUEUE / "queue_source.py") == manifest["queue_runner_sha256"],
        "six_jobs": len(state["jobs"]) == len(manifest["jobs"]) == 6,
        "seeds123_paired": sorted((j["seed"], j["method"]) for j in state["jobs"]) ==
                             [(seed, method) for seed in [1, 2, 3] for method in ["A1L", "A2"]]}
    if not all(global_checks.values()):
        raise ValueError(global_checks)
    shutil.copy2(audit_path, OUTPUT / "audit_training_checkpoint.py")
    shutil.copy2(QUEUE / "manifest.json", OUTPUT / "queue_manifest_snapshot.json")
    shutil.copy2(QUEUE / "state.json", OUTPUT / "queue_state_snapshot.json")
    reference_inventory = read(Path(manifest["source_inventory"]))
    baseline_meta = read(ROOT / "outputs/a2_paired_v2_clip007_2000iter_seed0/compiled/contact_objective_metadata.json")
    records = []
    for declared, job in zip(manifest["jobs"], state["jobs"]):
        name = f"{job['method'].lower()}_seed{job['seed']}"
        folder = OUTPUT / name; folder.mkdir()
        run = Path(job["output"]); config = read(run / "input_config.json")
        inventory = read(run / "source/inventory.json"); runtime = read(run / "runtime.json")
        summary = read(run / "summary.json"); result = read(run / "compiled/result.json")
        template = read(Path(manifest["templates"][job["method"]]["path"]))
        ignored = {"label", "seed", "output_dir", "max_attempt_seconds"}
        observation = job["actual_process_OS_observation"]
        started = timestamp(observation["creation_date_utc"])
        checks = {
            "manifest_job_matches_state": all(job[k] == v for k, v in declared.items()),
            "config_file_and_snapshot_hash": sha(Path(job["config"])) == job["config_sha256"] == sha(run / "input_config.json"),
            "template_hash_unchanged": sha(Path(manifest["templates"][job["method"]]["path"])) == manifest["templates"][job["method"]]["sha256"],
            "only_declared_template_changes": {k:v for k,v in config.items() if k not in ignored} == {k:v for k,v in template.items() if k not in ignored},
            "from_scratch_and_seed": config["seed"] == job["seed"] == read(run / "compiled/effective_training_config.json")["seed"]
                and read(run / "compiled/effective_training_config.json")["model_checkpoint"] is None,
            "guard_3600_prespecified": config["max_attempt_seconds"] == manifest["per_run_attempt_guard_seconds"] == 3600,
            "run_statuses_passed": job["status"] == runtime["status"] == summary["status"] == result["status"] == "passed",
            "actual_os_pid_matches_runtime": observation["status"] == "observed_exact_PID_and_identity" and
                    observation["pid"] == runtime["pid"] == job["actual_pid"] and observation["parent_pid"] == job["launcher_pid"],
            "os_command_identifies_config": job["config"].lower() in observation["command_line"].lower(),
            "os_creation_within_recorded_bounds": timestamp(job["actual_process_creation_conservative_bounds"]["lower_utc"]) <= started <=
                                                   timestamp(job["actual_process_creation_conservative_bounds"]["upper_utc"]),
            "frozen_dex_source_hashes_match_manifest_inventory": inventory["files"] == reference_inventory["files"],
            "paired_protocol_sha": sha(run / "source/paired_protocol.json") == manifest["paired_protocol_sha256"],
            "full_budget": result["final_iteration"] == 2000 and result["collected_sim_transitions"] == 648192 and
                    result["actual_optimizer_steps"] == {"wm":4000,"fail_head":4000,"actor":1800,"critic":7200},
            "matching_native_asset": sha(run / "compiled/nominal_model.mjb") == "24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd"}
        health = save_health(auditor, run, 2000, folder / "final2000_health")
        checks["independent_final_health_passed"] = health["status"] == "passed"
        previous = read(Path(job["logs"]) / "final2000_cpu_audit/report.json")
        checks["agrees_with_existing_final_audit"] = previous["checks"] == health["checks"] and previous["checkpoint_sha256"] == health["checkpoint_sha256"]
        archive = archives(run / "source/dex_source.zip", inventory["files"])
        checks["dex_archive_integrity"] = not archive["entry_mismatches"] and archive["sha256"] == inventory["dex_source_zip_sha256"]
        indexed = [json.loads(x) for x in (run / "compiled/checkpoint_index.jsonl").read_text().splitlines()]
        candidates = []
        for item in indexed:
            path = Path(item["checkpoint"]); elapsed = (timestamp(item["completed_utc"]) - started).total_seconds()
            candidates.append({**item, "os_creation_elapsed_seconds": elapsed, "strictly_before_1200s": 0 <= elapsed < 1200,
                               "sha_verified": sha(path) == item["checkpoint_sha256"],
                               "sidecar_equals_index": read(path.with_suffix(".meta.json")) == item})
        checks["all_completed_checkpoint_integrity"] = len(candidates) == 21 and all(x["sha_verified"] and x["sidecar_equals_index"] for x in candidates)
        eligible = [x for x in candidates if x["strictly_before_1200s"]]
        selected = max(eligible, key=lambda x:timestamp(x["completed_utc"])) if eligible else None
        after = [x for x in candidates if not x["strictly_before_1200s"]]
        selection = {"schema":"supermimic_chord_secondary_budget_selection:1", "method":job["method"], "seed":job["seed"],
                     "run":str(run), "selection_rule":"latest indexed complete checkpoint with completedUTC minus exact actual-child OS CreationDate strictly <1200 seconds; no quality selection",
                     "queue_manifest_sha256":sha(QUEUE / "manifest.json"), "queue_state_sha256":sha(QUEUE / "state.json"),
                     "actual_process_OS_observation":observation, "os_creation_utc":started.isoformat(),
                     "cutoff_utc":(started+timedelta(seconds=1200)).isoformat(), "cutoff_seconds":1200,
                     "checkpoint_index_sha256":sha(run / "compiled/checkpoint_index.jsonl"), "candidates":candidates,
                     "selected":selected, "first_after_cutoff":min(after,key=lambda x:timestamp(x["completed_utc"])) if after else None,
                     "scope":"2000-iteration-capped experiment; early completion does not imply 1200s consumed"}
        checks["secondary_checkpoint_available"] = selected is not None
        write(folder / "secondary1200_selection.json", selection)
        final = torch.load(run / "compiled/train/model_final.pt", map_location="cpu", weights_only=False)
        last = torch.load(run / "compiled/train/model_2000.pt", map_location="cpu", weights_only=False)
        differing = [key for key in last if not equal(last[key], final[key])]
        checks["final_vs_2000_only_wall_seconds"] = set(differing) <= {"wall_seconds"}
        checks["final_hash_matches_queue"] = sha(run / "compiled/train/model_final.pt") == job["final_checkpoint_sha256"]
        secondary_health = None
        if selected and selected["iteration"] != 2000:
            secondary_health = save_health(auditor, run, selected["iteration"], folder / "secondary_health")
            checks["secondary_health_passed"] = secondary_health["status"] == "passed"
        rows = [json.loads(x) for x in (run / "compiled/metrics.jsonl").read_text().splitlines()]
        prior = [x for x in rows if any(k.startswith("prior/") for k in x)]
        checks["complete_metric_epoch_grid"] = [x["iteration"] for x in rows] == list(range(0,2000,10))
        contact_archive = None; metadata_summary = None
        if job["method"] == "A2":
            frozen = read(run / "source/contact_objective/frozen_inputs.json")
            contact_archive = archives(run / "source/contact_objective/study_contact_source.zip", frozen["source_hashes"])
            checks["contact_archive_integrity"] = not contact_archive["entry_mismatches"]
            metadata = read(run / "compiled/contact_objective_metadata.json")
            paths = {"bundle_directory","reference_file","model_file","target_file"}
            checks["metadata_exact_except_run_paths"] = {k:v for k,v in metadata.items() if k not in paths} == {k:v for k,v in baseline_meta.items() if k not in paths}
            checks["prior_epoch_grid"] = [x["iteration"] for x in prior] == list(range(200,2000,10))
            metadata_summary = {k:metadata[k] for k in ["scientific_role","qualified_contact_predictor","target_defined_from_policy_contacts","target_frozen","target_npz_sha256","basis512_sha256","training_basis64_sha256","points_per_geom","sampled_hand_surface_points","dt_applied_inside_module"]}
        else:
            checks["A1L_no_prior_fields"] = len(prior) == 0
        checks["no_prior_in_wm_pretraining"] = not any(any(k.startswith("prior/") for k in x) for x in rows if x["iteration"]<200)
        artifact = {"status":"passed" if all(checks.values()) else "failed", "method":job["method"],"seed":job["seed"],"checks":checks,
                    "final_health":str(folder / "final2000_health/report.json"),"final_health_check_count":len(health["checks"]),
                    "full_optimizer_steps":health["actual_optimizer_steps"],"full_transitions":health["real_simulator_transitions"],
                    "final_checkpoint":str(run / "compiled/train/model_final.pt"),"final_sha256":sha(run / "compiled/train/model_final.pt"),
                    "final_vs_2000_differing_fields":differing,"dex_archive":archive,"contact_archive":contact_archive,
                    "contact_metadata":metadata_summary,"prior_logged_snapshots":len(prior),
                    "secondary_selection":str(folder / "secondary1200_selection.json"),"selected":selected,
                    "full_completed_os_elapsed_seconds":(timestamp(candidates[-1]["completed_utc"])-started).total_seconds(),
                    "recorded_training_process_seconds":summary["total_wall_seconds_including_failed_attempts"],
                    "queue_reserved_seconds_including_cpu_health":job["reserved_wall_seconds_including_CPU_health"],
                    "torch_peak_allocated_gib":result["torch_peak_allocated_gib"],
                    "claim_scope":"serialization, budgets, frozen inputs and time selection only; no new physical evaluation or benefit claim"}
        write(folder / "audit.json", artifact); records.append(artifact)
        print(json.dumps({"run":name,"status":artifact["status"],"selected_iteration":selected["iteration"] if selected else None,
                          "selected_os_elapsed":selected["os_creation_elapsed_seconds"] if selected else None},ensure_ascii=False),flush=True)
        del final, last
    all_checks = {**global_checks, "all_six_independent_audits_passed": all(x["status"] == "passed" for x in records)}
    report = {"schema":"supermimic_chord_replication_cpu_audit:1","status":"passed" if all(all_checks.values()) else "failed",
              "checked_utc":datetime.now(timezone.utc).isoformat(),"gpu_executed":False,"audit_source_sha256":sha(__file__),
              "queue_manifest_sha256":sha(QUEUE / "manifest.json"),"queue_state_sha256":sha(QUEUE / "state.json"),
              "checks":all_checks,"runs":records,"seed0_excluded":True,
              "interpretation":"seeds1,2,3 are paired replication seeds, not new motion/video samples; no evaluation quality used in selection; same updates are not equal compute"}
    write(OUTPUT / "manifest.json", report)
    write(OUTPUT / "secondary1200_manifest.json", {"schema":"supermimic_chord_replication_secondary_selection:1",
          "audit_manifest":str(OUTPUT / "manifest.json"),"status":report["status"],"cutoff_seconds":1200,
          "selection_rule":"strict completedUTC < exact actual-child OS CreationDate +1200 seconds, within2000-iteration-capped runs",
          "selected":[{"method":x["method"],"seed":x["seed"],"checkpoint":x["selected"]["checkpoint"],
                       "sha256":x["selected"]["checkpoint_sha256"],"iteration":x["selected"]["iteration"],
                       "os_elapsed_seconds":x["selected"]["os_creation_elapsed_seconds"],"selection_manifest":x["secondary_selection"]} for x in records]})
    lines = ["# Seeds 1–3 独立 CPU 审计", "", f"状态：{report['status']}。未使用 GPU、未改冻结源码，原 queue audits 保留。", "",
             "| 方法 | seed | final health | full iter / transitions | pre1200 iter | OS 完成秒 |", "|---|---:|---|---:|---:|---:|"]
    for x in records:
        lines.append(f"| {x['method']} | {x['seed']} | {x['status']} | 2000 / {x['full_transitions']} | {x['selected']['iteration']} | {x['selected']['os_creation_elapsed_seconds']:.6f} |")
    lines += ["", "每个完整 run 的 WM/failure-head 各4000、actor1800、critic7200次更新；模型、优化器、指标有限，冻结来源/接触输入与manifest一致。",
              "A1L 均提前完成2000迭代，沿用final；A2按时间可用规则均选1700。不能表述两臂各耗满1200秒。",
              "完整final和按时可用checkpoint的物理质量需独立评估；本报告不作成功率或接触收益结论。", ""]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
