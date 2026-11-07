"""Freeze CPU-only prospective execution inputs; never imports Torch or launches training."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {"path":str(Path(p).resolve()),"sha256":sha(p)}
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),"Never overwrite a frozen preparation artifact: "+str(p)
    p.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n",encoding="utf-8")

def main():
    arms={a:ROOT/f"outputs/c2_{a.lower()}_10iter_seed0_engineering_v1" for a in ["A1L","P_ONLY","A2"]}
    evidence={}
    for name,rel,script in [
        ("cpu_real_imag","native_control_contract_cpu_v2","check_c2_control_contract.py"),
        ("cuda_real_imag","native_control_contract_warp_v1","check_c2_control_contract.py"),
        ("pure_projection_AMP","target_inset_amp_cuda_v1","check_target_inset_amp.py"),
        ("legacy_format103_rejection","legacy_checkpoint_rejection_cpu_v1","check_legacy_checkpoint_rejection_cpu.py")]:
        evidence[name]={**pin(ROOT/f"outputs/{rel}/checks.json"),"script":str(ROOT/"scripts"/script),"script_sha256":sha(ROOT/"scripts"/script)}
    contract=read(arms["A1L"]/"source/c2_controller/controller_contract.json")
    local_paths={"adapter":ROOT/"src/c2_training_adapter.py","pure_projection":ROOT/"src/target_inset.py","penalty_wrapper":ROOT/"src/penalty_study/penalty_only_objective.py","entry":ROOT/"scripts/train_controlled_pen.py","original_trainer":ROOT/"source_baselines/original_train_controlled_pen.py"}
    # The baseline directory is an explicit archived source, never guessed by
    # silently choosing a file with the same basename.
    for label,source in contract["files"].items():
        if not local_paths[label].exists():
            matches=[p for p in ROOT.rglob(source["filename"]) if "outputs" not in p.relative_to(ROOT).parts and sha(p)==source["sha256"]]
            assert len(matches)==1,(label,matches);local_paths[label]=matches[0]
        assert sha(local_paths[label])==source["sha256"],label
    bindings={k:pin(p) for k,p in local_paths.items()}
    evidence["legacy_format103_rejection"]["source_binding_sha256"]=bindings["original_trainer"]["sha256"]
    estimates={}
    for arm,run in arms.items():
        evidence["smoke_"+arm]=pin(ROOT/f"outputs/engineering_health_v2/{arm.lower()}/report.json")
        metrics=run/"compiled/metrics.jsonl";rows=[json.loads(line) for line in metrics.read_text().splitlines()]
        assert [r["iteration"] for r in rows]==list(range(10))
        rates={phase:statistics.mean(rows[i]["profile/seconds/"+phase]-rows[i-1]["profile/seconds/"+phase] for i in range(4,10)) for phase in ["collect","wm","policy"]}
        summary=read(run/"summary.json");attempt_wall=summary["attempts"][-1]["attempt_wall_seconds"]
        fixed=attempt_wall-10*(rates["collect"]+rates["wm"])-8*rates["policy"]
        point=fixed+2000*(rates["collect"]+rates["wm"])+1800*rates["policy"]
        estimates[arm]={"low":.9*point,"high":1.3*point,"point":point,"phase_seconds_per_iteration":rates,"fixed_setup_compile_and_smoke_nonprofile_seconds":fixed,
            "engineering_metrics":pin(metrics),"engineering_summary":pin(run/"summary.json"),"phase_difference_iterations":[4,5,6,7,8,9],
            "excluded_first_policy_compile_iteration":2,"scope":"Engineering extrapolation, not a confidence interval. Profiled collect/WM projected2000, policy1800; smoke nonprofile/setup retained once. Frequent smoke save/log overhead is not multiplied200 times. Auditor/queue/entry overhead and cache uncertainty remain separate."}
    write(ROOT/"outputs/prospective_preparation_v1/runtime_estimates.json",estimates)
    v2=ROOT/"protocols/confirmatory_training_draft_v2.json";protocol=read(v2)
    protocol.update(schema="C2_020_prospective_training_execution_protocol:1",status="FROZEN_FOR_ROOT_REVIEW_NOT_TRAINING_AUTHORIZATION",formal_training_authorized=False,
        implementation_binding_status="REAL_IMAG_CPU_WARP_THREE_ARM_H8_SMOKE_AND_CPU_CHECKPOINT_HEALTH_MEASURED",
        supersedes_design_draft=pin(v2),source_bindings=bindings,engineering_evidence=evidence,
        queue_source=pin(ROOT/"scripts/run_matched_training_queue.py"),audit_source=pin(ROOT/"scripts/audit_c2_training_checkpoint.py"),
        lifecycle_cpu=pin(ROOT/"outputs/queue_cpu_lifecycle_v1/checks.json"),reviewed_stop_real_process_log=pin(ROOT/"outputs/queue_cpu_lifecycle_v1/reviewed_stop_real_process_checks.log"),
        root_raw_review=pin(ROOT/"outputs/root_engineering_raw_review_v1/report.json"),runtime_estimates=pin(ROOT/"outputs/prospective_preparation_v1/runtime_estimates.json"))
    protocol["training_process_caps"]={"native_attempt_seconds_all_arms":2400,"external_owned_child_seconds_all_arms":2700,"basis":"Same finite compute cap, unequal measured compute. Root final approval required before execute. Native cap excludes interpreter/setup; independent2700s timer starts beforePopen. CPU auditors may block parent but cannot disable timer. No quality stopping/retry/resume/automatic alternate checkpoint.","cleanup":"Reviewed10s taskkill plus5s rootreap; request at cap, bounded grace, unknown tree/GPU idleness retained and queue stops."}
    protocol["actual_u_evidence_status"]={"all_collection_batch_summaries":"MEASURED_IN_SEED0_SMOKES_REQUIRED_FOR_EVERY_FORMAL_CHECKPOINT_PREFIX","required_fields":["collect_call","iteration","steps","worlds","all_batch_exact","all_finite","within_inset_domain","max_abs_errors","clamped_components","total_components"],"error_keys":["cache_vs_actual","proposed_projection_vs_cache","residual_before_after_filter","replay_u_vs_actual","replay_c_vs_residual"],"checkpoint_link":"latest_actualu_evidence pins a complete contiguous JSONL prefix; sum steps*worlds equals saved simulator transitions. Later live append is outside the audited checkpoint. Sparse traces independently test numerical projection/cache/ctrl/replay and unchanged residual.","physical_scope":"Target bounds do not establish actual-q safety, force correctness or capacity qualification. Final physics evaluation remains separate."}
    protocol["freeze_authorization_rule"]="This file freezes reviewable execution design and evidence. Only a separately hash-bound queue manifest with explicit Root authorization can launch training. No formal seed has run during this CPU preparation. Goal pauses after first100health/ETA; the expressly authorized detached queue continues."
    protocol_path=ROOT/"protocols/final_execution_protocol_v1.json";write(protocol_path,protocol)
    source_draft=ROOT/"configs/formal_draft_root_v1/manifest.json";draft=read(source_draft);jobs=[];templates={}
    for old in draft["jobs"]:
        cfg=read(old["config"]);cfg["c2_protocol_file"]=str(protocol_path);cfg["c2_protocol_sha256"]=sha(protocol_path)
        target=ROOT/"configs/formal_v1"/Path(old["config"]).name;write(target,cfg)
        assert cfg["max_attempt_seconds"]==2400
        declared=read(ROOT/"protocols/confirmatory_schedule_draft_v1.json")["training_jobs"][old["position"]-1]
        jobs.append({"position":old["position"],"job_id":declared["job_id"],"model_id":declared["model_id"],"method":old["method"],"seed":old["seed"],"config":str(target),"config_sha256":sha(target),"output":old["output"]})
        templates.setdefault(old["method"],pin(target))
    inventory=arms["A1L"]/"source/inventory.json";paired=Path(read(jobs[0]["config"])["paired_protocol_file"])
    manifest={"schema":"C2_020_prospective_training_queue:1","status":"frozen_ready_for_authorization","formal_training_authorized":False,
        "caps_frozen_from_seed0_smoke":True,"background_queue_continues_when_Goal_paused":True,"checkpoint_format_version":103,"seeds":[4,5,6],
        "queue_runner_sha256":sha(ROOT/"scripts/run_matched_training_queue.py"),"python_executable":"D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe",
        "trainer":str(ROOT/"scripts/train_controlled_pen.py"),"trainer_sha256":bindings["entry"]["sha256"],
        "audit_script":str(ROOT/"scripts/audit_c2_training_checkpoint.py"),"audit_script_sha256":sha(ROOT/"scripts/audit_c2_training_checkpoint.py"),
        "source_inventory":str(inventory),"source_inventory_sha256":sha(inventory),"source_bindings":bindings,
        "nominal_model_sha256":protocol["unchanged_inputs"]["nominal_model_sha256"],"paired_protocol_file":str(paired),"paired_protocol_sha256":sha(paired),
        "prospective_protocol_file":str(protocol_path),"prospective_protocol_sha256":sha(protocol_path),
        "schedule_file":str(ROOT/"protocols/confirmatory_schedule_draft_v1.json"),"schedule_sha256":sha(ROOT/"protocols/confirmatory_schedule_draft_v1.json"),
        "gate_evidence":evidence,"method_attempt_seconds":{a:2400 for a in arms},"per_run_process_watchdog_seconds":2700,
        "runtime_estimate_seconds_by_method_from_engineering":estimates,"estimated_total_seconds_from_pilot_costs":sum(v["point"] for v in estimates.values())*3,
        "templates":templates,"jobs":jobs,"queue_output":str(ROOT/"outputs/confirmatory_queue_v1"),"draft_config_parent":pin(source_draft),
        "first100_scope":"CPU serialization/finiteness/source/controller/40192transitions/WM200+fail200/actor0critic0 health; no physical quality preview. Then notify Root with observed PID/OS CreationDate/ETA; user-requested Goal pause leaves already authorized detached queue running."}
    write(ROOT/"configs/confirmatory_queue_v1.json",manifest)
    print(json.dumps({"status":"CPU_frozen_for_review_no_execution","protocol":pin(protocol_path),"manifest":pin(ROOT/"configs/confirmatory_queue_v1.json"),"jobs":len(jobs),"estimates":{a:v["point"] for a,v in estimates.items()},"estimated_total_seconds":manifest["estimated_total_seconds_from_pilot_costs"]},indent=2))

if __name__=="__main__":main()
