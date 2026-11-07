"""Independent CPU audit of the three completed P_ONLY runs, no GPU/physics.

Exact OS-child timestamps select checkpoints; no evaluation quality is read.
The frozen P_ONLY auditor is reused in new folders, and observed saved state58
is separately passed through frozen A2 and P_ONLY for exact penalty tensor and
state-gradient equality. Original run/queue artifacts remain untouched.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import torch

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
QUEUE=ROOT/"outputs/penalty_only_seeds123_queue_v1"
sys.path.insert(0,str(ROOT/"src"))


def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def sha(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def write(p,v):
    p=Path(p)
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False,default=lambda value:value.item())+"\n",encoding="utf-8")
def time(v):return datetime.fromisoformat(v).astimezone(timezone.utc)
def module(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def archive_check(p,expected):
    with zipfile.ZipFile(p) as z:bad=[name for name,h in expected.items() if hashlib.sha256(z.read(name)).hexdigest()!=h]
    return {"sha256":sha(p),"entries_checked":len(expected),"mismatches":bad}


def main():
    torch.set_num_threads(2)
    assert torch.cuda.is_available() is False and torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
    m,s=read(QUEUE/"manifest.json"),read(QUEUE/"state.json")
    auditor=module("frozen_penalty_cpu_auditor",ROOT/"scripts/audit_training_checkpoint.py")
    trainer=module("frozen_penalty_cpu_factory",ROOT/"scripts/train_controlled_pen.py")
    global_checks={"queue_passed":s["status"]=="passed","queue_manifest_hash":sha(QUEUE/"manifest.json")==s["manifest_sha256"],
                   "auditor_hash":sha(ROOT/"scripts/audit_training_checkpoint.py")==m["audit_script_sha256"],
                   "trainer_hash":sha(ROOT/"scripts/train_controlled_pen.py")==m["trainer_sha256"],
                   "queue_source_hash":sha(QUEUE/"queue_source.py")==m["queue_runner_sha256"],
                   "followup_protocol_hash":sha(Path(m["followup_protocol_file"]))==m["followup_protocol_sha256"],
                   "three_seeds":len(s["jobs"])==3 and [j["seed"] for j in s["jobs"]]==[1,2,3]}
    assert all(global_checks.values()),global_checks
    for name in ["manifest.json","state.json"]:shutil.copy2(QUEUE/name,OUT/("queue_"+name))
    shutil.copy2(ROOT/"scripts/audit_training_checkpoint.py",OUT/"audit_training_checkpoint.py")
    base_inventory=read(Path(m["source_inventory"]))
    from dex_supermimic.rl.dex_supermimic.pen import unflatten_pen_memory
    records=[]
    for declared,j in zip(m["jobs"],s["jobs"]):
        run=Path(j["output"]);o=OUT/f"seed{j['seed']}";o.mkdir()
        cfg=read(run/"input_config.json");inv=read(run/"source/inventory.json");runtime=read(run/"runtime.json")
        result=read(run/"compiled/result.json");meta=read(run/"compiled/contact_objective_metadata.json")
        source=read(run/"source/contact_objective/frozen_inputs.json")
        osobs=j["actual_process_OS_observation"];start=time(osobs["creation_date_utc"])
        checks={"manifest_job_exact":all(j[k]==v for k,v in declared.items()),
                "config_hash":sha(Path(j["config"]))==j["config_sha256"]==sha(run/"input_config.json"),
                "actual_os_child_identity":osobs["status"]=="observed_exact_PID_and_identity" and osobs["pid"]==runtime["pid"]==j["actual_pid"] and osobs["parent_pid"]==j["launcher_pid"] and j["config"].lower() in osobs["command_line"].lower(),
                "exact_OS_within_bounds":time(j["actual_process_creation_conservative_bounds"]["lower_utc"])<=start<=time(j["actual_process_creation_conservative_bounds"]["upper_utc"]),
                "seed_method":cfg["method"]=="P_ONLY" and cfg["seed"]==j["seed"],
                "full_budget":result["final_iteration"]==2000 and result["collected_sim_transitions"]==648192 and result["actual_optimizer_steps"]=={"wm":4000,"fail_head":4000,"actor":1800,"critic":7200},
                "native_source_hashes":inv["files"]==base_inventory["files"]}
        final_health,metrics=auditor.audit(run,2000);(o/"final2000_health").mkdir();write(o/"final2000_health/report.json",final_health);(o/"final2000_health/metrics_snapshot.jsonl").write_bytes(metrics)
        checks["final52_health"]=final_health["status"]=="passed" and len(final_health["checks"])==52
        prior=read(Path(j["logs"])/"final2000_cpu_audit/report.json")
        checks["agrees_with_queue_audit"]=prior["checks"]==final_health["checks"] and prior["checkpoint_sha256"]==final_health["checkpoint_sha256"]
        indexed=[json.loads(x) for x in (run/"compiled/checkpoint_index.jsonl").read_text().splitlines()]
        candidates=[]
        for c in indexed:
            p=Path(c["checkpoint"]);elapsed=(time(c["completed_utc"])-start).total_seconds()
            candidates.append({**c,"os_creation_elapsed_seconds":elapsed,"strictly_before_1200":0<=elapsed<1200,
                               "sha_verified":sha(p)==c["checkpoint_sha256"],"sidecar_verified":read(p.with_suffix(".meta.json"))==c})
        checks["all21_completed_checkpoint_integrity"]=len(candidates)==21 and all(c["sha_verified"] and c["sidecar_verified"] for c in candidates)
        selected=max([c for c in candidates if c["strictly_before_1200"]],key=lambda c:time(c["completed_utc"]))
        selection={"schema":"penalty_only_secondary1200_selection:1","method":"P_ONLY","seed":j["seed"],"run":str(run),
                   "queue_manifest_sha256":sha(QUEUE/"manifest.json"),"queue_state_sha256":sha(QUEUE/"state.json"),"exact_OS_child":osobs,
                   "cutoff_utc":(start+timedelta(seconds=1200)).isoformat(),"rule":"latest complete indexed checkpoint strictly before actual-child OS CreationDate+1200s; no quality selection",
                   "candidates":candidates,"selected":selected,"first_after_cutoff":next(c for c in candidates if not c["strictly_before_1200"])}
        write(o/"secondary1200_selection.json",selection)
        h,mb=auditor.audit(run,selected["iteration"]);(o/"secondary_health").mkdir();write(o/"secondary_health/report.json",h);(o/"secondary_health/metrics_snapshot.jsonl").write_bytes(mb)
        checks["secondary52_health"]=h["status"]=="passed" and len(h["checks"])==52
        f=torch.load(run/"compiled/train/model_final.pt",map_location="cpu",weights_only=False);last=torch.load(run/"compiled/train/model_2000.pt",map_location="cpu",weights_only=False)
        differences=[k for k in last if not equal(last[k],f[k])];checks["final_vs2000_only_wall"]=set(differences)<={"wall_seconds"}
        checks["final_hash_matches_queue"]=sha(run/"compiled/train/model_final.pt")==j["final_checkpoint_sha256"]
        checks["from_scratch_seed"]=f["config"]["model_checkpoint"] is None and f["config"]["seed"]==j["seed"]
        archives={"dex":archive_check(run/"source/dex_source.zip",inv["files"]),"contact":archive_check(run/"source/contact_objective/study_contact_source.zip",source["source_hashes"])}
        checks["archive_integrity"]=not any(a["mismatches"] for a in archives.values())
        rows=[json.loads(x) for x in (run/"compiled/metrics.jsonl").read_text().splitlines()];pr=[x for x in rows if any(k.startswith("prior/") for k in x)]
        checks["metric_grid"]=[x["iteration"] for x in rows]==list(range(0,2000,10))
        checks["prior_grid"]=[x["iteration"] for x in pr]==list(range(200,2000,10))
        checks["support_coefficient_zero"]=all(x["prior/geometry_intent_support_coefficient"]==0 for x in pr)
        checks["penalty_coefficient_point1"]=all(np.isclose(x["prior/geometry_intent_penetration_coefficient"],.1,rtol=0,atol=1e-7) for x in pr)
        log_error=max(abs(x["prior/geometry_intent_bonus_unscaled"]+.1*x["prior/geometry_intent_penetration_prior"]) for x in pr)
        checks["mean_bonus_equals_minus_point1P"]=log_error<=8*np.finfo(np.float32).eps*max(1,max(x["prior/geometry_intent_penetration_prior"] for x in pr))
        # Exact tensor/gradient verification uses saved states, never simulator
        # future contact, reward updates or newly generated physics.
        wrapper=trainer.build_penalty_only_objective(source,"cpu")
        trace_path=run/"compiled/replay_control_trace.npz"
        with np.load(trace_path,allow_pickle=False) as trace:
            states=torch.tensor(np.concatenate([trace["x1"][0],trace["x1"][-1]]),dtype=torch.float32)
            mem=unflatten_pen_memory(torch.tensor(np.concatenate([trace["m"][0],trace["m"][-1]]),dtype=torch.float32))
        xa=states.clone().requires_grad_();xb=states.clone().requires_grad_()
        _,bt=wrapper.base(xa,mem);expected=-.1*bt["geometry_intent_penetration_prior"];actual,terms=wrapper(xb,mem)
        ga=torch.autograd.grad(expected.sum(),xa)[0];gb=torch.autograd.grad(actual.sum(),xb)[0]
        tensor_checks={"bonus_exact":torch.equal(actual,expected),"state_gradient_exact":torch.equal(ga,gb),
                       "values_and_gradients_finite":bool(torch.isfinite(actual).all() and torch.isfinite(gb).all()),
                       "support_coefficients_zero":bool((terms["geometry_intent_support_coefficient"]==0).all()),
                       "no_target_grad":all(not v.requires_grad and v.grad is None for v in wrapper.buffers())}
        checks["observed_state_tensor_parity"]=all(tensor_checks.values())
        np.savez_compressed(o/"penalty_tensor_parity.npz",state58=states.detach().numpy(),expected_bonus=expected.detach().numpy(),actual_bonus=actual.detach().numpy(),expected_state_gradient=ga.numpy(),actual_state_gradient=gb.numpy(),base_penetration_prior=bt["geometry_intent_penetration_prior"].detach().numpy())
        record={"status":"passed" if all(checks.values()) else "failed","method":"P_ONLY","seed":j["seed"],"checks":checks,
                "final_health":str(o/"final2000_health/report.json"),"secondary_health":str(o/"secondary_health/report.json"),
                "final_sha256":j["final_checkpoint_sha256"],"final_vs2000_differing_fields":differences,"archives":archives,
                "selected":selected,"selection_manifest":str(o/"secondary1200_selection.json"),"full_budget":final_health["actual_optimizer_steps"]|{"transitions":final_health["real_simulator_transitions"]},
                "metadata_contract":{k:meta[k] for k in ["schema","formula","support_coefficient","penetration_coefficient","penalty_wrapper_sha256","qualified_contact_predictor","dt_applied_inside_module","caller_dt_s","target_npz_sha256"]},
                "prior_logged_snapshots":len(pr),"prior_mean_formula_max_abs_rounding_error":log_error,"tensor_checks":tensor_checks,
                "tensor_samples":len(states),"tensor_nonzero_penalty_samples":int((actual.detach()!=0).sum()),"trace_sha256":sha(trace_path),
                "queue_GPU_slot_seconds_including_CPU_health":j["reserved_wall_seconds_including_CPU_health"],"training_recorded_seconds":read(run/"summary.json")["total_wall_seconds_including_failed_attempts"],
                "claim_scope":"health, budgets, exact shared-penalty tensors and time selection only; no physical outcome or three-arm causal conclusion"}
        write(o/"audit.json",record);records.append(record)
        print(json.dumps({"seed":j["seed"],"status":record["status"],"final_checks":len(final_health["checks"]),"secondary_checks":len(h["checks"]),"selected_iteration":selected["iteration"],"tensor_checks":tensor_checks}),flush=True)
        del f,last,wrapper,xa,xb,ga,gb,expected,actual
    global_checks["all_three_independent_audits_passed"]=all(r["status"]=="passed" for r in records)
    global_checks["CUDA_uninitialized_after"]=not torch.cuda.is_initialized()
    overall={"schema":"penalty_only_seeds123_cpu_audit:1","status":"passed" if all(global_checks.values()) else "failed","created_utc":datetime.now(timezone.utc).isoformat(),
             "gpu_executed":False,"cuda_mask":"-1","audit_source_sha256":sha(__file__),"checks":global_checks,"runs":records,
             "interpretation":"Post-outcome exploratory P_ONLY follow-up; historical A1L/A2 controls not contemporaneously randomized. No evaluation results read."}
    write(OUT/"manifest.json",overall)
    write(OUT/"secondary1200_manifest.json",{"schema":"penalty_only_secondary1200_manifest:1","status":overall["status"],"audit_manifest":str(OUT/"manifest.json"),
          "selected":[{"method":"P_ONLY","seed":r["seed"],"iteration":r["selected"]["iteration"],"checkpoint":r["selected"]["checkpoint"],"sha256":r["selected"]["checkpoint_sha256"],"os_elapsed_seconds":r["selected"]["os_creation_elapsed_seconds"],"selection_manifest":r["selection_manifest"]} for r in records]})
    raise SystemExit(0 if overall["status"]=="passed" else 1)


if __name__=="__main__":main()
