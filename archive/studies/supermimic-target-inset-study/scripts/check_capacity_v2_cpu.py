"""CPU-only status/gate contracts for C2's one post-physics sticky read."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
w=load(ROOT/"scripts/evaluate_target_inset_capacity_v2.py","capacity_worker")
q=load(ROOT/"scripts/run_target_inset_capacity_queue_v2.py","capacity_queue")
checks={"stdlib_only_no_Torch_Warp":not any(k in sys.modules for k in ("torch","warp","mujoco_warp"))}
checks["zero_is_OK"]=w.classify_sticky_overflow([0])=={"status":"OK","overflow_bitmask":0,"flags_by_world":[0],"read_error":None}
checks["nonzero_is_OVERFLOW"]=w.classify_sticky_overflow([2])["status"]=="OVERFLOW"
checks["bits_OR_all_worlds"]=w.classify_sticky_overflow([1,4])["overflow_bitmask"]==5
checks["unreadable_is_UNKNOWN"]=w.classify_sticky_overflow(read_error="read failure")["status"]=="UNKNOWN"
checks["missing_empty_invalid_never_zero"]=all(w.classify_sticky_overflow(v)["status"]=="UNKNOWN" and w.classify_sticky_overflow(v)["overflow_bitmask"] is None for v in (None,[],[-1],[0.0],["0"]))
plan={"plan_sha256_runtime":"plan","source_pins":{"worker":{"sha256":"worker"}}};job={"job_id":"job"};meta={"resets_during_rollout":0}
valid={**w.classify_sticky_overflow([0]),"resets_during_rollout":0,"read_count":1,"read_at_control_intervals":150,
       "actual_world_count":1,"job_id":"job","plan_sha256":"plan","worker_sha256":"worker"}
q.validate_capacity(plan,job,w,meta,valid);checks["queue_accepts_complete_zero_noreset"]=True
for label,patch in (("queue_rejects_OVERFLOW",{"status":"OVERFLOW","overflow_bitmask":1}),
                    ("queue_rejects_UNKNOWN",{"status":"UNKNOWN","overflow_bitmask":None}),
                    ("queue_rejects_reset",{"resets_during_rollout":1}),
                    ("queue_rejects_early_read",{"read_at_control_intervals":149}),
                    ("queue_rejects_multiple_reads",{"read_count":2}),
                    ("queue_rejects_wrong_identity",{"worker_sha256":"other"})):
    try:q.validate_capacity(plan,job,w,meta,{**valid,**patch})
    except ValueError:checks[label]=True
    else:checks[label]=False
planpath=ROOT/"configs/target-inset-capacity-v2.json"
q.preflight(planpath);checks["final_frozen_plan_preflight"]=True
out=ROOT/"outputs/capacity_v2_cpu_preparation";out.mkdir(parents=True,exist_ok=True)
record={"status":"passed" if all(checks.values()) else "failed","checks":checks,"count":len(checks),
        "scope":"pure CPU flag/status/identity/timing/reset gate fixtures only, no simulator/GPU/physics",
        "worker_sha256":w.sha(w.__file__),"queue_sha256":w.sha(q.__file__),"plan_sha256":w.sha(planpath)}
(out/"checks.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
