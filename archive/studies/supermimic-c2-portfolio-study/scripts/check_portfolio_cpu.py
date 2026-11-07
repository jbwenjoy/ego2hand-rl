"""Small CPU identity/order/queue contracts; no simulator or GPU execution."""
import ast
import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("portfolio_queue",ROOT/"scripts/run_c2_portfolio_queue.py")
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
planpath=ROOT/"configs/c2-portfolio-v1.json"
plan,w,_=q.preflight(planpath)
checks={"stdlib_no_Torch_Warp":not any(k in sys.modules for k in ("torch","warp","mujoco_warp"))}
checks["27job_lookup_real_methods_seeds"]=all(w.model_for_job(plan,j)["training_seed"]==j["training_seed"] and w.model_for_job(plan,j)["method"]==j["trained_method"] for j in plan["jobs"])
checks["3process_per9checkpoints"]=all(sum(j["model_id"]==m["model_id"] for j in plan["jobs"])==3 for m in plan["models"])
def rejected(call):
    try:call()
    except (ValueError,KeyError):return True
    return False
first=plan["jobs"][0];model=w.model_for_job(plan,first)
checks["unknown_model_rejected"]=rejected(lambda:w.model_for_job(plan,{**first,"model_id":"unknown"}))
checks["wrong_job_seed_rejected"]=rejected(lambda:w.model_for_job(plan,{**first,"training_seed":99}))
checks["wrong_job_method_rejected"]=rejected(lambda:w.model_for_job(plan,{**first,"trained_method":"wrong"}))
all_meta=True
for model in plan["models"]:
    job=next(j for j in plan["jobs"] if j["model_id"]==model["model_id"])
    label={"A1L":"a1l","A2":"a2","P_ONLY":"p_only"}[model["method"]]
    meta=w.read(Path("D:/workspace/supermimic-chord-study/outputs")/f"replication_eval_{label}_seed{model['training_seed']}_primary/rollout.json")
    record={"trained_method":model["method"],"model_id":model["model_id"],"training_seed":model["training_seed"],"checkpoint_sha256":model["checkpoint_sha256"]}
    q.validate_policy_records(plan,job,w,meta,record,record)
checks["actual9_archived_raw_metadata_seed_checkpoint2000"]=all_meta
meta={"checkpoint_sha256":model["checkpoint_sha256"],"seed":0,"checkpoint_training_config":{"seed":model["training_seed"]},"checkpoint_iteration":2000}
checks["wrong_actual_config_seed_rejected"]=rejected(lambda:q.validate_policy_records(plan,job,w,{**meta,"checkpoint_training_config":{"seed":99}},record,record))
checks["wrong_actual_checkpoint_rejected"]=rejected(lambda:q.validate_policy_records(plan,job,w,{**meta,"checkpoint_sha256":"wrong"},record,record))
checks["wrong_condition_policy_rejected"]=rejected(lambda:q.validate_policy_records(plan,job,w,meta,{**record,"trained_method":"wrong"},record))
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);bad=copy.deepcopy(plan);bad["jobs"][0],bad["jobs"][1]=bad["jobs"][1],bad["jobs"][0]
    (root/"wrong_order.json").write_text(json.dumps(bad),encoding="utf-8")
    checks["proposal_order_change_rejected"]=rejected(lambda:w.validate(root/"wrong_order.json"))
    class Lifecycle:
        def __init__(self,fail):self.calls=0;self.fail=fail
        def gpu_lock(self):return contextlib.nullcontext()
        def reject_project_gpu_workers(self):pass
        def write(self,p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v),encoding="utf-8")
        def watch(self,*args,**kwargs):
            self.calls+=1;assert 0<kwargs["hard_seconds"]<=45
            return {"status":"failed" if self.fail else "completed","exit_code":1 if self.fail else 0}
    saved=q.health;q.health=lambda *_:{"task_success":False,"quality_used_for_control":False}
    try:
        for fail in (True,False):
            output=root/("fail" if fail else "all_false");test={**plan,"output_root":str(output)};life=Lifecycle(fail)
            try:q.run(planpath,test,w,life)
            except ValueError:assert fail
            state=w.read(output/"state.json")
            checks["one_fail_no_retry_next26" if fail else "all27_taskfalse_retained"]=(life.calls==1 and sum(j["status"]=="NOT_RUN" for j in state["jobs"])==26) if fail else life.calls==27 and state["status"]=="completed_27_jobs"
    finally:q.health=saved
old=Path("D:/workspace/supermimic-target-inset-study/scripts/evaluate_target_inset_capacity_v2.py")
factories=[next(n for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))) if isinstance(n,ast.FunctionDef) and n.name=="factory") for p in (old,Path(w.__file__))]
checks["entire_physics_factory_AST_unchanged"]=ast.dump(factories[0],include_attributes=False)==ast.dump(factories[1],include_attributes=False)
checks["pure_function_copy_exact"]=w.sha(ROOT/"src/target_inset.py")==w.sha(Path("D:/workspace/supermimic-target-inset-study/src/target_inset.py"))
out=ROOT/"outputs/portfolio_cpu_preparation";out.mkdir(parents=True,exist_ok=True)
report={"status":"passed" if all(checks.values()) else "failed","count":len(checks),"checks":checks,"scope":"CPU metadata/order/identity fixtures and fake queue; no GPU or physical evaluation","plan_sha256":w.sha(planpath),"worker_sha256":w.sha(w.__file__),"queue_sha256":w.sha(q.__file__)}
(out/"checks.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
print(json.dumps(report,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
