"""Small CPU action/queue contracts with fake children; no simulator execution."""
import ast
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("inset_queue",ROOT/"scripts/run_target_inset_queue.py")
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
plan_path=ROOT/"configs/target-inset-v1.json"
plan,worker,_=q.preflight(plan_path)
checks={"stdlib_preflight_no_torch_or_warp":not any(k in sys.modules for k in ("torch","warp","mujoco_warp"))}
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    bad={**plan,"jobs":[{**j} for j in plan["jobs"]]}
    bad["jobs"][0]["original_evaluator_args"]=["--seed","1"]
    (root/"bad.json").write_text(json.dumps(bad),encoding="utf-8")
    try:worker.validate(root/"bad.json")
    except ValueError:checks["changed_original_eval_CLI_rejected"]=True
    else:checks["changed_original_eval_CLI_rejected"]=False
    class Lifecycle:
        def __init__(self,fail):self.calls=0;self.fail=fail
        def gpu_lock(self):return contextlib.nullcontext()
        def reject_project_gpu_workers(self):pass
        def write(self,path,value):
            Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value),encoding="utf-8")
        def watch(self,*args,**kwargs):
            self.calls+=1;assert 0<kwargs["hard_seconds"]<=45
            return {"status":"failed" if self.fail else "completed","exit_code":1 if self.fail else 0}
        def observe_actual_pid(self,*args):return {"status":"CPU_mock_only"}
    original_health=q.health
    q.health=lambda *_:{"task_success":False,"quality_used_for_control":False}
    try:
        for fail in (True,False):
            life=Lifecycle(fail);output=root/("fail" if fail else "all_false")
            test={**plan,"output_root":str(output)}
            try:q.run(plan_path,test,worker,life)
            except ValueError:assert fail
            state=worker.read(output/"state.json")
            if fail:checks["one_engineering_fail_no_retry_nextjob"]=life.calls==1 and sum(j["status"]=="NOT_RUN" for j in state["jobs"])==8
            else:checks["nine_quality_false_jobs_all_retained"]=life.calls==9 and state["status"]=="completed_9_jobs" and all(j["health"]["task_success"] is False for j in state["jobs"])
    finally:q.health=original_health

import torch
sys.path.insert(0,str(ROOT/"src"))
from target_inset import target_inset
native_path=Path("D:/workspace/dex-supermimic-research/src/dex_supermimic/tasks/pen_spin/mdp/actions.py")
tree=ast.parse(native_path.read_text(encoding="utf-8"))
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SingleHandResidualEMAAction")
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="apply_actions")
ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(native_path),"exec"),ns)
lo,hi=torch.zeros(1,20),torch.ones(1,20)
for margin in (0,.005,.020):
    proposed=torch.linspace(0,1,20)[None];residual=torch.linspace(-.3,.3,20)[None]
    filtered=target_inset(proposed,lo,hi,margin);writes=[]
    fake=SimpleNamespace(_targets=filtered.clone(),_processed_actions=residual.clone(),_target_ids=list(range(20)),
        limit_target=lambda value:value.clamp(min=lo,max=hi),
        _entity=SimpleNamespace(set_joint_position_target=lambda value,**_:writes.append(value.clone())))
    for _ in range(4):ns["apply_actions"](fake)
    checks[f"native_apply4_preserves_inset_and_EMA_margin{margin}"]=all(torch.equal(v,filtered) for v in writes) and torch.equal(fake._processed_actions,residual)
checks["CPU_mask_no_CUDA_initialization"]=not torch.cuda.is_initialized() and torch.cuda.device_count()==0
out=ROOT/"outputs/target_inset_glue_cpu_v1";out.mkdir(parents=True,exist_ok=True)
result={"status":"passed" if all(checks.values()) else "failed","count":len(checks),"checks":checks,
        "scope":"stdlib source/CLI and fake queue contracts; actual native apply_actions AST executed on CPU fakeentity tensors, not physical rollout",
        "plan_sha256":worker.sha(plan_path),"worker_sha256":worker.sha(worker.__file__),"queue_sha256":worker.sha(q.__file__),"CUDA_initialized":torch.cuda.is_initialized()}
(out/"checks.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
