"""Tiny CPU contracts for the repeat-driver glue; fake children, no simulator."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("repeat_driver", ROOT / "scripts/run_pen_evaluator_repeats.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plan_path = ROOT / "configs/evaluator-repeatability-a1l-seed1-v1.json"
plan, _ = m.preflight(plan_path)
checks = {"default_preflight_no_torch_warp": not any(k in sys.modules for k in ("torch","warp","mujoco_warp"))}

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    bad = dict(plan); bad["evaluation_seed"] = 1
    (root/"bad.json").write_text(json.dumps(bad),encoding="utf-8")
    try:m.preflight(root/"bad.json")
    except ValueError:checks["changed_eval_seed_rejected"] = True
    else:checks["changed_eval_seed_rejected"] = False
    fake = root/"fake_artifact";fake.mkdir()
    for name in ("rollout.npz","reference.npz","reference.json","model.mjb"):(fake/name).write_bytes(b"mock CPU contract only")
    meta={"recorded_frames":151,"recorded_control_intervals":150,"rollout_censored":False,"seed":0,
          "checkpoint_sha256":"fake_checkpoint","resets_during_rollout":0,"object_state_writes_after_initialization":0}
    (fake/"rollout.json").write_text(json.dumps(meta),encoding="utf-8")
    (fake/"evaluation.json").write_text(json.dumps({"task_success":False}),encoding="utf-8")
    fake_plan={**plan,"checkpoint":{"sha256":"fake_checkpoint"},"reference":{"sha256":m.sha(fake/"reference.npz")},
               "expected_model_sha256":m.sha(fake/"model.mjb"),"plan_sha256_runtime":"fake_plan"}
    result=m.child_artifacts(fake_plan,{"output_dir":str(fake),"repeat_id":"repeat1"})
    checks["task_false_is_complete_not_engineering_failure"] = result["complete_engineering_horizon"] and result["task_success"] is False

    class FakeLifecycle:
        def __init__(self,fail):self.calls=0;self.fail=fail
        def gpu_lock(self):return contextlib.nullcontext()
        def reject_project_gpu_workers(self):pass
        def write(self,path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding="utf-8")
        def watch(self,*args,**kwargs):
            self.calls+=1
            assert 0 < kwargs["hard_seconds"] <= 45
            return {"status":"failed" if self.fail else "completed","exit_code":1 if self.fail else 0}
    for failing in (True,False):
        output=root/("fail" if failing else "all_false")
        test_plan={**plan,"output_root":str(output),"repeats":[{**r,"output_dir":str(output/f"repeat_{i:02d}")} for i,r in enumerate(plan["repeats"],1)]}
        lifecycle=FakeLifecycle(failing)
        original=m.child_artifacts
        m.child_artifacts=lambda *_:{"task_success":False,"quality_used_for_control":False}
        try:
            try:m.run(plan_path,test_plan,lifecycle)
            except ValueError:
                assert failing
        finally:m.child_artifacts=original
        state=m.read(output/"state.json")
        if failing:
            checks["engineering_failure_stops_once_no_retry"] = lifecycle.calls==1 and state["status"]=="engineering_failed" and sum(j["status"]=="NOT_RUN" for j in state["jobs"])==4
        else:
            checks["five_task_false_results_all_retained"] = lifecycle.calls==5 and state["status"]=="completed_5_repeats" and all(j["artifacts"]["task_success"] is False for j in state["jobs"])
out=ROOT/"outputs/evaluator_repeatability_cpu_v1";out.mkdir(parents=True,exist_ok=True)
record={"status":"passed" if all(checks.values()) else "failed","checks":checks,"count":len(checks),"scope":"fake CPU glue contracts only, no subprocess/GPU/physics","driver_sha256":m.sha(m.__file__),"plan_sha256":m.sha(plan_path)}
(out/"checks.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
print(json.dumps(record,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
