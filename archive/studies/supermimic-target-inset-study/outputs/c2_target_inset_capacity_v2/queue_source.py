"""C2 capacity-audit v2 nine-job queue; CPU preflight default, no retry."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.util
from pathlib import Path
import shutil
import sys
import time

STARTED=time.monotonic()
ROOT=Path(__file__).resolve().parents[1]
STUDY=Path("D:/workspace/supermimic-chord-study")
DEFAULT_PLAN=ROOT/"configs/target-inset-capacity-v2.json"


def import_path(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def preflight(plan_path):
    worker=import_path(ROOT/"scripts/evaluate_target_inset_capacity_v2.py","inset_worker_contract_v2")
    plan=worker.validate(plan_path)
    worker.require(Path(plan["source_pins"]["queue"]["path"]).resolve()==Path(__file__).resolve(),"Wrong queue")
    for index,job in enumerate(plan["jobs"]):
        worker.require(job["block_id"]==index//3+1 and job["position"]==index%3+1,"Latin block/position changed")
        worker.require(job["margin_rad"]==job["margin_mrad"]/1000,"Margin units mismatch")
        expected=[plan["python"],"-u",plan["source_pins"]["worker"]["path"],"--plan",str(Path(plan_path).resolve()),
                  "--job-id",job["job_id"],"--output",job["output_dir"],"--run"]
        worker.require(job["command"]==expected,"Frozen worker command changed")
    lifecycle=import_path(plan["source_pins"]["lifecycle"]["path"],"reviewed_inset_lifecycle")
    lifecycle.ROOT=STUDY
    return plan,worker,lifecycle


def validate_capacity(plan,job,worker,meta,capacity):
    worker.require(capacity["status"]=="OK" and capacity["overflow_bitmask"]==0,"Sticky capacity not qualified")
    worker.require(meta["resets_during_rollout"]==capacity["resets_during_rollout"]==0,"Reset invalidates sticky scope")
    worker.require(capacity["read_count"]==1 and capacity["read_at_control_intervals"]==150,"Wrong observation timing/count")
    worker.require(capacity["actual_world_count"]==1 and capacity["flags_by_world"]==[0],"Wrong world flags")
    worker.require(capacity["job_id"]==job["job_id"] and capacity["plan_sha256"]==plan["plan_sha256_runtime"],"Capacity identity mismatch")
    worker.require(capacity["worker_sha256"]==plan["source_pins"]["worker"]["sha256"],"Capacity worker mismatch")


def health(plan,job,worker):
    output=Path(job["output_dir"])
    required=("rollout.npz","rollout.json","evaluation.json","model.mjb","reference.npz","reference.json","condition.json","control_targets_50hz.npz")
    worker.require(all((output/n).is_file() for n in required),"Missing complete child artifact")
    meta=worker.read(output/"rollout.json");condition=worker.read(output/"condition.json")
    runtime=worker.read(Path(job["wrapper_metadata_dir"])/"runtime.json")
    worker.require(meta["recorded_frames"]==151 and meta["recorded_control_intervals"]==150 and not meta["rollout_censored"],"Incomplete fixed3s horizon")
    worker.require(meta["checkpoint_sha256"]==plan["checkpoint"]["sha256"] and meta["seed"]==0,"Checkpoint/seed mismatch")
    worker.require(worker.sha(output/"model.mjb")==plan["expected_model_sha256"],"Model changed")
    worker.require(worker.sha(output/"reference.npz")==plan["reference"]["sha256"],"Reference changed")
    worker.require(condition["job_id"]==job["job_id"] and condition["margin_rad"]==job["margin_rad"] and condition["method"]=="C2_TARGET_INSET","Condition mismatch")
    worker.require(condition["control_npz_sha256"]==worker.sha(output/"control_targets_50hz.npz"),"Control log hash mismatch")
    worker.require(runtime["status"]=="completed" and runtime["control_intervals"]==150,"Wrapper incomplete")
    capacity_path=output/"capacity_audit.json"
    worker.require(capacity_path.is_file(),"Capacity audit missing")
    capacity=worker.read(capacity_path)
    validate_capacity(plan,job,worker,meta,capacity)
    worker.require(capacity["control_npz_sha256"]==worker.sha(output/"control_targets_50hz.npz") and capacity["rollout_npz_sha256"]==worker.sha(output/"rollout.npz"),"Capacity artifact linkage mismatch")
    worker.require(condition["capacity_audit_sha256"]==worker.sha(capacity_path) and runtime["capacity_audit"]==capacity,"Capacity record mismatch")
    return {"engineering_passed":True,"task_success":worker.read(output/"evaluation.json").get("task_success"),
            "quality_used_for_control":False,"artifact_sha256":{n:worker.sha(output/n) for n in required},
            "capacity_audit_sha256":worker.sha(capacity_path),"capacity_audit":capacity}


def run(plan_path,plan,worker,lifecycle):
    root=Path(plan["output_root"]);worker.require(not root.exists(),"No overwrite/retry")
    root.mkdir(parents=True)
    state={"status":"starting","pid":__import__("os").getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),
           "plan_sha256":worker.sha(plan_path),"training_updates":0,"method":"C2_TARGET_INSET",
           "jobs":[{**j,"status":"NOT_RUN"} for j in plan["jobs"]]}
    write=lambda:lifecycle.write(root/"state.json",state)
    write();shutil.copyfile(plan_path,root/"frozen_plan.json");shutil.copyfile(__file__,root/"queue_source.py")
    plan["plan_sha256_runtime"]=state["plan_sha256"]
    try:
        with lifecycle.gpu_lock():
            lifecycle.reject_project_gpu_workers()
            worker.require(worker.sha(plan_path)==state["plan_sha256"],"Plan changed after preflight")
            state["status"]="running";write()
            for job in state["jobs"]:
                remaining=240-(time.monotonic()-STARTED);cap=min(45,remaining-45)
                worker.require(cap>0,"Whole-queue scheduling budget exhausted")
                job.update(status="running",effective_child_cap_seconds=cap);write()
                observer=lambda pid,parent:lifecycle.observe_actual_pid(pid,parent,plan["source_pins"]["worker"]["path"],plan_path,job["output_dir"])
                process=lifecycle.watch(job["command"],Path(job["wrapper_metadata_dir"]),root/(job["job_id"]+"_process"),
                                        hard_seconds=cap,observer=observer)
                job["process"]=process
                worker.require(time.monotonic()-STARTED<=240,"Whole-queue budget exceeded after child")
                worker.require(process["status"]=="completed" and process["exit_code"]==0,"Engineering failure; no retry/next job")
                job["health"]=health(plan,job,worker)
                worker.require(time.monotonic()-STARTED<=240,"Whole-queue budget exceeded after health")
                job["status"]="completed";write()
            state["status"]="completed_9_jobs"
    except BaseException as error:
        for job in state["jobs"]:
            if job["status"]=="running":job["status"]="engineering_failed"
        state.update(status="engineering_failed",error=repr(error));raise
    finally:
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(),whole_queue_wall_seconds=time.monotonic()-STARTED,
                     whole_budget_scope="240s scheduling/stop-request budget with45s cleanup reserve; child independent timer<=45s, not OS hard realtime")
        write()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,default=DEFAULT_PLAN);p.add_argument("--run",action="store_true");a=p.parse_args()
    plan,worker,lifecycle=preflight(a.plan.resolve(strict=True))
    if a.run:run(a.plan.resolve(strict=True),plan,worker,lifecycle)
    else:print(__import__("json").dumps({"status":"passed_CPU_preflight","GPU_started":False,"plan_sha256":worker.sha(a.plan),"jobs":9,"source_pins":len(plan["source_pins"])},indent=2))


if __name__=="__main__":main()
