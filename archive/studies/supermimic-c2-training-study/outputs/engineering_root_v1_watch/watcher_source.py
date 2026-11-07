"""Three fixed seed0 C2 engineering smokes. Default is CPU preflight only."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

STARTED=time.monotonic()
ROOT=Path(__file__).resolve().parents[1]
LIFECYCLE=Path("D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py")
ENTRY=ROOT/"scripts/train_controlled_pen.py"
CONFIGS=[ROOT/"configs/engineering_root_v1"/(arm+".json") for arm in ("a1l","p_only","a2")]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",action="store_true");args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("reviewed_engineering_lifecycle",LIFECYCLE)
    life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
    life.ROOT=ROOT
    life.require(life.sha(LIFECYCLE)=="40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa","Reviewed lifecycle source changed")
    source_guard={"entry_sha256":life.sha(ENTRY),"adapter_sha256":life.sha(ROOT/"src/c2_training_adapter.py"),
                  "pure_sha256":life.sha(ROOT/"src/target_inset.py")}
    life.require(source_guard["entry_sha256"]=="458f52dd54c208d4f521c34d4737f9930e1aa179474cb197bcc11e001cf38fe8" and source_guard["adapter_sha256"]=="4ec7f2c27b8e2a041b8d6929b4f345a02cdabe801215aa28d0340905fb34e6b0","Reviewed C2 training source changed")
    jobs=[]
    for file,arm in zip(CONFIGS,("A1L","P_ONLY","A2")):
        config=life.read(file)
        life.require(config["method"]==arm and config["seed"]==0 and config["run_role"]=="engineering_smoke","Engineering arm/seed/role mismatch")
        training=config["training"]
        life.require(training["max_iterations"]==10 and training["wm_only_iters"]==2 and training["horizon"]==8 and training["num_envs"]==128,"Only fixed real10iter/H8 smoke is allowed")
        life.require(config["compile_fallback"] is False and config["max_attempt_seconds"]==600,"No fallback/retry or changed native cap")
        life.require(config["c2_controller"]=={"kind":"C2_TARGET_INSET","margin_rad":.020,"checkpoint_format_version":103,"resume":"from_scratch_only"},"Controller contract mismatch")
        target=Path(config["output_dir"])
        target=target.resolve() if target.is_absolute() else (ROOT/target).resolve()
        jobs.append({"arm":arm,"config":str(file),"config_sha256":life.sha(file),"output":str(target),"status":"NOT_RUN"})
    if not args.run:
        print(json.dumps({"status":"CPU_preflight_passed","GPU_started":False,"jobs":jobs,
            "entry_sha256":life.sha(ENTRY),"adapter_sha256":life.sha(ROOT/"src/c2_training_adapter.py"),
            "per_child_hard_cap_seconds":720,"native_attempt_cap_seconds":600,"whole_scheduling_budget_seconds":2300},indent=2));return
    output=ROOT/"outputs/engineering_root_v1_watch"
    life.require(not output.exists(),"No overwrite/retry of prior engineering queue")
    output.mkdir(parents=True)
    state={"status":"running","jobs":jobs,"training_seeds":[0],"formal_training":False,"automatic_retry":False,
           "source_guard":source_guard,"lifecycle_sha256":life.sha(LIFECYCLE),"watcher_sha256":life.sha(__file__)}
    write=lambda:life.write(output/"state.json",state)
    write();shutil.copy2(__file__,output/"watcher_source.py")
    try:
        with life.gpu_lock():
            life.reject_project_gpu_workers()
            for job in jobs:
                life.require(life.sha(job["config"])==job["config_sha256"],"Engineering config changed")
                remaining=2300-(time.monotonic()-STARTED);cap=min(720,remaining-45)
                life.require(cap>0,"Engineering scheduling budget exhausted")
                job["status"]="running";write()
                command=[sys.executable,"-u",str(ENTRY),"--config",job["config"],"--run"]
                def observe(pid,parent):
                    # CLI pins config; output is read from that pinned config,
                    # not a separate command-line argument.
                    record=life.observe_actual_pid(pid,parent,ENTRY,job["config"],job["config"])
                    record["identity_basis"]="exact PID/parent/entry/config; output is config-bound"
                    return record
                job["process"]=life.watch(command,Path(job["output"]),output/job["arm"].lower(),hard_seconds=cap,observer=observe)
                life.require(job["process"]["status"]=="completed" and job["process"]["exit_code"]==0,"Engineering child failed; stop without retry")
                result=life.read(Path(job["output"])/"compiled/result.json")
                life.require(result["status"]=="passed" and result["actual_optimizer_steps"]=={"wm":20,"fail_head":20,"actor":8,"critic":32} and result["collected_sim_transitions"]==11392,"Engineering update/sample budget mismatch")
                job.update(status="passed",actual_updates=result["actual_optimizer_steps"],real_transitions=result["collected_sim_transitions"]);write()
            state["status"]="passed_three_seed0_smokes"
    except BaseException as error:
        for job in jobs:
            if job["status"]=="running":job["status"]="engineering_failed"
        state.update(status="engineering_failed",error=repr(error));raise
    finally:
        state["whole_wall_seconds"]=time.monotonic()-STARTED;write()


if __name__=="__main__":main()
