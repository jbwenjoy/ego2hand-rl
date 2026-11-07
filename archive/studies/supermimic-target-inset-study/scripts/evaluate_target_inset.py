"""Original evaluator with a zero-shot absolute-target inset and common logger.

No training/WM modification. The original t0 initialization is completed before
the first actual action interval enables the filter. Default is CPU preflight.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = Path("D:/workspace/supermimic-chord-study")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(plan_path):
    p = read(plan_path)
    require(p["method"] == "C2_TARGET_INSET" and p["training_updates"] == 0, "Wrong condition")
    require(p["training_seed"] == 1 and p["evaluation_seed"] == 0, "Frozen seeds changed")
    require([j["margin_mrad"] for j in p["jobs"]] == [0,5,20,20,0,5,5,20,0], "Latin order changed")
    require(len(p["jobs"]) == len({j["output_dir"] for j in p["jobs"]}) == 9, "Nine distinct jobs required")
    require(len({j["job_id"] for j in p["jobs"]}) == 9,"Nine distinct job identities required")
    require(p["per_child_cap_seconds"] == 45 and p["whole_queue_budget_seconds"] == 240, "Budget changed")
    require(p["automatic_retry"] is False and p["task_quality_stops_queue"] is False, "No retry/quality stopping")
    require(p["m0"] is None and p["a_brake"] is None and not p["C1_enabled"], "C1 remains disabled")
    for name,pin in p["source_pins"].items():
        require(sha(pin["path"]) == pin["sha256"], "Source changed: " + name)
    for name in ("checkpoint", "reference", "reference_metadata"):
        require(sha(p[name]["path"]) == p[name]["sha256"], "Input changed: " + name)
    for job in p["jobs"]:
        expected=["--reference",p["reference"]["path"],"--checkpoint",p["checkpoint"]["path"],
                  "--seed","0","--device","cuda:0","--output",job["output_dir"]]
        require(job["original_evaluator_args"]==expected,"Original evaluator arguments changed")
    return p


def import_path(path, name):
    spec = importlib.util.spec_from_file_location(name,path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run(plan_path, plan, job):
    started = time.perf_counter()
    output, metadata_dir = Path(job["output_dir"]), Path(job["wrapper_metadata_dir"])
    require(not output.exists() and not metadata_dir.exists(), "No overwrite or retry")
    metadata_dir.mkdir(parents=True)
    runtime = {"status":"running","pid":os.getpid(),"parent_pid":os.getppid(),
               "started_utc":datetime.now(timezone.utc).isoformat(),"plan_sha256":sha(plan_path),
               "job_id":job["job_id"],"margin_rad":job["margin_rad"]}
    def runtime_write():
        temporary=metadata_dir/"runtime.json.tmp"
        temporary.write_text(json.dumps(runtime,indent=2)+"\n",encoding="utf-8")
        os.replace(temporary,metadata_dir/"runtime.json")
    runtime_write()
    import numpy as np
    import torch
    import mjlab.envs
    sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(STUDY/"scripts"));sys.path.insert(0,str(STUDY/"src"))
    from target_inset import target_inset, validate_inset_domain
    evaluation = import_path(plan["source_pins"]["evaluator"]["path"],"frozen_inset_evaluator")
    native_class = mjlab.envs.ManagerBasedRlEnv
    rows,initial,bounds = [],{},{}

    def factory(*args,**kwargs):
        env = native_class(*args,**kwargs)
        term = env.action_manager.get_term("joint_pos")
        require(not term._mask_enabled,"Mask must remain disabled")
        original_process, original_step = term.process_actions, env.step
        active = False
        pending = None
        margin = job["margin_rad"]
        lo,hi = term.target_position_limits
        validate_inset_domain(lo,hi,margin)
        innerlo,innerhi = lo.to(torch.float32)+margin,hi.to(torch.float32)-margin
        aids = term._position_ctrl_ids.detach().cpu().numpy().astype(int)
        bounds.update(position_ctrl_ids=aids,joint_names=np.asarray(env.scene["robot"].joint_names),
                      lower_bound_rad=lo[0].cpu().numpy().copy(),upper_bound_rad=hi[0].cpu().numpy().copy(),
                      inset_lower_rad=innerlo[0].cpu().numpy().copy(),inset_upper_rad=innerhi[0].cpu().numpy().copy())

        def process(actions):
            nonlocal pending
            original_process(actions)
            if active:
                proposed = term._targets.detach().clone()
                ema_before = term._processed_actions.detach().clone()
                executed = target_inset(proposed,lo,hi,margin)
                term._targets.copy_(executed)
                require(torch.equal(ema_before,term._processed_actions),"EMA residual was modified")
                pending = {"u_proposed_original_bounded_rad":proposed[0].cpu().numpy().copy(),
                           "u_executed_filtered_rad":executed[0].cpu().numpy().copy(),
                           "ema_residual_before_filter_rad":ema_before[0].cpu().numpy().copy(),
                           "ema_residual_after_filter_rad":term._processed_actions[0].cpu().numpy().copy()}

        def step(action):
            nonlocal active,pending
            if not initial:
                # Original evaluator already performed reset/phase pin/zero
                # process/apply/write/forward/sense. Capture without a solve.
                initial.update(initial_qpos=env.sim.wp_data.qpos.numpy()[0].copy(),
                               initial_qvel=env.sim.wp_data.qvel.numpy()[0].copy(),
                               initial_ctrl=env.sim.wp_data.ctrl.numpy()[0].copy(),
                               initial_cached_target_rad=term._targets[0].detach().cpu().numpy().copy())
                initial["u_prev_rad"] = initial["initial_ctrl"][aids].copy()
            pending = None;active = True
            try:
                result = original_step(action)
            finally:
                active = False
            require(pending is not None,"Original process_actions did not execute")
            actual = env.sim.wp_data.ctrl.numpy()[0,aids].copy()
            cached = term._targets[0].detach().cpu().numpy().copy()
            require(np.array_equal(actual.astype(np.float32),pending["u_executed_filtered_rad"]),"Actual ctrl mismatches filtered target")
            require(np.array_equal(cached,pending["u_executed_filtered_rad"]),"Target cache changed during hold")
            rows.append({**pending,"u_cached_after_step_rad":cached,"u_actual_ctrl_readback_rad":actual,
                         "time_start_s":len(rows)*.02,"time_end_s":(len(rows)+1)*.02})
            return result
        term.process_actions = process
        env.step = step
        return env

    old_argv = sys.argv;mjlab.envs.ManagerBasedRlEnv = factory
    try:
        sys.argv = [plan["source_pins"]["evaluator"]["path"],*job["original_evaluator_args"]]
        evaluation.main()
        require(len(rows)==150,"Incomplete150 action intervals")
        arrays = {k:np.asarray([row[k] for row in rows]) for k in rows[0]}
        arrays.update(initial);arrays.update(bounds)
        np.savez_compressed(output/"control_targets_50hz.npz",**arrays)
        condition = {"schema":"C2_target_inset_zero_shot:1","method":"C2_TARGET_INSET",**job,
                     "training_seed":1,"evaluation_seed":0,"plan_sha256":sha(plan_path),
                     "source_hashes":{k:v["sha256"] for k,v in plan["source_pins"].items()},
                     "control_npz_sha256":sha(output/"control_targets_50hz.npz"),
                     "initialization":"original physical state and ctrl preserved; filter enabled only inside first real env.step",
                     "proposal":"absolute bounded _targets after original process_actions; not pre-hard-limit raw proposal",
                     "memory":"_processed_actions/c1/c2 remain original EMA residual; frozen_target becomes filtered absolute _targets; no new rate memory",
                     "hold":"original apply_actions called4 times/20ms, inset is subset of original hard/ctrl limits",
                     "logging":"same50Hz copies and FP32 clamp path for all margins including zero; no5ms hook or extra forward",
                     "WM_imagined_filter_validated":False,"dynamics_guarantee":False,"C1_enabled":False,"m0":None,"a_brake":None}
        (output/"condition.json").write_text(json.dumps(condition,indent=2)+"\n",encoding="utf-8")
        runtime["status"]="completed"
    except BaseException as error:
        runtime.update(status="engineering_failed",error=repr(error));raise
    finally:
        sys.argv=old_argv;mjlab.envs.ManagerBasedRlEnv=native_class
        runtime.update(finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-started,
                       control_intervals=len(rows));runtime_write()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--plan",type=Path,required=True);p.add_argument("--job-id",required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--run",action="store_true");a=p.parse_args()
    plan=validate(a.plan);job=next(j for j in plan["jobs"] if j["job_id"]==a.job_id)
    require(a.output.resolve()==Path(job["output_dir"]).resolve(),"Worker output differs from frozen job")
    if a.run:
        try:run(a.plan,plan,job)
        except BaseException as error:
            runtime_path=Path(job["wrapper_metadata_dir"])/"runtime.json"
            if runtime_path.is_file():
                failed=read(runtime_path);failed.update(status="engineering_failed",error=repr(error))
                runtime_path.write_text(json.dumps(failed,indent=2)+"\n",encoding="utf-8")
            raise
    else:print(json.dumps({"status":"passed_CPU_preflight","GPU_started":False,"job_id":job["job_id"]}))


if __name__=="__main__":main()
