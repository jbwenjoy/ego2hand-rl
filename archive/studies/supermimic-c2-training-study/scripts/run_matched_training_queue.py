"""Single-GPU, detached-capable queue for nine prospective C2_020 matched-seed runs.

The parent uses only the standard library. Health checks run in separate CPU
processes. Goal pause has no effect on this process or its training child.
There is no resume, automatic retry, or physical-quality-based stopping.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[1]
LIFECYCLE=Path("D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py")
LIFECYCLE_SHA="40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


def validate_gate_evidence(manifest):
    """Read measured evidence; manifest booleans cannot stand in for a run."""
    bindings=manifest["source_bindings"]
    expected={"cpu_real_imag","cuda_real_imag","smoke_A1L","smoke_P_ONLY","smoke_A2","pure_projection_AMP","legacy_format103_rejection"}
    assert set(manifest["gate_evidence"])==expected
    for name,pin in manifest["gate_evidence"].items():
        assert digest(pin["path"])==pin["sha256"],"Gate artifact changed: "+name
        proof=read(pin["path"])
        assert proof.get("status")=="passed"
        if name in {"cpu_real_imag","cuda_real_imag"}:
            assert proof["adapter_sha256"]==bindings["adapter"]["sha256"]
            assert proof["pure_projection_sha256"]==bindings["pure_projection"]["sha256"]
            assert digest(pin["script"])==proof["script_sha256"]==pin["script_sha256"]
            assert proof["model_sha256"]==manifest["nominal_model_sha256"]
            assert proof["training_updates"]==0 and proof["GPU_requested"]==(name=="cuda_real_imag")
            cpu=proof["cpu"]
            assert cpu["passed"] is True and len(cpu["groups"])==16 and all(g["passed"] for g in cpu["groups"])
            assert digest(cpu["native_actor_control_module"])==cpu["native_actor_control_sha256"]
            assert digest(cpu["fixture_helper"])==cpu["fixture_helper_sha256"]
            if name=="cuda_real_imag":
                assert proof["warp"]["passed"] is True and len(proof["warp"]["groups"])==16
                assert all(g["passed"] for g in proof["warp"]["groups"])
        elif name.startswith("smoke_"):
            method=name.removeprefix("smoke_")
            assert proof["audit_script_sha256"]==manifest["audit_script_sha256"]
            assert proof["iteration"]==10 and proof["real_simulator_transitions"]==11392
            assert proof["actual_optimizer_steps"]=={"wm":20,"fail_head":20,"actor":8,"critic":32}
            assert proof["checkpoint_format"]==103 and all(proof["checks"].values())
            run=Path(proof["run"]);cfg=read(run/"input_config.json")
            assert cfg["seed"]==0 and cfg["method"]==method and cfg["run_role"]=="engineering_smoke"
            assert cfg["training"]["horizon"]==cfg["training"]["max_horizon"]==8 and cfg["training"]["amp"] is True
            contract=read(run/"source/c2_controller/controller_contract.json")
            assert contract["input_config_sha256"]==digest(run/"input_config.json")
            assert contract["base_method"]==method and contract["training_seed"]==0
            for label,source in contract["files"].items():
                assert source["sha256"]==bindings[label]["sha256"]
                assert digest(run/"source/c2_controller"/source["filename"])==source["sha256"]
            assert digest(proof["checkpoint"])==proof["checkpoint_sha256"]
            assert digest(proof["actual_final_checkpoint"]["checkpoint"])==proof["actual_final_checkpoint"]["sha256"]
            coverage=proof["C2_actual_u_evidence"]
            assert coverage["status"]=="PASSED_CHECKPOINT_PREFIX" and coverage["prefix_last_collect_call"]==11
            assert coverage["verified_real_transitions"]==11392 and all(coverage["checks"].values())
            rows=[json.loads(line) for line in (run/"compiled/metrics.jsonl").read_text().splitlines()]
            policy=[row for row in rows if int(row["iteration"])>=2]
            assert len(policy)==8 and all(finite(row) and row["training/horizon"]==8 for row in policy)
            if method!="A1L":
                assert all("prior/geometry_intent_bonus_unscaled" in row for row in policy)
                if method=="P_ONLY":
                    assert all(row["prior/geometry_intent_support_coefficient"]==0 and abs(row["prior/geometry_intent_penetration_coefficient"]-.1)<1e-7 for row in policy)
                    assert all(abs(row["prior/geometry_intent_bonus_unscaled"]+.1*row["prior/geometry_intent_penetration_prior"])<1e-6 for row in policy)
        elif name=="pure_projection_AMP":
            assert proof["pure_sha256"]==bindings["pure_projection"]["sha256"]
            assert digest(pin["script"])==proof["script_sha256"]==pin["script_sha256"]
            assert proof["cpu"]["passed"] is True and proof["cuda"]["passed"] is True
            assert proof["GPU_requested"] is True and proof["training_updates"]==0
        else:
            assert digest(pin["script"])==proof["source_sha256"]==pin["script_sha256"]
            assert pin["source_binding_sha256"]==bindings["original_trainer"]["sha256"]
            assert proof["CUDA_initialized"] is False and len(proof["rows"])==3
            assert {r["arm"] for r in proof["rows"]}=={"a1l","p_only","a2"}
            for row in proof["rows"]:
                assert row["format103"] is True and digest(row["checkpoint"])==row["checkpoint_sha256"]
                assert set(row["guards"])=={"load","load_model_only","native_play"}
                assert all(g["rejected"] is True for g in row["guards"].values())


def validate(manifest, for_execute=False):
    assert manifest["schema"] == "C2_020_prospective_training_queue:1"
    pending = manifest.get("status") != "frozen_ready_for_authorization" or not manifest.get("caps_frozen_from_seed0_smoke")
    if pending:
        assert not for_execute, "Draft/pending runtime gates cannot execute"
        return {"status":"PENDING_NOT_EXECUTABLE", "GPU_started":False,
                "reason":"Source/config/gates/caps await measured seed0 smoke and Root review"}
    if for_execute:
        assert manifest.get("formal_training_authorized") is True, "Formal queue not authorized"
        assert manifest.get("background_queue_continues_when_Goal_paused") is True
    assert digest(__file__) == manifest["queue_runner_sha256"], "Queue changed"
    assert manifest["checkpoint_format_version"] == 103 and manifest["seeds"] == [4,5,6]
    for key, sha_key in (("trainer","trainer_sha256"),("audit_script","audit_script_sha256"),
                         ("source_inventory","source_inventory_sha256"),
                         ("paired_protocol_file","paired_protocol_sha256"),
                         ("prospective_protocol_file","prospective_protocol_sha256"),
                         ("schedule_file","schedule_sha256")):
        assert digest(manifest[key])==manifest[sha_key], "Frozen file changed: "+key
    schedule=read(manifest["schedule_file"])["training_jobs"]
    assert len(manifest["jobs"])==len(schedule)==9
    inventory=read(manifest["source_inventory"])
    for relative,expected in inventory["files"].items():
        assert digest(Path(inventory["dex_root"])/relative)==expected, "Native source changed: "+relative
    for name,pin in manifest["source_bindings"].items():
        assert digest(pin["path"])==pin["sha256"], "Extension source changed: "+name
    assert Path(manifest["python_executable"]).is_file()
    validate_gate_evidence(manifest)
    caps=manifest["method_attempt_seconds"]
    assert set(caps)=={"A1L","P_ONLY","A2"} and all(isinstance(v,(int,float)) and 0<v<100000 for v in caps.values())
    assert manifest["per_run_process_watchdog_seconds"]>=max(caps.values())
    estimates=manifest["runtime_estimate_seconds_by_method_from_engineering"]
    assert set(estimates)==set(caps) and all(0<estimate["low"]<=estimate["high"] and finite(estimate) for estimate in estimates.values())
    assert finite(manifest["estimated_total_seconds_from_pilot_costs"])
    configurations=[]
    for position,(job,declared) in enumerate(zip(manifest["jobs"],schedule),1):
        assert job["position"]==position and job["method"]==declared["trained_method"] and job["seed"]==declared["training_seed"]
        assert job["model_id"]==declared["model_id"] and digest(job["config"])==job["config_sha256"]
        cfg=read(job["config"]);template=manifest["templates"][job["method"]]
        assert digest(template["path"])==template["sha256"];base=read(template["path"])
        normalized=dict(cfg)
        for key in ("label","seed","output_dir","max_attempt_seconds"):normalized[key]=base[key]
        assert normalized==base, "Unpermitted formal config change"
        assert cfg["method"]==job["method"] and cfg["seed"]==job["seed"]
        assert cfg["c2_controller"]=={"kind":"C2_TARGET_INSET","margin_rad":.020,"checkpoint_format_version":103,"resume":"from_scratch_only"}
        assert cfg["c2_protocol_sha256"]==manifest["prospective_protocol_sha256"]
        assert digest(ROOT/cfg["paired_protocol_file"])==manifest["paired_protocol_sha256"]
        assert cfg["max_attempt_seconds"]==caps[job["method"]] and cfg["compile_fallback"] is False
        training=cfg["training"]
        assert not ({"checkpoint","model_checkpoint"}&set(training))
        for key,value in {"max_iterations":2000,"num_envs":128,"wm_only_iters":200,"horizon":8,"max_horizon":8,"wm_batches":2,"critic_epochs":2,"critic_minibatches":2}.items():
            assert training.get(key)==value, "Formal budget/config mismatch: "+key
        destination=(ROOT/cfg["output_dir"]).resolve()
        assert destination==Path(job["output"]).resolve() and destination.is_relative_to(ROOT/"outputs")
        assert (job["method"] in {"P_ONLY","A2"})==bool(cfg.get("contact_objective"))
        if cfg.get("contact_objective"):
            c=cfg["contact_objective"];bundle=Path(c["bundle_dir"]);reference=Path(c["reference_file"])
            files={"manifest_sha256":bundle/"manifest.json","target_npz_sha256":bundle/"contact_targets.npz",
                "rules_sha256":bundle/"rules.json","raw_geometry_sha256":bundle/"raw_geometry.json","time_windows_sha256":bundle/"time_windows.json",
                "reference_npz_sha256":reference,"reference_json_sha256":reference.with_suffix(".json"),"model_sha256":Path(c["model_file"]),
                "objective_module_sha256":ROOT/"src/chord_study/contact_objective.py","geometry_module_sha256":ROOT/"src/chord_study/contact_geometry.py"}
            assert all(digest(path)==c["expected_hashes"][key] for key,path in files.items())
            assert c["per_control_step_scale"]==.02 and c["qualification_status"]=="local_proxy_qualification_failed_exploratory_ablation"
        configurations.append(cfg)
    assert len({j["output"] for j in manifest["jobs"]})==9
    assert all(c["training"]==configurations[0]["training"] and c["reference"]==configurations[0]["reference"] for c in configurations)
    return {"status":"passed_CPU_frozen_queue_contract", "GPU_started":False, "checked_utc":utc(),
            "native_files_verified":len(inventory["files"]),"job_count":9,"no_GPU_or_torch_import":"torch" not in sys.modules,
            "formal_execution_authorized":manifest.get("formal_training_authorized") is True}


@contextmanager
def gpu_queue_lock():
    import msvcrt
    # One physical GPU is shared with the original study; use its exact lock.
    path = Path("D:/workspace/supermimic-chord-study/.locks/single_gpu_training.lock")
    path.parent.mkdir(exist_ok=True)
    with path.open("a+b") as stream:
        if not path.stat().st_size:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def reject_external_GPU_work():
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.CommandLine -match 'train_controlled_pen|controlled_pen_rollout|evaluate_pen_checkpoint|verify_warp_hold|check_a2_cuda|probe_a2_whole|isolate_a2_cuda|reproduce_a2' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", query], capture_output=True, text=True, check=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
    assert not result.stdout.strip(), f"Existing training/evaluation process: {result.stdout.strip()}"


def complete_checkpoint(run, iteration):
    try:
        path = run / f"compiled/train/model_{iteration}.pt"
        meta = read(path.with_suffix(".meta.json"))
        data = (run / "compiled/checkpoint_index.jsonl").read_bytes()
        data = data[:data.rfind(b"\n")+1]
        return path.is_file() and meta in [json.loads(line) for line in data.decode().splitlines()]
    except (OSError, ValueError):
        return False


def audit(manifest, job, iteration, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".stdout.log").open("w", encoding="utf-8") as out, destination.with_suffix(".stderr.log").open("w", encoding="utf-8") as err:
        process = subprocess.run([manifest["python_executable"], manifest["audit_script"], "--run", job["output"],
                                  "--iteration", str(iteration), "--output-dir", str(destination), "--queue-manifest", manifest["manifest_path"]],
                                 stdout=out, stderr=err, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}, timeout=120)
    assert process.returncode == 0 and read(destination / "report.json")["status"] == "passed", f"CPU health audit failed: {destination}"


def first100_handoff(output,manifest,job,record,destination,elapsed,child_running=True):
    """A health/ETA handoff only; the detached child/queue remain running."""
    health=read(destination/"report.json")
    estimates=manifest["runtime_estimate_seconds_by_method_from_engineering"]
    remaining={bound:sum(estimates[j["method"]][bound] for j in manifest["jobs"]) - elapsed for bound in ("low","high")}
    handoff={"status":"FIRST100_HEALTH_PASSED_BACKGROUND_CONTINUES","created_utc":utc(),
        "queue_pid":os.getpid(),"actual_training_pid":record.get("actual_pid"),
        "actual_process_OS_observation":record.get("actual_process_OS_observation"),
        "first_model":job["model_id"],"checkpoint":health["checkpoint"],"checkpoint_sha256":health["checkpoint_sha256"],
        "checkpoint_health":str(destination),"phase":health["phase"],"actual_optimizer_steps":health["actual_optimizer_steps"],
        "real_simulator_transitions":health["real_simulator_transitions"],
        "remaining_queue_seconds_engineering_range":{k:max(0,v) for k,v in remaining.items()},
        "ETA_scope":"Seed0 engineering phase estimates minus actual elapsed, not20timesWM-only100 wall; actor/prior cost and cache uncertainty retained.",
        "Goal_action_for_Root":"Report scoped health/actual PID/ETA then pause Goal as user requested; this function does not call Goal tools or terminate the expressly authorized background queue.",
        "background_queue_continues":True,"physical_task_quality_assessed":False}
    handoff["training_child_alive_at_handoff"]=child_running
    write(output/"first100_handoff.json",handoff)
    print(json.dumps(handoff),flush=True)


def stop_owned_child(process):
    """Reuse the reviewed10s-taskkill/5s-reap primitive, never foreign PIDs."""
    assert digest(LIFECYCLE)==LIFECYCLE_SHA,"Reviewed lifecycle changed"
    spec=importlib.util.spec_from_file_location("reviewed_C2_queue_stop",LIFECYCLE)
    lifecycle=importlib.util.module_from_spec(spec);spec.loader.exec_module(lifecycle)
    return lifecycle.stop_owned(process)


class OwnedDeadline:
    """Independent stop request remains active during blocking CPU audit/CIM."""
    def __init__(self,seconds,stop=stop_owned_child):
        assert math.isfinite(seconds) and seconds>0
        self.seconds=seconds;self.stop=stop;self.process=None;self.details={}
        self.expired=threading.Event();self.done=threading.Event();self.cleanup_finished=threading.Event();self.stop_lock=threading.Lock()
        self.before=time.monotonic();self.timer=threading.Timer(seconds,self.timeout);self.timer.daemon=True

    def start(self):
        self.timer.start()  # Must be called before Popen: includes launch/setup.

    def cleanup(self,wait=.1):
        if self.process is None:return
        if not self.stop_lock.acquire(timeout=wait):return
        try:self.details["cleanup"]=self.stop(self.process)
        except BaseException as error:self.details["cleanup_error"]=repr(error)
        finally:self.stop_lock.release();self.cleanup_finished.set()

    def timeout(self):
        if self.expired.is_set():return
        if self.done.is_set() or (self.process is not None and self.process.poll() is not None):return
        self.expired.set();self.details.update(kill_requested_utc=utc(),kill_requested_elapsed_s=time.monotonic()-self.before)
        if self.process is not None:self.cleanup()
        else:self.details["launch_in_progress_at_deadline"]=True

    def attach(self,process):
        self.process=process
        if self.expired.is_set():self.cleanup()

    def finish(self,on_error=False):
        self.done.set();self.timer.cancel()
        error_stop=on_error and self.process is not None and self.process.poll() is None
        if error_stop and not self.expired.is_set():self.cleanup()
        self.timer.join(timeout=16)
        cleanup=self.details.get("cleanup",{})
        uncertain=(self.expired.is_set() or error_stop) and not cleanup.get("owned_tree_termination_confirmed",False)
        return {"expired":self.expired.is_set(),"details":dict(self.details),"cleanup_unverified":uncertain,
                "scope":"Stop requested at independent deadline; bounded cleanup only, not OS hard-real-time. Unknown descendants/GPU idleness are never treated as confirmed."}


def observe_actual_process(pid, launcher_pid, trainer, config):
    """One CIM observation; absence never masquerades as an OS timestamp."""
    query = (f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}'; "
             "if($null -eq $p){throw 'Actual child PID no longer observable'}; "
             "[ordered]@{pid=$p.ProcessId;parent_pid=$p.ParentProcessId;"
             "creation_date_utc=$p.CreationDate.ToUniversalTime().ToString('o');"
             "creation_date_local=$p.CreationDate.ToString('o');command_line=$p.CommandLine;"
             "executable_path=$p.ExecutablePath}|ConvertTo-Json -Compress")
    try:
        completed = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
                                   capture_output=True, text=True, check=True, timeout=15, creationflags=subprocess.CREATE_NO_WINDOW)
        result = json.loads(completed.stdout)
        command = result["command_line"].lower().replace("/", "\\")
        assert result["pid"] == int(pid) and result["parent_pid"] == launcher_pid
        assert str(trainer).lower().replace("/", "\\") in command and str(config).lower().replace("/", "\\") in command
        return {"status": "observed_exact_PID_and_identity", "observed_utc": utc(), **result}
    except Exception as error:
        return {"status": "missing_or_identity_unverified", "observed_utc": utc(), "error": repr(error),
                "strict_cutoff_use": "Use conservative launch/runtime-observation bounds; not an exact OS start timestamp."}


def execute_impl(manifest_path, manifest):
    output = Path(manifest["queue_output"])
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "state.json").exists(), "Queue already started; no automatic retry/resume"
    for job in manifest["jobs"]:
        assert not Path(job["output"]).exists(), f"Refusing prior run output: {job['output']}"
    state = {"status": "running", "pid": os.getpid(), "started_utc": utc(), "manifest_sha256": digest(manifest_path),
             "jobs": [], "pause_Goal_leaves_background_running": True,
             "ETA_seconds_from_seed0_costs_excluding_audit_overhead": manifest["estimated_total_seconds_from_pilot_costs"]}
    with gpu_queue_lock():
        assert not (output / "state.json").exists(), "Queue already started"
        shutil.copy2(manifest_path, output / "manifest.json")
        shutil.copy2(__file__, output / "queue_source.py")
        write(output / "state.json", state)
        for job in manifest["jobs"]:
            process = None
            deadline = None
            record = {**job, "status": "starting", "started_utc": utc()}
            state["jobs"].append(record)
            state["current_job"] = job["position"]
            write(output / "state.json", state)
            try:
                validate(manifest)
                reject_external_GPU_work()
                logs = output / f"job_{job['position']:02d}_{job['method'].lower()}_seed{job['seed']}"
                logs.mkdir()
                started = time.perf_counter()
                deadline=OwnedDeadline(manifest["per_run_process_watchdog_seconds"])
                deadline.start()
                with (logs / "stdout.log").open("w", encoding="utf-8") as out, (logs / "stderr.log").open("w", encoding="utf-8") as err:
                    record["Popen_prelaunch_utc"] = utc()
                    process = subprocess.Popen([manifest["python_executable"], "-u", manifest["trainer"], "--config", job["config"], "--run"],
                                               cwd=ROOT, stdout=out, stderr=err,
                                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
                    deadline.attach(process)
                    record.update(status="running", launcher_pid=process.pid, logs=str(logs), Popen_postlaunch_utc=utc())
                    first_audited = False
                    while process.poll() is None:
                        elapsed = time.perf_counter()-started
                        if deadline.expired.is_set() and (deadline.cleanup_finished.is_set() or elapsed>manifest["per_run_process_watchdog_seconds"]+20):
                            raise TimeoutError("Independent child deadline expired; no retry/next job")
                        run = Path(job["output"])
                        try:
                            runtime = read(run / "runtime.json")
                            record.update(actual_pid=runtime["pid"], runtime_started_utc=runtime["started_utc"])
                            if "actual_process_OS_observation" not in record:
                                observed = utc()
                                record["runtime_first_observed_utc"] = observed
                                record["actual_process_creation_conservative_bounds"] = {
                                    "lower_utc": record["Popen_prelaunch_utc"], "upper_utc": observed,
                                    "scope": "Actual Python child is spawned asynchronously by venv launcher; Popen postlaunch alone is not its upper bound."}
                                record["actual_process_OS_observation"] = observe_actual_process(
                                    runtime["pid"], process.pid, manifest["trainer"], job["config"])
                        except (OSError, ValueError):
                            pass
                        metrics_path = run / "compiled/metrics.jsonl"
                        if metrics_path.exists():
                            data = metrics_path.read_bytes(); data = data[:data.rfind(b"\n")+1]
                            assert all(finite(json.loads(line)) for line in data.decode().splitlines()), "Nonfinite metric row"
                        if not first_audited and complete_checkpoint(run, 100):
                            destination = logs / "first100_cpu_audit"
                            audit(manifest, job, 100, destination)
                            if deadline.expired.is_set():raise TimeoutError("Deadline expired during CPU health; no live-background handoff")
                            if process.poll() is not None and process.returncode!=0:raise RuntimeError("Training failed during CPU health")
                            first_audited = True
                            record.update(first_checkpoint_health="passed", first_checkpoint_audit=str(destination),
                                          first_checkpoint_checked_utc=utc())
                            if job["position"]==1:
                                first100_handoff(output,manifest,job,record,destination,time.perf_counter()-started,process.poll() is None)
                        record["reserved_wall_seconds_so_far"] = time.perf_counter()-started
                        write(output / "state.json", state)
                        time.sleep(1)
                record["deadline"]=deadline.finish()
                assert not record["deadline"]["expired"], "Independent child deadline expired"
                assert process.returncode == 0, f"Training exited {process.returncode}"
                if not first_audited:
                    audit(manifest, job, 100, logs / "first100_cpu_audit")
                    record["first_checkpoint_health"] = "passed"
                audit(manifest, job, 2000, logs / "final2000_cpu_audit")
                summary = read(Path(job["output"]) / "summary.json")
                attempt = summary["attempts"][-1]
                assert summary["status"] == "passed" and attempt["final_iteration"] == 2000
                assert attempt["actual_optimizer_steps"] == {"wm":4000,"fail_head":4000,"actor":1800,"critic":7200}
                assert attempt["collected_sim_transitions"] == 648192
                final_health=read(logs/"final2000_cpu_audit/report.json")
                assert final_health["actual_final_checkpoint"]["sha256"]==attempt["checkpoint_sha256"]
                assert Path(final_health["actual_final_checkpoint"]["checkpoint"]).resolve()==Path(attempt["checkpoint"]).resolve()
                record.update(status="passed", finished_utc=utc(), reserved_wall_seconds_including_CPU_health=time.perf_counter()-started,
                              training_STARTED_wall_seconds=summary["total_wall_seconds_including_failed_attempts"],
                              final_checkpoint=attempt["checkpoint"], final_checkpoint_sha256=attempt["checkpoint_sha256"])
                write(output / "state.json", state)
            except BaseException as error:
                bounded=deadline.finish(on_error=True) if deadline is not None else {"expired":False,"cleanup_unverified":False}
                record["deadline"]=bounded
                label="timeout_cleanup_unverified" if bounded["expired"] and bounded["cleanup_unverified"] else "failed_cleanup_unverified" if bounded["cleanup_unverified"] else "timeout" if bounded["expired"] else "failed"
                record.update(status=label, error=repr(error), finished_utc=utc())
                state.update(status=label+"_stopped_no_retry", finished_utc=utc())
                if bounded["cleanup_unverified"]:
                    state.update(GPU_idleness_confirmed=False,shared_lock_release_scope="Released only after bounded cleanup attempt; descendants/GPU idleness unconfirmed. No retry or next job.")
                write(output / "state.json", state)
                raise
    state.update(status="passed", finished_utc=utc())
    write(output / "state.json", state)


def execute(manifest_path, manifest):
    try:
        validate(manifest, for_execute=True)
        manifest={**manifest,"manifest_path":str(manifest_path)}
        execute_impl(manifest_path, manifest)
    except BaseException as error:
        output = Path(manifest["queue_output"])
        output.mkdir(parents=True, exist_ok=True)
        try:
            state = read(output / "state.json")
        except (OSError, ValueError):
            state = {"pid": os.getpid(), "jobs": []}
        failure = {"pid": os.getpid(), "status": "failed_no_child_started", "error": repr(error), "finished_utc": utc()}
        if state.get("pid") != os.getpid():
            # Never overwrite a prior/competing queue's state while reporting a
            # failed launch of this distinct process.
            write(output / f"startup_failure_pid{os.getpid()}.json", failure)
        else:
            if "cleanup_unverified" in state.get("status",""):
                failure["status"]=state["status"]
                failure["GPU_idleness_confirmed"]=False
            elif any(job.get("launcher_pid") for job in state.get("jobs", [])):
                failure["status"] = "failed_stopped_no_retry"
            state.update(failure)
            write(output / "state.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = read(args.manifest)
    if args.validate_only:
        result = validate(manifest)
        print(json.dumps(result, indent=2)); return
    execute(args.manifest.resolve(), manifest)


if __name__ == "__main__":
    main()
