"""Single-GPU, detached-capable queue for six frozen matched-seed runs.

The parent uses only the standard library. Health checks run in separate CPU
processes. Goal pause has no effect on this process or its training child.
There is no resume, automatic retry, or physical-quality-based stopping.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


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


def validate(manifest):
    assert manifest["schema"] == "supermimic_chord_replication_queue:1"
    assert digest(__file__) == manifest["queue_runner_sha256"], "Queue runner changed"
    for key, sha_key in (("trainer", "trainer_sha256"), ("audit_script", "audit_script_sha256"),
                         ("source_inventory", "source_inventory_sha256"),
                         ("paired_protocol_file", "paired_protocol_sha256")):
        assert digest(manifest[key]) == manifest[sha_key], f"Frozen {key} changed"
    inventory = read(manifest["source_inventory"])
    for relative, expected in inventory["files"].items():
        assert digest(Path(inventory["dex_root"]) / relative) == expected, f"Dex source changed: {relative}"
    assert Path(manifest["python_executable"]).is_file()
    assert len(manifest["jobs"]) == 6
    configurations = []
    for position, job in enumerate(manifest["jobs"], 1):
        assert job["position"] == position and digest(job["config"]) == job["config_sha256"]
        cfg = read(job["config"])
        template = manifest["templates"][job["method"]]
        assert digest(template["path"]) == template["sha256"]
        base = read(template["path"])
        normalized = dict(cfg)
        for key in ("label", "seed", "output_dir", "max_attempt_seconds"):
            normalized[key] = base[key]
        assert normalized == base, f"Nonpermitted config difference in {job['config']}"
        assert cfg["training"] == base["training"] and cfg["reference"] == base["reference"]
        assert cfg["method"] == job["method"] and cfg["seed"] == job["seed"]
        assert cfg["training"]["max_iterations"] == 2000 and cfg["max_attempt_seconds"] == 3600
        assert cfg["run_role"] == "primary_matched_arm" and cfg["compile_fallback"] is False
        destination = (ROOT / cfg["output_dir"]).resolve()
        assert destination == Path(job["output"]).resolve() and destination.is_relative_to(ROOT / "outputs")
        assert (cfg["method"] == "A2") == bool(cfg.get("contact_objective"))
        if cfg.get("contact_objective"):
            contact = cfg["contact_objective"]
            bundle, reference = Path(contact["bundle_dir"]), Path(contact["reference_file"])
            files = {"manifest_sha256": bundle / "manifest.json", "target_npz_sha256": bundle / "contact_targets.npz",
                     "rules_sha256": bundle / "rules.json", "raw_geometry_sha256": bundle / "raw_geometry.json",
                     "time_windows_sha256": bundle / "time_windows.json", "reference_npz_sha256": reference,
                     "reference_json_sha256": reference.with_suffix(".json"), "model_sha256": Path(contact["model_file"]),
                     "objective_module_sha256": ROOT / "src/chord_study/contact_objective.py",
                     "geometry_module_sha256": ROOT / "src/chord_study/contact_geometry.py"}
            assert all(digest(file) == contact["expected_hashes"][name] for name, file in files.items())
        configurations.append(cfg)
    for i, seed in enumerate(manifest["seeds"]):
        pair = manifest["jobs"][2*i:2*i+2]
        assert {j["method"] for j in pair} == {"A1L", "A2"} and all(j["seed"] == seed for j in pair)
    assert all(cfg["training"] == configurations[0]["training"] and cfg["reference"] == configurations[0]["reference"] for cfg in configurations)
    return {"status": "passed", "checked_utc": utc(), "native_files_verified": len(inventory["files"]),
            "job_count": 6, "no_GPU_or_torch_import": "torch" not in sys.modules,
            "jobs": [{k: j[k] for k in ("position", "seed", "method", "config_sha256")} for j in manifest["jobs"]]}


@contextmanager
def gpu_queue_lock():
    import msvcrt
    path = ROOT / ".locks/single_gpu_training.lock"
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
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", query], capture_output=True, text=True, check=True)
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
                                  "--iteration", str(iteration), "--output-dir", str(destination)],
                                 stdout=out, stderr=err, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, timeout=120)
    assert process.returncode == 0 and read(destination / "report.json")["status"] == "passed", f"CPU health audit failed: {destination}"


def stop_owned_child(process):
    if process.poll() is None:
        # Only the still-live Popen PID created by this queue and its descendants.
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False)
        process.wait(timeout=30)


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
                                   capture_output=True, text=True, check=True, timeout=15)
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
                with (logs / "stdout.log").open("w", encoding="utf-8") as out, (logs / "stderr.log").open("w", encoding="utf-8") as err:
                    record["Popen_prelaunch_utc"] = utc()
                    process = subprocess.Popen([manifest["python_executable"], "-u", manifest["trainer"], "--config", job["config"]],
                                               cwd=ROOT, stdout=out, stderr=err,
                                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW)
                    record.update(status="running", launcher_pid=process.pid, logs=str(logs), Popen_postlaunch_utc=utc())
                    first_audited = False
                    while process.poll() is None:
                        elapsed = time.perf_counter()-started
                        assert elapsed <= manifest["per_run_process_watchdog_seconds"], "Finite process watchdog exceeded"
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
                            first_audited = True
                            record.update(first_checkpoint_health="passed", first_checkpoint_audit=str(destination),
                                          first_checkpoint_checked_utc=utc())
                        record["reserved_wall_seconds_so_far"] = time.perf_counter()-started
                        write(output / "state.json", state)
                        time.sleep(1)
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
                record.update(status="passed", finished_utc=utc(), reserved_wall_seconds_including_CPU_health=time.perf_counter()-started,
                              training_STARTED_wall_seconds=summary["total_wall_seconds_including_failed_attempts"],
                              final_checkpoint=attempt["checkpoint"], final_checkpoint_sha256=attempt["checkpoint_sha256"])
                write(output / "state.json", state)
            except BaseException as error:
                if process is not None:
                    stop_owned_child(process)
                record.update(status="failed", error=repr(error), finished_utc=utc())
                state.update(status="failed_stopped_no_retry", finished_utc=utc())
                write(output / "state.json", state)
                raise
    state.update(status="passed", finished_utc=utc())
    write(output / "state.json", state)


def execute(manifest_path, manifest):
    try:
        validate(manifest)
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
            if any(job.get("launcher_pid") for job in state.get("jobs", [])):
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
