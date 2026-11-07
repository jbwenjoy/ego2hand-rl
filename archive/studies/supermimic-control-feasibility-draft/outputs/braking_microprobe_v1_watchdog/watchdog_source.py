"""One-shot hidden worker watchdog; default mode is CPU-only preflight.

Only --run starts the hash-bound GPU worker. The shared project GPU lock is
held until the owned child is reaped. A timer independent of polling/CIM work
requests process-tree termination at 240 seconds, including worker setup.
No retry, artifact deletion, calibration fit or inference is performed here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
LOCK = Path("D:/workspace/supermimic-chord-study/.locks/single_gpu_training.lock")
HARD_SECONDS = 240.0
CLEANUP_GRACE_SECONDS = 20.0
HIDDEN = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt" else 0


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


@contextmanager
def gpu_lock(path=LOCK):
    """The same nonblocking one-byte Windows lock used by both training queues."""
    import msvcrt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def reject_project_gpu_workers():
    # Advisory detection complements the common lock. It never kills somebody
    # else's process. The filename boundary excludes this run_* supervisor.
    pattern = r"[\\/](train_controlled_pen|controlled_pen_rollout|evaluate_pen_checkpoint|verify_warp_hold|probe_world_model_gradients|braking_microprobe)\.py(?:[\s\"']|$)|check_a2_cuda|probe_a2_whole|isolate_a2_cuda|reproduce_a2"
    query = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' "
             "-and $_.ProcessId -ne " + str(os.getpid()) + " -and $_.CommandLine -match '" +
             pattern.replace("'", "''") + "' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
                            capture_output=True, text=True, check=True, timeout=10, creationflags=HIDDEN)
    require(not result.stdout.strip(), "Another project GPU worker is present: " + result.stdout.strip())


def matching_pin(plan, path):
    path = Path(path).resolve()
    matches = [p for p in plan.get("source_pins", {}).values() if Path(p["path"]).resolve() == path]
    require(len(matches) == 1, f"Exactly one frozen source pin is required for {path}")
    require(sha(path) == matches[0]["sha256"], f"Frozen source changed: {path}")
    return matches[0]


def preflight(plan_path, runtime, python):
    plan = read(plan_path)
    require(plan["process_wall_cap_seconds"] == HARD_SECONDS, "Whole-worker hard cap must remain 240s")
    require(plan["total_branches"] == 54 and plan["branch_duration_s"] == .2, "Only fixed 54 x .2s probe is allowed")
    require(plan["training_updates"] == 0 and plan["m0"] is None and plan["a_brake"] is None
            and plan["parameter_enablement"] is False and plan["automatic_retry"] is False,
            "Training, parameter enablement or retry is forbidden")
    matching_pin(plan, runtime)
    matching_pin(plan, __file__)
    checked = subprocess.run([str(python), str(runtime), "--plan", str(plan_path), "--validate-only"],
                             capture_output=True, text=True, timeout=30, creationflags=HIDDEN,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
    require(checked.returncode == 0, "Worker CPU preflight failed: " + checked.stderr + checked.stdout)
    return {"status": "passed", "checked_utc": utc(), "GPU_started": False,
            "plan_sha256": sha(plan_path), "runtime_sha256": sha(runtime), "watchdog_sha256": sha(__file__),
            "worker_validate_stdout": checked.stdout, "worker_validate_stderr": checked.stderr,
            "hard_wall_seconds": HARD_SECONDS, "shared_lock": str(LOCK)}


def observe_actual_pid(pid, launcher_pid, runtime, plan_path, output):
    """CIM identity plus exact OS CreationDate, or explicitly bounded unknown."""
    query = (f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId = {int(pid)}'; "
             "if($null -eq $p){throw 'Actual worker not observable'}; "
             "[ordered]@{pid=$p.ProcessId;parent_pid=$p.ParentProcessId;"
             "creation_date_utc=$p.CreationDate.ToUniversalTime().ToString('o');"
             "command_line=$p.CommandLine;executable_path=$p.ExecutablePath}|ConvertTo-Json -Compress")
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
                                capture_output=True, text=True, timeout=3, check=True, creationflags=HIDDEN)
        item = json.loads(result.stdout)
        require(item["pid"] == int(pid), "PID mismatch")
        expected_parent = os.getpid() if pid == launcher_pid else launcher_pid
        require(item["parent_pid"] == expected_parent, "Worker parent/ownership mismatch")
        command = item["command_line"].lower().replace("/", "\\")
        require(all(str(Path(p).resolve()).lower().replace("/", "\\") in command
                    for p in (runtime, plan_path, output)), "Worker command identity mismatch")
        return {"status": "exact_OS_identity_observed", "observed_utc": utc(), **item}
    except Exception as error:
        return {"status": "missing_or_identity_unverified", "observed_utc": utc(), "error": repr(error),
                "scope": "Use prelaunch/first-runtime-observation bounds, never label them exact OS creation"}


def stop_owned(process):
    """Kill only our still-live Popen root and descendants, never foreign PIDs."""
    if process.poll() is not None:
        return {"already_exited": True, "root_reaped": True,
                "owned_tree_termination_confirmed": False}
    result = {"used_Popen_kill_fallback": False, "owned_tree_termination_confirmed": False}
    if os.name == "nt":
        try:
            killed = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                    capture_output=True, text=True, timeout=10, creationflags=HIDDEN)
            result.update(taskkill_exit_code=killed.returncode, stdout=killed.stdout, stderr=killed.stderr)
            result["owned_tree_termination_confirmed"] = killed.returncode == 0
        except BaseException as error:
            result["taskkill_error"] = repr(error)
    else:
        result["tree_scope"] = "non-Windows test only; no descendant guarantee"
    if process.poll() is None:
        result["used_Popen_kill_fallback"] = True
        try:
            process.kill()  # Our retained Popen handle; never an arbitrary PID.
        except BaseException as error:
            result["Popen_kill_error"] = repr(error)
    try:
        process.wait(timeout=5)
        result["root_reaped"] = True
    except subprocess.TimeoutExpired:
        result["root_reaped"] = False
        result["owned_tree_termination_confirmed"] = False
    if not result["owned_tree_termination_confirmed"]:
        result["scope"] = "Only root convergence was attempted; descendants/GPU idleness are NOT confirmed"
    return result


def watch(command, output, logs, *, hard_seconds=HARD_SECONDS, observer=None):
    """Internal lifecycle primitive; tests supply only tiny standard-library CPU children."""
    output, logs = Path(output), Path(logs)
    record = {"status": "starting", "supervisor_pid": os.getpid(), "created_utc": utc(),
              "command": list(map(str, command)), "worker_output": str(output), "hard_seconds": hard_seconds,
              "automatic_retry": False, "inference_qualification": "not assessed by watchdog"}
    write(logs / "state.json", record)
    process = None
    expired = threading.Event()
    done = threading.Event()
    holder = {}
    timer_result = {}
    timer_lock = threading.Lock()
    cleanup_finished = threading.Event()
    before = time.monotonic()

    def timeout():
        child = holder.get("process")
        if done.is_set() or (child is not None and child.poll() is not None):
            cleanup_finished.set()
            return
        expired.set()
        timer_result["kill_requested_utc"] = utc()
        timer_result["kill_requested_elapsed_s"] = time.monotonic() - before
        if child is not None:
            try:
                with timer_lock:
                    timer_result["cleanup"] = stop_owned(child)
            except BaseException as error:
                timer_result["cleanup_error"] = repr(error)
            finally:
                cleanup_finished.set()

    timer = threading.Timer(hard_seconds, timeout)
    timer.daemon = True
    timer.start()  # Includes Popen/venv and all imports/setup, not only physics.
    try:
        record["Popen_prelaunch_utc"] = utc()
        with (logs / "stdout.log").open("w", encoding="utf-8") as stdout, (logs / "stderr.log").open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(list(map(str, command)), cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=HIDDEN)
            holder["process"] = process
            record.update(status="running", launcher_pid=process.pid, Popen_postlaunch_utc=utc())
            if expired.is_set():
                with timer_lock:
                    stop_owned(process)
            while process.poll() is None:
                # The independent timer remains active during this I/O/CIM.
                if expired.is_set() and (cleanup_finished.is_set()
                                        or time.monotonic()-before > hard_seconds+CLEANUP_GRACE_SECONDS):
                    raise TimeoutError("Owned root still live after bounded cleanup; descendants/GPU idleness unconfirmed")
                if "actual_process_OS_observation" not in record:
                    try:
                        runtime = read(output / "runtime.json")
                        pid = int(runtime["pid"])
                        observed = utc()
                        record.update(actual_pid=pid, runtime_metadata=runtime,
                                      actual_creation_conservative_bounds={"lower_utc": record["Popen_prelaunch_utc"], "upper_utc": observed})
                        if observer is not None:
                            record["actual_process_OS_observation"] = observer(pid, process.pid)
                    except (OSError, ValueError, KeyError, TypeError):
                        pass
                record["elapsed_s"] = time.monotonic() - before
                write(logs / "state.json", record)
                done.wait(.1)
            process.wait()
        done.set(); timer.cancel(); timer.join(timeout=12)
        cleanup = timer_result.get("cleanup", {})
        uncertain = expired.is_set() and not cleanup.get("owned_tree_termination_confirmed", False)
        record.update(status=("timeout_cleanup_unverified" if uncertain else "timeout") if expired.is_set()
                      else ("completed" if process.returncode == 0 else "failed"),
                      exit_code=process.returncode, finished_utc=utc(), elapsed_s=time.monotonic()-before,
                      timeout=expired.is_set(), timeout_details=dict(timer_result), partial_artifacts_preserved=True)
        if "actual_process_OS_observation" not in record:
            record["actual_process_OS_observation"] = {"status": "unobserved", "scope": "No exact actual-child OS creation timestamp"}
        write(logs / "state.json", record)
        return record
    except BaseException as error:
        done.set(); timer.cancel()
        # The timeout thread has its own bounded cleanup. Do not block on its
        # lock or wait indefinitely if OS termination cannot be confirmed.
        if process is not None and not expired.is_set():
            try:
                with timer_lock:
                    record["cleanup"] = stop_owned(process)
            except BaseException as cleanup_error:
                record["cleanup_error"] = repr(cleanup_error)
        timer.join(timeout=12)
        record.update(status="timeout" if expired.is_set() else "failed", error=repr(error),
                      timeout=expired.is_set(), timeout_details=dict(timer_result),
                      finished_utc=utc(), elapsed_s=time.monotonic()-before, partial_artifacts_preserved=True)
        if expired.is_set() and not timer_result.get("cleanup", {}).get("owned_tree_termination_confirmed", False):
            record["status"] = "timeout_cleanup_unverified"
            record["GPU_idleness_confirmed"] = False
        elif process is not None and record.get("cleanup") and not record["cleanup"].get("owned_tree_termination_confirmed", False):
            record["status"] = "failed_cleanup_unverified"
            record["GPU_idleness_confirmed"] = False
        write(logs / "state.json", record)
        raise


def run(plan_path, output, runtime, python):
    output = Path(output).resolve()
    logs = output.with_name(output.name + "_watchdog")
    require(not output.exists(), "Refusing an existing worker output; no retry/overwrite")
    # Exclusive directory creation avoids competing launchers overwriting an
    # active launch's shared state, even if neither acquired the GPU lock yet.
    logs.mkdir(parents=True, exist_ok=False)
    startup = {"status": "preflight", "supervisor_pid": os.getpid(), "started_utc": utc(),
               "plan": str(plan_path), "worker_output": str(output), "logs": str(logs),
               "watchdog_sha256": sha(__file__), "automatic_retry": False}
    write(logs / "state.json", startup)
    try:
        result = preflight(plan_path, runtime, python)
        write(logs / "preflight.json", result)
        shutil.copy2(plan_path, logs / "plan_snapshot.json")
        shutil.copy2(__file__, logs / "watchdog_source.py")
        with gpu_lock():
            reject_project_gpu_workers()
            # Guard against source/plan changes between preflight and launch.
            require(sha(plan_path) == result["plan_sha256"], "Plan changed after preflight")
            matching_pin(read(plan_path), runtime)
            command = [str(python), "-u", str(runtime), "--plan", str(plan_path), "--output", str(output)]
            observed = lambda pid, parent: observe_actual_pid(pid, parent, runtime, plan_path, output)
            state = watch(command, output, logs, observer=observed)
            if state["status"].endswith("cleanup_unverified"):
                state["shared_lock_release_scope"] = "Released after bounded failed cleanup; GPU idleness NOT confirmed. No retry or next job."
                write(logs / "state.json", state)
        return state
    except BaseException as error:
        previous = read(logs / "state.json")
        if previous.get("status") not in ("failed", "timeout", "completed", "timeout_cleanup_unverified", "failed_cleanup_unverified"):
            previous.update(status="failed_no_child_started", error=repr(error), finished_utc=utc())
            write(logs / "state.json", previous)
        elif str(previous.get("status", "")).endswith("cleanup_unverified"):
            previous["shared_lock_release_scope"] = "Bounded cleanup ended without confirming GPU idleness; no retry or next job."
            write(logs / "state.json", previous)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / "calibration_plan_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/braking_microprobe_v1")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run", action="store_true", help="Explicitly run one reviewed GPU worker; otherwise CPU preflight only")
    args = parser.parse_args()
    plan, runtime = args.plan.resolve(strict=True), (ROOT / "scripts/braking_microprobe.py").resolve(strict=True)
    python = args.python.resolve(strict=True)
    if not args.run:
        result = preflight(plan, runtime, python)
        print(json.dumps(result, indent=2)); return
    state = run(plan, args.output, runtime, python)
    print(json.dumps(state, indent=2))
    raise SystemExit(0 if state["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
