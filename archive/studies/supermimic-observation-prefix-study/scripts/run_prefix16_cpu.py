"""One owned, hash-bound CPU geometry solve. No retries, GPU or simulation.

This supervisor imports only the standard library. It never imports the worker
or creates its formal output; the child owns that new directory. A timeout kills
only the Popen process handle this supervisor created, never a searched PID.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    # Supervisor snapshots have distinct names and are written exactly once.
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def process_observation(pid):
    if os.name != "nt":
        raise RuntimeError("This protocol requires actual Windows CIM identity")
    if type(pid) is not int or pid <= 0:
        raise ValueError("Expected positive owned PID")
    # Only the integer PID is interpolated. No paths or user text enter shell code.
    cmd = (
        f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId = {pid}\"; "
        "if ($null -eq $p) { 'null' } else { "
        "[PSCustomObject]@{pid=[int]$p.ProcessId;parent_pid=[int]$p.ParentProcessId;"
        "creation_utc=$p.CreationDate.ToUniversalTime().ToString('o');"
        "command_line=$p.CommandLine;executable_path=$p.ExecutablePath} | "
        "ConvertTo-Json -Compress }"
    )
    done = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=5, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    if done.returncode:
        raise RuntimeError("CIM query failed: " + done.stderr[-1000:])
    return json.loads(done.stdout.strip())


def validate(plan_path):
    p = read(plan_path)
    required = {
        "schema": "prefix16_geometry_cpu_plan:1", "optimization_authorized": True,
        "GPU_authorized": False, "training_authorized": False,
        "frames": 16, "joints": 20, "max_nfev": 80,
        "soft_wall_seconds": 110, "owned_process_seconds": 120,
        "automatic_retry": False,
    }
    for key, value in required.items():
        if p.get(key) != value or type(p.get(key)) is not type(value):
            raise ValueError("Plan contract differs: " + key)
    if p["source_pins"]["supervisor"] != {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)}:
        raise ValueError("Supervisor source binding differs")
    for name, pin in {**p["inputs"], **p["source_pins"], "protocol": p["protocol"]}.items():
        if sha(pin["path"]) != pin["sha256"]:
            raise ValueError("Source/input drift: " + name)
    protocol = read(p["protocol"]["path"])
    if p["inputs"] != protocol["inputs"]:
        raise ValueError("Input catalog must match frozen proposal exactly")
    if protocol["GPU_authorized"] is not False or protocol["training_authorized"] is not False:
        raise ValueError("Only CPU proposal is accepted")
    worker = Path(p["source_pins"]["worker"]["path"]).resolve()
    worker.relative_to(ROOT / "scripts")
    output = Path(p["output"]).resolve()
    output.relative_to(ROOT / "outputs")
    if output == ROOT / "outputs":
        raise ValueError("Output must be a new child directory")
    watchdog = output.with_name(output.name + "_watchdog")
    if output.exists() or watchdog.exists():
        raise FileExistsError("Existing outcome/attempt must be retained; no retry")
    python = Path(p["python"]).resolve()
    if python != Path(sys.executable).resolve():
        raise ValueError("Invoke the supervisor with the pinned Python")
    return p, worker, output, watchdog


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    plan_path = args.plan.resolve()
    p, worker, output, watchdog = validate(plan_path)
    if not args.run:
        print(json.dumps({"status": "SUPERVISOR_PLAN_PREFLIGHT_PASS", "physics_steps": 0,
                          "optimization_started": False, "GPU_used": False,
                          "plan_sha256": sha(plan_path)}))
        return
    watchdog.mkdir(parents=True, exist_ok=False)
    (watchdog / "plan_snapshot.json").write_bytes(plan_path.read_bytes())
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    command = [str(Path(sys.executable).resolve()), "-u", str(worker),
               "--plan", str(plan_path), "--output", str(output), "--run"]
    state = {"status": "initializing", "started_utc": utc(), "supervisor_pid": os.getpid(),
             "supervisor_sha256": sha(__file__), "plan_sha256": sha(plan_path),
             "command": command, "hard_seconds": 120, "automatic_retry": False,
             "GPU_authorized": False, "physics_steps_by_supervisor": 0,
             "environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
             "actual_process_OS_observation": None, "timeout": False,
             "identity_errors": [], "worker_restarted": False}
    write_new(watchdog / "launch.json", state)
    child = None
    began = time.monotonic()
    error = None
    try:
        with (watchdog / "stdout.log").open("xb") as stdout, (watchdog / "stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            state["actual_pid"] = child.pid
            state["launched_utc"] = utc()
            # Re-observe this same owned handle only; an observation error never
            # triggers a second worker or changes the hard deadline.
            for _ in range(3):
                try:
                    obs = process_observation(child.pid)
                    if obs is not None:
                        if obs["pid"] != child.pid or obs["parent_pid"] != os.getpid():
                            raise RuntimeError("Actual OS identity has wrong ownership")
                        if str(worker).lower() not in obs["command_line"].lower() or str(plan_path).lower() not in obs["command_line"].lower():
                            raise RuntimeError("Actual OS command differs")
                        state["actual_process_OS_observation"] = obs
                        write_new(watchdog / "actual_OS_identity.json", obs)
                        break
                except Exception as exc:
                    state["identity_errors"].append(repr(exc))
                if child.poll() is not None:
                    break
                time.sleep(.05)
            state["status"] = "running"
            write_new(watchdog / "observed_launch.json", state)
            while child.poll() is None:
                if time.monotonic() - began >= 120:
                    state["timeout"] = True
                    # Popen.kill uses the owned process handle on Windows.
                    child.kill()
                    break
                time.sleep(.05)
            state["exit_code"] = child.wait(timeout=5)
            state["status"] = "hard_timeout_partial" if state["timeout"] else "completed" if state["exit_code"] == 0 else "failed_child"
    except BaseException as exc:
        error = exc
        state["error"] = repr(exc)
        state["status"] = "supervisor_error_partial"
    finally:
        if child is not None:
            try:
                if child.poll() is None:
                    try:
                        child.kill()
                    except OSError as exc:
                        state["cleanup_terminate_error"] = repr(exc)
                    try:
                        child.wait(timeout=5)
                    except (subprocess.TimeoutExpired, OSError) as exc:
                        state["cleanup_wait_error"] = repr(exc)
                state["owned_handle_terminal"] = child.poll() is not None
            except OSError as exc:
                state["cleanup_handle_error"] = repr(exc)
                state["owned_handle_terminal"] = False
            try:
                post = process_observation(child.pid)
                state["post_terminal_OS_observation"] = post
                prior = state.get("actual_process_OS_observation")
                state["same_owned_identity_remaining"] = bool(post and prior and post["creation_utc"] == prior["creation_utc"])
            except Exception as exc:
                state["post_terminal_OS_query_error"] = repr(exc)
                state["same_owned_identity_remaining"] = None
        state["finished_utc"] = utc()
        state["owned_scope_wall_seconds"] = time.monotonic() - began
        state["authoritative_worker_output_exists"] = output.is_dir()
        state["latest_checkpoint_exists"] = (output / "latest_checkpoint.json").is_file()
        state["identity_qualified"] = state.get("actual_process_OS_observation") is not None and state.get("same_owned_identity_remaining") is False and state.get("owned_handle_terminal") is True
        state["result_hashes"] = {}
        if output.is_dir() and state.get("owned_handle_terminal") is True:
            for f in output.iterdir():
                if f.is_file():
                    try:
                        state["result_hashes"][f.name] = sha(f)
                    except OSError as exc:
                        state.setdefault("result_hash_errors", {})[f.name] = repr(exc)
        if state.get("owned_handle_terminal") is not True:
            state["status"] = "cleanup_unverified_worker_may_be_live"
        write_new(watchdog / "state.json", state)
    print(json.dumps({k: state.get(k) for k in ("status", "actual_pid", "exit_code", "timeout",
        "identity_qualified", "owned_scope_wall_seconds", "latest_checkpoint_exists")}, indent=2))
    if error is not None:
        raise error
    if state["status"] != "completed" or not state["identity_qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
