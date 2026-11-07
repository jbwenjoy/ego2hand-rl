"""Five fixed-checkpoint fresh-process evaluations; default is CPU preflight.

This post-hoc reproducibility diagnostic does not add training seeds, modify the
original evaluations, or stop on task quality. Only --run starts GPU children.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

STARTED = time.monotonic()
STUDY = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = STUDY / "configs/evaluator-repeatability-a1l-seed1-v1.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preflight(plan_path):
    plan = read(plan_path)
    require(plan["repeat_count"] == len(plan["repeats"]) == 5, "Exactly five repeats required")
    require(plan["training_seed"] == 1 and plan["evaluation_seed"] == 0, "Frozen seeds changed")
    require(plan["per_child_cap_seconds"] == 45 and plan["whole_driver_cap_seconds"] == 240, "Frozen caps changed")
    require(plan["training_updates"] == 0 and plan["automatic_retry"] is False, "No training/retry")
    require(plan["task_quality_controls_execution"] is False and plan["engineering_fail_stop"] is True,
            "Only engineering failure may stop the fixed repeat sequence")
    require(plan["cleanup_and_metadata_reserve_seconds"] == 45, "Conservative cleanup reserve changed")
    require(plan["reference_repeat_id"] == "repeat1", "Reference repeat is prespecified")
    for label, pin in plan["source_pins"].items():
        require(sha(pin["path"]) == pin["sha256"], "Source changed: " + label)
    require(Path(plan["source_pins"]["driver"]["path"]).resolve() == Path(__file__).resolve(), "Wrong driver")
    for label in ("checkpoint", "reference", "reference_metadata"):
        pin = plan[label]
        require(sha(pin["path"]) == pin["sha256"], "Input changed: " + label)
    for label in ("archive", "instrumented_donor"):
        item = plan[label]
        for name, digest in item["sha256"].items():
            require(sha(Path(item["path"]) / name) == digest, label + " changed: " + name)
    expected_ids = ["repeat" + str(i) for i in range(1, 6)]
    require([r["repeat_id"] for r in plan["repeats"]] == expected_ids, "Fixed repeat order changed")
    require(len({r["output_dir"] for r in plan["repeats"]}) == 5, "Distinct outputs required")
    for item in plan["repeats"]:
        command = [plan["python"], "-u", plan["source_pins"]["evaluator"]["path"],
                   "--reference", plan["reference"]["path"], "--checkpoint", plan["checkpoint"]["path"],
                   "--seed", "0", "--device", "cuda:0", "--output", item["output_dir"]]
        require(item["command"] == command, "Original evaluator CLI changed")
        require(Path(item["output_dir"]).parent == Path(plan["output_root"]), "Output outside fixed root")
    spec = importlib.util.spec_from_file_location("reviewed_repeat_lifecycle", plan["source_pins"]["lifecycle"]["path"])
    lifecycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lifecycle)
    # Explicitly preserve the original evaluation manager's cwd. This changes
    # only the imported lifecycle library's launch cwd, never its frozen file.
    lifecycle.ROOT = STUDY
    return plan, lifecycle


def child_artifacts(plan, item):
    output = Path(item["output_dir"])
    files = ("rollout.npz", "rollout.json", "evaluation.json", "model.mjb", "reference.npz", "reference.json")
    require(all((output / n).is_file() for n in files), "Missing complete evaluation artifact")
    meta, result = read(output / "rollout.json"), read(output / "evaluation.json")
    require(meta["recorded_frames"] == 151 and meta["recorded_control_intervals"] == 150,
            "Engineering failure: incomplete fixed3s horizon")
    require(meta["rollout_censored"] is False, "Engineering failure: censored physical rollout")
    require(meta["seed"] == 0 and meta["checkpoint_sha256"] == plan["checkpoint"]["sha256"], "Checkpoint/seed changed")
    require(sha(output / "reference.npz") == plan["reference"]["sha256"], "Reference changed")
    require(sha(output / "model.mjb") == plan["expected_model_sha256"], "Nominal model changed")
    require(meta["resets_during_rollout"] == 0 and meta["object_state_writes_after_initialization"] == 0,
            "Reset/object-write contract violated")
    condition = {"schema": "fixed_checkpoint_reproducibility_diagnostic:1", "repeat_id": item["repeat_id"],
                 "training_seed": 1, "evaluation_seed": 0, "plan_sha256": plan["plan_sha256_runtime"],
                 "checkpoint_sha256": plan["checkpoint"]["sha256"], "evaluator_sha256": plan["source_pins"]["evaluator"]["sha256"],
                 "instrumentation": "original evaluator only; no hook/monkeypatch/native manager modification",
                 "new_independent_training_seed": False, "task_quality_selects_or_stops": False}
    (output / "condition.json").write_text(json.dumps(condition, indent=2) + "\n", encoding="utf-8")
    return {"complete_engineering_horizon": True, "recorded_frames": 151,
            "task_success": result.get("task_success"), "quality_used_for_control": False,
            "artifact_sha256": {name: sha(output / name) for name in (*files, "condition.json")}}


def run(plan_path, plan, lifecycle):
    root = Path(plan["output_root"])
    require(not root.exists(), "Refusing existing diagnostic directory; no overwrite or retry")
    root.mkdir(parents=True)
    state = {"status": "starting", "started_utc": datetime.now(timezone.utc).isoformat(),
             "plan_sha256": sha(plan_path), "driver_sha256": sha(__file__),
             "training_seed": 1, "evaluation_seed": 0, "training_updates": 0,
             "jobs": [{**r, "status": "NOT_RUN"} for r in plan["repeats"]]}
    write = lambda: lifecycle.write(root / "state.json", state)
    write()
    shutil.copyfile(plan_path, root / "frozen_plan.json")
    shutil.copyfile(__file__, root / "driver_source.py")
    plan["plan_sha256_runtime"] = state["plan_sha256"]
    try:
        with lifecycle.gpu_lock():
            lifecycle.reject_project_gpu_workers()
            require(sha(plan_path) == state["plan_sha256"], "Plan changed after preflight")
            state["status"] = "running"; write()
            for job in state["jobs"]:
                remaining = 240 - (time.monotonic() - STARTED)
                # Reserve bounded watchdog cleanup and final metadata time;
                # child whole-process time is never greater than45 seconds.
                allowance = min(45, remaining - plan["cleanup_and_metadata_reserve_seconds"])
                require(allowance > 0, "Whole-driver budget exhausted; no next child")
                job.update(status="running", effective_child_cap_seconds=allowance,
                           remaining_whole_driver_seconds_before_launch=remaining)
                write()
                logs = root / (job["repeat_id"] + "_process")
                result = lifecycle.watch(job["command"], Path(job["output_dir"]), logs,
                                         hard_seconds=allowance)
                job["process"] = result
                require(time.monotonic() - STARTED <= 240, "Whole-driver scheduling budget exceeded after child")
                require(result["status"] == "completed" and result["exit_code"] == 0,
                        "Engineering child failure; fail-stop without retry")
                job["artifacts"] = child_artifacts(plan, job)
                require(time.monotonic() - STARTED <= 240, "Whole-driver scheduling budget exceeded after artifact check")
                job["status"] = "completed"
                write()
            state["status"] = "completed_5_repeats"
    except BaseException as error:
        for job in state["jobs"]:
            if job["status"] == "running":
                job["status"] = "engineering_failed"
        state.update(status="engineering_failed", error=repr(error))
        raise
    finally:
        state["finished_utc"] = datetime.now(timezone.utc).isoformat()
        state["whole_driver_wall_seconds"] = time.monotonic() - STARTED
        state["within_whole_budget"] = state["whole_driver_wall_seconds"] <= 240
        state["whole_budget_scope"] = "240s scheduling/stop-request budget with45s cleanup reserve; child lifecycle has independent timer, no claim of OS hard real-time termination"
        write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    plan, lifecycle = preflight(args.plan.resolve(strict=True))
    if args.run:
        run(args.plan.resolve(strict=True), plan, lifecycle)
    else:
        print(json.dumps({"status": "passed_CPU_preflight", "GPU_started": False,
                          "plan_sha256": sha(args.plan), "driver_sha256": sha(__file__),
                          "source_pins": len(plan["source_pins"]), "repeat_count": 5,
                          "fresh_process": "five independent subprocess.Popen calls; actual venv worker PID may be unobserved and is never equated to launcher PID",
                          "cwd": str(STUDY), "per_child_cap_seconds": 45, "whole_driver_cap_seconds": 240}, indent=2))


if __name__ == "__main__":
    main()
