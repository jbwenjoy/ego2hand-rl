"""Serial GPU evaluation of an immutable model list, without quality selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan)
    assert plan["schema"] == "supermimic_replication_evaluation_plan:1"
    assert digest(__file__) == plan["runner_sha256"]
    destination = Path(plan["output"])
    if destination.exists():
        raise FileExistsError(destination)
    for key in ("reference", "evaluator", "cost_runner", "source_inventory"):
        assert digest(plan[key]) == plan[key+"_sha256"]
    inventory = read(plan["source_inventory"])
    for relative, sha in inventory["files"].items():
        assert digest(Path(inventory["dex_root"])/relative) == sha
    for job in plan["jobs"]:
        assert digest(job["checkpoint"]) == job["checkpoint_sha256"]
        assert not Path(job["output"]).exists() and not Path(job["cost_output"]).exists()
    destination.mkdir(parents=True)
    shutil.copy2(args.plan, destination/"plan.json")
    shutil.copy2(__file__, destination/Path(__file__).name)
    shutil.copy2(plan["evaluator"], destination/"executed_evaluator.py")
    state = {"status":"running", "pid":os.getpid(), "started_utc":datetime.now(timezone.utc).isoformat(),
             "plan_sha256":digest(args.plan), "jobs":[], "quality_based_selection":False}
    write(destination/"state.json", state)
    for job in plan["jobs"]:
        record = {**job, "status":"running"}
        state["jobs"].append(record)
        write(destination/"state.json", state)
        # No automatic retry. On an engineering failure, inspect actual child
        # processes before any subsequent GPU work, including timeout cases.
        command = [plan["python"], plan["cost_runner"], "--output", job["cost_output"],
                   "--stage", "evaluation", "--cache", "warm", "--timeout-s", "120", "--gpu-count", "1", "--",
                   plan["python"], plan["evaluator"], "--reference", plan["reference"],
                   "--checkpoint", job["checkpoint"], "--seed", str(plan["evaluation_seed"]), "--output", job["output"]]
        code = subprocess.run(command, cwd=ROOT).returncode
        if code:
            record.update(status="engineering_failure", exit_code=code)
            state.update(status="stopped_engineering_failure_no_retry")
            write(destination/"state.json", state)
            raise SystemExit(code)
        evaluation = read(Path(job["output"])/"evaluation.json")
        metadata = read(Path(job["output"])/"rollout.json")
        assert metadata["checkpoint_sha256"] == job["checkpoint_sha256"]
        assert metadata["reference_bundle_sha256"] == plan["reference_sha256"]
        assert metadata["script_sha256"] == plan["evaluator_sha256"]
        condition = {"experimental_method":job["method"], "training_seed":job["training_seed"],
                     "evaluation_seed":plan["evaluation_seed"], "budget":job["budget"],
                     "training_run":job["training_run"], "checkpoint_sha256":job["checkpoint_sha256"],
                     "scope":"The raw metadata's legacy A1 policy label names the common Actor class; this file binds the actual training arm."}
        write(Path(job["output"])/"condition.json", condition)
        record.update(status="evaluation_recorded", task_success=evaluation["task_success"],
                      frames=evaluation["frames"], censored=evaluation["rollout_censored"],
                      cost=read(Path(job["cost_output"])/"cost.json"))
        write(destination/"state.json", state)
    state.update(status="completed_all_evaluations", finished_utc=datetime.now(timezone.utc).isoformat())
    write(destination/"state.json", state)


if __name__ == "__main__":
    main()
