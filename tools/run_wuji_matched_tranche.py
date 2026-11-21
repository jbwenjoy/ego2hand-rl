"""Round-robin training and frozen-protocol evaluation with a shared single-GPU budget."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import shutil
import sys
import time

from benchmark_wuji_parallel import (
    SEQUENCES, PREFILL, environment, parquet_paths, run_child, store, summarize_run, train_command,
)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src,"sha256").hexdigest()


def local_path(value):
    foreign=PureWindowsPath(value)
    if len(foreign.drive)==2 and foreign.drive.endswith(":"):
        return Path("/mnt") / foreign.drive[0].lower() / Path(*foreign.parts[1:])
    if foreign.drive.lower() in (r"\\wsl.localhost\ubuntu-24.04",r"\\wsl$\ubuntu-24.04"):
        return Path("/") / Path(*foreign.parts[1:])
    path=Path(value)
    if not path.is_absolute():
        raise ValueError(f"Artifact pin must use an absolute source path: {value}")
    return path


def manifest_pins(node):
    if isinstance(node,dict):
        if {"path","bytes","sha256"} <= node.keys():
            yield node
        else:
            for item in node.values():
                yield from manifest_pins(item)
    elif isinstance(node,list):
        for item in node:
            yield from manifest_pins(item)


def verify_inputs(manifest, snapshot=None):
    unique = {}
    for pin in manifest_pins(read(manifest)):
        src=local_path(pin["path"])
        if not src.exists() or src.stat().st_size!=pin["bytes"] or digest(src)!=pin["sha256"]:
            raise ValueError(f"Frozen input mismatch: {src}")
        unique[pin["sha256"]]=src
    if not unique:
        raise ValueError("Empty training/evaluation input closure")
    if snapshot is not None:
        snapshot.mkdir(parents=True,exist_ok=True)
        for sha, src in unique.items():
            target=snapshot / sha[:2] / sha
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                temp=target.with_suffix(".partial")
                shutil.copyfile(src,temp)
                if digest(temp)!=sha:
                    raise ValueError(f"Input changed while being snapshotted: {src}")
                temp.replace(target)
            if digest(target)!=sha:
                raise ValueError(f"Frozen byte snapshot is incomplete: {target}")
    return len(unique)


def multiple(value, worlds):
    return max(worlds, math.ceil(value / worlds) * worlds)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def continuation_path(base, result):
    """Resume using a completed result, or a fresh sibling after a failed attempt."""
    failed = []
    for number in range(1, 1000):
        path = base if number == 1 else base.with_name(f"{base.name}_a{number:02d}")
        process = path.with_suffix(".process.json")
        if (path / result).is_file() and process.is_file() and read(process).get("returncode") == 0:
            return path, failed
        if not path.exists() and not process.exists() and not path.with_suffix(".stdout.log").exists():
            return path, failed
        if process.exists():
            old=read(process)
            if not old["finished"] and old.get("worker_pid"):
                proc=Path("/proc")/str(old["worker_pid"])/"cmdline"
                if proc.exists() and proc.read_bytes().rstrip(b"\0").split(b"\0")==[v.encode() for v in old["command"]]:
                    raise RuntimeError(f"Recorded worker still running for {path}; avoid a duplicate GPU job")
            failed.append({"attempt":str(path), **old})
    raise RuntimeError(f"Too many failed attempts for {base}")


def prepare(args):
    started=time.perf_counter()
    probe = read(args.probes)
    scores = {n: math.sqrt(probe[f"single{n}"]["warm_env_steps_per_s"] *
                           probe[f"three{n}"]["warm_env_steps_per_s"]) for n in (96, 192)}
    # Prefer the smaller batch when the measured geometric-mean advantage is below 5%.
    n = 192 if scores[192] > 1.05 * scores[96] else 96
    single = probe[f"single{n}"]
    shared = probe[f"three{n}"]
    wm = read(args.probes.with_name("hybrid_probe.json"))
    physics=read(args.probes.with_name("memory_preflight.json"))["physics"]
    if digest(Path(physics["path"]))!=physics["sha256"]:
        raise ValueError("The measured solver contract changed")
    if Path(wm["path"]).name != f"hybrid{n}":
        raise ValueError("Hybrid probe must use the selected world count")
    # All streams share the same schedule vs per-clip samples. Leave nominal zero-assist
    # training time towards the end even if the hybrid's additional work lowers throughput.
    horizon = multiple(0.7 * 900 * min(single["warm_env_steps_per_s"], shared["warm_env_steps_per_s"],
                                      wm["warm_env_steps_per_s"]), n)
    refs = parquet_paths(args.repo)
    timings = {"single":single["warm_env_steps_per_s"], "shared_mf":shared["warm_env_steps_per_s"],
               "shared_wm":wm["warm_env_steps_per_s"]}
    args.out.mkdir(parents=True, exist_ok=True)
    manifest=args.out / "inputs.json"
    if manifest.exists() and digest(manifest)!=digest(args.inputs):
        raise ValueError("Use a new plan directory for a new input closure")
    if manifest.resolve()!=args.inputs:
        shutil.copyfile(args.inputs,manifest)
    frozen=args.bulk / "inputs-by-sha256"
    entries=verify_inputs(manifest,frozen)
    plan = dict(
        created_utc=stamp(), worlds=n, probes=str(args.probes), probe_sha256=digest(args.probes), physics=physics,
        hybrid_probe_sha256=digest(args.probes.with_name("hybrid_probe.json")),
        inputs=str(manifest),inputs_sha256=digest(manifest),snapshot=str(frozen),unique_input_digests=entries,
        world_selection_geometric_mean_sps=scores, single_clip_curriculum_horizon=horizon,
        nominal_seconds_per_single_cumulative=[60, 360, 900],
        nominal_seconds_per_shared_cumulative=[180, 1080, 2700],
        estimated_training_rates=timings, references=[dict(sequence=s,parquet=str(p),sha256=digest(p))
                                                        for s,p in zip(SEQUENCES,refs,strict=True)],
        protocol_path=str(args.repo / "configs/hocap_carton_evaluation.json"),
        protocol_sha256=digest(args.repo / "configs/hocap_carton_evaluation.json"),
        checkpoint_policy="Stop each segment at a complete actor/critic+optimizer/replay/auxiliary save; score its actor immediately.",
        score_frequency="Strict eval at each segment endpoint. Internal checkpoints are unscored until explicitly evaluated.",
        cost_scope="All exclusive card wall, including failed jobs, setup/compilation, checkpoint IO and strict eval. Warm time is the observed post-prefill/post-compile synchronized learner window excluding checkpoint IO.",
        hybrid_rate_note="Initial WM rate measured with active native, model and imagined updates; recalibrated from each measured segment.",
        canonical_initial_state=str(args.bulk / "initial_native_state"),
        bulk=str(args.bulk), more_precise_clock="GPU sample timestamps and learner timing.jsonl in each run.")
    plan["plan_and_snapshot_wall_s"]=time.perf_counter()-started
    store(args.out / "plan.json", plan)
    return plan


def train_segment(args, plan, key, round_id, previous, records):
    verified_phase(plan)
    refs = parquet_paths(args.repo)
    single = key.startswith("single")
    idx = int(key[-1]) if single else None
    bank = [refs[idx]] if single else refs
    kind = "single" if single else key
    worlds = plan["worlds"]
    allocated = sum(r["useful_warm_s"] for r in records)
    target = (plan["nominal_seconds_per_single_cumulative"] if single else
              plan["nominal_seconds_per_shared_cumulative"])[round_id]
    left = max(0., target - allocated)
    rate = records[-1]["measured"]["warm_env_steps_per_s"] if records else plan["estimated_training_rates"][kind]
    pre = records[-1]["measured"]["total_steps"] if records else 0
    # Early processes pay the same warm-up exclusion regardless of method, including the
    # largest WM warm-up; later processes start measuring after their first synchronized log.
    warm_after = (max(2 * PREFILL, 104 * worlds) if not pre else pre + 8 * worlds)
    # The native target provides ample room for a timed stop and an always-valid terminal checkpoint.
    upper = multiple(max(24 * worlds + warm_after, warm_after + 1.45 * left * rate),worlds)
    cadence_seconds = min(max(12., left / 5), 45. if single else 90.)
    cadence = multiple(cadence_seconds * rate, worlds)
    destination, failures = continuation_path(args.bulk / f"r{round_id+1}_{key}", "result.json")
    cmd = train_command(args.repo, args.upstream, destination, bank, worlds, upper, cadence,
                        seed=42 + (idx or 0), resume_dir=previous, hybrid=(key == "shared_wm"))
    cmd.extend(["--curriculum-horizon-steps", str(plan["single_clip_curriculum_horizon"] * len(bank))])
    for override in plan["physics"]["overrides"]:
        cmd.extend(["--override",override])
    if key == "shared_wm":
        cmd.extend(["--wm-fit-horizon", "3"])
    if not previous:
        cmd.extend(["--shared-initial-state", plan["canonical_initial_state"]])
        if key == "single0" and round_id == 0 and not (Path(plan["canonical_initial_state"]) / "initial.json").exists():
            cmd.append("--create-shared-initial-state")
    phase_env = environment(args.repo, args.upstream)
    phase_env["WUJI_PHYSICS_SCHEME"]=plan["physics"]["scheme"]
    phase_env.update(WUJI_SEGMENT_WARM_SECONDS=str(left), WUJI_SEGMENT_WARM_AFTER_STEP=str(warm_after), WUJI_RECORD_VOC="1")
    live = dict(stage="training", round=round_id+1, stream=key, utc=stamp(), pid=os.getpid(),
                segment_remaining_target_warm_s=left, predicted_sps=rate,
                checkpoint_interval_steps=cadence, bulk=str(destination), command=cmd,
                progress=str(destination / "timing.jsonl"))
    store(args.out / "live.json", live)
    print("TRAIN", json.dumps(live), flush=True)
    reused=(destination / "result.json").is_file()
    if not reused:
        run_child(cmd, destination, phase_env)
    verified_phase(plan)
    measurement = summarize_run(destination, pre)
    checks = [json.loads(line) for line in (destination / "timing.jsonl").read_text().splitlines()]
    warm = [e for e in checks if e["event"] == "checked_training_budget"]
    if not warm:
        raise ValueError("No complete measured checkpoint after warmup")
    step = measurement["total_steps"]
    checkpoint = destination / f"checkpoints/policy_{step}.safetensors"
    return dict(round=round_id+1, stream=key, useful_warm_s=warm[-1]["warm_s"],
                inputs_sha256=plan["inputs_sha256"],physics_sha256=plan["physics"]["sha256"],
                planned_warm_s=left, measured=measurement, checkpoint=str(checkpoint),
                train_all_in_s=measurement["wall_s"], native_added_steps=step-pre,
                source_sequences=[SEQUENCES[idx]] if single else list(SEQUENCES), checks=[],
                reused_training_result=reused, failed_train_attempts=failures)


def evaluate(args, plan, item):
    for sequence, parquet in zip(SEQUENCES,parquet_paths(args.repo),strict=True):
        if sequence not in item["source_sequences"]:
            continue
        dest, failures = continuation_path(args.bulk / f"eval_r{item['round']}_{item['stream']}_{sequence}", "report.json")
        verified_phase(plan)
        cmd = [sys.executable,"-u",str(args.repo / "tools/evaluate_flash_wuji_quality.py"),
               "--checkpoint",item["checkpoint"],"--parquet",str(parquet),"--protocol",plan["protocol_path"],
               "--out",str(dest),"--warp-cache",str(args.upstream.parent / "cache/warp")]
        store(args.out / "live.json", dict(stage="strict_evaluation", round=item["round"],stream=item["stream"],
              sequence=sequence, utc=stamp(),pid=os.getpid(),bulk=str(dest),command=cmd))
        reused=(dest / "report.json").is_file()
        if not reused:
            run_child(cmd, dest, environment(args.repo,args.upstream))
        verified_phase(plan)
        score = read(dest / "report.json")
        # Retain native early termination, raw replay divergence and each actual gate.
        item["checks"].append(dict(sequence=sequence, passed=score["passed_numeric_development_gates"],
            gates=score["gates"], metrics=score["metrics"], terminal=score["terminal"],
            physical_repeat=score.get("physical_repeat",{"status":"not_recorded"}),
            eval_all_in_s=read(dest.with_suffix(".process.json"))["elapsed_s"], report=str(dest / "report.json"),
            report_sha256=digest(dest / "report.json"), inputs_sha256=plan["inputs_sha256"],
            evaluator_sha256=digest(args.repo / "tools/evaluate_flash_wuji_quality.py"),
            reused_evaluation_result=reused,failed_eval_attempts=failures))


def verified_phase(plan):
    manifest=Path(plan["inputs"])
    if digest(manifest)!=plan["inputs_sha256"] or digest(Path(plan["protocol_path"]))!=plan["protocol_sha256"]:
        raise ValueError("Frozen inputs or quality protocol changed; a corrected evaluator requires a separately versioned re-score.")
    verify_inputs(manifest)


def run(args):
    if (args.out / "obsolete_physics.json").exists():
        raise ValueError("This plan was withdrawn after the native physics audit; its original checkpoints are diagnostic only.")
    plan = read(args.out / "plan.json")
    verified_phase(plan)
    if (plan["bulk"] != str(args.bulk) or plan["probe_sha256"] != digest(args.probes)
            or plan["protocol_sha256"] != digest(Path(plan["protocol_path"]))
            or plan["physics"]["sha256"] != digest(Path(plan["physics"]["path"]))):
        raise ValueError("Plan, probe closure or frozen evaluator protocol changed")
    if any(digest(Path(r["parquet"])) != r["sha256"] for r in plan["references"]):
        raise ValueError("Reference changed after the plan was prepared")
    args.bulk.mkdir(parents=True, exist_ok=True)
    ledger = args.out / "ledger.json"
    rows = read(ledger) if ledger.exists() else []
    orders = [("single0","shared_mf","shared_wm","single1","single2"),
              ("shared_wm","single2","single1","single0","shared_mf"),
              ("shared_mf","single1","single0","single2","shared_wm")]
    for round_id, order in enumerate(orders):
        for key in order:
            if any(i["round"] == round_id+1 and i["stream"] == key for i in rows):
                continue
            previous = [i for i in rows if i["stream"] == key]
            latest = Path(previous[-1]["measured"]["path"]) if previous else None
            segment_started=time.monotonic()
            record = train_segment(args,plan,key,round_id,latest,previous)
            evaluate(args,plan,record)
            child=sum(check["eval_all_in_s"] for check in record["checks"] if not check["reused_evaluation_result"])
            if not record["reused_training_result"]:
                child+=record["train_all_in_s"]
            record["coordinator_overhead_s_this_invocation"]=max(0.,time.monotonic()-segment_started-child)
            rows.append(record)
            store(ledger,rows)
            print("SCORED",key,round_id+1,[(e["sequence"],e["passed"]) for e in record["checks"]],flush=True)
    store(args.out / "live.json",dict(stage="completed",utc=stamp(),scored_segments=len(rows)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("plan","run"))
    parser.add_argument("--inputs",type=lambda s:Path(s).resolve(strict=True))
    for name in ("repo","upstream","out","bulk","probes"):
        parser.add_argument("--"+name,required=True,type=lambda s:Path(s).resolve())
    opts=parser.parse_args()
    if opts.action == "plan":
        if opts.inputs is None:
            parser.error("--inputs must pin the artifacts, calibrated physics, training code and evaluator")
        if (opts.out / "plan.json").exists():
            raise FileExistsError(opts.out / "plan.json")
        print(json.dumps(prepare(opts),indent=2))
    else:
        opts.bulk.mkdir(parents=True,exist_ok=True)
        with (opts.bulk / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(opts)
