"""Measure single and three-reference native FlashSAC at matched updates per real sample."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


SEQUENCES = ("subject1_G04_2", "subject3_G10_3", "subject4_G16_4")
REPLAY_CAPACITY = 131072
PREFILL = 3072
PROBE_STEPS = 24576


def store(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def parquet_paths(repo):
    folder = repo / "runs/hocap_wuji_flashchord_v1/assets/human_motion_data/hocap/processed"
    refs = [(folder / f"sequence_id={seq}/robot_name=wuji_hand2/data.parquet").resolve(strict=True)
            for seq in SEQUENCES]
    return refs


def train_command(repo, upstream, out, references, worlds, steps, checkpoint_interval,
                  seed=42, resume_dir=None, hybrid=False):
    if worlds % len(references) or worlds % 32 or steps % worlds or checkpoint_interval % worlds:
        raise ValueError("Unbalanced sharding, update ratio or partial final collection")
    cmd = [sys.executable, "-u", str(repo / "tools/time_wuji_flashsac.py"), "train",
           "--upstream", str(upstream), "--assets", str(repo / "runs/flashchord_wuji_asset_v1"),
           "--out", str(out), "--worlds-per-reference", str(worlds // len(references)),
           "--device", "cuda:0", "--steps", str(steps), "--log-interval", "8",
           "--checkpoint-every", str(checkpoint_interval)]
    for ref in references:
        cmd.extend(["--parquet", str(ref)])
    # One critic/update opportunity per 32 native transitions at every world count.
    for override in (f"training.replay.capacity={REPLAY_CAPACITY}", f"training.replay.minimum_size={PREFILL}",
                     "training.replay.batch_size=128", f"training.updates_per_collection={worlds // 32}",
                     "training.optimization.learning_rate_schedule_updates=100000",
                     f"reset.seed={seed}", "training.seed=42"):
        cmd.extend(["--override", override])
    if hybrid:
        cmd.append("--hybrid")
    if resume_dir:
        result = json.loads((resume_dir / "result.json").read_bytes())
        prior = result["learner_steps"]
        cmd.extend(["--resume-state", str(resume_dir / f"checkpoints/state_{prior}.safetensors"),
                    "--resume-replay", str(resume_dir / f"checkpoints/replay_{prior}.safetensors")])
    return cmd


def run_child(cmd, output, env):
    output.parent.mkdir(parents=True, exist_ok=True)
    log = output.with_suffix(".stdout.log")
    if output.exists() or log.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parents[1]
    sources = [
        *sorted((repo/"src/supermimic_research/runtime").glob("flash*.py")),
        *sorted((repo/"src/supermimic_research/learning").glob("flash*.py")),
        *[repo/"tools"/n for n in ("time_wuji_flashsac.py","run_wuji_multiref_flashsac.py",
          "run_wuji_flashsac_smoke.py","evaluate_flash_wuji_quality.py","run_wuji_matched_tranche.py",
          "benchmark_wuji_parallel.py")]]
    proc_info = dict(command=cmd, utc_start=datetime.now(timezone.utc).isoformat(), finished=False,
        code_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        warm_seconds=env.get("WUJI_SEGMENT_WARM_SECONDS"),warm_after_step=env.get("WUJI_SEGMENT_WARM_AFTER_STEP"))
    info = output.with_suffix(".process.json")
    store(info, proc_info)
    start = time.monotonic()
    # One persistent board sampler; fixed common sampling period for every measured arm.
    try:
        with output.with_suffix(".board.csv").open("x") as board, log.open("x") as stream:
            poller = subprocess.Popen(["/usr/lib/wsl/lib/nvidia-smi",
                "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits", "--loop-ms=500"], stdout=board, stderr=subprocess.DEVNULL)
            try:
                proc = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
                proc_info["worker_pid"]=proc.pid
                store(info,proc_info)
                code=proc.wait()
            finally:
                poller.terminate()
                poller.wait(timeout=10)
    except BaseException as error:
        proc_info.update(elapsed_s=time.monotonic()-start,error=repr(error))
        store(info,proc_info)
        raise
    proc_info.update(finished=True, elapsed_s=time.monotonic() - start, returncode=code)
    store(info, proc_info)
    if code:
        raise RuntimeError(f"Process exited {code}. Inspect {log}")
    return proc_info


def summarize_run(output, previous_steps=0):
    records = [json.loads(l) for l in (output / "timing.jsonl").read_text().splitlines()]
    ready = next(r for r in records if r["event"] == "native_rl_env_ready")
    start = next(r["wall_s"] for r in records if r["event"] == "xla_compile_start")
    end = next(r["wall_s"] for r in records if r["event"] == "xla_compile_end")
    budget_started = next((r["step"] for r in records if r["event"]=="budget_warm_start"),0)
    windows = [r for r in records if r["event"] == "synchronized_log"
               and r["step"] >= max(PREFILL * 2, previous_steps, budget_started)]
    if len(windows) < 2:
        raise ValueError("Insufficient synchronized learner windows to compute steady rate")
    # Subtract completed native checkpoint writes inside the measured interval.
    saves = [r for r in records if r["event"] in ("native_checkpoint_start","sidecar_checkpoint_end")]
    inside = sum(b["wall_s"] - a["wall_s"] for a,b in zip(saves[::2], saves[1::2], strict=True)
                 if windows[0]["wall_s"] < a["wall_s"] and b["wall_s"] < windows[-1]["wall_s"])
    warm_s = windows[-1]["wall_s"] - windows[0]["wall_s"] - inside
    steps = windows[-1]["step"] - windows[0]["step"]
    result = json.loads((output / "result.json").read_bytes())
    final_save=next(r for r in reversed(records) if r["event"]=="native_checkpoint_start")
    telemetry = []
    for row in output.with_suffix(".board.csv").read_text().splitlines():
        try:
            telemetry.append([float(x.strip()) for x in row.split(",")[1:]])
        except ValueError:
            continue
    if not telemetry:
        raise ValueError("Headroom and power measurement unavailable")
    peak = np.asarray(telemetry).max(axis=0)
    return dict(path=str(output), wall_s=json.loads(output.with_suffix(".process.json").read_bytes())["elapsed_s"],
                setup_through_env_ready_s=ready["wall_s"], xla_compile_or_cache_load_s=end-start,
                synchronized_warm_steps=steps, synchronized_warm_s=warm_s, warm_env_steps_per_s=steps/warm_s,
                sampled_peak_board_MiB=peak[1], sampled_peak_power_W=peak[2], sampled_peak_gpu_percent=peak[0],
                board_samples=len(telemetry), logical_replay=ready["logical_replay"],
                total_steps=result["learner_steps"], optimizer_updates=result["optimizer_updates"],
                accepted_critic_adam_updates=final_save.get("accepted_critic_adam_updates"),
                accepted_actor_adam_updates=final_save.get("accepted_actor_adam_updates"),
                native_critic_updates=result["native_sac_critic_updates"], native_actor_updates=result["native_sac_actor_updates"])


def environment(repo, upstream):
    env = os.environ.copy()
    base = upstream.parent
    env.update(PYTHONPATH=str(repo / "src"), XDG_CACHE_HOME=str(base / "cache"),
               CUDA_CACHE_PATH=str(base / "cache/cuda"), TMPDIR=str(base / "tmp"),
               XLA_PYTHON_CLIENT_PREALLOCATE="false", WANDB_MODE="disabled", PYTHONUNBUFFERED="1")
    return env


def measure_physics(path):
    manifest=json.loads(path.read_bytes())
    overrides=manifest["overrides"]
    if not overrides or not all(isinstance(v,str) and v for v in overrides):
        raise ValueError("Specify the common validated solver overrides")
    if manifest.get("validated") is not True:
        raise ValueError("Throughput measurements require a validated contact solver")
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),overrides=overrides,
                scheme=manifest["scheme"])


def probe(repo, upstream, output, physics, steps):
    refs = parquet_paths(repo)
    env = environment(repo, upstream)
    env["WUJI_PHYSICS_SCHEME"]=physics["scheme"]
    output.mkdir(parents=True, exist_ok=True)
    total, free = map(int, subprocess.check_output(
        ["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True).splitlines()[0].split(","))
    dims = json.loads((repo / "runs/wuji_flashsac_health_v1/train04/replay_memory.json").read_bytes())["spec"]
    obs = dims["observation_dim"] + 3
    stride = 2 * obs * 4 + 4 * (dims["action_dim"] + dims["objective_term_count"] + 2 * dims["critic_context_dim"]) + 2
    estimate = {str(n): (REPLAY_CAPACITY + n * (dims["n_step"] - 1)) * stride + 16 for n in (96,192)}
    store(output / "memory_preflight.json", dict(physics=physics,steps_per_probe=steps,native_dimension_source=dims, augmented_observation_dim=obs,
          bytes_per_transition=stride, replay_capacity=REPLAY_CAPACITY, replay_bytes=estimate,
          total_MiB=total, initial_free_MiB=free, scope="Replay persistent array budget. Other allocations measured during each run."))
    if max(estimate.values()) >= free * 2**20:
        raise MemoryError("Logical replay alone exceeds available GPU memory")
    rows = {}
    for name, worlds, bank in (("single96",96,refs[:1]), ("three96",96,refs),
                               ("three192",192,refs), ("single192",192,refs[:1])):
        destination = output / name
        cmd = train_command(repo, upstream, destination, bank, worlds, steps, steps)
        for override in physics["overrides"]:
            cmd.extend(["--override",override])
        if not (destination / "result.json").is_file():
            run_child(cmd, destination, env)
        rows[name] = summarize_run(destination)
        store(output / "probes.json", rows)
        print(name, json.dumps(rows[name]), flush=True)


def probe_hybrid(repo, upstream, output, worlds, physics):
    refs = parquet_paths(repo)
    env = environment(repo,upstream)
    env["WUJI_PHYSICS_SCHEME"]=physics["scheme"]
    warm_after=104 * worlds
    env.update(WUJI_SEGMENT_WARM_SECONDS="12",WUJI_SEGMENT_WARM_AFTER_STEP=str(warm_after))
    dest = output / f"hybrid{worlds}"
    cmd=train_command(repo,upstream,dest,refs,worlds,worlds*384,worlds*16,hybrid=True)
    for override in physics["overrides"]:
        cmd.extend(["--override",override])
    cmd.extend(["--wm-fit-horizon","3"])
    if not (dest/"result.json").is_file():
        run_child(cmd,dest,env)
    row=summarize_run(dest)
    stat=json.loads((dest/"result.json").read_bytes())["auxiliary"]
    if stat["imagined_actor_updates"]<=0 or stat["world_model_updates"]<=0:
        raise ValueError("Hybrid throughput requires applied model and imagined policy updates")
    row["auxiliary"]=stat
    store(output/"hybrid_probe.json",row)
    print(json.dumps(row),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "upstream", "out"):
        p.add_argument("--" + name, required=True, type=lambda x: Path(x).resolve())
    p.add_argument("--hybrid-worlds",type=int,choices=(96,192))
    p.add_argument("--physics",type=lambda s:Path(s).resolve(strict=True),required=True,
                   help="Frozen JSON containing validated solver overrides and their calibration evidence")
    p.add_argument("--probe-steps",type=int,default=PROBE_STEPS)
    args = p.parse_args()
    physics=measure_physics(args.physics)
    if (args.out/"physics_scope_correction.json").is_file():
        raise ValueError("Use a new run directory for corrected contact physics")
    if args.hybrid_worlds:
        if json.loads((args.out/"memory_preflight.json").read_bytes())["physics"]!=physics:
            raise ValueError("Physics changed between single, shared and hybrid probes")
        probe_hybrid(args.repo,args.upstream,args.out,args.hybrid_worlds,physics)
    else:
        probe(args.repo, args.upstream, args.out,physics,args.probe_steps)
