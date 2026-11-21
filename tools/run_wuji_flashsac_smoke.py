"""Run an instrumented, bounded Wuji experiment through the tagged FlashCHORD entry points."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def append(path, value):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def load_entry(path):
    spec = importlib.util.spec_from_file_location("tagged_flashchord_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LocalMetrics:
    def __init__(self, out):
        self.path = out / "learner_metrics.jsonl"

    def log(self, metrics, step):
        # The upstream sink includes empty-episode means. Preserve those as null.
        import math
        append(self.path, {"step": int(step), "metrics": {
            k: float(v) if math.isfinite(v) else None for k, v in metrics.items()
        }})

    def close(self):
        pass


def record_device(stop, out, start):
    command = ["/usr/lib/wsl/lib/nvidia-smi",
               "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"]
    while not stop.is_set():
        try:
            fields = subprocess.check_output(command, text=True, timeout=6).splitlines()[0].split(",")
            util, memory, power = [float(v.strip()) for v in fields]
            append(out / "device.jsonl", dict(wall_s=time.monotonic() - start, gpu_percent=util,
                                              board_memory_MiB=memory, board_power_W=power))
        except (OSError, ValueError, subprocess.SubprocessError, IndexError) as error:
            append(out / "device.jsonl", dict(wall_s=time.monotonic() - start, error=str(error)))
        stop.wait(2)


def config_for(args):
    from hydra import compose, initialize_config_dir
    group_dir = args.upstream / "src/flash_chord/configs"
    with initialize_config_dir(config_dir=str(group_dir), version_base="1.3"):
        if args.mode == "evaluate":
            return compose(config_name="evaluate_policy", overrides=[
                f"evaluation.checkpoint='{args.checkpoint}'", f"evaluation.world_count={args.worlds}",
                "evaluation.start_frame=0", "evaluation.reset_mode=explicit", "logging.mode=disabled",
                f"evaluation.metrics_output='{args.out / 'official_evaluation.json'}'"])
        overrides = [
            "experiment=sharpa_flash_sac", "embodiment._target_=supermimic_research.runtime.flash_wuji.WujiHand2",
            "~embodiment.config", f"+embodiment.asset_dir='{args.assets}'",
            "action._target_=supermimic_research.runtime.flash_wuji.WujiActionConfig",
            "objective._target_=supermimic_research.runtime.flash_wuji_objective.make_right_objective",
            "observation.sides=[right]", "observation.contact_sides=[right]", "+env.contact.sides=[right]",
            "action.wrist_kp=38.5", "action.wrist_tau_max=2.309401", "action.delay.max_steps=0",
            f"task.parquet='{args.parquet}'", "task.motion_speed=1.0",
            f"scene.world_count={args.worlds}", "scene.support=true", "collision.ground=false",
            "collision.robot_support_collision=true",
            "scene.contact_friction=0.7", "scene.object_scale_min=1.0", "scene.object_scale_max=1.0",
            "scene.object_free_joint_damping=0", "sim.fps=30", "sim.substeps=40",
            f"training.total_environment_steps={args.steps}", "training.replay.capacity=4096",
            "training.replay.minimum_size=256", "training.replay.batch_size=128",
            "training.updates_per_collection=1", "training.checkpoint.save_replay=true",
            f"training.checkpoint.save_interval_environment_steps={args.steps}",
            "logging.mode=disabled", "logging.log_interval=4", "voc_scale_override=0.0",
            f"output_dir='{args.out / 'checkpoints'}'",
        ]
        if args.mode == "restore":
            overrides.extend(["training.resume=true", f"training.checkpoint_path='{args.checkpoint}'",
                              f"training.checkpoint.load_replay_path='{args.replay}'"])
        return compose(config_name="train_flash_sac", overrides=overrides + args.override)


def run(args, start):
    from omegaconf import OmegaConf
    cfg = config_for(args)
    dump(args.out / "resolved.json", OmegaConf.to_container(cfg, resolve=True))
    if args.mode == "compose":
        if args.parquet.is_file():
            from flash_chord.data.loader import load_reference
            ref = load_reference(str(args.parquet))
            dump(args.out / "reference.json", dict(frames=ref.num_frames, fps=ref.fps, sides=list(ref.sides)))
        return
    import warp as wp
    wp.config.kernel_cache_dir = str(args.upstream.parent / "cache/warp")
    if args.mode == "evaluate":
        script = load_entry(args.upstream / "scripts/evaluate_policy.py")
        script.main.__wrapped__(cfg)
        return
    from flash_chord.training.flash_sac.replay import ReplaySpec, replay_memory
    from flash_chord.training.flash_sac import runner
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    script = load_entry(args.upstream / "scripts/train_flash_sac.py")
    metrics = LocalMetrics(args.out)

    def event(name, **data):
        append(args.out / "phases.jsonl", dict(event=name, wall_s=time.monotonic() - start, **data))

    # The native entry builds the embodiment, reference, Warp graph and checkpoint metadata.
    # Inspect its actual dimensions immediately before the native runner allocates learner arrays.
    def observe_training(env, config, **kwargs):
        spec = ReplaySpec.from_config(
            config.replay, discount=config.discount, world_count=env.world_count,
            observation_dim=env.observation_dim, action_dim=env.action_dim,
            objective_term_count=len(env.objective_term_names), critic_context_dim=env.critic_context_dim)
        sizes = {str(cap): asdict(replay_memory(replace(spec, capacity=cap)))
                 for cap in [config.replay.capacity, 4194304]}
        dump(args.out / "replay_memory.json", dict(spec=asdict(spec), bytes=sizes))
        event("scene_ready", world_count=env.world_count)
        kwargs["logger"] = metrics
        return runner.run_training(env, config, **kwargs)

    # Measure XLA compile separately from environment construction and sampling.
    compile_executables = runner._compile_executables
    def observe_compile(*a, **kw):
        event("learner_compile_start")
        answer = compile_executables(*a, **kw)
        event("learner_compile_end")
        return answer

    runner._compile_executables = observe_compile
    script.run_training = observe_training
    try:
        event("environment_build_start")
        script.main.__wrapped__(cfg)
    finally:
        runner._compile_executables = compile_executables
    event("training_finished")
    ckpt_dir = args.out / "checkpoints"
    metadata = {}
    for kind in ("state", "replay", "policy"):
        path = ckpt_dir / f"{kind}_{args.steps}.safetensors"
        m = read_checkpoint_metadata(path)
        metadata[kind] = {key: m.get(key) for key in ("environment_steps", "global_update_step", "voc_scale",
                                                      "replay_included", "checkpoint_set_id")}
    dump(args.out / "checkpoint_health.json", metadata)
    if int(metadata["state"]["global_update_step"]) <= 0:
        raise ValueError("Expected a nonzero number of native optimizer updates")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("compose", "train", "restore", "evaluate"))
    for key in ("upstream", "parquet", "assets", "out"):
        p.add_argument("--" + key, required=True, type=lambda s: Path(s).resolve())
    p.add_argument("--steps", type=int, default=1024)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--replay", type=Path)
    p.add_argument("--override", action="append", default=[])
    args = p.parse_args()
    if args.mode in ("restore", "evaluate") and not (args.checkpoint and args.checkpoint.is_file()):
        p.error("A saved checkpoint is required")
    if args.mode == "restore" and not (args.replay and args.replay.is_file()):
        p.error("A matching saved replay file is required for resume")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    start = time.monotonic()
    args_doc = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    info = dict(arguments=args_doc, utc_start=datetime.now(timezone.utc).isoformat(), complete=False)
    dump(args.out / "attempt.json", info)
    stop = threading.Event()
    thread = threading.Thread(target=record_device, args=(stop, args.out, start), daemon=True)
    if args.mode != "compose":
        thread.start()
    try:
        run(args, start)
        info["complete"] = True
    except BaseException:
        info["error"] = traceback.format_exc()
        raise
    finally:
        info["elapsed_s"] = time.monotonic() - start
        stop.set()
        if thread.is_alive():
            thread.join(timeout=7)
        dump(args.out / "attempt.json", info)


if __name__ == "__main__":
    main()
