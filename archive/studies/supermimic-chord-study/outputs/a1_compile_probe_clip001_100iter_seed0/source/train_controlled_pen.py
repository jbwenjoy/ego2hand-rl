"""Bounded, recorded A1 training with the existing Dex SuperMimic implementation.

Example: isolated-dex-python scripts/train_controlled_pen.py --config probe.json
Input is JSON; each run freezes its config, single-clip derivative, source,
physics/control context, update counts, checkpoints and phase timing. A compile
failure can restart an equal-budget eager attempt from the same seed. This
entry point contains no CHORD objective and rejects method=A2 until implemented.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import dataclasses
import gc
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

STARTED = time.perf_counter()
STUDY_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve(value):
    path = Path(value).expanduser()
    return (STUDY_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def git(path, *arguments):
    return subprocess.check_output(["git", "-C", str(path), *arguments], text=True, encoding="utf-8").strip()


def prepare_reference(config, output):
    """Copy a slice, without resampling/pose correction; retain original lineage."""
    import numpy as np

    allowed = {"clips_root", "source_clip_id", "start_frame", "duration_s"}
    if set(config) - allowed:
        raise ValueError(f"Unknown reference keys: {sorted(set(config) - allowed)}")
    root = resolve(config["clips_root"])
    source_id = config.get("source_clip_id", "01_single_axis/clip_001")
    components = source_id.split("/")
    if len(components) != 2 or any(p in {"", ".", ".."} or "\\" in p for p in components):
        raise ValueError("source_clip_id must be a safe group/clip ID")
    group, local = components
    index_path = root / group / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig"))
    entry = next(item for item in index["clips"] if item["clip_id"] == local)
    source = root / group / entry["file"]
    if source.resolve().parent != (root / group).resolve() or digest(source) != entry["sha256"]:
        raise ValueError("Source clip path/hash does not match the original catalog")
    dt = float(index["frame_dt"])
    if abs(dt - .02) > 1e-8:
        raise ValueError("This controller requires original 50 Hz data")
    start = int(config.get("start_frame", 0))
    duration = float(config.get("duration_s", 3.0))
    intervals = round(duration / dt)
    if start < 0 or intervals < 2 or abs(intervals * dt - duration) > 1e-8:
        raise ValueError("start_frame/duration_s must define an integer, positive 50 Hz interval")
    stop = start + intervals + 1  # Includes both endpoints: 3s is 151 samples.
    with np.load(source, allow_pickle=False) as bundle:
        count = len(bundle["reference_qpos"])
        if stop > count:
            raise ValueError(f"Requested [{start},{stop}) exceeds {count} source frames")
        arrays = {key: bundle[key][start:stop].copy() if bundle[key].ndim and len(bundle[key]) == count else bundle[key].copy()
                  for key in bundle.files}
    derived_id = f"{local}_f{start:06d}_n{stop - start:04d}"
    clips = output / "reference/clips"
    group_root = clips / "controlled"
    group_root.mkdir(parents=True)
    derived = group_root / f"{derived_id}.npz"
    np.savez_compressed(derived, **arrays)
    write_json(group_root / "index.json", {
        "schema": "wuji_pen_spin_clips:1", "group_id": "controlled", "group_name": "Frozen single-clip derivative",
        "frame_dt": dt, "clip_count": 1,
        "clips": [{"clip_id": derived_id, "file": derived.name, "sha256": digest(derived)}],
    })
    recipe = output / "reference/recipe.json"
    write_json(recipe, {"schema": "wuji_pen_spin_recipe:1", "clips": [f"controlled/{derived_id}"]})
    provenance = {
        "source_kind": "upstream robot-hand reference; not raw egocentric video",
        "source_clip_id": source_id, "source_file": str(source), "source_sha256": digest(source),
        "source_index_sha256": digest(index_path), "source_frame_range_half_open": [start, stop],
        "frame_dt_s": dt, "frames": stop - start, "control_intervals": intervals, "duration_s": duration,
        "resampling": False, "joint_or_object_trajectory_correction": False,
        "derived_file": str(derived), "derived_sha256": digest(derived),
        "derived_clip_id": f"controlled/{derived_id}", "recipe_sha256": digest(recipe),
    }
    write_json(output / "reference/provenance.json", provenance)
    return recipe, clips, provenance


def freeze_source(dex_root, output):
    """Keep exact Python/config code while recording original Git provenance."""
    target = output / "source"
    target.mkdir()
    shutil.copy2(__file__, target / Path(__file__).name)
    files = sorted((dex_root / "src/dex_supermimic").rglob("*.py"))
    files += [dex_root / name for name in ["pyproject.toml", "uv.lock", "WINDOWS.md"]]
    inventory = {}
    with zipfile.ZipFile(target / "dex_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in files:
            relative = file.relative_to(dex_root).as_posix()
            bundle.write(file, relative)
            inventory[relative] = digest(file)
    (target / "dex_changes.diff").write_text(git(dex_root, "diff", "--binary"), encoding="utf-8")
    source = {
        "dex_root": str(dex_root), "dex_commit": git(dex_root, "rev-parse", "HEAD"),
        "dex_status": git(dex_root, "status", "--short"), "python_executable": sys.executable,
        "script_sha256": digest(__file__), "dex_source_zip_sha256": digest(target / "dex_source.zip"),
        "files": inventory,
    }
    write_json(target / "inventory.json", source)
    return source


def run_attempt(input_config, dex_root, recipe, clips, output, compiled):
    import numpy as np
    import torch
    import warp as wp
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.config import DexSuperMimicCfg
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from dex_supermimic.rl.dex_supermimic.pen import PEN_MEMORY_DIMS
    from dex_supermimic.tasks.pen_spin.task import make_env_cfg
    import wuji_description

    output.mkdir()
    started = time.perf_counter()
    seed = int(input_config.get("seed", 0))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.suppress_errors = False
    if not torch.cuda.is_available():
        raise RuntimeError("An NVIDIA CUDA device is required")
    wp.set_device("cuda:0")
    defaults = dict(num_envs=128, max_iterations=100, buffer_steps=256, warmup_steps=64,
                    wm_only_iters=20, wm_window=8, wm_batches=2, wm_batch=256, wm_hidden=(512, 512, 512),
                    pi_batch=128, horizon=8, max_horizon=8, critic_epochs=2, critic_minibatches=2,
                    holdout_fraction=.1, save_interval=100, log_interval=10, profile=True, amp=True,
                    compile=True, compile_policy=True, compile_policy_dynamic=False, action_noise=.1)
    requested = dict(input_config.get("training", {}))
    forbidden = {"task", "algorithm", "recipe_path", "clips_root", "device", "logger", "seed", "contact_weight", "failure_head", "checkpoint", "model_checkpoint"}
    known = {field.name for field in dataclasses.fields(DexSuperMimicCfg)}
    if set(requested) - known or set(requested) & forbidden:
        raise ValueError(f"Unknown/controlled training keys: {sorted((set(requested) - known) | (set(requested) & forbidden))}")
    defaults.update(requested)
    defaults.update(compile=compiled, compile_policy=compiled and bool(defaults["compile_policy"]), profile=True,
                    task="hand2-pen-spin", algorithm="world_model", recipe_path=str(recipe), clips_root=str(clips),
                    device="cuda:0", seed=seed, logger="none", contact_weight=.1, failure_head=False)
    defaults["wm_hidden"] = tuple(defaults["wm_hidden"])
    cfg = DexSuperMimicCfg(**defaults)
    if not 1 <= cfg.max_iterations <= 100000 or not 0 <= cfg.wm_only_iters < cfg.max_iterations:
        raise ValueError("Use a finite positive iteration budget with at least one policy update")
    if cfg.save_interval > 100:
        raise ValueError("Checkpoint interval must be <=100 iterations")
    write_json(output / "effective_training_config.json", cfg.to_dict())

    class ControlledPenEnv(ManagerBasedRlEnv):
        """Align dt=0 resets with the evaluator without additional pose writes."""
        reset_worlds = 0

        def reset(self, *, seed=None, env_ids=None, options=None):
            obs, info = super().reset(seed=seed, env_ids=env_ids, options=options)
            ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
            motion = self.command_manager.get_term("motion")
            motion.time_steps[ids] = 0
            motion._episode_start_steps[ids] = 0
            term = self.action_manager.get_term("joint_pos")
            term._targets[ids] = motion.joint_pos[ids] - term._entity.data.encoder_bias[ids][:, term._target_ids]
            term._targets_valid[ids] = True
            term.apply_actions()
            self.scene.write_data_to_sim()
            self.sim.forward(); self.sim.sense()
            self.obs_buf = self.observation_manager.compute(update_history=False)
            self.reset_worlds += len(ids)
            return self.obs_buf, info

    class RecordedRunner(DexSuperMimicRunner):
        def __init__(self, *args, **kwargs):
            self.phase_seconds = defaultdict(float)
            self.phase_calls = defaultdict(int)
            self.actual_optimizer_steps = defaultdict(int)
            self.deadline = started + float(input_config.get("max_attempt_seconds", 900))
            super().__init__(*args, **kwargs)
            for name, optimizer in self.optimizers.items():
                optimizer.register_step_post_hook(lambda _opt, _args, _kwargs, name=name: self.actual_optimizer_steps.__setitem__(name, self.actual_optimizer_steps[name] + 1))
            self.base_writer = self.writer
            runner = self
            class Writer:
                def log_scalars(self, values, step):
                    runner.base_writer.log_scalars(values, step)
                    with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"iteration": step, **values}) + "\n")
                def close(self):
                    runner.base_writer.close()
            self.writer = Writer()

        @contextmanager
        def phase(self, name):
            if time.perf_counter() > self.deadline:
                raise TimeoutError("Bounded training attempt exceeded max_attempt_seconds")
            torch.cuda.synchronize(); tick = time.perf_counter()
            try:
                yield
            finally:
                torch.cuda.synchronize()
                self.phase_seconds[name] += time.perf_counter() - tick
                self.phase_calls[name] += 1

        def collect(self, steps):
            with self.phase("collect_including_warmup"):
                return super().collect(steps)

        def fit_world_model(self, holdout=False):
            with self.phase("wm_holdout" if holdout else "wm_train"):
                return super().fit_world_model(holdout)

        def policy_update(self):
            with self.phase("policy"):
                return super().policy_update()

        def save(self, path, next_iteration=None):
            with self.phase("checkpoint"):
                return super().save(path, next_iteration)

    env_cfg = make_env_cfg(cfg.num_envs, 50, play=True, recipe_path=recipe, clips_root=clips)
    env_cfg.seed = seed; env_cfg.auto_reset = False
    motion_cfg = env_cfg.commands["motion"]
    motion_cfg.first_frame_hold_prob = 0.; motion_cfg.first_frame_hold_s = (0., 0.)
    motion_cfg.trajectory_sampling_mode = "uniform"; motion_cfg.debug_vis = False
    env_cfg.actions["joint_pos"].mask_prob = 0.
    env_cfg.terminations["object_position_error"].params["threshold"] = .05
    from mjlab.managers import TerminationTermCfg, SceneEntityCfg
    from dex_supermimic.tasks.pen_spin import mdp
    env_cfg.terminations["object_orientation_error"] = TerminationTermCfg(
        func=mdp.object_shaft_direction_error_exceeded,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=("right_palm_link",)),
                "object_cfg": SceneEntityCfg("object"), "threshold_deg": 60.}, time_out=False)
    if set(env_cfg.events) != {"reset_to_reference_state", "pen_dimensions"}:
        raise ValueError("Unexpected DR/exogenous event in controlled scene")
    for key in ("diameter_range_m", "total_length_range_m"):
        bounds = env_cfg.events["pen_dimensions"].params[key]
        if bounds[0] != bounds[1]:
            raise ValueError("Controlled pen dimensions must be fixed")
    if any(group.enable_corruption for group in env_cfg.observations.values()):
        raise ValueError("Observation noise must be disabled")
    env = None; runner = None
    result = {
        "status": "running", "method": "A1 upstream SuperMimic; no CHORD objective",
        "compiled_requested": compiled, "seed": seed, "training_budget_iterations": cfg.max_iterations,
        "contact_head": "15-dimensional finger self-collision; not hand-object wrench/contact",
        "contact_weight": .1, "failure_head_used_in_policy": False, "auxiliary_failure_head_fit_retained": True,
        "initial_grasp_status": input_config.get("initial_grasp_status", "unqualified; throughput/compile probe only"),
        "training_episode_reset": "first frame, no hold/RSI; phase corrected to zero at dt=0 reset",
        "training_terminations": {"object_position_m": .05, "shaft_angle_deg": 60.},
        "versions": {p: metadata.version(p) for p in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "triton-windows")},
        "asset_revision": wuji_description.REVISION,
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        env_started = time.perf_counter()
        env = ControlledPenEnv(env_cfg, device=cfg.device)
        result["environment_init_seconds"] = time.perf_counter() - env_started
        runner_started = time.perf_counter()
        runner = RecordedRunner(env, cfg, output / "train")
        result["runner_init_seconds"] = time.perf_counter() - runner_started
        assert runner.contact_dim == 15 and bool((runner.observations.disturbance(runner.observations.auxiliary()) == 0).all())
        assert bool((env.command_manager.get_term("motion").time_steps == 0).all())
        before = {name: [p.detach().cpu().clone() for p in module.parameters()] for name, module in runner.modules().items()}
        before_contact = runner.wm.net[-1].weight[-15:].detach().cpu().clone()
        body_fields = ("body_mass", "body_inertia", "body_iquat", "geom_size", "geom_pos", "geom_friction", "actuator_gainprm", "actuator_biasprm")
        physical = {name: getattr(env.sim.model, name).detach().cpu().numpy() for name in body_fields}
        physical["reader_context"] = runner.reader.xi.detach().cpu().numpy()
        np.savez_compressed(output / "fixed_physics_context.npz", **physical)
        mujoco.mj_saveModel(env.sim.mj_model, str(output / "nominal_model.mjb"), None)
        write_json(output / "context_names.json", {"reader_context": runner.reader.context_names, "memory_dims": PEN_MEMORY_DIMS,
                                                    "physical_snapshot": "actual Warp model fields; nominal_model.mjb is the host model before per-world fixed pen dimensions"})
        train_started = time.perf_counter()
        runner.learn()
        torch.cuda.synchronize()
        result["learn_wall_seconds"] = time.perf_counter() - train_started
        changes = {}
        for name, module in runner.modules().items():
            params = list(module.parameters())
            if not all(bool(torch.isfinite(p).all()) for p in params):
                raise FloatingPointError(f"Nonfinite final parameters: {name}")
            changes[name] = sum(float((p.detach().cpu() - old).double().square().sum()) for p, old in zip(params, before[name])) ** .5
        for name in ("actor", "wm", "critic"):
            if changes[name] == 0 or runner.actual_optimizer_steps[name] == 0:
                raise AssertionError(f"No real {name} optimizer updates")
        contact_delta = float((runner.wm.net[-1].weight[-15:].detach().cpu() - before_contact).double().square().sum()) ** .5
        if contact_delta == 0:
            raise AssertionError("Self-collision contact head was not supervised")
        rows = (runner.buffer.ptr - runner.buffer.size + torch.arange(min(runner.buffer.size, 8), device=runner.device)) % runner.buffer.capacity
        trace = {key: value[rows, :min(cfg.num_envs, 4)].detach().cpu().numpy() for key, value in runner.buffer.data.items()}
        trace["reader_context"] = runner.reader.xi[:min(cfg.num_envs, 4)].detach().cpu().numpy()
        np.savez_compressed(output / "replay_control_trace.npz", **trace)
        checkpoint = output / "train/model_final.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert saved["iteration"] == cfg.max_iterations
        result.update(status="passed", final_iteration=saved["iteration"], reference_signature=saved["reference_signature"],
                      parameter_delta_l2=changes, self_collision_head_weight_delta_l2=contact_delta,
                      checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
                      nominal_model_sha256=digest(output / "nominal_model.mjb"),
                      compiled_counters={group: {key: int(value) for key, value in entries.items()} for group, entries in torch._dynamo.utils.counters.items()},
                      collected_sim_transitions=runner.sim_steps, reset_world_count=env.reset_worlds,
                      torch_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
        return result
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["attempt_wall_seconds"] = time.perf_counter() - started
        if runner is not None:
            result.update(phase_seconds=dict(runner.phase_seconds), phase_calls=dict(runner.phase_calls),
                          actual_optimizer_steps=dict(runner.actual_optimizer_steps), last_started_iteration=runner.iteration)
            runner.close()
        if env is not None:
            env.close()
        write_json(output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    allowed = {"schema", "label", "method", "seed", "output_dir", "dex_root", "reference", "training", "compile_fallback", "max_attempt_seconds", "initial_grasp_status"}
    if set(config) - allowed or config.get("schema") != "supermimic_chord_train:1":
        raise ValueError("Invalid configuration schema/keys")
    if config.get("method", "A1") != "A1":
        raise ValueError("This entry point implements A1 only; A2 requires an actual contact objective")
    dex_root = resolve(config.get("dex_root", "D:/workspace/dex-supermimic-research"))
    output = resolve(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a prior experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output / "input_config.json")
    source = freeze_source(dex_root, output)
    recipe, clips, provenance = prepare_reference(config["reference"], output)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(dex_root / ".cache/torchinductor/controlled_pen"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(dex_root / ".cache/triton/controlled_pen"))
    result = {"label": config.get("label"), "config_sha256": digest(config_path), "source": source,
              "reference": provenance, "attempts": [], "claim_scope": "bounded training/compile/throughput; quality requires independent no-reset physics evaluation"}
    compile_requested = bool(config.get("training", {}).get("compile", True))
    modes = [True, False] if compile_requested and config.get("compile_fallback", True) else [compile_requested]
    for compiled in modes:
        attempt = output / ("compiled" if compiled else "eager")
        try:
            completed = run_attempt(config, dex_root, recipe, clips, attempt, compiled)
            result["attempts"].append(completed)
            result["status"] = "passed"
            break
        except Exception as error:
            details = {"status": "failed", "compiled": compiled, "error": repr(error), "traceback": traceback.format_exc()}
            write_json(attempt / "failure.json", details)
            result["attempts"].append({**details, "partial_result": json.loads((attempt / "result.json").read_text()) if (attempt / "result.json").exists() else None})
            message = repr(error).lower()
            compiler_error = any(word in message for word in ("compile", "inductor", "triton", "backendcompiler", "invalidcudacode", "max_attempt_seconds"))
            if not compiled or not compiler_error or len(modes) == 1:
                result["status"] = "failed"
                result["total_wall_seconds_including_failed_attempts"] = time.perf_counter() - STARTED
                write_json(output / "summary.json", result)
                raise
            print(f"Compiled attempt failed; restarting same seeded budget eagerly: {error!r}", flush=True)
            import torch
            torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
    result["total_wall_seconds_including_failed_attempts"] = time.perf_counter() - STARTED
    write_json(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "output": str(output), "attempt_modes": [a.get("compiled_requested", a.get("compiled")) for a in result["attempts"]],
                      "total_wall_seconds": result["total_wall_seconds_including_failed_attempts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
