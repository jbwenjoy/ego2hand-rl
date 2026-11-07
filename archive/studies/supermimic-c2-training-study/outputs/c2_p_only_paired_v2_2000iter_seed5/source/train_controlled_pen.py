"""Bounded, recorded A1 training with the existing Dex SuperMimic implementation.

Example: isolated-dex-python scripts/train_controlled_pen.py --config probe.json
Input is JSON; each run freezes its config, single-clip derivative, source,
physics/control context, update counts, checkpoints and phase timing. A compile
failure can restart an equal-budget eager attempt from the same seed. Optional
A2 shaping is an explicitly unqualified geometric-intent prior used only in
policy imagination; actual collection and native WM fitting are unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import dataclasses
from datetime import datetime, timezone
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
sys.path.insert(0,str(STUDY_ROOT / "src"))


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


def freeze_paired_protocol(config, output, provenance, source):
    if not config.get("paired_protocol_file"):
        return
    path = resolve(config["paired_protocol_file"])
    protocol = json.loads(path.read_text(encoding="utf-8-sig"))
    frozen = provenance.get("frozen_evaluation_reference")
    if protocol.get("schema") != "supermimic_chord_paired_contact_pilot:2" or not frozen:
        raise ValueError("Matched pilot requires the version2 paired protocol and frozen evaluation reference")
    if frozen["npz_sha256"] != protocol["reference_npz_sha256"] or frozen["task_acceptance_sha256"] != protocol["task_acceptance_sha256"]:
        raise ValueError("Training reference/task criteria disagree with the paired protocol")
    shutil.copy2(path, output / "source/paired_protocol.json")
    source["paired_protocol"] = {"file": str(path), "sha256": digest(path),
                                 "run_role": config.get("run_role", "primary_matched_arm")}
    write_json(output / "source/inventory.json", source)


def validate_paired_budget(cfg, config, output):
    snapshot = output.parent / "source/paired_protocol.json"
    if not snapshot.exists():
        return
    protocol = json.loads(snapshot.read_text(encoding="utf-8-sig"))
    common = protocol["shared_settings"]
    field_map = {"WM_hidden": "wm_hidden", "WM_window": "wm_window", "WM_batches": "wm_batches", "WM_batch": "wm_batch",
                 "policy_batch": "pi_batch", "horizon": "horizon", "critic_epochs": "critic_epochs", "critic_minibatches": "critic_minibatches",
                 "amp": "amp", "compile": "compile"}
    for declared, field in field_map.items():
        actual = getattr(cfg, field)
        actual = list(actual) if isinstance(actual, tuple) else actual
        if actual != common[declared]:
            raise ValueError(f"Matched pilot setting {field} differs from the frozen protocol")
    if config.get("run_role", "primary_matched_arm") == "engineering_smoke":
        return
    budget = protocol["primary_budget"]
    actual = {"iterations": cfg.max_iterations, "num_envs": cfg.num_envs, "warmup_steps": cfg.warmup_steps,
              "wm_only_iters": cfg.wm_only_iters, "wm_updates": cfg.max_iterations * cfg.wm_batches,
              "actor_updates": cfg.max_iterations - cfg.wm_only_iters,
              "critic_updates": (cfg.max_iterations - cfg.wm_only_iters) * cfg.critic_epochs * cfg.critic_minibatches,
              "real_transitions": cfg.num_envs * (max(cfg.warmup_steps, cfg.wm_window) + int(cfg.max_iterations / cfg.utd))}
    if any(actual[key] != value for key, value in budget.items() if key != "type"):
        raise ValueError(f"Resolved sample/update budget differs from frozen primary budget: {actual}")


def cache_inventory(path):
    path = Path(path)
    files = [p for p in path.rglob("*") if p.is_file()] if path.exists() else []
    return {"path": str(path), "existed": path.exists(), "file_count": len(files),
            "bytes": sum(p.stat().st_size for p in files)}


@contextmanager
def imagination_contact_reward(runner, objective, scale=.02):
    """Temporarily shape policy imagination; preserve native callable/terms.

    The None/A1 branch performs no tensor or random-number operation. Even an
    exception restores reward_step before the next collect or WM-fit call.
    """
    native = runner.reward_step
    if objective is None:
        yield None
        return
    collected_terms = defaultdict(list)

    def shaped(x1, memory, action, residual, contact, torque, soft=False):
        reward, cage, native_terms = native(x1, memory, action, residual, contact, torque, soft=soft)
        bonus, terms = objective(x1, memory)
        if bonus.shape != reward.shape:
            raise ValueError("Geometric-intent bonus must match native reward shape")
        for name, value in terms.items():
            collected_terms[name].append(value.detach().float().mean())
        return reward + scale * bonus, cage, native_terms

    runner.reward_step = shaped
    try:
        yield collected_terms
    finally:
        runner.reward_step = native


def freeze_contact_inputs(config, output):
    """Validate/copy an independent fixed target bundle before any training."""
    allowed = {"kind", "bundle_dir", "reference_file", "model_file", "expected_hashes", "hyperparameters",
               "per_control_step_scale", "qualification_status"}
    if set(config) - allowed or config.get("kind") != "CHORD-inspired geometric intent shaping":
        raise ValueError("A2 requires the explicitly labelled geometric-intent prior configuration")
    if config.get("qualification_status") != "local_proxy_qualification_failed_exploratory_ablation":
        raise ValueError("Retain the failed local-proxy qualification in the A2 configuration")
    if config.get("per_control_step_scale") != .02:
        raise ValueError("A2 applies the unscaled geometric bonus exactly once at dt=0.02")
    hyper = config.get("hyperparameters", {})
    if set(hyper) != {"bandwidth_m", "tolerance", "var"}:
        raise ValueError("Explicit fixed bandwidth_m, tolerance and var are required")
    expected = config["expected_hashes"]
    required = {"manifest_sha256", "target_npz_sha256", "reference_npz_sha256", "reference_json_sha256", "model_sha256",
                "rules_sha256", "raw_geometry_sha256", "time_windows_sha256", "objective_module_sha256", "geometry_module_sha256"}
    if set(expected) != required or any(not isinstance(value, str) or len(value) != 64 for value in expected.values()):
        raise ValueError("A2 must freeze all target/model/reference/rule and objective-code SHA256 values")
    bundle = resolve(config["bundle_dir"])
    reference = resolve(config["reference_file"])
    model = resolve(config["model_file"])
    source = STUDY_ROOT / "src/chord_study"
    files = {
        "manifest_sha256": bundle / "manifest.json", "target_npz_sha256": bundle / "contact_targets.npz",
        "reference_npz_sha256": reference, "reference_json_sha256": reference.with_suffix(".json"), "model_sha256": model,
        "rules_sha256": bundle / "rules.json", "raw_geometry_sha256": bundle / "raw_geometry.json",
        "time_windows_sha256": bundle / "time_windows.json", "objective_module_sha256": source / "contact_objective.py",
        "geometry_module_sha256": source / "contact_geometry.py",
    }
    for key, path in files.items():
        if digest(path) != expected[key]:
            raise ValueError(f"A2 frozen {key} mismatch")
    target = output / "source/contact_objective"
    target.mkdir()
    copied_bundle = target / "bundle"
    copied_bundle.mkdir()
    for name in ("manifest.json", "contact_targets.npz", "rules.json", "raw_geometry.json", "time_windows.json"):
        shutil.copy2(bundle / name, copied_bundle / name)
    copied_reference = target / "reference.npz"
    copied_model = target / "model.mjb"
    shutil.copy2(reference, copied_reference)
    shutil.copy2(reference.with_suffix(".json"), copied_reference.with_suffix(".json"))
    shutil.copy2(model, copied_model)
    source_hashes = {}
    with zipfile.ZipFile(target / "study_contact_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob("*.py")):
            relative = path.relative_to(STUDY_ROOT / "src").as_posix()
            archive.write(path, relative)
            source_hashes[relative] = digest(path)
    spec = {"bundle_dir": str(copied_bundle), "reference_file": str(copied_reference), "model_file": str(copied_model),
            "expected_hashes": dict(expected), "hyperparameters": dict(hyper), "per_control_step_scale": .02,
            "scientific_label": config["kind"], "qualification_status": config["qualification_status"],
            "source_hashes": source_hashes}
    write_json(target / "frozen_inputs.json", spec)
    return spec


def build_contact_objective(spec, device):
    """Construct only from frozen files; preserve all training RNG streams."""
    import numpy as np
    import torch
    sys.path.insert(0, str(STUDY_ROOT / "src"))
    from chord_study.contact_objective import GeometricContactObjective

    source = STUDY_ROOT / "src/chord_study"
    for key, name in (("objective_module_sha256", "contact_objective.py"), ("geometry_module_sha256", "contact_geometry.py")):
        if digest(source / name) != spec["expected_hashes"][key]:
            raise ValueError(f"A2 objective code changed after freezing: {name}")
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        module = GeometricContactObjective.from_bundle(
            spec["bundle_dir"], reference_file=spec["reference_file"], model_file=spec["model_file"],
            expected_reference_sha256=spec["expected_hashes"]["reference_npz_sha256"],
            device=device, dtype=torch.float32, **spec["hyperparameters"])
    finally:
        random.setstate(python_rng); np.random.set_state(numpy_rng); torch.set_rng_state(torch_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    module.eval().requires_grad_(False)
    for key, expected in spec["expected_hashes"].items():
        if key.endswith("_module_sha256"):
            continue
        if module.metadata.get(key) != expected:
            raise ValueError(f"A2 constructed metadata disagrees with frozen {key}")
    if module.metadata.get("dt_applied_inside_module") is not False or module.metadata.get("target_defined_from_policy_contacts") is not False:
        raise ValueError("A2 target independence or dt contract changed")
    return module


def copy_frozen_evaluation(config, output):
    if not config.get("frozen_evaluation_reference"):
        return None
    path = resolve(config["frozen_evaluation_reference"])
    sidecar = path.with_suffix(".json")
    destination = output / "reference/frozen_evaluation_reference.npz"
    shutil.copy2(path, destination)
    shutil.copy2(sidecar, destination.with_suffix(".json"))
    meta = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    return {"source": str(path), "npz_sha256": digest(path), "json_sha256": digest(sidecar),
            "task_acceptance_sha256": meta.get("task_acceptance_sha256"),
            "use": "frozen external evaluation contract only; not used to alter training rewards"}


def prepare_reference(config, output):
    """Copy a slice, without resampling/pose correction; retain original lineage."""
    import numpy as np

    allowed = {"clips_root", "source_clip_id", "start_frame", "duration_s", "recipe_path", "provenance_path", "frozen_evaluation_reference"}
    if set(config) - allowed:
        raise ValueError(f"Unknown reference keys: {sorted(set(config) - allowed)}")
    root = resolve(config["clips_root"])
    if config.get("recipe_path"):
        from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
        original_recipe = resolve(config["recipe_path"])
        selection = resolve_recipe(original_recipe, clips_root=root, verify_clip_hashes=True)
        selected = [clip for group in selection.clip_sets for clip in group.clips]
        if len(selected) != 1:
            raise ValueError("Controlled training requires exactly one frozen clip")
        clip = selected[0]
        index_path = clip.npz_path.parent / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8-sig"))
        entry = next(item for item in index["clips"] if item["file"] == clip.npz_path.name)
        with np.load(clip.npz_path, allow_pickle=False) as bundle:
            count = len(bundle["reference_qpos"])
        dt = float(index["frame_dt"])
        if abs(dt - .02) > 1e-8 or abs((count - 1) * dt - float(config.get("duration_s", 3.))) > 1e-8:
            raise ValueError("Frozen recipe must match the declared 50Hz duration")
        clips = output / "reference/clips"
        group_root = clips / clip.group_id
        group_root.mkdir(parents=True)
        copied = group_root / clip.npz_path.name
        shutil.copy2(clip.npz_path, copied)
        copied_index = dict(index, clip_count=1, clips=[entry])
        write_json(group_root / "index.json", copied_index)
        recipe = output / "reference/recipe.json"
        write_json(recipe, {"schema": "wuji_pen_spin_recipe:1", "clips": [clip.clip_id]})
        provenance = {"source_kind": "frozen external single-clip catalog; parent lineage retained below",
                      "source_recipe": str(original_recipe), "source_recipe_sha256": digest(original_recipe),
                      "source_index_sha256": digest(index_path), "source_file": str(clip.npz_path),
                      "source_sha256": digest(clip.npz_path), "derived_file": str(copied), "derived_sha256": digest(copied),
                      "derived_clip_id": clip.clip_id, "recipe_sha256": digest(recipe),
                      "frame_dt_s": dt, "frames": count, "control_intervals": count - 1,
                      "duration_s": (count - 1) * dt, "resampling": False, "joint_or_object_trajectory_correction": False}
        if config.get("provenance_path"):
            parent = resolve(config["provenance_path"])
            shutil.copy2(parent, output / "reference/parent_provenance.json")
            provenance["parent_provenance"] = {"source": str(parent), "sha256": digest(parent)}
        provenance["frozen_evaluation_reference"] = copy_frozen_evaluation(config, output)
        write_json(output / "reference/provenance.json", provenance)
        return recipe, clips, provenance
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
    provenance["frozen_evaluation_reference"] = copy_frozen_evaluation(config, output)
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


def run_attempt(input_config, dex_root, recipe, clips, output, compiled, objective_spec=None):
    from mjlab.envs import ManagerBasedRlEnv
    from dex_supermimic.rl.dex_supermimic.pen import PenMDP
    from dex_supermimic.rl.dex_supermimic.runner import DexSuperMimicRunner
    from c2_training_adapter import training_controller
    metadata={"schema":"C2_020_training_controller:1","trained_method":input_config["method"],
        "base_method":input_config["method"],"training_seed":input_config.get("seed",0),
        "run_role":input_config.get("run_role"),"config_sha256":digest(output.parent/"input_config.json"),
        "paired_protocol_sha256":digest(resolve(input_config["paired_protocol_file"])) if input_config.get("paired_protocol_file") else None,
        "study_protocol_sha256":input_config.get("c2_protocol_sha256"),
        "native_source_inventory_sha256":digest(output.parent/"source/inventory.json"),
        "objective":input_config.get("contact_objective"),"record_actual_u":True,
        "adapter_sha256":digest(STUDY_ROOT/"src/c2_training_adapter.py"),
        "pure_projection_sha256":digest(STUDY_ROOT/"src/target_inset.py"),
        "entry_sha256":digest(__file__),"no_reward_or_WM_loss_change":True}
    with training_controller(ManagerBasedRlEnv,PenMDP,DexSuperMimicRunner,metadata):
        return _native_run_attempt(input_config,dex_root,recipe,clips,output,compiled,objective_spec)


def build_penalty_only_objective(spec,device):
    from penalty_study.penalty_only_objective import PenaltyOnlyPrior
    if digest(STUDY_ROOT/"src/penalty_study/penalty_only_objective.py")!="cfa53515fb7dc12ae49a2aec2d9ca37603061c1752fc8875f879cf5ea235c914":
        raise ValueError("Frozen P_ONLY objective source changed")
    base=build_contact_objective(spec,device)
    module=PenaltyOnlyPrior(base).eval().requires_grad_(False)
    module.metadata["penalty_wrapper_sha256"]=digest(STUDY_ROOT/"src/penalty_study/penalty_only_objective.py")
    return module


def _native_run_attempt(input_config, dex_root, recipe, clips, output, compiled, objective_spec=None):
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
    validate_paired_budget(cfg, input_config, output)
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
            self.contact_objective = None
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
            report_progress = self.phase_calls[name] == 0 or self.iteration % 10 == 0
            if report_progress:
                write_json(output / "progress.json", {"phase": name, "status": "running", "iteration": self.iteration,
                    "elapsed_seconds": tick - started, "actual_optimizer_steps": dict(self.actual_optimizer_steps)})
            try:
                yield
            finally:
                torch.cuda.synchronize()
                self.phase_seconds[name] += time.perf_counter() - tick
                self.phase_calls[name] += 1
                if report_progress:
                    write_json(output / "progress.json", {"phase": name, "status": "returned", "iteration": self.iteration,
                        "elapsed_seconds": time.perf_counter() - started, "phase_seconds": dict(self.phase_seconds),
                        "actual_optimizer_steps": dict(self.actual_optimizer_steps)})

        def collect(self, steps):
            with self.phase("collect_including_warmup"):
                return super().collect(steps)

        def fit_world_model(self, holdout=False):
            with self.phase("wm_holdout" if holdout else "wm_train"):
                return super().fit_world_model(holdout)

        def policy_update(self):
            with self.phase("policy"):
                if self.contact_objective is None:
                    return super().policy_update()
                with imagination_contact_reward(self, self.contact_objective, scale=.02) as terms:
                    stats = super().policy_update()
                stats.update({"prior/" + name: torch.stack(values).mean() for name, values in terms.items()})
                return stats

        def save(self, path, next_iteration=None):
            with self.phase("checkpoint"):
                super().save(path, next_iteration)
                path = Path(path)
                record = {"checkpoint": str(path.resolve()), "checkpoint_sha256": digest(path),
                          "iteration": self.iteration if next_iteration is None else next_iteration,
                          "process_elapsed_seconds_at_completion": time.perf_counter() - STARTED,
                          "completed_utc": datetime.now(timezone.utc).isoformat(),
                          "real_simulator_transitions": self.sim_steps,
                          "actual_optimizer_steps": dict(self.actual_optimizer_steps),
                          "selection_rule": "latest completed checkpoint before fixed elapsed cutoff; no quality-based selection"}
                write_json(path.with_suffix(".meta.json"), record)
                with (output / "checkpoint_index.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")

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
        "status": "running", "condition": input_config.get("method", "A1"),
        "method": "P_ONLY exact shared penetration prior, support0" if input_config["method"]=="P_ONLY" else ("A2 CHORD-inspired geometric intent shaping; unqualified geometric prior" if objective_spec else "A1 upstream SuperMimic; no CHORD objective"),
        "control_contract": "C2_020 common FP32 inset on real/imagined actualu; EMA residual unchanged",
        "compiled_requested": compiled, "seed": seed, "training_budget_iterations": cfg.max_iterations,
        "contact_head": "15-dimensional finger self-collision; not hand-object wrench/contact",
        "contact_weight": .1, "failure_head_used_in_policy": False, "auxiliary_failure_head_fit_retained": True,
        "initial_grasp_status": input_config.get("initial_grasp_status", "unqualified; throughput/compile probe only"),
        "training_episode_reset": "first frame, no hold/RSI; phase corrected to zero at dt=0 reset",
        "training_terminations": {"object_position_m": .05, "shaft_angle_deg": 60.},
        "versions": {p: metadata.version(p) for p in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "triton-windows")},
        "asset_revision": wuji_description.REVISION,
        "compiler_cache_paths": {key: os.environ.get(key) for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")},
        "warp_cache_directory": str(wp.config.kernel_cache_dir),
    }
    try:
        torch.cuda.reset_peak_memory_stats()
        env_started = time.perf_counter()
        env = ControlledPenEnv(env_cfg, device=cfg.device)
        result["environment_init_seconds"] = time.perf_counter() - env_started
        runner_started = time.perf_counter()
        runner = RecordedRunner(env, cfg, output / "train")
        result["runner_init_seconds"] = time.perf_counter() - runner_started
        if objective_spec is not None:
            setup_started = time.perf_counter()
            runner.contact_objective = (build_penalty_only_objective if input_config["method"]=="P_ONLY" else build_contact_objective)(objective_spec, cfg.device)
            if runner.contact_objective.frame_count != len(runner.mdp.motion._frames["joint_pos"]):
                raise ValueError("A2 fixed contact timeline does not match the single training reference")
            if runner.contact_objective.metadata["joint_names"] != list(runner.reader.robot.joint_names):
                raise ValueError("A2 contact/prior joint ordering differs from the training control contract")
            result["contact_objective_setup_seconds"] = time.perf_counter() - setup_started
            result["contact_objective_metadata"] = runner.contact_objective.metadata
            result["geometric_intent_per_control_step_scale"] = .02
            write_json(output / "contact_objective_metadata.json", runner.contact_objective.metadata)
        assert runner.contact_dim == 15 and bool((runner.observations.disturbance(runner.observations.auxiliary()) == 0).all())
        assert bool((env.command_manager.get_term("motion").time_steps == 0).all())
        before = {name: [p.detach().cpu().clone() for p in module.parameters()] for name, module in runner.modules().items()}
        fingerprints={}
        for name,values in before.items():
            hashed=hashlib.sha256()
            for value in values:
                hashed.update(str((str(value.dtype),tuple(value.shape))).encode())
                hashed.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            fingerprints[name]={"sha256":hashed.hexdigest(),"tensors":len(values),
                "parameters":sum(v.numel() for v in values),"l2":sum(float(v.double().square().sum()) for v in values)**.5}
        write_json(output/"initial_parameters.json",fingerprints)
        runner.c2_initial_parameter_fingerprints=fingerprints
        result["initial_parameter_fingerprints"]=fingerprints
        before_contact = runner.wm.net[-1].weight[-15:].detach().cpu().clone()
        body_fields = ("body_mass", "body_inertia", "body_iquat", "geom_size", "geom_pos", "geom_friction", "actuator_gainprm", "actuator_biasprm")
        physical = {name: getattr(env.sim.model, name).detach().cpu().numpy() for name in body_fields}
        physical["reader_context"] = runner.reader.xi.detach().cpu().numpy()
        np.savez_compressed(output / "fixed_physics_context.npz", **physical)
        mujoco.mj_saveModel(env.sim.mj_model, str(output / "nominal_model.mjb"), None)
        if objective_spec is not None and digest(output / "nominal_model.mjb") != objective_spec["expected_hashes"]["model_sha256"]:
            raise ValueError("A2 frozen collision model does not match the actual controlled scene")
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
        assert saved["format_version"]==103 and saved["c2_controller"]["margin_rad"]==.020
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
    parser.add_argument("--run", action="store_true", help="Explicitly execute the bounded GPU attempt; default is CPU configuration preflight")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    allowed = {"schema", "label", "method", "seed", "output_dir", "dex_root", "reference", "training", "compile_fallback", "max_attempt_seconds", "initial_grasp_status", "contact_objective", "paired_protocol_file", "run_role", "c2_controller", "c2_protocol_file", "c2_protocol_sha256"}
    if set(config) - allowed or config.get("schema") != "supermimic_chord_train:1":
        raise ValueError("Invalid configuration schema/keys")
    method = config.get("method", "A1")
    if method not in {"A1L", "A2", "P_ONLY"}:
        raise ValueError("C2 training arms are A1L/A2/P_ONLY")
    if (method in {"A2","P_ONLY"}) != bool(config.get("contact_objective")):
        raise ValueError("A2/P_ONLY require frozen objective inputs; A1L omits them")
    if config.get("c2_controller")!={"kind":"C2_TARGET_INSET","margin_rad":.020,"checkpoint_format_version":103,"resume":"from_scratch_only"}:
        raise ValueError("Explicit immutable C2_020 controller contract is required")
    if {"checkpoint","model_checkpoint"} & set(config.get("training",{})):
        raise ValueError("C2 training is from scratch only")
    if bool(config.get("c2_protocol_file"))!=bool(config.get("c2_protocol_sha256")):
        raise ValueError("Protocol path/hash must be declared together")
    if config.get("c2_protocol_file") and digest(resolve(config["c2_protocol_file"]))!=config["c2_protocol_sha256"]:
        raise ValueError("C2 study protocol hash mismatch")
    if not args.run:
        print(json.dumps({"status":"CPU_preflight_only","GPU_started":False,"method":method,"seed":config.get("seed"),"controller":config["c2_controller"]},indent=2))
        return
    dex_root = resolve(config.get("dex_root", "D:/workspace/dex-supermimic-research"))
    output = resolve(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a prior experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    runtime = {"status": "running", "pid": os.getpid(), "parent_pid": os.getppid(), "python_executable": sys.executable,
               "started_utc": datetime.now(timezone.utc).isoformat(), "input_config": str(config_path), "output": str(output)}
    write_json(output / "runtime.json", runtime)
    shutil.copy2(config_path, output / "input_config.json")
    source = freeze_source(dex_root, output)
    c2_source=output/"source/c2_controller"
    c2_source.mkdir()
    c2_files={"adapter":STUDY_ROOT/"src/c2_training_adapter.py","pure_projection":STUDY_ROOT/"src/target_inset.py",
        "penalty_wrapper":STUDY_ROOT/"src/penalty_study/penalty_only_objective.py",
        "entry":Path(__file__),"original_trainer":STUDY_ROOT/"source_baselines/original_train_controlled_pen.py"}
    frozen_files={}
    for name,path in c2_files.items():
        filename="original_train_controlled_pen.py" if name=="original_trainer" else path.name
        shutil.copy2(path,c2_source/filename)
        frozen_files[name]={"filename":filename,"sha256":digest(path)}
    c2_contract={"schema":"C2_020_training_controller:1","base_method":method,"training_seed":config.get("seed",0),
        "run_role":config.get("run_role"),"controller":config["c2_controller"],"input_config_sha256":digest(config_path),
        "files":frozen_files,"study_protocol_sha256":config.get("c2_protocol_sha256"),
        "actualu_evidence":"percollect allbatch exact cache/actual/replay.u+residual/replay.c; traces calls1/2/3/every100, last<=4steps/env0:4"}
    if config.get("c2_protocol_file"):
        shutil.copy2(resolve(config["c2_protocol_file"]),c2_source/"study_protocol.json")
    write_json(c2_source/"controller_contract.json",c2_contract)
    source["c2_controller"]={**c2_contract,"contract_sha256":digest(c2_source/"controller_contract.json")}
    write_json(output/"source/inventory.json",source)
    recipe, clips, provenance = prepare_reference(config["reference"], output)
    freeze_paired_protocol(config, output, provenance, source)
    objective_spec = freeze_contact_inputs(config["contact_objective"], output) if method in {"A2","P_ONLY"} else None
    if objective_spec is not None:
        frozen_evaluation = provenance.get("frozen_evaluation_reference")
        if not frozen_evaluation or frozen_evaluation["npz_sha256"] != objective_spec["expected_hashes"]["reference_npz_sha256"] or frozen_evaluation["json_sha256"] != objective_spec["expected_hashes"]["reference_json_sha256"]:
            raise ValueError("A2 target bundle must use the same frozen reference/protocol copied for this training run")
    inherited_cache = {key: os.environ.get(key) for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")}
    # Override only this child process's cache variables; do not inherit other
    # applications' cache roots or modify the user's persistent environment.
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(dex_root / ".cache/torchinductor/controlled_pen")
    os.environ["TRITON_CACHE_DIR"] = str(dex_root / ".cache/triton/controlled_pen")
    caches_before = {key: cache_inventory(os.environ[key]) for key in inherited_cache}
    write_json(output / "cache_context_before.json", {"inherited_environment": inherited_cache, "effective": caches_before})
    result = {"label": config.get("label"), "config_sha256": digest(config_path), "source": source,
              "reference": provenance, "attempts": [], "claim_scope": "bounded training/compile/throughput; quality requires independent no-reset physics evaluation"}
    compile_requested = bool(config.get("training", {}).get("compile", True))
    modes = [True, False] if compile_requested and config.get("compile_fallback", True) else [compile_requested]
    for compiled in modes:
        attempt = output / ("compiled" if compiled else "eager")
        try:
            completed = run_attempt(config, dex_root, recipe, clips, attempt, compiled, objective_spec=objective_spec)
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
                write_json(output / "runtime.json", {**runtime, "status": "failed", "finished_utc": datetime.now(timezone.utc).isoformat()})
                raise
            print(f"Compiled attempt failed; restarting same seeded budget eagerly: {error!r}", flush=True)
            import torch
            torch._dynamo.reset(); gc.collect(); torch.cuda.empty_cache()
    result["total_wall_seconds_including_failed_attempts"] = time.perf_counter() - STARTED
    result["cache_context"] = {"before": caches_before, "after": {key: cache_inventory(os.environ[key]) for key in inherited_cache}}
    write_json(output / "summary.json", result)
    write_json(output / "runtime.json", {**runtime, "status": result["status"], "finished_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": result["status"], "output": str(output), "attempt_modes": [a.get("compiled_requested", a.get("compiled")) for a in result["attempts"]],
                      "total_wall_seconds": result["total_wall_seconds_including_failed_attempts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
