"""Export true one-step, pre-reset dynamics labels from the tagged Wuji FlashCHORD environment."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import warp as wp

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint_metadata import policy_schema
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.flash_multiref import EXTENT_BLOCK, scan_extents_m


def collect(config_path, output, controls, seed, physics_path):
    start = time.monotonic()
    cfg = OmegaConf.load(config_path)
    physics = json.loads(physics_path.read_bytes())
    if physics.get("validated") is not True:
        raise ValueError("Native world-model labels require validated solver physics")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(physics["overrides"]))
    # The saved config is resolved; shared interpolation links are materialized in it.
    cfg.env.sim = OmegaConf.to_container(cfg.sim, resolve=True)
    # The standard CHORD reset applies full assistance for a settling window even
    # with a zero curriculum target. Free-object dynamics labels exclude this.
    cfg.env.reset.voc_decay_steps = 0
    cfg.env.voc_scale = 0.0
    cfg.env.voc.max_force = 0.0
    cfg.env.voc.max_torque = 0.0
    training = instantiate_typed(cfg.training, TrainingConfig)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    with wp.ScopedDevice(training.device):
        setup = setup_scene(
            parquet=cfg.task.parquet, control_fps=env_config.sim.fps, motion_speed=cfg.task.motion_speed,
            embodiment=instantiate_typed(cfg.embodiment, Embodiment), collision=instantiate_typed(cfg.collision, CollisionPolicy),
            world_count=cfg.scene.world_count, include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min, object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed, contact_friction=cfg.scene.contact_friction,
            object_free_joint_damping=cfg.scene.object_free_joint_damping,
            source_frame_playback=cfg.task.source_frame_playback,
            motion_start_frame=cfg.task.motion_start_frame, motion_end_frame=cfg.task.motion_end_frame)
        native = RLEnv(setup.scene, setup.reference, config=env_config)
        receipt = native.solver.wuji_conditioning
        if receipt["scheme"]!=physics["scheme"]:
            raise ValueError("Active native solver calibration disagrees with the physics manifest")
        np.testing.assert_allclose(native.solver.mjw_model.stat.meaninertia.numpy(), receipt["conditioned"], rtol=1e-6)
        native.reset()
        native.capture_step()
        env = WarpRLEnv(native, publish_diagnostics=False, publish_step_metrics=True)
        stage = replace(training.curriculum.stages[0], voc_scale=0.0)
        env.set_curriculum(stage)
        weights = np.asarray(stage.weights_for(env.objective_term_names), dtype=np.float32)
        extents = np.broadcast_to(scan_extents_m(setup.parquet_path), (env.world_count, 3))
        def observation(v):
            return np.concatenate((np.asarray(v), extents), axis=-1)
        obs = observation(env.reset())
        context = np.array(env.initial_critic_context)
        phase = env.critic_context_names.index("normalized_reference_phase")
        rng = np.random.default_rng(seed)
        worlds = np.arange(env.world_count)
        rows = []
        collection_start = time.monotonic()
        for tick in range(controls):
            # A zero control group and two low-amplitude opposite perturbations.
            perturb = rng.uniform(-0.12, 0.12, env.action_dim).astype(np.float32)
            action = np.outer(np.array([0, 1, -1], dtype=np.float32)[worlds % 3], perturb)
            source_frame = np.array(native.timestep.numpy(), copy=True)
            applied_voc = np.array(native.reset_policy.applied_voc_scale.numpy(), copy=True)
            if np.any(applied_voc != 0):
                raise ValueError("Physical labels require zero actual virtual object control")
            result = env.step(jnp.asarray(action))
            row = dict(obs=obs, critic_context=context, action=action, source_frame=source_frame, applied_voc=applied_voc,
                       next_obs=observation(result.terminal_observation),
                       next_critic_context=np.array(result.terminal_critic_context),
                       post_reset_obs=observation(result.observation),
                       post_reset_critic_context=np.array(result.critic_context),
                       reward=np.array(result.reward), objective_terms=np.array(result.objective_terms),
                       terminated=np.array(result.terminated), truncated=np.array(result.truncated))
            row["post_reset_source_frame"] = native.timestep.numpy().copy()
            row["terminal_source_frame_from_phase"] = np.rint(
                row["next_critic_context"][:, phase] * (setup.reference.num_frames - 1)).astype(np.int32)
            np.testing.assert_allclose(row["objective_terms"] @ weights * env.frame_dt,
                                       row["reward"], atol=2e-5, rtol=1e-5)
            if not all(np.isfinite(a).all() for a in row.values()):
                raise ValueError(f"Nonfinite official transition at control {tick}")
            rows.append(row)
            obs, context = row["post_reset_obs"], row["post_reset_critic_context"]
        fields = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
        fields.update(world_id=worlds, control_index=np.arange(controls), weights=weights,
                      dt=np.float32(env.frame_dt), scan_extents_m=np.array(extents[0]))
        path = output / "transitions.npz"
        np.savez_compressed(path, **fields)
        report = dict(
            config=file_pin(config_path), collector=file_pin(__file__), parquet=file_pin(setup.parquet_path),
            physics=file_pin(physics_path),active_solver=receipt,
            sequence=setup.reference.metadata.sequence_id, worlds=env.world_count, controls=controls,
            samples=controls * env.world_count, seed=seed, extent_block=EXTENT_BLOCK,
            native_observation_dim=env.observation_dim, policy_schema=policy_schema(native.action, native.observation_strategy),
            objective_names=list(env.objective_term_names), critic_context_names=list(env.critic_context_names),
            axes="Control index, world index, feature. Source box, right hand, native environment and action schema identical across groups.",
            next_obs="Pre-reset terminal observation for the applied action. post_reset_obs feeds the subsequent action.",
            reset_transitions=int(np.count_nonzero(fields["terminated"] | fields["truncated"])),
            free_object_overrides={"env.reset.voc_decay_steps": 0, "env.voc.max_force": 0,
                                   "env.voc.max_torque": 0, "env.voc_scale": 0},
            controls_s=time.monotonic() - collection_start, wall_s=time.monotonic() - start,
            data=file_pin(path))
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--controls", type=int, default=32)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--warp-cache", type=str, required=True)
    p.add_argument("--physics", type=Path, required=True)
    args = p.parse_args()
    if args.controls <= 0:
        p.error("--controls must be positive")
    wp.config.kernel_cache_dir = args.warp_cache
    args.output.mkdir(parents=True, exist_ok=False)
    collect(args.config.resolve(), args.output.resolve(), args.controls, args.seed, args.physics.resolve())
