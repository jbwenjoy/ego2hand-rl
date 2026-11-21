"""Batch independent FlashCHORD scenes behind one native FlashSAC learner."""
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from flash_chord.assets.registry import resolve_asset_path
from flash_chord.data.loader import load_reference
from flash_chord.training.checkpoint_metadata import critic_schema, policy_schema
from flash_chord.training.environment import WarpRLEnv


EXTENT_BLOCK = "source_object_extent_xyz_m"


def scan_extents_m(parquet):
    reference = load_reference(parquet)
    mesh, = reference.object_mesh_paths()
    path = Path(resolve_asset_path(mesh, reference_path=parquet))
    with path.open(encoding="utf8") as source:
        vertices = np.asarray([[float(v) for v in line.split()[1:4]] for line in source if line.startswith("v ")])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(f"Cannot extract metric vertices from exported scan {path}")
    extent = vertices.max(0)-vertices.min(0)
    if (extent <= 0).any():
        raise ValueError("Positive three-dimensional object bounds required")
    return extent.astype(np.float32)


@dataclass(frozen=True)
class GeometryObservationLayout:
    observation_dim: int
    block_names: tuple
    block_ranges: tuple

    @classmethod
    def from_native(cls, native):
        width = native.observation_dim
        return cls(width+3, (*native.block_names,EXTENT_BLOCK), (*native.block_ranges,(width,width+3)))


class MultiReferenceEnv:
    """World IDs and reference assignment are fixed for the entire n-step replay stream."""

    def __init__(self, shards, extents):
        self.shards = tuple(shards)
        if not self.shards or len(shards) != len(extents):
            raise ValueError("Require one source geometry descriptor for each native scene")
        first = self.shards[0]
        self.per_shard = first.world_count
        if self.per_shard <= 0:
            raise ValueError("Empty scene shard")
        self.world_count = self.per_shard*len(shards)
        self.env = first.env  # Used by NVIDIA's optional scalar default curriculum.
        self.layout = GeometryObservationLayout.from_native(self.env.observation_strategy)
        self.observation_dim = self.layout.observation_dim
        self.action_dim = first.action_dim
        self.critic_context_dim = first.critic_context_dim
        self.frame_dt = first.frame_dt
        self.policy_schema = policy_schema(first.env.action, self.layout)
        self.critic_context_names = first.critic_context_names
        self.critic_schema = critic_schema(first.env.action, self.layout, first.critic_context_names)
        self.objective_term_names = first.objective_term_names
        self.termination_cause_names = first.termination_cause_names
        self.tracking_error_names = first.tracking_error_names
        self.reset_to_first_frame_probability = first.reset_to_first_frame_probability
        self.immediate_first_frame_probability = first.immediate_first_frame_probability
        for shard in shards:
            layout = GeometryObservationLayout.from_native(shard.env.observation_strategy)
            if policy_schema(shard.env.action,layout) != self.policy_schema:
                raise ValueError("Distinct action or observation layout across training scenes")
            if critic_schema(shard.env.action,layout,shard.critic_context_names) != self.critic_schema:
                raise ValueError("Distinct asymmetric-critic context layout")
            if (shard.world_count != self.per_shard or shard.frame_dt != self.frame_dt
                    or shard.tracking_error_names != self.tracking_error_names
                    or shard.objective_term_names != self.objective_term_names
                    or shard.termination_cause_names != self.termination_cause_names):
                raise ValueError("Unbalanced or incompatible native transition layouts")
        extent = np.asarray(extents, dtype=np.float32)
        if extent.shape != (len(shards), 3) or not np.isfinite(extent).all() or np.any(extent<=0):
            raise ValueError("Invalid source object extents")
        self.extent_by_world = jnp.asarray(np.repeat(extent,self.per_shard,axis=0))

    def _augment(self, obs):
        return jnp.concatenate((obs, self.extent_by_world),axis=-1)

    @property
    def initial_critic_context(self):
        return jnp.concatenate([s.initial_critic_context for s in self.shards])

    def reset(self):
        return self._augment(jnp.concatenate([s.reset() for s in self.shards]))

    def set_curriculum(self, stage):
        for shard in self.shards:
            shard.set_curriculum(stage)
        self.reset_to_first_frame_probability = self.shards[0].reset_to_first_frame_probability
        self.immediate_first_frame_probability = self.shards[0].immediate_first_frame_probability

    def transition_logging_inputs(self):
        entries = [shard.transition_logging_inputs() for shard in self.shards]
        return tuple(jnp.concatenate(parts) for parts in zip(*entries,strict=True))

    def step(self, action):
        if action.shape != (self.world_count,self.action_dim):
            raise ValueError(f"Wrong batched action shape {action.shape}")
        actions = jnp.split(action,len(self.shards),axis=0)
        steps = [shard.step(a) for shard,a in zip(self.shards,actions,strict=True)]
        # Flax VectorStep is a pytree. All optional metrics must be present or absent
        # on the same shards. No source frame or collection count is a policy feature.
        joined = jax.tree.map(lambda *arrays: jnp.concatenate(arrays,axis=0), *steps)
        if joined.terminal_observation is None:
            raise ValueError("Native FlashSAC requires terminal observations before auto-reset")
        return joined.replace(observation=self._augment(joined.observation),
                              terminal_observation=self._augment(joined.terminal_observation))


def run_shared_training(env, training, *, transition_hook=None, initializer=None, **runner_kwargs):
    """Run the unmodified FlashSAC collector; optionally extend the post-compile transition call.

    The optional callable receives (native_consume, learner, transition, *runtime_args)
    and must return (updated_learner, next_action, metrics). The incoming learner holds
    the pre-action observation. Transition.next_observation is the terminal/pre-reset
    successor. A hook owns the checkpoint and timing of any auxiliary model state.
    A separate OS process per run isolates the temporary entry-point binding.
    """
    from dataclasses import replace
    from contextlib import ExitStack
    from unittest.mock import patch
    from flash_chord.training.flash_sac import runner

    if transition_hook is None and initializer is None:
        return runner.run_training(env, training, **runner_kwargs)
    compile_native = runner._compile_executables
    create_native = runner.create_learner

    def initialize(*args, **kwargs):
        learner,kernels=create_native(*args,**kwargs)
        return initializer(learner),kernels

    def with_hook(*args, **kwargs):
        kernels = compile_native(*args, **kwargs)
        def consume(learner, transition, *runtime_args):
            return transition_hook(kernels.consume_transition, learner, transition, *runtime_args)
        return replace(kernels, consume_transition=consume)

    with ExitStack() as stack:
        if transition_hook is not None:
            stack.enter_context(patch.object(runner,"_compile_executables",with_hook))
        if initializer is not None:
            stack.enter_context(patch.object(runner,"create_learner",initialize))
        return runner.run_training(env, training, **runner_kwargs)


def build_native_shard(cfg, parquet, *, worlds, capture, seed_offset=0):
    from dataclasses import replace
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    config = instantiate_typed(cfg.env,RLEnvConfig)
    # Independent pseudorandom streams keep seeded initial frames from tracking
    # in lockstep across references; record each shard seed in the run metadata.
    if seed_offset:
        config = replace(config,reset=replace(config.reset,seed=config.reset.seed+seed_offset))
    setup = setup_scene(parquet=parquet, control_fps=config.sim.fps, motion_speed=cfg.task.motion_speed,
                        embodiment=instantiate_typed(cfg.embodiment,Embodiment),
                        collision=instantiate_typed(cfg.collision,CollisionPolicy), world_count=worlds,
                        include_support=cfg.scene.support, decompose_objects=cfg.scene.decompose_objects,
                        object_scale_min=cfg.scene.object_scale_min,object_scale_max=cfg.scene.object_scale_max,
                        object_scale_seed=cfg.scene.object_scale_seed,contact_friction=cfg.scene.contact_friction,
                        object_free_joint_damping=cfg.scene.object_free_joint_damping,
                        source_frame_playback=cfg.task.source_frame_playback,
                        motion_start_frame=cfg.task.motion_start_frame,motion_end_frame=cfg.task.motion_end_frame)
    env = RLEnv(setup.scene,setup.reference,config=config)
    env.reset()
    if capture:
        env.capture_step()
        env.reset()
    return WarpRLEnv(env,publish_diagnostics=False,publish_step_metrics=False), setup
