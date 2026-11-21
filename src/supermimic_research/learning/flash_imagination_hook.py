"""Attach learned-model updates to the same native transitions used by the shared FlashSAC learner."""

from functools import partial
from pathlib import Path

from flax import serialization, struct
import jax
import jax.numpy as jnp
import optax

from flash_chord.training.flash_sac.network import Actor, Critic, project_unit_parameters
from flash_chord.training.flash_sac.reward_normalization import reward_scale_denominator
from supermimic_research.learning.flash_imagination import (
    Dynamics, NativeSteps, Scales, fit_scales, imagined_actor_loss, model_update, sequence_model_update,
)


@struct.dataclass
class AuxiliaryState:
    model_params: object
    model_optimizer: object
    actor_optimizer: object
    scale: Scales
    history: NativeSteps
    key: jax.Array
    model_updates: jax.Array
    actor_updates: jax.Array
    skipped_updates: jax.Array
    last_model_loss: jax.Array
    last_actor_loss: jax.Array


@jax.jit
def insert(history, new, slot):
    return jax.tree.map(lambda buf, row: buf.at[slot].set(row), history, new)


def all_finite(tree):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)]))


class ImaginationHook:
    """The auxiliary learner runs before the official JAX learner consumes/donates its buffers."""

    def __init__(self, training, *, history=32, batch=128, horizon=3, every=4,
                 model_warmup=16, geometry_tail=3, seed=318, fit_horizon=1):
        if min(history, batch, horizon, every, model_warmup) < 1 or geometry_tail < 0 or not 1 <= fit_horizon <= history:
            raise ValueError("Invalid world-model training options")
        self.config = training
        self.options = dict(history=history, batch=batch, horizon=horizon, every=every,
                            model_warmup=model_warmup, geometry_tail=geometry_tail, seed=seed, fit_horizon=fit_horizon)
        self.collected = 0
        self.state = None
        self._model_opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(3e-4))
        self._actor_opt = optax.chain(optax.clip_by_global_norm(.5), optax.adam(1e-5))

    def initialize(self, batch, actor_params):
        o = self.options
        key, model_key = jax.random.split(jax.random.key(o["seed"]))
        width = batch.state().shape[1]
        self._world = Dynamics(width + batch.terms.shape[1])
        model_params = self._world.init(model_key, batch.state()[:1], batch.action[:1])["params"]
        dtype = getattr(jnp, self.config.compute_dtype) if self.config.mixed_precision else jnp.float32
        self._actor = Actor(action_dim=batch.action.shape[1], config=self.config.network, dtype=dtype)
        lo, hi = self.config.reward_normalization.value_support
        self._critic = Critic(value_min=lo, value_max=hi, config=self.config.network, dtype=dtype)
        ring = jax.tree.map(lambda x: jnp.zeros((o["history"], *x.shape), dtype=x.dtype), batch)
        self.state = AuxiliaryState(
            model_params, self._model_opt.init(model_params), self._actor_opt.init(actor_params),
            fit_scales(batch), ring, jax.random.key_data(key), jnp.int32(0), jnp.int32(0), jnp.int32(0),
            jnp.float32(0), jnp.float32(0),
        )
        self._compiled = jax.jit(self._advance, static_argnames=("imagine",))

    def _advance(self, state, actor, critic, weights, dt, denominator, ring_head, *, imagine):
        key, sample_key = jax.random.split(state.key)
        flat = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), state.history)
        idx = jax.random.randint(sample_key, (self.options["batch"],), 0, flat.action.shape[0])
        batch = jax.tree.map(lambda x: x[idx], flat)
        fit_horizon = self.options["fit_horizon"]
        if fit_horizon == 1:
            params, model_opt, (loss, *_) = model_update(self._world, self._model_opt, state.model_params,
                                                        state.model_optimizer, state.scale, batch)
        else:
            history, worlds = state.history.action.shape[:2]
            key, sequence_key = jax.random.split(key)
            starts = jax.random.randint(sequence_key, (self.options["batch"],), 0, (history-fit_horizon+1)*worlds)
            slots = (ring_head + starts[None,:]//worlds + jnp.arange(fit_horizon)[:,None]) % history
            sequence = jax.tree.map(lambda x: x[slots,(starts%worlds)[None,:]],state.history)
            params, model_opt, (loss, *_) = sequence_model_update(self._world, self._model_opt, state.model_params,
                                                                  state.model_optimizer, state.scale, sequence)
        valid = all_finite((loss, params, model_opt))
        params, model_opt = jax.lax.cond(valid, lambda: (params, model_opt),
                                          lambda: (state.model_params, state.model_optimizer))
        state = state.replace(model_params=params, model_optimizer=model_opt, key=key,
                              model_updates=state.model_updates+valid.astype(jnp.int32),
                              skipped_updates=state.skipped_updates+(~valid).astype(jnp.int32), last_model_loss=loss)
        if imagine:
            func = partial(imagined_actor_loss, actor=self._actor, actor_stats=actor.batch_stats,
                           critic=self._critic, critic_variables={"params": critic.params, "batch_stats": critic.batch_stats},
                           model=self._world, model_params=params, scales=state.scale, starts=batch,
                           objective_weights=weights, dt=dt, discount=self.config.discount,
                           reward_denominator=denominator, action_transform=self.config.network.action_transform,
                           action_scale=self.config.network.action_scale, steps=self.options["horizon"],
                           geometry_tail=self.options["geometry_tail"])
            val, grad = jax.value_and_grad(func)(actor.params)
            updates, actor_opt = self._actor_opt.update(grad, state.actor_optimizer, actor.params)
            next_params = project_unit_parameters(optax.apply_updates(actor.params, updates),
                                                  self.config.network.projection_epsilon)
            valid_actor = valid & all_finite((val, grad, next_params, actor_opt))
            next_params, actor_opt = jax.lax.cond(valid_actor, lambda: (next_params, actor_opt),
                                                  lambda: (actor.params, state.actor_optimizer))
            state = state.replace(actor_optimizer=actor_opt, actor_updates=state.actor_updates+valid_actor.astype(jnp.int32),
                                  skipped_updates=state.skipped_updates+(~valid_actor).astype(jnp.int32),
                                  last_actor_loss=val)
            actor = actor.replace(params=next_params)
        return state, actor

    def __call__(self, consume, learner, tr, weights, dt, *runtime_args):
        batch = NativeSteps(learner.observation, learner.critic_context, tr.action, tr.next_observation,
                            tr.next_critic_context, tr.objective_terms, tr.terminated, tr.truncated)
        if self.state is None:
            self.initialize(batch, learner.actor.params)
        o = self.options
        hist = insert(self.state.history, batch, jnp.int32(self.collected % o["history"]))
        self.collected += 1
        self.state = self.state.replace(history=hist)
        if self.collected == o["history"]:
            flat = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), hist)
            self.state = self.state.replace(scale=fit_scales(flat))
        if self.collected >= o["history"] and self.collected % o["every"] == 0:
            normalizer = self.config.reward_normalization
            denom = (reward_scale_denominator(learner.reward_normalizer, bound=normalizer.normalized_return_bound,
                                              epsilon=normalizer.epsilon) if normalizer.enabled else jnp.float32(1.))
            imagined = self.collected >= o["history"] + o["model_warmup"] * o["every"]
            self.state, actor = self._compiled(self.state, learner.actor, learner.target_critic,
                                               weights, dt, denom, jnp.int32(self.collected % o["history"]), imagine=imagined)
            learner = learner.replace(actor=actor)
        return consume(learner, tr, weights, dt, *runtime_args)

    def save(self, path, *, native_environment_steps):
        path = Path(path)
        if self.state is None:
            raise ValueError("No native transitions collected")
        payload = serialization.msgpack_serialize(dict(
            native_environment_steps=int(native_environment_steps), options=self.options,
            collected=self.collected, state=serialization.to_state_dict(jax.device_get(self.state)),
        ))
        temp = path.with_suffix(path.suffix+".tmp")
        temp.write_bytes(payload)
        temp.replace(path)

    def restore(self, path, batch_template, actor_params, *, native_environment_steps):
        raw = serialization.msgpack_restore(Path(path).read_bytes())
        stored_options = {"fit_horizon": 1, **raw["options"]}
        if stored_options != self.options or raw["native_environment_steps"] != int(native_environment_steps):
            raise ValueError("Auxiliary configuration or native checkpoint step mismatch")
        self.initialize(batch_template, actor_params)
        self.state = jax.device_put(serialization.from_state_dict(self.state, raw["state"]))
        self.collected = raw["collected"]
