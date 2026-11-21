"""One-step world-model fitting and short differentiable rollouts for the Wuji FlashSAC policy."""

from typing import NamedTuple

from flax import linen as nn
from flax import struct
import jax
import jax.numpy as jnp
import optax

from flash_chord.training.flash_sac.network import deterministic_action


@struct.dataclass
class NativeSteps:
    observation: jax.Array
    context: jax.Array
    action: jax.Array
    next_observation: jax.Array  # Pre-reset terminal snapshot, also on non-terminal steps.
    next_context: jax.Array
    terms: jax.Array
    terminated: jax.Array
    truncated: jax.Array

    def state(self):
        return jnp.concatenate((self.observation, self.context), axis=-1)

    def targets(self):
        next_state = jnp.concatenate((self.next_observation, self.next_context), axis=-1)
        return jnp.concatenate((next_state - self.state(), self.terms), axis=-1)


class Scales(NamedTuple):
    state_center: jax.Array
    state_scale: jax.Array
    output_center: jax.Array
    output_scale: jax.Array


def fit_scales(batch):
    """Calibrate from training transitions only; validation and holdout do not enter."""
    state, outputs = batch.state(), batch.targets()
    return Scales(state.mean(0), jnp.maximum(state.std(0), .01), outputs.mean(0),
                  jnp.maximum(outputs.std(0), .001))


class Dynamics(nn.Module):
    output_dim: int
    width: int = 192

    @nn.compact
    def __call__(self, state_features, action):
        h = jnp.concatenate((state_features, action), axis=-1)
        h = nn.silu(nn.Dense(self.width)(h))
        h = h + nn.Dense(self.width)(nn.silu(nn.Dense(self.width)(nn.LayerNorm()(h))))
        # Separate stop heads distinguish task failure from reaching a window boundary.
        return nn.Dense(self.output_dim)(h), nn.Dense(2)(h)


def predictions(module, params, scales, state, action):
    norm = (state - scales.state_center) / scales.state_scale
    norm_output, stop_logits = module.apply({"params": params}, norm, action)
    output = norm_output * scales.output_scale + scales.output_center
    delta, terms = jnp.split(output, [state.shape[-1]], axis=-1)
    return state + delta, terms, stop_logits


def training_loss(module, params, scales, batch):
    state = batch.state()
    predicted, stop = module.apply({"params": params}, (state - scales.state_center) / scales.state_scale,
                                   batch.action)
    truth = (batch.targets() - scales.output_center) / scales.output_scale
    state_err = optax.huber_loss(predicted[:, :state.shape[-1]], truth[:, :state.shape[-1]]).mean()
    term_err = optax.huber_loss(predicted[:, state.shape[-1]:], truth[:, state.shape[-1]:]).mean()
    flags = jnp.stack((batch.terminated, batch.truncated), axis=-1).astype(state.dtype)
    stop_err = optax.sigmoid_binary_cross_entropy(stop, flags).mean()
    return state_err + term_err + stop_err, (state_err, term_err, stop_err)


def model_update(module, optimizer, params, opt_state, scales, batch):
    (loss, components), grads = jax.value_and_grad(training_loss, argnums=1, has_aux=True)(
        module, params, scales, batch)
    updates, next_opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), next_opt_state, (loss, *components)


def sequence_model_update(module, optimizer, params, opt_state, scales, sequence):
    """Fit an unroll to actual contiguous actions from each world, ending its loss at reset."""
    def loss_fn(current):
        batch0 = jax.tree.map(lambda x: x[0], sequence)
        state = batch0.state()
        valid = jnp.ones(state.shape[0], dtype=jnp.float32)
        counts = jnp.float32(0)
        sums = jnp.zeros((3,), dtype=jnp.float32)
        for k in range(sequence.action.shape[0]):
            batch = jax.tree.map(lambda x: x[k], sequence)
            nxt, terms, stop = predictions(module,current,scales,state,batch.action)
            truth = jnp.concatenate((batch.next_observation,batch.next_context),axis=-1)
            feature_loss = optax.huber_loss((nxt-truth)/scales.output_scale[:nxt.shape[1]]).mean(-1)
            term_loss = optax.huber_loss((terms-batch.terms)/scales.output_scale[nxt.shape[1]:]).mean(-1)
            flags = jnp.stack((batch.terminated,batch.truncated),axis=-1)
            end_loss = optax.sigmoid_binary_cross_entropy(stop,flags).mean(-1)
            sums += jnp.stack((feature_loss,term_loss,end_loss)) @ valid
            counts += valid.sum()
            valid *= jnp.logical_not(jnp.logical_or(batch.terminated,batch.truncated)).astype(jnp.float32)
            state = nxt
        components = sums / counts
        return components.sum(), components
    (loss, components), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, next_opt_state = optimizer.update(grads,opt_state,params)
    return optax.apply_updates(params,updates), next_opt_state, (loss,*components)


def imagined_actor_loss(actor_params, *, actor, actor_stats, critic, critic_variables, model,
                        model_params, scales, starts, objective_weights, dt, discount,
                        reward_denominator, action_transform, action_scale, steps, geometry_tail=0):
    """Differentiate through modeled physics to actor parameters; critic supplies the tail value."""
    if steps < 1 or not 0 <= geometry_tail <= starts.observation.shape[-1]:
        raise ValueError("Invalid imagined horizon or static object-feature width")
    obs_dim = starts.observation.shape[-1]
    initial_state = jax.lax.stop_gradient(starts.state())
    frozen_model = jax.tree.map(jax.lax.stop_gradient, model_params)
    frozen_critic = jax.tree.map(jax.lax.stop_gradient, critic_variables)

    def action(state):
        dist = actor.apply({"params": actor_params, "batch_stats": actor_stats}, state[:, :obs_dim], training=False)
        return deterministic_action(dist, action_transform=action_transform, action_scale=action_scale)

    def body(carry, _):
        state, survival, total = carry
        nxt, terms, logits = predictions(model, frozen_model, scales, state, action(state))
        if geometry_tail:
            sl = slice(obs_dim - geometry_tail, obs_dim)
            nxt = nxt.at[:, sl].set(initial_state[:, sl])
        reward = (terms @ objective_weights) * dt / reward_denominator
        # Stop at failure or sequence end. Keep the current transition's reward.
        continuation = jnp.prod(jax.nn.sigmoid(-logits), axis=-1)
        return (nxt, survival * discount * continuation, total + survival * reward), None

    shape = initial_state.shape[:1]
    (last, weight, total), _ = jax.lax.scan(body, (initial_state, jnp.ones(shape), jnp.zeros(shape)),
                                            None, length=steps)
    q = critic.apply(frozen_critic, last, action(last), training=False).q
    return -jnp.mean(total + weight * jnp.min(q, axis=0))
