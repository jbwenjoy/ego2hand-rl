"""Integration checks for terminal snapshots and auxiliary restart at a native checkpoint."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("flash_chord"), "requires the tagged FlashCHORD environment")
class FlashImaginationTest(unittest.TestCase):
    def test_terminal_snapshot_and_restart(self):
        import jax
        import jax.numpy as jnp
        from flash_chord.training.flash_sac.config import NetworkConfig, TrainingConfig
        from flash_chord.training.flash_sac.learner import OptimizedNetworkState, TargetCriticState
        from flash_chord.training.flash_sac.network import Actor, Critic
        from flash_chord.training.flash_sac.reward_normalization import create_reward_normalizer
        from flax import struct
        from supermimic_research.learning.flash_imagination import NativeSteps
        from supermimic_research.learning.flash_imagination_hook import ImaginationHook

        @struct.dataclass
        class Learner:
            observation: object
            critic_context: object
            actor: object
            target_critic: object
            reward_normalizer: object

        net = replace(NetworkConfig(), actor_hidden_dim=16, critic_hidden_dim=16, atom_count=11)
        cfg = TrainingConfig(world_count=4, mixed_precision=False, network=net)
        obs = jnp.arange(32, dtype=jnp.float32).reshape(4,8)*.02
        context, action = obs[:, :2], obs[:, 2:5]
        av = Actor(action_dim=3, config=net).init(jax.random.key(1), obs, training=False)
        cv = Critic(-5,5,net).init(jax.random.key(2), jnp.concatenate((obs,context),axis=1), action, training=False)
        parent = Learner(obs, context, OptimizedNetworkState(av["params"], av["batch_stats"], (), 0),
                         TargetCriticState(cv["params"], cv["batch_stats"]), create_reward_normalizer(4))
        transition = SimpleNamespace(action=action, next_observation=obs+.01, next_critic_context=context+.015,
                                     post_reset_observation=-100*obs, terminated=jnp.array([0,0,0,1],bool),
                                     truncated=jnp.zeros((4,),bool), objective_terms=jnp.column_stack((obs[:,0],action[:,0])))
        kwargs = dict(history=2,batch=4,horizon=2,fit_horizon=2,every=1,model_warmup=1,geometry_tail=0)
        hook = ImaginationHook(cfg, **kwargs)
        weights, dt = jnp.ones(2), jnp.float32(1/30)
        def apply(hook, learner):
            return hook(lambda state,*unused: state, learner, transition, weights, dt)
        state = parent
        for _ in range(4):
            state = apply(hook,state)
        self.assertEqual(int(hook.state.model_updates),3)
        self.assertEqual(int(hook.state.actor_updates),2)
        self.assertEqual(int(hook.state.skipped_updates),0)
        np.testing.assert_array_equal(hook.state.history.next_observation[0], transition.next_observation)
        changes = jax.tree.leaves(jax.tree.map(lambda x,y:jnp.max(jnp.abs(x-y)), state.actor.params, parent.actor.params))
        self.assertGreater(max(map(float,changes)),0)
        template = NativeSteps(obs,context,action,transition.next_observation,transition.next_critic_context,
                               transition.objective_terms,transition.terminated,transition.truncated)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"auxiliary.msgpack"
            hook.save(path,native_environment_steps=16)
            resumed=ImaginationHook(cfg,**kwargs)
            with self.assertRaises(ValueError):
                resumed.restore(path,template,state.actor.params,native_environment_steps=12)
            resumed.restore(path,template,state.actor.params,native_environment_steps=16)
        first,second=apply(hook,state),apply(resumed,state)
        for left,right in zip(jax.tree.leaves(first),jax.tree.leaves(second),strict=True):
            np.testing.assert_array_equal(left,right)


if __name__ == "__main__":
    unittest.main()
