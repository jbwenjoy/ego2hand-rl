"""Exercise balanced replay rows, terminal/post-reset observations and schema checks on CPU."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["JAX_PLATFORMS"]="cpu"
os.environ["CUDA_VISIBLE_DEVICES"]=""

import jax.numpy as jnp
import numpy as np

from flash_chord.training.environment import VectorStep
from supermimic_research.runtime.flash_multiref import MultiReferenceEnv, scan_extents_m


class SentinelShard:
    world_count = 2
    observation_dim = 1
    action_dim = 1
    frame_dt = 1/30.
    critic_context_dim = 1
    critic_context_names = ("reset_state",)
    objective_term_names = ("tracking",)
    termination_cause_names = ("miss",)
    tracking_error_names = ("distance",)
    reset_to_first_frame_probability = 0.
    immediate_first_frame_probability = 0.

    def __init__(self, i):
        self.index = i
        self.rows = jnp.arange(2) + 2*i
        actor = SimpleNamespace(block_names=("residual",),block_ranges=((0,1),),action_dim=1,
                                input_mode="normalized",input_mapping="rational")
        observation = SimpleNamespace(block_names=("command",),block_ranges=((0,1),),observation_dim=1)
        self.env=SimpleNamespace(action=actor,observation_strategy=observation)
        self.initial_critic_context=self.rows[:,None]

    def reset(self):
        return self.rows[:,None]

    def step(self, action):
        self.received = np.asarray(action).copy()
        return VectorStep(observation=action+100,terminal_observation=action+200,reward=None,
                          terminated=self.rows%2==0,truncated=self.rows%2==1,
                          episode_return=None,episode_length=None,episode_reference_progress=None,
                          objective_terms=self.rows[:,None],critic_context=action+300,
                          terminal_critic_context=action+400)

    def transition_logging_inputs(self):
        return self.rows,self.rows[:,None]+500


def main(paths, out):
    ext=[scan_extents_m(p) for p in paths]
    shards=[SentinelShard(i) for i in range(len(paths))]
    joined=MultiReferenceEnv(shards,ext)
    obs=joined.reset()
    rows=np.arange(joined.world_count)
    expected=np.repeat(ext,2,axis=0)
    for j in (0,1):
        action=jnp.array(rows[:,None]+j*10,dtype=jnp.float32)
        transition=joined.step(action)
        np.testing.assert_array_equal(np.concatenate([s.received for s in shards]),action)
        np.testing.assert_array_equal(transition.observation[:,0],action[:,0]+100)
        np.testing.assert_array_equal(transition.terminal_observation[:,0],action[:,0]+200)
        for a in (obs,transition.observation,transition.terminal_observation):
            np.testing.assert_array_equal(a[:,-3:],expected)
        np.testing.assert_array_equal(transition.terminated,rows%2==0)
        np.testing.assert_array_equal(transition.truncated,rows%2==1)
        np.testing.assert_array_equal(transition.critic_context,action+300)
        np.testing.assert_array_equal(transition.terminal_critic_context,action+400)
        a,b=joined.transition_logging_inputs()
        np.testing.assert_array_equal(a,rows)
        np.testing.assert_array_equal(b[:,0],rows+500)
    bad=SentinelShard(0)
    bad.env.action.block_names=("different_action",)
    try:
        MultiReferenceEnv([*shards,bad],[*ext,ext[0]])
    except ValueError:
        pass
    else:
        raise AssertionError("Semantically distinct actions were pooled")
    if len(shards)>1:
        one=MultiReferenceEnv(shards[:1],ext[:1])
        assert one.policy_schema==joined.policy_schema and one.critic_schema==joined.critic_schema
    out.write_text(json.dumps(dict(physical_step_test=False, cpu_batch_contract=True,
                                  source_extents_by_reference=[r.tolist() for r in ext],
                                  world_order=rows.tolist(),roundtrip_actions_per_world=2,
                                  schema_compatible_with_single_shard=True),indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("parquet",nargs="+",type=Path)
    a=p.parse_args()
    main(a.parquet,a.output)
