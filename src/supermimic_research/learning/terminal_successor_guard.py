"""Offline prototype for accepting already-committed FlashSAC rows at the RL/WM boundary."""
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from flash_chord.training.flash_sac.replay import ReplayBatch


class InvalidCommittedBatch(ValueError):
    def __init__(self, report):
        super().__init__(f"Rejected committed replay rows {report['rejected_row_ids']}")
        self.report=report


@dataclass(frozen=True)
class GuardedBatch:
    rl_batch: ReplayBatch
    report: dict
    observed_one_step_label_ids: np.ndarray


def finite_candidate_or_raise(candidate, metrics):
    """Commit boundary for the CPU prototype; leave the old immutable learner untouched on rejection."""
    bad=[]
    for category,tree in (("learner",candidate),("metrics",metrics)):
        for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
            if not np.isfinite(np.asarray(value)).all():
                bad.append(category+jax.tree_util.keystr(path))
    if bad:
        raise FloatingPointError("Nonfinite proposed update: "+", ".join(bad))
    return candidate


def observed_physics_rows(raw, decision):
    """Select only unchanged raw one-step successors; an empty selection means no WM update."""
    idx=decision.observed_one_step_label_ids
    data=(raw.observation[idx],raw.action[idx],raw.next_observation[idx])
    if any(not np.isfinite(np.asarray(v)).all() for v in data):
        raise ValueError("Nonfinite dynamics label in original observations")
    return data


def validate_committed(batch, *, n_step, successor_origin, objective_weights, frame_dt, observed_pre_reset_one_step=False):
    """Validate before the native learner touches mutable BN/optimizer state.

    A finite current-state duplicate may serve as network padding only for a true
    terminal row with a nonfinite successor. Its native Bellman mask stays zero.
    The raw replay is immutable and the separate WM selector sees raw inputs only.
    """
    if not isinstance(batch,ReplayBatch):
        raise TypeError("Only already-committed ReplayBatch is supported; pending n-step windows, collector resets and post-reset rollout states require separate handling")
    if successor_origin!="tagged_pre_reset":
        raise ValueError("Post-reset or unverified successor semantics cannot be inferred from a committed array")
    if not isinstance(n_step,int) or n_step<1:raise ValueError("Invalid declared replay horizon")
    if observed_pre_reset_one_step and n_step!=1:
        raise ValueError("An aggregated n-step successor is not a one-control physics label")
    raw={name:np.asarray(getattr(batch,name)) for name in batch.__dataclass_fields__}
    t=raw['terminated'];c=raw['truncated']
    if (t.dtype!=np.bool_ or c.dtype!=np.bool_ or c.shape!=t.shape or t.ndim!=1 or not len(t)):
        raise ValueError("Committed native transition flags must be two boolean vectors")
    for name in ('observation','action','discounted_objective_terms','next_observation','critic_context','next_critic_context'):
        if raw[name].ndim!=2 or raw[name].shape[0]!=len(t) or not np.issubdtype(raw[name].dtype,np.floating):
            raise ValueError(f"Invalid float batch shape for {name}")
    if raw['next_observation'].shape!=raw['observation'].shape or raw['next_critic_context'].shape!=raw['critic_context'].shape:
        raise ValueError("Current and successor feature dimensions differ")
    finite={k:np.isfinite(a).all(axis=1) for k,a in raw.items() if a.ndim==2}
    w=np.asarray(objective_weights,dtype=np.float32)
    if w.shape!=(raw['discounted_objective_terms'].shape[1],) or not np.isfinite(w).all() or not np.isfinite(frame_dt) or frame_dt<=0:
        raise ValueError("Invalid reward weights or time step")
    with np.errstate(invalid='ignore',over='ignore'):
        reward=np.float32(frame_dt)*np.einsum('bt,t->b',raw['discounted_objective_terms'].astype(np.float32),w)
    valid_current=finite['observation']&finite['critic_context']&finite['action']&finite['discounted_objective_terms']&np.isfinite(reward)
    valid_successor=finite['next_observation']&finite['next_critic_context']
    true_terminal=t&~c
    placeholder=valid_current&true_terminal&~valid_successor
    reject=~valid_current | (~valid_successor & ~true_terminal)
    wm_ids=np.flatnonzero(valid_current&valid_successor) if n_step==1 and observed_pre_reset_one_step else np.empty(0,dtype=int)
    report=dict(rows=len(t),true_terminal_rows=int(true_terminal.sum()),rejected_rows=int(reject.sum()),
                rejected_row_ids=np.flatnonzero(reject).tolist(),rl_placeholder_rows=int(placeholder.sum()),
                rl_placeholder_ids=np.flatnonzero(placeholder).tolist(),wm_eligible_rows=len(wm_ids),
                invalid_fields={k:np.flatnonzero(~v).tolist() for k,v in finite.items() if not v.all()},
                invalid_scalar_reward_ids=np.flatnonzero(~np.isfinite(reward)).tolist(),
                bootstrap_needed_but_invalid_ids=np.flatnonzero(~t&~valid_successor).tolist(),
                ambiguous_terminal_timeout_invalid_ids=np.flatnonzero(t&c&~valid_successor).tolist())
    if reject.any():raise InvalidCommittedBatch(report)
    if not placeholder.any():return GuardedBatch(batch,report,wm_ids)
    bad=jnp.asarray(placeholder)[:,None]
    padded=batch.replace(next_observation=jnp.where(bad,batch.observation,batch.next_observation),
                         next_critic_context=jnp.where(bad,batch.critic_context,batch.next_critic_context))
    return GuardedBatch(padded,report,wm_ids)
