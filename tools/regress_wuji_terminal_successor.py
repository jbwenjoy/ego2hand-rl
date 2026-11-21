"""CPU-only paired update using actual archived r2_single2 terminal row, with no training run."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from safetensors import safe_open

from flash_chord.configuration import instantiate_typed
from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.flash_sac import learner as L
from flash_chord.training.flash_sac.checkpoint import training_checkpoint,restore_learner_checkpoint
from flash_chord.training.flash_sac.config import TrainingConfig
from flash_chord.training.flash_sac.network import Actor,Critic
from flash_chord.training.flash_sac.replay import ReplayBatch,ReplayStorage,ReplayInsert
from flash_chord.training.flash_sac.update import categorical_td_target
from supermimic_research.learning.terminal_successor_guard import (
    validate_committed,InvalidCommittedBatch,finite_candidate_or_raise,observed_physics_rows)


def digest(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()


def archived_batch(path,indices):
    with safe_open(path,framework="np") as reader:
        arrays={}
        for i,name in enumerate(ReplayStorage.__dataclass_fields__):
            values=reader.get_tensor(f"state.{i:04d}")[indices]
            arrays[name]=jnp.asarray(values)
    return ReplayBatch(**arrays)


def state_health(tree):
    bad={}
    total=0
    for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        a=np.asarray(value); count=int(np.count_nonzero(~np.isfinite(a)));total+=1
        if count:bad[jax.tree_util.keystr(path)]=count
    return dict(leaves=total,bad_leaves=len(bad),nonfinite=bad)


def exact_tree(a,b):
    aa,ta=jax.tree.flatten(a);bb,tb=jax.tree.flatten(b)
    return ta==tb and all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(aa,bb))


def tree_hash(tree):
    h=hashlib.sha256()
    for v in jax.tree.leaves(tree):
        a=np.asarray(v);h.update(a.tobytes(order="C"))
    return h.hexdigest()


def replace_row(batch,slot,row):
    return batch.replace(**{n:getattr(batch,n).at[slot].set(getattr(row,n)[0]) for n in ReplayBatch.__dataclass_fields__})


def run(args):
    start=time.perf_counter();args.out.mkdir(exist_ok=False,parents=True)
    if jax.default_backend()!="cpu":raise ValueError("Diagnostic must run on CPU")
    directory=args.checkpoints
    parent=directory/"state_456960.safetensors"; clean_path=directory/"replay_456960.safetensors"
    poisoned=directory/"replay_548352.safetensors"
    meta=read_checkpoint_metadata(parent)
    saved=checkpoint_config(meta);config=instantiate_typed(OmegaConf.create(saved["training"]),TrainingConfig)
    healthy_templates=json.loads((directory.parent/"critic_schema.json").read_bytes())
    obs=healthy_templates["observation"]["dimension"];context=healthy_templates["context"]["dimension"]
    names=tuple(x["name"] for x in healthy_templates["context"]["blocks"])
    dim=healthy_templates["action"]["dimension"];n=192
    # Keep tagged sample size, n-step, network, fp16 scale, optimizer schedule and weights.
    # Only shrink the fixture's committed storage from the original 131072 to 192 rows.
    config=replace(config,replay=replace(config.replay,capacity=n,minimum_size=n))
    base,_=L.create_learner(config,jnp.zeros((config.world_count,obs),dtype=jnp.float32),
                  jnp.zeros((config.world_count,context),dtype=jnp.float32),action_dim=dim,
                  objective_term_count=len(OBJECTIVE_TERM_NAMES),critic_context_names=names,device=jax.devices("cpu")[0])
    base,_=restore_learner_checkpoint(parent,base)
    stable=training_checkpoint(base)
    before=tree_hash(stable)
    assert not state_health(stable)["bad_leaves"] and int(base.global_update_step)%config.optimization.actor_update_period==0
    stage=config.curriculum.stages[int(meta["curriculum_stage"])]
    weights=jnp.asarray(stage.weights_for(OBJECTIVE_TERM_NAMES),dtype=jnp.float32)
    dt=1./float(saved["sim"]["fps"])
    dtype=jnp.dtype(config.compute_dtype) if config.mixed_precision else jnp.float32
    actor=Actor(action_dim=dim,config=config.network,dtype=dtype)
    lo,hi=config.reward_normalization.value_support
    critic=Critic(value_min=lo,value_max=hi,config=config.network,dtype=dtype)
    optimizer=L._create_adam(config)
    @jax.jit
    def native_attempt(s):
        updated,metrics=L._optimizer_attempt(s,L._zero_metrics(),weights,jnp.asarray(dt,dtype=jnp.float32),
                  actor,critic,optimizer,config,bool(config.mixed_precision and jnp.dtype(dtype)==jnp.dtype(jnp.float16)))
        return training_checkpoint(updated),metrics
    def try_update(batch, initial=base):
        storage=ReplayStorage(**{k:getattr(batch,k) for k in ReplayStorage.__dataclass_fields__})
        replay=base.replay.replace(storage=storage,size=jnp.asarray(n,jnp.int32),write_index=jnp.asarray(0,jnp.int32))
        state,metrics=native_attempt(initial.replace(replay=replay))
        jax.block_until_ready(state)
        return state,metrics
    clean=archived_batch(clean_path,np.arange(n,dtype=int))
    real=archived_batch(poisoned,np.array([3677]))
    assert bool(real.terminated[0]) and not bool(real.truncated[0])
    assert not np.isfinite(np.asarray(real.next_observation)).all()
    sample_key=jax.random.split(base.key)[1]
    sample_indices=np.asarray(jax.random.randint(sample_key,(config.replay.batch_size,),0,n,dtype=jnp.int32))
    chosen=int(sample_indices[0]);introduced=replace_row(clean,chosen,real)
    params=dict(n_step=config.replay.n_step,successor_origin="tagged_pre_reset",objective_weights=weights,frame_dt=dt)
    clean_guard=validate_committed(clean,**params);protected=validate_committed(introduced,**params)
    assert clean_guard.rl_batch is clean
    assert not np.isfinite(np.asarray(introduced.next_observation[chosen])).all() and np.isfinite(protected.rl_batch.next_observation[chosen]).all()
    assert np.array_equal(np.asarray(protected.rl_batch.discounted_objective_terms),np.asarray(introduced.discounted_objective_terms))
    assert protected.report["rl_placeholder_ids"]==[chosen] and len(protected.observed_one_step_label_ids)==0
    baseline,baseline_metrics=try_update(clean);clean_checked,clean_metrics=try_update(clean_guard.rl_batch)
    raw_state,raw_metrics=try_update(introduced);fixed_state,fixed_metrics=try_update(protected.rl_batch)
    # Fail closed on a bad proposal. The original restored state remains available and unchanged.
    proposed={}
    for name,state,metrics in (("allfinite_upstream",baseline,baseline_metrics),("allfinite_guard",clean_checked,clean_metrics),
                               ("bad_terminal_upstream",raw_state,raw_metrics),("bad_terminal_guard",fixed_state,fixed_metrics)):
        h=state_health(state); mh=state_health(metrics)
        proposed[name]=dict(learner=h,stats=mh,commit_allowed=(h["bad_leaves"]==0 and mh["bad_leaves"]==0),
                changed_actor_params=not exact_tree(stable.actor.params,state.actor.params),
                changed_critic_params=not exact_tree(stable.critic.params,state.critic.params),
                changed_actor_optimizer=not exact_tree(stable.actor.optimizer_state,state.actor.optimizer_state),
                changed_critic_optimizer=not exact_tree(stable.critic.optimizer_state,state.critic.optimizer_state),
                changed_actor_bn=not exact_tree(stable.actor.batch_stats,state.actor.batch_stats),
                changed_target_critic_bn=not exact_tree(stable.target_critic.batch_stats,state.target_critic.batch_stats))
    assert exact_tree(baseline,clean_checked) and exact_tree(baseline_metrics,clean_metrics)
    assert proposed['allfinite_upstream']['commit_allowed'] and not proposed['bad_terminal_upstream']['commit_allowed']
    assert proposed['bad_terminal_guard']['commit_allowed'] and proposed['bad_terminal_guard']['changed_actor_bn']
    refused=False
    try:finite_candidate_or_raise(raw_state,raw_metrics)
    except FloatingPointError:refused=True
    assert refused
    assert finite_candidate_or_raise(fixed_state,fixed_metrics) is fixed_state
    assert not any(len(v) for v in observed_physics_rows(introduced,protected))
    # Secondary diagnostic only: the archived 2^22 fp16 loss scale can itself cause a native
    # skipped gradient. Repeat on the same data/weights at scale 1024 to exercise BOTH optimizers.
    lower=base.replace(loss_scale=base.loss_scale.replace(scale=jnp.asarray(1024.,dtype=jnp.float32)))
    low_state,low_metrics=try_update(protected.rl_batch,lower)
    finite_candidate_or_raise(low_state,low_metrics)
    low_actor=not exact_tree(lower.actor.optimizer_state,low_state.actor.optimizer_state)
    low_critic=not exact_tree(lower.critic.optimizer_state,low_state.critic.optimizer_state)
    assert low_actor and low_critic
    invalid_cases={
       "continuing_bad_successor":real.replace(terminated=jnp.zeros_like(real.terminated)),
       "timeout_bad_successor":real.replace(terminated=jnp.zeros_like(real.terminated),truncated=jnp.ones_like(real.truncated)),
       "ambiguous_termination_bad_successor":real.replace(truncated=jnp.ones_like(real.truncated)),
       "invalid_reward_component":real.replace(discounted_objective_terms=real.discounted_objective_terms.at[0,0].set(jnp.nan)),
       "overflow_even_with_finite_reward_components":real.replace(discounted_objective_terms=real.discounted_objective_terms.at[0,0].set(jnp.finfo(jnp.float32).max)),
       "invalid_current_action":real.replace(action=real.action.at[0,0].set(jnp.nan)),
       "invalid_current_observation":real.replace(observation=real.observation.at[0,0].set(jnp.inf)),
       "invalid_current_context":real.replace(critic_context=real.critic_context.at[0,0].set(jnp.nan))}
    rejects={}
    for name,data in invalid_cases.items():
        try:validate_committed(data,**params)
        except InvalidCommittedBatch as e:rejects[name]=e.report
        else:raise AssertionError(f"{name} incorrectly accepted")
    # An otherwise ordinary truncated-but-finite row must still bootstrap, unchanged.
    finite_timeout=clean.replace(truncated=clean.truncated.at[0].set(True),terminated=clean.terminated.at[0].set(False))
    assert validate_committed(finite_timeout,**params).rl_batch is finite_timeout
    unsupported=False
    try:validate_committed(introduced,**{**params,"observed_pre_reset_one_step":True})
    except ValueError:unsupported=True
    assert unsupported
    pending_refused=False
    try:validate_committed(None,**params)
    except TypeError:pending_refused=True
    assert pending_refused
    post_reset_refused=False
    try:validate_committed(real,**{**params,"successor_origin":"post_reset"})
    except ValueError:post_reset_refused=True
    assert post_reset_refused
    # Pure native Bellman target test: if terminated, changing finite successor value
    # probabilities or entropy cannot change the target beyond summation roundoff.
    atoms=config.network.atom_count
    probs=jax.nn.log_softmax(jnp.arange(atoms,dtype=jnp.float32)[None,:]/30)
    rewards=jnp.asarray([.75],dtype=jnp.float32);term=jnp.array([True])
    target_args=dict(bootstrap_discount=base.replay.spec.bootstrap_discount,value_min=lo,value_max=hi)
    def target(p,ent,flags):return np.asarray(categorical_td_target(p,rewards,flags,jnp.array([ent],jnp.float32),**target_args))
    delta=float(abs(target(probs,4.,term)-target(probs[:,::-1],-6.,term)).max())
    boot_delta=float(abs(target(probs,4.,~term)-target(probs[:,::-1],-6.,~term)).max())
    assert delta<2e-6 and boot_delta>.001
    assert tree_hash(training_checkpoint(base))==before
    report=dict(scope='Offline committed replay fixture, no environment step, no policy saved or scored, no WM fit.',
       parent=dict(path=parent.name,sha256=digest(parent),optimizer_step=int(base.global_update_step)),
       healthy_replay_sha256=digest(clean_path),source_bad_row=dict(archive=poisoned.name,sha256=digest(poisoned),slot=3677),
       fixture=dict(ring_rows=n,sampled_with_replacement=config.replay.batch_size,chosen_slot=chosen,
          occurrences_in_native_sample=int((sample_indices==chosen).sum()),n_step=config.replay.n_step,compute_dtype=str(dtype),
          parent_loss_scale=float(base.loss_scale.scale),reward_term_weights=np.asarray(weights).tolist()),
       good_batch=clean_guard.report,bad_terminal_batch=protected.report,native_proposals=proposed,
       whole_candidate_rejection_tested=refused,scale_1024_sensitivity=dict(actor_adam_changed=low_actor,
            critic_adam_changed=low_critic,all_learner_and_metric_leaves_finite=True),
       all_finite_guard_exactly_matches_native_state_and_metrics=True,rejected_cases=rejects,
       true_terminal_target_max_delta=delta,continuing_target_max_delta=boot_delta,
       world_model_rows_used=0,pre_reset_single_step_provenance_required=True,
       parent_state_unmodified_sha256=before,implementation_sha256=digest(__file__),
       guard_source_sha256=digest(Path(__file__).resolve().parents[1]/'src/supermimic_research/learning/terminal_successor_guard.py'),
       upstream_learner_sha256=digest(L.__file__),pending_not_accepted=pending_refused,post_reset_not_accepted=post_reset_refused,
       live_rollout_repair_tested=False,elapsed_s=time.perf_counter()-start)
    (args.out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('CPU REGRESSION',[(k,v["learner"]["bad_leaves"],v["changed_actor_params"],v["changed_critic_params"]) for k,v in proposed.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoints',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
