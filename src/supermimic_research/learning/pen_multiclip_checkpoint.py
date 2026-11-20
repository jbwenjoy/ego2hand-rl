"""Ownership-preserving checkpoints for matched, alternating two-clip PPO."""
import copy
import os
from pathlib import Path
import torch
from .native_pen_ppo import cpu_tree
from .pen_matched_recovery import saved_bytes_equal
from .owned_pen_ppo import validate_recorded_rollout
from ..runtime import owned_pen_state as native
from ..runtime.pen_portable_state import rng_state,restore_rng

CLIPS=('clip_001','clip_003')
FIELDS=('iteration','optimizer_steps','training_transitions','calibration_transitions',
        'normalizers_ready','_rollout_complete','_captured_rows','memory_iteration')
SCHEMA='two_clip_native_PPO_frozen_parent_norm:1'


def owners(profile):
    if profile=='shared':return {clip:'shared' for clip in CLIPS}
    if profile=='specialists':return {clip:'specialist_'+clip[-3:] for clip in CLIPS}
    raise ValueError(profile)


def validate_identity(learners,profile):
    a,b=(learners[clip] for clip in CLIPS)
    for key in ('actor','critic','optimizer'):
        if (getattr(a,key) is getattr(b,key)) != (profile=='shared'):
            raise ValueError(f'Invalid ownership: {key}')
    for learner in (a,b):
        params=tuple(learner.actor.parameters())+tuple(learner.critic.parameters())
        actual=[p for g in learner.optimizer.param_groups for p in g['params']]
        if list(map(id,params))!=list(map(id,actual)):raise ValueError('Adam parameter order differs')


def snapshot(learners,profile,receipt,index,protocol):
    validate_identity(learners,profile)
    owner=owners(profile);models={};collectors={}
    for clip,learner in learners.items():
        if learner._collecting or learner.adapter._collection_open or learner.normalization_batch is not None:
            raise ValueError('Expected frozen-normalizer collection boundary')
        if not saved_bytes_equal(cpu_tree(learner.observations),cpu_tree(learner.adapter.observation())):
            raise ValueError('Stored observation differs from actual observations')
        if owner[clip] not in models:
            models[owner[clip]]=dict(actor=cpu_tree(learner.actor.state_dict()),critic=cpu_tree(learner.critic.state_dict()),
                                     optimizer=cpu_tree(learner.optimizer.state_dict()))
        collectors[clip]=dict(native=native.capture(learner.adapter,receipt),metadata=learner.metadata(),
                              observations=cpu_tree(learner.observations),memory=cpu_tree(learner.memory),
                              fields={k:getattr(learner,k) for k in FIELDS})
    return dict(schema=SCHEMA,profile=profile,completed_batches=index,protocol=protocol,
                owners=owner,models=models,collectors=collectors,rng=rng_state())


def _to(value,device):
    if isinstance(value,torch.Tensor):return value.to(device).clone()
    if isinstance(value,dict):return {k:_to(v,device) for k,v in value.items()}
    return copy.deepcopy(value)


def check_bytes(expected,actual,label):
    if not saved_bytes_equal(expected,actual):raise ValueError(f'Checkpoint mismatch: {label}')


def save_checked(path,state):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    tmp=path.with_suffix('.partial')
    torch.save(state,tmp);loaded=torch.load(tmp,map_location='cpu',weights_only=False)
    check_bytes(state,loaded,'disk roundtrip')
    os.replace(tmp,path)
    return loaded


def restore(learners,saved,profile,protocol):
    if saved['schema']!=SCHEMA or saved['profile']!=profile or saved['protocol']!=protocol or saved['owners']!=owners(profile):
        raise ValueError('Inconsistent protocol/ownership')
    validate_identity(learners,profile);devices={c:l.device for c,l in learners.items()}
    if set(learners)!=set(CLIPS) or set(saved['models'])!=set(saved['owners'].values()):
        raise ValueError('Missing owner or clip')
    seen=set();reports={}
    for clip in CLIPS:
        learner=learners[clip];ck=saved['collectors'][clip];key=saved['owners'][clip]
        if learner.metadata()!=ck['metadata']:raise ValueError('Clip metadata differs')
        audit=native.restore(learner.adapter,ck['native'])
        if not all(audit[k] for k in ('all_array_bytes_equal','all_scalar_values_equal','pointers_preserved')):
            raise ValueError('Native restore failed')
        if key not in seen:
            model=saved['models'][key]
            learner.actor.load_state_dict(model['actor']);learner.critic.load_state_dict(model['critic'])
            learner.optimizer.load_state_dict(copy.deepcopy(model['optimizer']))
            for name in ('actor','critic','optimizer'):
                check_bytes(model[name],cpu_tree(getattr(learner,name).state_dict()),key+'.'+name)
            seen.add(key)
        learner.memory=_to(ck['memory'],devices[clip])
        learner.observations=_to(ck['observations'],devices[clip])
        learner.normalization_batch=None
        for k in FIELDS:setattr(learner,k,ck['fields'][k])
        if learner.memory is not None:
            validate_recorded_rollout(learner.memory,learner.cfg.ppo_gamma)
            check_bytes(ck['memory'],cpu_tree(learner.memory),'collector memory')
        check_bytes(ck['observations'],cpu_tree(learner.adapter.observation()),'native observation after restore')
        reports[clip]=dict(native=audit,fields=ck['fields'],stored_observations_equal=True,memory_equal=True)
    restore_rng(saved['rng']);check_bytes(saved['rng'],rng_state(),'global RNG')
    validate_identity(learners,profile)
    return dict(clips=reports,all_models_and_adam_byte_equal=True,ownership=saved['owners'],rng_equal=True,
                scope='Restored both independently constructed runtime instances, collectors and optimizer ownership.')
