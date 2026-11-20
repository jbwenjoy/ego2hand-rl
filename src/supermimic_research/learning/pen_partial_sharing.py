"""Explicit partial sharing for a two-task PPO learner and its restore boundary."""
import copy
import torch

from .pen_multiclip_checkpoint import CLIPS,FIELDS,check_bytes,_to
from .native_pen_ppo import cpu_tree
from .owned_pen_ppo import validate_recorded_rollout
from ..runtime import owned_pen_state as native
from ..runtime.pen_portable_state import rng_state,restore_rng

PROFILES=('shared_adam','local_adam')


def wire(a,b,parent,profile):
    """Each step uses one ordinary AdamW over the active Actor and Critic."""
    if profile not in PROFILES: raise ValueError(profile)
    a.actor.load_state_dict(parent['modules']['actor'])
    b.actor=a.actor
    for learner in (a,b):
        learner.critic.load_state_dict(parent['modules']['critic'])
        params=tuple(learner.actor.parameters())+tuple(learner.critic.parameters())
        learner.optimizer=torch.optim.AdamW(params,lr=learner.cfg.ppo_lr,weight_decay=learner.cfg.weight_decay)
        learner.optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        learner.normalizers_ready=True
    bind_actor_state(a,b,profile)
    validate(a,b,profile)


def bind_actor_state(a,b,profile):
    if profile=='shared_adam':
        for param in a.actor.parameters():
            b.optimizer.state[param]=a.optimizer.state[param]


def validate(a,b,profile):
    if a.actor is not b.actor or a.critic is b.critic or a.optimizer is b.optimizer:
        raise ValueError('Incorrect module ownership')
    for learner in (a,b):
        params=tuple(learner.actor.parameters())+tuple(learner.critic.parameters())
        owned=[p for group in learner.optimizer.param_groups for p in group['params']]
        if list(map(id,params))!=list(map(id,owned)):
            raise ValueError('Optimizer parameter-order mismatch')
    for param in a.actor.parameters():
        left=a.optimizer.state[param];right=b.optimizer.state[param]
        if (left is right)!=(profile=='shared_adam') or set(left)!=set(right):
            raise ValueError('Actor Adam state ownership mismatch')
        for key in left:
            if isinstance(left[key],torch.Tensor) and (left[key].data_ptr()==right[key].data_ptr())!=(profile=='shared_adam'):
                raise ValueError('Actor Adam tensor alias mismatch')
    for pa,pb in zip(a.critic.parameters(),b.critic.parameters(),strict=True):
        if pa.data_ptr()==pb.data_ptr() or a.optimizer.state[pa] is b.optimizer.state[pb]:
            raise ValueError('Critic parameter/state alias mismatch')


def capture(learners,profile,protocol,receipt,count):
    a,b=(learners[c] for c in CLIPS)
    validate(a,b,profile)
    saved=dict(schema='partial_multiclip_ppo:1',profile=profile,protocol=protocol,count=count,
               actor=cpu_tree(a.actor.state_dict()),clips={},rng=rng_state())
    for clip,l in learners.items():
        if l._collecting or l.adapter._collection_open or l.normalization_batch is not None:
            raise ValueError('Not at a frozen-normalization boundary')
        check_bytes(cpu_tree(l.observations),cpu_tree(l.adapter.observation()),clip+' observation')
        saved['clips'][clip]=dict(critic=cpu_tree(l.critic.state_dict()),optimizer=cpu_tree(l.optimizer.state_dict()),
            native=native.capture(l.adapter,receipt),metadata=l.metadata(),memory=cpu_tree(l.memory),
            observations=cpu_tree(l.observations),fields={k:getattr(l,k) for k in FIELDS})
    return saved


def restore(learners,profile,protocol,saved):
    if saved['schema']!='partial_multiclip_ppo:1' or saved['profile']!=profile or saved['protocol']!=protocol or set(saved['clips'])!=set(CLIPS):
        raise ValueError('Checkpoint identity mismatch')
    a,b=(learners[k] for k in CLIPS)
    a.actor.load_state_dict(saved['actor'])
    audits={}
    for key,l in learners.items():
        s=saved['clips'][key]
        if l.metadata()!=s['metadata']:raise ValueError('Clip metadata mismatch')
        r=native.restore(l.adapter,s['native'])
        if not all(r[k] for k in ('all_array_bytes_equal','all_scalar_values_equal','pointers_preserved')):raise ValueError('Native restore mismatch')
        l.critic.load_state_dict(s['critic'])
        l.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
        l.observations=_to(s['observations'],l.device)
        l.memory=_to(s['memory'],l.device)
        for k in FIELDS:setattr(l,k,s['fields'][k])
        l.normalization_batch=None
        if l.memory is not None:validate_recorded_rollout(l.memory,l.cfg.ppo_gamma)
        check_bytes(s['memory'],cpu_tree(l.memory),key+' restored memory')
        check_bytes(s['observations'],cpu_tree(l.adapter.observation()),key+' restored observation')
        audits[key]=r
    bind_actor_state(a,b,profile);validate(a,b,profile)
    for key,l in learners.items():
        s=saved['clips'][key]
        check_bytes(saved['actor'],cpu_tree(l.actor.state_dict()),key+' actor')
        check_bytes(s['critic'],cpu_tree(l.critic.state_dict()),key+' critic')
        check_bytes(s['optimizer'],cpu_tree(l.optimizer.state_dict()),key+' Adam')
    restore_rng(saved['rng']);check_bytes(saved['rng'],rng_state(),'restored RNG')
    return dict(native=audits,actor_critic_adam_memory_observations_exact=True,rng_exact=True,ownership_verified=profile)
