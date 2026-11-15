"""Functional saved-observation reader using the maintained feature encoders."""
from types import SimpleNamespace
import torch
from supermimic_research.runtime.pen_native_policy import NativePenFeatures
from supermimic_research.runtime.pen_reference_cache import FrozenPenReferenceCache
from supermimic_research.runtime.portable_inventory import path_value


def feature_tensors(runtime):
    f=runtime.features;m=f.motion;d=f.robot.data;c=runtime.reference_cache
    return dict(bias=d.encoder_bias,default_q=d.default_joint_pos,default_qd=d.default_joint_vel,
        raw=f.action._raw_actions,EMA=f.action._processed_actions,phase=m.time_steps,trajectory=m.trajectory_ids,
        starts=m._trajectory_starts,ends=m._trajectory_ends,qids=f.qids,dofs=f.dofs,local_ids=f.local_joints,prefix=c.prefix,world_ids=c.world_ids)


def feature_sources(runtime,inventory):
    paths={}
    for key,value in feature_tensors(runtime).items():
        matches=[p for p,v in inventory['tensors'].items() if v is value]
        if not matches:
            matches=[p for p,v in inventory['tensors'].items() if (v.data_ptr(),v.shape,v.stride(),v.dtype)==(value.data_ptr(),value.shape,value.stride(),value.dtype)]
        if not matches:raise ValueError('Uninventoried observation input '+key)
        paths[key]=sorted(matches,key=repr)[0]
    owner=next(p for p in inventory['objects'] if path_value(runtime,p) is runtime.features)
    cache_owner=next(p for p in inventory['objects'] if path_value(runtime,p) is runtime.reference_cache)
    static={k:dict(path=owner+(('attr',k),),value=getattr(runtime.features,k)) for k in ['body','palm','qa']}
    static['frames']=dict(path=cache_owner+(('attr','frames'),),value=runtime.reference_cache.frames)
    return dict(schema='owned_observation_saved_sources:1',tensors=paths,static=static)


@torch.no_grad()
def saved_observation(runtime,native_state,sources):
    """No live state mutation or alternate geometry formulas; CPU cold path."""
    values={k:native_state['python']['tensors'][p] for k,p in sources['tensors'].items()}
    live=feature_tensors(runtime)
    static={k:record['value'] for k,record in sources['static'].items()}
    if any(native_state['python']['scalars'][record['path']]!=record['value'] for record in sources['static'].values()):raise ValueError('Saved static feature selector changed')
    for key in ['qids','dofs','local_ids','starts','ends','world_ids']:
        if not torch.equal(values[key],live[key].cpu()):raise ValueError('Saved immutable feature indices changed '+key)
    m=SimpleNamespace(time_steps=values['phase'],trajectory_ids=values['trajectory'],_trajectory_starts=values['starts'],_trajectory_ends=values['ends'])
    cls=type(runtime.motion)
    for key in ['trajectory_starts','trajectory_ends','trajectory_lengths','frame_ids']:setattr(m,key,getattr(cls,key).fget(m))
    raw=native_state['native']['data']['arrays'];data=SimpleNamespace(**{k:torch.from_numpy(raw[k].copy()) for k in ['qpos','qvel','xpos','xquat']})
    f=NativePenFeatures.__new__(NativePenFeatures);f.env=SimpleNamespace(sim=SimpleNamespace(data=data,mj_model=runtime.env.sim.mj_model))
    f.robot=SimpleNamespace(data=SimpleNamespace(encoder_bias=values['bias'],default_joint_pos=values['default_q'],default_joint_vel=values['default_qd']))
    f.action=SimpleNamespace(_raw_actions=values['raw'],_processed_actions=values['EMA']);f.motion=m
    for key in ['body','palm','qa']:setattr(f,key,static[key])
    f.qids=values['qids'];f.dofs=values['dofs'];f.local_joints=values['local_ids'];f.reference={'joint_pos':torch.empty(static['frames'],20)}
    cache=FrozenPenReferenceCache.__new__(FrozenPenReferenceCache);cache.prefix=values['prefix'];cache.world_ids=values['world_ids']
    policy=cache.policy(f)
    return dict(policy=policy,critic=f.critic(policy))
