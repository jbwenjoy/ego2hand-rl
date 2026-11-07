"""Parameter-free geometry-model output contract with explicit momentum closure.

Physical head layout is joint_delta40, COM-ballistic-residual3, WORLD-left
rotation_delta3, and22 source-BODY/moving-COM contact impulse6 blocks. This is
NOT either old52 latent chart. The network and its loss weights are not defined.
Position/R heads remain separate: impulse alone does not determine a pose.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
from . import momentum, rotation
from . import state_codec as scalar
SCHEMA='phone_joint40_COM_residual3_world_dR3_pair22_impulse6:1'

def encode_observed_numpy(source58,target58,pair_impulse22x6,duration_s,gravity_world):
    """Supervised target preparation, never a model forward input provider.

    Impulse targets must come from separately decoded actual contacts. Endpoint
    velocities are NOT used to fabricate contact impulse targets.
    """
    x=np.asarray(source58);y=np.asarray(target58);J=np.asarray(pair_impulse22x6);t=np.asarray(duration_s);g=np.asarray(gravity_world)
    if x.ndim!=2 or x.shape!=y.shape or x.shape[1]!=58 or J.shape!=(len(x),22,6) or t.shape!=(len(x),):raise ValueError('Batched matching physical targets')
    if not all(np.isfinite(v).all() for v in [x,y,J,t,g]) or np.any(t<=0):raise ValueError('Known executed targets only; unavailable NaNs excluded explicitly')
    R0=x[:,43:52].reshape(-1,3,3);R1=y[:,43:52].reshape(-1,3,3)
    dr=np.array([scalar.log_world(b@a.T) for a,b in zip(R0,R1)])
    residual=y[:,40:43]-x[:,40:43]-x[:,52:55]*t[:,None]-.5*g*t[:,None]**2
    return dict(joint_delta40=y[:,:40]-x[:,:40],COM_ballistic_residual3_m=residual,world_rotation_delta3_rad=dr,
        pair_impulse_source_BODY_moving_COM22x6=J.copy())

def decode_physical_heads(source58,heads,duration_s,mass_kg,inertia_body,gravity_world):
    """Decode independent pose heads and conditional momentum velocities.

    No joint clipping, pose projection, torque/force oracle, hidden contact
    state or trainable parameter. Mean-force/contact-timing heads, if later
    required by a policy, need independent supervision and are not inferred.
    """
    required={'joint_delta40','COM_ballistic_residual3_m','world_rotation_delta3_rad','pair_impulse_source_BODY_moving_COM22x6'}
    if set(heads)!=required:raise ValueError('Exact new physical-head schema')
    x=source58;lead=x.shape[:-1]
    shapes={'joint_delta40':lead+(40,),'COM_ballistic_residual3_m':lead+(3,),'world_rotation_delta3_rad':lead+(3,),'pair_impulse_source_BODY_moving_COM22x6':lead+(22,6)}
    for key,shape in shapes.items():
        h=heads[key]
        if h.shape!=shape or h.device!=x.device or h.dtype!=x.dtype:raise ValueError('Same-device/dtype physical heads: '+key)
    if duration_s.shape!=lead:raise ValueError('Explicit actual horizon per row')
    R0=x[...,43:52].reshape(lead+(3,3));R1=rotation.so3_exp_torch(heads['world_rotation_delta3_rad'])@R0
    joint=x[...,:40]+heads['joint_delta40']
    COM=x[...,40:43]+x[...,52:55]*duration_s[...,None]+.5*gravity_world*duration_s[...,None]**2+heads['COM_ballistic_residual3_m']
    impulse=heads['pair_impulse_source_BODY_moving_COM22x6'].sum(dim=-2)
    velocity=momentum.velocities_from_source_impulse(x,R1,impulse,mass_kg,inertia_body,gravity_world,duration_s)
    return torch.cat([joint,COM,R1.flatten(-2),velocity],dim=-1)
