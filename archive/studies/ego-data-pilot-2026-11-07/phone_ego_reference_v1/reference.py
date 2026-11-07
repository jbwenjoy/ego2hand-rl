"""Object-only fixed-palm motion intent from proper F64 raw HOT3D poses.

No hand-joint retarget, gravity registration, contact inference or feasibility
claim is made. Left-human to right-robot morphology is an independent problem.
All object increments are expressed in the same canonical initial phone frame.
"""
from pathlib import Path
import hashlib,importlib.util
import numpy as np

P=Path(__file__).resolve().parent.parent
CODEC=P/'phone_transition_dataset_v1/transition_codec_v1.py'
if hashlib.sha256(CODEC.read_bytes()).hexdigest()!='1bf33a7fd27ae51d38a71c1f7e095dbe3047c4828ee6e65c199117bf536f4cac':
    raise ValueError('Frozen scalar world-left codec changed')
sp=importlib.util.spec_from_file_location('reference_world_left_codec',CODEC);codec=importlib.util.module_from_spec(sp);sp.loader.exec_module(codec)


def validate_transform(T):
    T=np.asarray(T,dtype=np.float64)
    if T.shape[-2:]!=(4,4) or not np.isfinite(T).all():raise ValueError('Finite4x4 proper rigid transforms')
    if np.max(abs(T[...,3,:]-np.array([0,0,0,1])))>1e-12:raise ValueError('Homogeneous last row')
    R=T[...,:3,:3]
    if np.max(abs(R.swapaxes(-1,-2)@R-np.eye(3)))>1e-10 or np.max(abs(np.linalg.det(R)-1))>1e-10:
        raise ValueError('Proper rotation required, no matrix projection/mirroring')
    return T


def pose_from_raw(record):
    q=np.asarray(record['quaternion_wxyz'],dtype=np.float64);p=np.asarray(record['translation_xyz'],dtype=np.float64)
    norm=float(np.linalg.norm(q))
    if q.shape!=(4,) or p.shape!=(3,) or not np.isfinite(np.r_[q,p]).all() or abs(norm-1)>1e-6:
        raise ValueError('Raw source near-unit quaternion and metric translation')
    # Explicit source quaternion representation normalization; no affine/R repair.
    T=np.eye(4);T[:3,:3]=codec.quat_R(q/norm);T[:3,3]=p
    return validate_transform(T),norm


def inverse_rigid(T):
    T=validate_transform(T);out=np.zeros_like(T);R=T[...,:3,:3]
    out[...,:3,:3]=R.swapaxes(-1,-2);out[...,:3,3]=-(R.swapaxes(-1,-2)@T[...,:3,3,None])[...,0];out[...,3,3]=1
    return out


def relative_object_increments(T_world_wrist,T_world_phone):
    W=validate_transform(T_world_wrist);O=validate_transform(T_world_phone)
    if W.shape!=O.shape or W.ndim!=3:raise ValueError('Matching batched wrist/object source poses')
    rel=inverse_rigid(W)@O
    delta=inverse_rigid(rel[0])@rel
    return validate_transform(rel),validate_transform(delta)


def reanchor_object_reference(initial_world_body,delta_initial_object,body_ipos):
    initial=validate_transform(initial_world_body);delta=validate_transform(delta_initial_object)
    ipos=np.asarray(body_ipos,dtype=np.float64)
    if initial.shape!=(4,4) or delta.ndim!=3 or ipos.shape!=(3,):raise ValueError('Initial rigid pose, deltaN4x4 and actual BODY COM offset')
    body=validate_transform(initial@delta)
    COM=body[:,:3,3]+np.einsum('nij,j->ni',body[:,:3,:3],ipos)
    return body,COM


def interpolate_increments(source_time_s,delta_initial_object,query_time_s):
    t=np.asarray(source_time_s,dtype=np.float64);D=validate_transform(delta_initial_object);query=np.asarray(query_time_s,dtype=np.float64)
    if t.ndim!=1 or len(t)!=len(D) or np.any(np.diff(t)<=0) or query.ndim!=1 or not np.isfinite(query).all():raise ValueError('Strict source timestamps and finite queries')
    if np.any(query<t[0]) or np.any(query>t[-1]):raise ValueError('No extrapolation or implicit duration rescaling')
    out=[]
    for v in query:
        i=min(int(np.searchsorted(t,v,side='right'))-1,len(t)-2);i=max(i,0)
        alpha=float((v-t[i])/(t[i+1]-t[i]));d=np.eye(4)
        d[:3,3]=(1-alpha)*D[i,:3,3]+alpha*D[i+1,:3,3]
        world_log=codec.log_world(D[i+1,:3,:3]@D[i,:3,:3].T)
        d[:3,:3]=codec.exp_world(alpha*world_log)@D[i,:3,:3];out.append(d)
    return validate_transform(np.asarray(out))


def interval_motion_descriptors(time_s,body_world,COM_world):
    t=np.asarray(time_s);B=validate_transform(body_world);C=np.asarray(COM_world);dt=np.diff(t)
    if len(B)!=len(t) or C.shape!=(len(t),3) or np.any(dt<=0):raise ValueError('Same-clock body/COM samples')
    v=np.diff(C,axis=0)/dt[:,None]
    omega=np.array([codec.log_world(B[i+1,:3,:3]@B[i,:3,:3].T)/dt[i] for i in range(len(dt))])
    return dict(interval_mean_COM_velocity_world_m_s=v,interval_geodesic_angular_velocity_world_rad_s=omega,
        descriptor_scope='Finite-interval reference descriptors, not measured instantaneous twist or force ground truth')
