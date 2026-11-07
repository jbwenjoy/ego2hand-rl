"""Pure saved-state codec. Physical world-tangent52, not native WM chart52."""
from pathlib import Path
import math
import numpy as np

from . import state_contract as state

def quat_R(q):
    """F64 quaternion sandwich matrix; raw q retained, no normalization/repair."""
    w,x,y,z=np.asarray(q,dtype=float)
    if not np.isfinite(q).all() or abs(w*w+x*x+y*y+z*z-1)>1e-12:raise ValueError('Saved unit quaternion required')
    return np.array([[w*w+x*x-y*y-z*z,2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),w*w-x*x+y*y-z*z,2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),w*w-x*x-y*y+z*z]])
def actual_state(qpos,qvel,ipos):
    q=np.asarray(qpos,dtype=float);v=np.asarray(qvel,dtype=float)
    assert q.shape==(27,) and v.shape==(26,)
    return state.to_com_world_state(q[:20],v[:20],q[20:23],quat_R(q[23:27]),v[20:23],v[23:26],np.asarray(ipos,dtype=float))
def snapshot_state(snapshot,ipos):
    s=snapshot['state'];R=np.asarray(s['R_body_world'],dtype=float)
    x=np.r_[s['q20'],s['qd20'],s['COM_world_m'],R.ravel(),s['v_COM_world_m_s'],s['omega_world_rad_s']].astype(float)
    assert x.shape==(58,) and np.isfinite(x).all()
    raw=state.from_com_world_state(x,np.asarray(ipos,dtype=float))
    assert np.max(abs(raw['p_root_world']-s['phone_root_world_m']))<=2e-12
    assert np.max(abs(raw['v_root_world']-s['v_root_world_m_s']))<=2e-12
    return x
def skew(v):
    x,y,z=v;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
def exp_world(v):
    v=np.asarray(v,dtype=float);theta=float(np.linalg.norm(v));K=skew(v)
    if theta<1e-7:a=1-theta**2/6+theta**4/120;b=.5-theta**2/24+theta**4/720
    else:a=math.sin(theta)/theta;b=(1-math.cos(theta))/theta**2
    return np.eye(3)+a*K+b*(K@K)
def log_world(R):
    """Principal Log on actual near-SO3 F64 operands; no matrix projection."""
    R=np.asarray(R,dtype=float)
    if R.shape!=(3,3) or not np.isfinite(R).all() or np.max(abs(R.T@R-np.eye(3)))>1e-10 or abs(np.linalg.det(R)-1)>1e-10:raise ValueError('SO3 arithmetic domain')
    v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2
    sine=float(np.linalg.norm(v));cosine=float((np.trace(R)-1)/2);theta=math.atan2(sine,cosine)
    if theta>=math.pi-1e-6:raise ValueError('Half-turn Log branch unresolved: stop whole build, no row filtering')
    return v if sine<1e-12 else (theta/sine)*v
def delta52(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    rot=log_world(y[43:52].reshape(3,3)@x[43:52].reshape(3,3).T)
    return np.r_[y[:43]-x[:43],rot,y[52:]-x[52:]]
def apply_delta52(x,d):
    x=np.asarray(x,dtype=float);d=np.asarray(d,dtype=float);out=x.copy()
    out[:43]+=d[:43];out[43:52]=(exp_world(d[43:46])@x[43:52].reshape(3,3)).ravel();out[52:]+=d[46:]
    return out
def codec_roundtrip(x,ipos):
    r=state.from_com_world_state(x,np.asarray(ipos,dtype=float))
    return state.to_com_world_state(r['q20'],r['qd20'],r['p_root_world'],r['R_body_world'],r['v_root_world'],r['omega_body'],np.asarray(ipos,dtype=float))
