"""Explicit public Actor observation; native solver state is not an input."""
import numpy as np
from .transition_records import state_rotation

SCHEMA='mug_actor144_state55_EMArad20_prevA20_phase1_relative_future3x15_valid3:1'
OFFSETS=(1,3,5)


def actor_observation(state55,EMA20,previous_action20,k,reference,*,task='REFERENCE'):
    s,m,a=[np.asarray(x,dtype=np.float64) for x in [state55,EMA20,previous_action20]]
    if s.shape!=(55,) or m.shape!=(20,) or a.shape!=(20,) or not np.isfinite(np.r_[s,m,a]).all() or np.any(abs(a)>1) or type(k) is not int or not 0<=k<100 or task not in ('HOLD','REFERENCE'):raise ValueError('Actual state55, physical-radian EMA20, previous bounded action20 and phase')
    R=state_rotation(s)
    if np.max(abs(R.T@R-np.eye(3)))>1e-8:raise ValueError('Raw proper observed orientation required')
    future=[];valid=[]
    for offset in OFFSETS:
        index=0 if task=='HOLD' else min(k+offset,100);valid.append(float(k+offset<=100));T=reference['T_world_from_BODY'][index];relative_R=R.T@T[:3,:3]
        goal_v=np.zeros(3) if task=='HOLD' else reference['COM_velocity_world_m_s'][index];goal_w=np.zeros(3) if task=='HOLD' else reference['angular_velocity_world_rad_s'][index]
        future.append(np.r_[R.T@(reference['COM_world_m'][index]-s[40:43]),relative_R[:,0],relative_R[:,1],R.T@(goal_v-s[49:52]),R.T@(goal_w-s[52:55])])
    result=np.r_[s,m,a,k/100.,np.concatenate(future),valid]
    if result.shape!=(144,) or not np.isfinite(result).all():raise ValueError('Finite declared public144 packet')
    return result
