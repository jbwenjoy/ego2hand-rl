"""Explicit compound H20 input/target chart shared by new diagnostic callers."""
import numpy as np
from scipy.spatial.transform import Rotation


def state_rotation(state):
    a,b=state[...,43:46],state[...,46:49]
    return np.stack([a,b,np.cross(a,b)],axis=-1)


def transition_xy(transition,dt):
    s,n=transition['source55'],transition['next55']
    delta=np.r_[n[:43]-s[:43],Rotation.from_matrix(state_rotation(n)@state_rotation(s).T).as_rotvec(),n[49:]-s[49:]]
    return (np.r_[s,transition['warmstart26_before'],transition['decision']['actual_control20'],dt],
            np.r_[delta,transition['warmstart26_after'],transition['mean_native_wrench_source_BODY_COM'],transition['mean_native_group_Fn_N']])
