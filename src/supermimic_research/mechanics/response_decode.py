"""Decode declared world-left delta52 without repairing or clipping physical state."""
import numpy as np
from scipy.spatial.transform import Rotation
from .transition_records import state_rotation


def decode_response(source,prediction):
    s,p=np.asarray(source,float),np.asarray(prediction,float)
    if s.shape!=(55,) or p.shape!=(87,) or not np.isfinite(np.r_[s,p]).all():raise ValueError('Finite explicit state55 and response87')
    R=Rotation.from_rotvec(p[43:46]).as_matrix()@state_rotation(s)
    return np.r_[s[:43]+p[:43],R[:,0],R[:,1],s[49:]+p[46:52]]


def momentum_defect(source,next_state,prediction,mass,inertia_BODY,gravity,H=.02):
    R0,R1=state_rotation(source),state_rotation(next_state)
    delta_p=mass*(next_state[49:52]-source[49:52])
    delta_L=R1@inertia_BODY@R1.T@next_state[52:55]-R0@inertia_BODY@R0.T@source[52:55]
    return np.r_[delta_p-H*(R0@prediction[78:81]+mass*gravity),delta_L-H*(R0@prediction[81:84])]
