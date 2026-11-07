"""Zero-velocity native response objective, separate from force residual scales."""
import numpy as np
from .native_projection import ForwardForceProjection,common_penalties


def zero_velocity_COM_acceleration(model,data,body,free_dof):
    if np.any(data.qvel):raise ValueError('This projection explicitly requires zero velocity')
    R=data.xmat[body].reshape(3,3);angular=R@data.qacc[free_dof+3:free_dof+6]
    linear=data.qacc[free_dof:free_dof+3]+np.cross(angular,data.xipos[body]-data.xpos[body])
    return linear,angular


class AccelerationProjection(ForwardForceProjection):
    def residual(self,z,s):
        linear,angular=zero_velocity_COM_acceleration(self.model,self.data,self.oid,self.dadr)
        return np.r_[s['qacc26'][:20]/.5,linear/.005,angular/.1,common_penalties(s,z)]
