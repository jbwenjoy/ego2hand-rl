"""Nominal interval-average WORLD COM contact wrench from reference momentum."""
import numpy as np
from scipy.spatial.transform import Rotation

from .reference_contact_wrench import reference_contact_wrench


def mean_reference_wrench(COM,quaternion_wxyz,mass,inertia_BODY,gravity,dt=.02):
    prior=reference_contact_wrench(COM,quaternion_wxyz,mass,inertia_BODY,gravity,dt)
    count=len(prior['velocity_world']);R=Rotation.from_quat(np.asarray(quaternion_wxyz)[:count,[1,2,3,0]]).as_matrix()
    I=np.asarray(inertia_BODY);world_I=R@I@R.transpose(0,2,1)
    angular_momentum=np.einsum('nij,nj->ni',world_I,prior['omega_world'])
    targets=np.zeros((len(COM)-1,6));valid=np.zeros(len(COM)-1,dtype=bool)
    targets[:count-1,:3]=mass*(np.diff(prior['velocity_world'],axis=0)/dt-np.asarray(gravity))
    targets[:count-1,3:]=np.diff(angular_momentum,axis=0)/dt;valid[:count-1]=True
    return dict(schema='nominal_mean_interval_WORLD_COM_wrench:1',wrench_world=targets,valid_interval=valid,
        velocity_world=prior['velocity_world'],omega_world=prior['omega_world'],angular_momentum_world=angular_momentum,
        estimator=prior['estimator'],scope='Mean interval force/torque by reference momentum change; nominal simulator prior, not measured source force. Original clock; last3 control intervals invalid, no extrapolation.')
