"""Fixed-wrist reference conversion and instantaneous pyramidal hand wrench screen."""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation


def fixed_reference(wrist,object_pose,fixed):
    wrist_R=Rotation.from_quat(wrist[:,[4,5,6,3]])
    fixed_R=Rotation.from_quat(fixed[[4,5,6,3]])
    transform=fixed_R*wrist_R.inv()
    return (fixed[:3]+transform.apply(object_pose[:,:3]-wrist[:,:3]),
            (transform*Rotation.from_quat(object_pose[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]],transform)


def contact_rays(contact,sign,com):
    frame=contact.frame.reshape(3,3)
    dim=contact.dim
    directions=[]
    for j in range(1,dim):
        for side in (-1,1):
            local=np.zeros(6);local[0]=1;local[j]=side*contact.friction[j-1]
            directions.append(local)
    if dim==1:directions=[np.array([1.,0,0,0,0,0])]
    rays=[]
    for local in directions:
        force=sign*frame.T@local[:3]
        torque=sign*frame.T@local[3:]+np.cross(contact.pos-com,force)
        rays.append(np.r_[force,torque])
    return rays


def min_normal(rays,target):
    if not len(rays):
        return 0. if np.linalg.norm(target)<1e-12 else None
    A=np.array(rays).T
    # Scale torque rows for numerical conditioning; recheck in SI units.
    scale=np.array([1,1,1,100,100,100])
    res=linprog(np.ones(A.shape[1]),A_eq=scale[:,None]*A,b_eq=scale*target,bounds=(0,None),method='highs')
    if res.status==2:return None
    if not res.success:raise RuntimeError(res.message)
    residual=A@res.x-target
    if max(abs(residual[:3]))>1e-6 or max(abs(residual[3:]))>1e-8:raise ValueError('Invalid force-cone solution')
    return float(res.fun)
