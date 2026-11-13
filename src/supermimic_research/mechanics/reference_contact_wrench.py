"""Nominal inverse-dynamics contact targets in the object's BODY COM frame."""
import numpy as np
from scipy.spatial.transform import Rotation


def reference_contact_wrench(COM,quaternion_wxyz,mass_kg,inertia_BODY,gravity_world,dt):
    p=np.asarray(COM,float);q=np.asarray(quaternion_wxyz,float);I=np.asarray(inertia_BODY,float);g=np.asarray(gravity_world,float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<4 or q.shape!=(len(p),4) or I.shape!=(3,3) or g.shape!=(3,):raise ValueError('Finite reference shapes and at least4 original-clock samples')
    if not np.isfinite(np.r_[p.ravel(),q.ravel(),I.ravel(),g,mass_kg,dt]).all() or mass_kg<=0 or dt<=0:raise ValueError('Finite positive mass/dt')
    if np.max(abs(np.linalg.norm(q,axis=1)-1))>1e-5 or not np.allclose(I,I.T,rtol=0,atol=1e-12) or np.linalg.eigvalsh(I).min()<=0:raise ValueError('Unit reference quaternions and symmetric positive inertia')
    R=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();count=len(p)-3
    velocity=(-11*p[:count]+18*p[1:count+1]-9*p[2:count+2]+2*p[3:count+3])/(6*dt)
    acceleration=(2*p[:count]-5*p[1:count+1]+4*p[2:count+2]-p[3:count+3])/dt**2
    local=[]
    for j in [1,2,3]:
        rot=Rotation.from_matrix(R[j:j+count]@np.swapaxes(R[:count],-1,-2)).as_rotvec()
        if np.max(np.linalg.norm(rot,axis=1))>=np.pi-1e-3:raise ValueError('Reference window crosses ambiguous SO3 log branch')
        local.append(rot)
    r1,r2,r3=local
    omega=(18*r1-9*r2+2*r3)/(6*dt);alpha=(-5*r1+4*r2-r3)/dt**2
    world_I=R[:count]@I@np.swapaxes(R[:count],-1,-2)
    Iomega=np.einsum('nij,nj->ni',world_I,omega)
    force=mass_kg*(acceleration-g);torque=np.einsum('nij,nj->ni',world_I,alpha)+np.cross(omega,Iomega)
    body=np.c_[np.einsum('nji,nj->ni',R[:count],force),np.einsum('nji,nj->ni',R[:count],torque)]
    return dict(frame_indices=np.arange(count),time_s=np.arange(count)*dt,wrench_BODY_COM=body,force_world=force,torque_COM_world=torque,
        velocity_world=velocity,acceleration_world=acceleration,omega_world=omega,alpha_world=alpha,
        valid_source_mask=np.arange(len(p))<count,
        estimator='Four-sample forward stencil at ORIGINAL frame: position derivatives and local-world SO3 log derivatives at each frame. No smoothing/extrapolation; last3 frames invalid.')
