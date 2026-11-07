"""Unit-normal candidate force cones, distinct from observed native contacts."""
import numpy as np


def unit_normal_envelope(points_COM_relative,normals,*,mu=.7,edges=4):
    """BODY-axis rays, normal force=1; moment arms are unscaled metres.

    Tangent phase is fixed by a BODY-axis Frisvad frame. Rotating an existing
    envelope preserves that phase; rebuilding from rotated normals may not.
    These candidate rays have no hand-load, force capacity or contact truth.
    """
    p,n=np.asarray(points_COM_relative,float),np.asarray(normals,float)
    if p.shape!=n.shape or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(np.r_[p.ravel(),n.ravel()]).all():raise ValueError('Finite explicit[K,3] geometry')
    if not np.allclose(np.linalg.norm(n,axis=1),1,atol=1e-7) or not np.isfinite(mu) or mu<0 or edges not in (0,4,8):raise ValueError('Unit inward normals and explicit0/4/8-edge cone')
    forces=[n]
    if edges:
        sign=np.where(n[:,2]>=0,1.,-1.);a=-1/(sign+n[:,2]);b=n[:,0]*n[:,1]*a
        t1=np.c_[1+sign*n[:,0]**2*a,sign*b,-sign*n[:,0]];t1/=np.linalg.norm(t1,axis=1)[:,None]
        t2=np.c_[b,sign+n[:,1]**2*a,-n[:,1]];t2/=np.linalg.norm(t2,axis=1)[:,None]
        for theta in np.arange(edges)*2*np.pi/edges:forces.append(n+mu*(np.cos(theta)*t1+np.sin(theta)*t2))
    F=np.stack(forces,axis=1);T=np.cross(p[:,None,:],F);W=np.concatenate([F,T],axis=2).reshape(-1,6).T
    return W,np.repeat(np.arange(len(p)),len(forces))


def rotate_wrenches(wrench,rotation):
    W,R=np.asarray(wrench,float),np.asarray(rotation,float)
    if W.ndim!=2 or W.shape[0]!=6 or R.shape!=(3,3) or not np.isfinite(np.r_[W.ravel(),R.ravel()]).all() or np.max(abs(R.T@R-np.eye(3)))>3e-6 or np.linalg.det(R)<0:raise ValueError('6D COM wrenches and proper rigid rotation')
    return np.r_[R@W[:3],R@W[3:]]
