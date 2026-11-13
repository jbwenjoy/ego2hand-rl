"""Pure-array source-BODY COM impulse decoder; host validation belongs at boundaries.

Pose heads remain independent. Small numerical impulse defects are explicit
supervised WORLD labels, never substituted for measured contact wrenches.
"""
import numpy as np
import torch


def exp_world_unchecked(v):
    a,b,c=v.unbind(-1);z=torch.zeros_like(a)
    K=torch.stack([z,-c,b,c,z,-a,-b,a,z],dim=-1).reshape(v.shape[:-1]+(3,3))
    t2=v.square().sum(-1);safe=t2.clamp_min(1e-6).sqrt();s=t2.clamp_max(1e-6)
    A=torch.where(t2<1e-6,1-s/6+s.square()/120-s.pow(3)/5040,torch.sinc(safe/torch.pi))
    B=torch.where(t2<1e-6,.5-s/24+s.square()/720-s.pow(3)/40320,.5*torch.sinc(safe/(2*torch.pi)).square())
    return torch.eye(3,dtype=v.dtype,device=v.device)+A[...,None,None]*K+B[...,None,None]*(K@K)


def rotation55_torch(s):
    a,b=s[...,43:46],s[...,46:49]
    return torch.stack([a,b,torch.linalg.cross(a,b)],dim=-1)


def mv(M,v):return (M@v.unsqueeze(-1)).squeeze(-1)


def inverse3x3_unchecked(M):
    a,b,c=M.unbind(-1);bc=torch.linalg.cross(b,c);det=(a*bc).sum(-1)
    return torch.stack([bc,torch.linalg.cross(c,a),torch.linalg.cross(a,b)],dim=-2)/det[...,None,None]


def close_velocities(x,heads,mass,inertia_BODY,inverse_inertia_BODY,gravity,defect_world=None,H=.02):
    # Same-device F64 physical arithmetic, static3x3 products, no solve/check/sync.
    s,h=x[...,:55].double(),heads.double();R0=rotation55_torch(s);R1=exp_world_unchecked(h[...,43:46])@R0
    if defect_world is None:defect_world=torch.zeros_like(h[...,46:52])
    force,tau=mv(R0,h[...,78:81]),mv(R0,h[...,81:84]);I0=R0@inertia_BODY@R0.transpose(-1,-2)
    velocity=s[...,49:52]+H*(force/mass+gravity)+defect_world[...,:3]/mass
    momentum=mv(I0,s[...,52:55])+H*tau+defect_world[...,3:]
    inverse_R1=inverse3x3_unchecked(R1)
    omega=mv(inverse_R1.transpose(-1,-2)@inverse_inertia_BODY@inverse_R1,momentum)
    return torch.cat([h[...,:46],velocity-s[...,49:52],omega-s[...,52:55],h[...,52:]],dim=-1)


def observed_defect_numpy(x,y,mass,inertia_BODY,gravity,H=.02):
    from .response_decode import decode_response,momentum_defect
    return np.array([momentum_defect(s[:55],decode_response(s[:55],p),p,mass,inertia_BODY,gravity,H) for s,p in zip(x,y)])
