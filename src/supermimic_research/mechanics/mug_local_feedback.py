"""Phase-dependent small-error feedback around free-mug trajectory witnesses."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def displacement(m,reference,current):
    """Tangent position error for the free body and revolute hand joints."""
    dq=np.empty(m.nv)
    mujoco.mj_differentiatePos(m,dq,1.,reference,current)
    return dq


def tangent_errors(m,nominal,states):
    q=nominal[1:1+m.nq];v=nominal[1+m.nq:]
    return np.stack([np.r_[displacement(m,q,s[1:1+m.nq]),s[1+m.nq:]-v] for s in states])


def local_dynamics(m,workers,engine,start,warm,cmd,end,qa,da):
    """Centered differences, using the feasible chord for clipped actuator probes."""
    nx=2*m.nv;nu=m.nu
    scale=np.r_[np.full(m.nv,.0005),np.full(m.nv,.005),np.full(nu,.002)]
    scale[da:da+3]=.00005;scale[m.nv+da:m.nv+da+3]=.0005
    diff=np.r_[np.diag(scale),-np.diag(scale)]
    states=np.tile(start,(len(diff),1))
    for s,dx in zip(states,diff):
        mujoco.mj_integratePos(m,s[1:1+m.nq],dx[:m.nv].copy(),1.)
        s[1+m.nq:]+=dx[m.nv:nx]
    cs=np.clip(cmd+diff[:,nx:],*m.actuator_ctrlrange.T)
    candidate,_=engine.rollout(m,workers,states,control=np.repeat(cs[:,None,:],64,axis=1),
        initial_warmstart=np.tile(warm,(len(states),1)))
    final=candidate[:,-1]
    error=tangent_errors(m,end,final)
    span=scale*2
    span[nx:]=np.diag(cs[nx:nx+nu]-cs[2*nx+nu:])
    assert (span>0).all() and np.isfinite(error).all()
    J=((error[:len(span)]-error[len(span):])/span[:,None]).T
    return J[:,:nx],J[:,nx:],cs[nx:nx+nu],cs[2*nx+nu:],len(diff)*64


def learned_object_rows(chart,net,start,warm,plus,minus,end,B):
    """Replace only the mug free-joint output rows of the native action Jacobian."""
    import torch
    cmd=np.vstack([plus,minus])
    with torch.inference_mode():
        pred=net(torch.from_numpy(chart.features(start,warm,cmd))).numpy()
    com,R,load=chart.decode(start,pred)
    m=chart.model;body=chart.body;mass=m.body_mass[body]
    Q=Rotation.from_quat(m.body_iquat[body,[1,2,3,0]]).as_matrix()
    inertia=Q@np.diag(m.body_inertia[body])@Q.T
    _,r0,v0,wb0=chart.motion(start)
    v=v0+.02*(load[:,:3]/mass+m.opt.gravity)
    Lworld=r0.apply(inertia@wb0)+.02*load[:,3:]
    wb=np.linalg.solve(inertia,R.inv().apply(Lworld).T).T
    lever=R.apply(chart.com)
    states=np.tile(end,(len(cmd),1))
    qa=1+chart.qa;da=1+m.nq+chart.da
    states[:,qa:qa+3]=com-lever
    states[:,qa+3:qa+7]=R.as_quat()[:,[3,0,1,2]]
    states[:,da:da+3]=v-np.cross(R.apply(wb),lever)
    states[:,da+3:da+6]=wb
    errs=tangent_errors(m,end,states)
    jac=((errs[:m.nu]-errs[m.nu:])/np.diag(plus-minus)[:,None]).T
    rows=np.r_[chart.da:chart.da+6,m.nv+chart.da:m.nv+chart.da+6]
    changed=B.copy();changed[rows]=jac[rows]
    return changed


def gains(A,B,m,da):
    """Finite-horizon Riccati recursion in MuJoCo's tangent coordinates."""
    q=np.r_[np.full(m.nv,1/(20*.10**2)),np.full(m.nv,1/(20*4.**2))]
    q[da:da+3]=1/(3*.005**2);q[da+3:da+6]=1/(3*np.deg2rad(8)**2)
    q[m.nv+da:m.nv+da+3]=1/(3*.20**2);q[m.nv+da+3:m.nv+da+6]=1/(3*3.**2)
    Q=np.diag(q);R=np.eye(m.nu)/(20*.06**2)
    S=Q.copy();result=[]
    for a,b in zip(A[::-1],B[::-1]):
        K=np.linalg.solve(R+b.T@S@b,b.T@S@a)
        closed=a-b@K
        S=Q+K.T@R@K+closed.T@S@closed
        S=(S+S.T)*.5
        if not np.isfinite(S).all():raise FloatingPointError('Nonfinite tracking gain')
        result.append(K)
    return np.array(result[::-1])
