"""Torque-bounded closest achievable contact wrench and minimum-change servo allocation."""
import mujoco
import numpy as np
from scipy.optimize import linprog
from .multiclip_wrench_screen import contact_rays


def closest_servo(m,d,body,hand,target,neutral,prior_offset):
    loads=[];reactions=[]
    for c in d.contact:
        g,h=map(int,c.geom);on=m.geom_bodyid[[g,h]]==body
        if c.dist>0 or on.sum()!=1:continue
        other=h if on[0] else g
        if other not in hand:continue
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jac(m,d,jp,jr,c.pos,int(m.geom_bodyid[other]))
        arm=c.pos-d.xipos[body]
        for ray in contact_rays(c,1 if on[1] else -1,d.xipos[body]):
            loads.append(ray);reactions.append((jp.T@ray[:3]+jr.T@(ray[3:]-np.cross(arm,ray[:3])))[:20])
    if not loads:return None,'empty',None
    A=np.asarray(loads).T;B=np.asarray(reactions).T;K=m.actuator_gainprm[:,0];bias=(d.qfrc_bias-d.qfrc_passive)[:20]
    lo=np.maximum(m.actuator_forcerange[:,0],K*(m.actuator_ctrlrange[:,0]-neutral));hi=np.minimum(m.actuator_forcerange[:,1],K*(m.actuator_ctrlrange[:,1]-neutral))
    # Report force error in N-equivalent using a 5 cm moment arm.
    scale=np.r_[np.ones(3),np.full(3,20.)];n=len(loads)
    eq=np.c_[scale[:,None]*A,np.eye(6),-np.eye(6),np.zeros((6,20))]
    C=np.r_[B,-B,B,-B];slack=np.r_[np.zeros((40,20)),-np.eye(20),-np.eye(20)]
    ub=np.c_[C,np.zeros((80,12)),slack];rhs=np.r_[hi-bias,bias-lo,K*prior_offset-bias,bias-K*prior_offset]
    c1=np.r_[np.zeros(n),np.ones(12),np.zeros(20)]
    def solve(cost,mat,r):
        return linprog(cost,A_ub=mat,b_ub=r,A_eq=eq,b_eq=scale*target,bounds=(0,None),method='highs')
    p=solve(c1,ub,rhs)
    if not p.success:return None,f'first_{p.status}',None
    q=solve(np.r_[np.full(n,.001),np.zeros(12),1/m.actuator_forcerange[:,1]],np.vstack([ub,c1]),np.r_[rhs,p.fun+1e-6])
    if not q.success:return None,f'second_{q.status}',None
    alpha=q.x[:n];tau=bias+B@alpha
    residual=A@alpha-target;err=np.abs(scale*residual).sum()
    if err>p.fun+2e-6 or np.any(tau<lo-1e-7) or np.any(tau>hi+1e-7):raise ValueError('Force allocation failed readback')
    detail=dict(L1_N_equivalent=float(err),force_error_N=float(np.linalg.norm(residual[:3])),torque_error_Nm=float(np.linalg.norm(residual[3:])),normal_sum_N=float(alpha.sum()))
    return neutral+tau/K,('ok' if p.fun<=1e-7 else 'approximate'),detail
