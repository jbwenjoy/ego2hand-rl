"""Hypothetical three-finger cone balance, not a native force prescription."""
import numpy as np
from scipy.optimize import linprog


def triplet_balance(wrench,target,ray_groups,*,minimum_normal_N=.1,radius_m=.1):
    W,t,g=np.asarray(wrench,float),np.asarray(target,float),np.asarray(ray_groups)
    if W.ndim!=2 or W.shape[0]!=6 or t.shape!=(6,) or g.shape!=(W.shape[1],) or not set(g)=={1,2,3} or not np.isfinite(np.r_[W.ravel(),t]).all():raise ValueError('Finite6D three-group cone')
    if not minimum_normal_N>=0 or not radius_m>0:raise ValueError('Explicit minimum normal and moment scale')
    eq=W.copy();eq[3:]/=radius_m;rhs=t.copy();rhs[3:]/=radius_m
    group_matrix=np.array([g==i for i in [1,2,3]],float)
    answer=linprog(np.ones(W.shape[1]),A_eq=eq,b_eq=rhs,A_ub=-group_matrix,b_ub=np.full(3,-minimum_normal_N),bounds=(0,None),method='highs')
    if not answer.success:return dict(success=False,status=int(answer.status),normal_coefficients_N=None,coefficients=None,group_normal_coefficients_N=None)
    lam=answer.x;error=W@lam-t;normal=group_matrix@lam
    if np.min(lam)<-1e-8 or np.max(abs(error[:3]))>1e-8 or np.max(abs(error[3:]))>1e-9 or np.min(normal)<minimum_normal_N-1e-8:raise ValueError('Triplet primal failed independent residual bounds')
    return dict(success=True,status=int(answer.status),normal_coefficients_N=float(lam.sum()),coefficients=lam,
        group_normal_coefficients_N=normal,force_error_N=error[:3],moment_error_Nm=error[3:])
