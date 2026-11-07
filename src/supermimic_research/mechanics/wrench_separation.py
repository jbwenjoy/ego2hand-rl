"""Numerical separating witnesses for a fixed nonnegative wrench-ray cone."""
import numpy as np
from scipy.optimize import linprog


def separating_witness(wrench,target,*,radius_m=.1):
    """Find y with W.T@y>=0, target@y=-1 in explicit force/moment units.

    A checked witness separates only this frozen ray cone. It says nothing
    about changing contacts, global task feasibility or real source forces.
    """
    W,t=np.asarray(wrench,float),np.asarray(target,float)
    if W.ndim!=2 or W.shape[0]!=6 or t.shape!=(6,) or not np.isfinite(np.r_[W.ravel(),t]).all() or not np.isfinite(radius_m) or radius_m<=0:raise ValueError('Finite6D rays/target and positive moment scale')
    scaled=W.copy();scaled[3:]/=radius_m;s=t.copy();s[3:]/=radius_m
    # Split the free dual vector into positive and negative parts; minimize L1.
    answer=linprog(np.ones(12),A_ub=np.c_[-scaled.T,scaled.T] if W.shape[1] else None,
        b_ub=np.zeros(W.shape[1]) if W.shape[1] else None,A_eq=np.r_[s,-s][None],b_eq=[-1.],bounds=(0,None),method='highs')
    if not answer.success:return dict(verified=False,solver_status=int(answer.status),message=str(answer.message),witness=None)
    y=answer.x[:6]-answer.x[6:];dots=scaled.T@y;target_dot=float(s@y)
    minimum=float(dots.min()) if len(dots) else None
    valid=(minimum is None or minimum>=-1e-9) and abs(target_dot+1)<1e-9 and np.isfinite(y).all()
    if not valid:raise ValueError('Solver returned an invalid numerical separating witness')
    dimensional=y.copy();dimensional[3:]/=radius_m
    return dict(verified=True,solver_status=int(answer.status),radius_m=radius_m,
        witness_scaled6=y.tolist(),witness_dual_to_F_N_T_Nm=dimensional.tolist(),minimum_ray_dot=minimum,target_dot=target_dot,
        separating_tolerance=1e-9,scope='Checked numerical certificate for the supplied frozen native rays only; not global infeasibility, an applied twist, force estimate or physical policy gradient.')
