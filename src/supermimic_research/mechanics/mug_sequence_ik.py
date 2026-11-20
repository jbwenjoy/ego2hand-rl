"""Track object-relative human fingertip displacement from an identified robot grasp."""
import numpy as np
from scipy.optimize import least_squares
from .retargeting import HandTipIK


def anchored_targets(poses,source_tips,initial_robot_tips):
    local0=(initial_robot_tips-poses[0,:3,3])@poses[0,:3,:3]
    relative=local0[None]+source_tips-source_tips[:1]
    return np.einsum('nij,nkj->nki',poses[:,:3,:3],relative)+poses[:,None,:3,3]


def solve_trajectory(model,qinit,targets):
    engine=HandTipIK(model,qinit,targets[0],regularization=0.)
    q0=qinit[:20].copy();prev=q0.copy()
    np.testing.assert_allclose(engine.state(q0),targets[0],atol=1e-8)
    q=[q0];tips=[engine.state(q0)];ok=[True]
    for desired in targets[1:]:
        # Distances in cm; posture priors in radians around grasp and preceding solution.
        engine.targets=desired
        def residual(v):return np.r_[engine.residual(v)[:15],.05*(v-q0),.1*(v-prev)]
        def jac(v):return np.vstack([engine.jacobian(v)[:15],.05*np.eye(20),.1*np.eye(20)])
        result=least_squares(residual,prev,jac=jac,bounds=(engine.lo,engine.hi),max_nfev=100)
        prev=result.x.copy();q.append(prev);tips.append(engine.state(prev));ok.append(bool(result.success))
    return np.array(q),np.array(tips),ok
