"""Pure R/F algebra and decision gates; no geometry or optimizer imports."""
import numpy as np
from scipy import sparse

DEPTH_LIMIT_MM=2.999
DEPTH_ARITHMETIC_TOL_MM=1e-9
LINEAR_ARITHMETIC_TOL_NATIVE=1e-12
BOX_ARITHMETIC_TOL_RAD=1e-12
R_TRUST_RADIUS_RAD=.05
R_INITIAL_SLACK_BUFFER_MM=1e-6
GN_MU=1e-4
F_STEP_RADIUS_RAD=.02
DIRECTION_ZERO_RAD=1e-12
ARMIJO_C1=1e-4
ALPHAS=(1.,.5,.25,.125,.0625,.03125)


def finite(x,shape=None):
    x=np.asarray(x,dtype=np.float64)
    if (shape is not None and x.shape!=shape) or not np.isfinite(x).all():raise ValueError('Nonfinite or wrong shape')
    return x


def restoration_box(seed,lower,upper):
    seed,lower,upper=(finite(x,(340,)) for x in (seed,lower,upper))
    lo=np.maximum(lower,seed-R_TRUST_RADIUS_RAD);hi=np.minimum(upper,seed+R_TRUST_RADIUS_RAD)
    if np.any(lo>hi):raise ValueError('Empty restoration trust box; never project seed')
    return lo,hi


def restoration_z0(seed,depth_mm):
    seed=finite(seed,(340,));depth=finite(depth_mm,(122,))
    return np.r_[seed,max(0.,float(np.max(depth-DEPTH_LIMIT_MM)))+R_INITIAL_SLACK_BUFFER_MM]


def restoration_objective(z):return float(finite(z,(341,))[-1])
def restoration_gradient(z):finite(z,(341,));return np.r_[np.zeros(340),1.]
def restoration_geometry_slack(z,depth_mm):return DEPTH_LIMIT_MM+finite(z,(341,))[-1]-finite(depth_mm,(122,))
def restoration_geometry_jacobian(depth_J):
    C=sparse.csr_matrix(depth_J)
    if C.shape!=(122,340) or not np.isfinite(C.data).all():raise ValueError('Wrong restoration geometry Jacobian')
    return np.c_[-C.toarray(),np.ones(122)]


def primal_audit(x,depth_mm,contract,*,trust_lower=None,trust_upper=None):
    """Original physical-planning inequalities, independent of R slack/status."""
    x=finite(x,(340,));depth=finite(depth_mm,(122,))
    lr=np.asarray(contract['A_ub']@(x-contract['x0'])-contract['b_centered']).ravel()
    if not np.isfinite(lr).all():raise ValueError('Nonfinite original linear residual')
    lower=finite(contract['lower'],(340,));upper=finite(contract['upper'],(340,))
    box=np.maximum(np.maximum(lower-x,x-upper),0.)
    dv=depth-DEPTH_LIMIT_MM
    trust=None
    if (trust_lower is None)!=(trust_upper is None):raise ValueError('Both restoration trust bounds required')
    if trust_lower is not None:
        trust=np.maximum(np.maximum(finite(trust_lower,(340,))-x,x-finite(trust_upper,(340,))),0.)
    exact={'depth':bool(np.all(dv<=0)),'linear':bool(np.all(lr<=0)),'box':bool(np.all(box==0)),'restoration_trustbox':True if trust is None else bool(np.all(trust==0))}
    declared={'depth':bool(np.all(dv<=DEPTH_ARITHMETIC_TOL_MM)),'linear':bool(np.all(lr<=LINEAR_ARITHMETIC_TOL_NATIVE)),'box':bool(np.all(box<=BOX_ARITHMETIC_TOL_RAD)),'restoration_trustbox':True if trust is None else bool(np.all(trust<=BOX_ARITHMETIC_TOL_RAD))}
    return {'passed':all(declared.values()),'exact_raw_all_pass':all(exact.values()),'exact_raw':exact,'declared_arithmetic':declared,
            'max_depth_mm':float(depth.max()),'max_depth_minus_2_999_mm':float(dv.max()),'sum_positive_depth_violation_mm':float(np.maximum(dv,0).sum()),'max_linear_Ax_minus_b_native':float(lr.max()),'max_coefficient_box_excess_rad':float(box.max()),'max_R_trustbox_excess_rad':None if trust is None else float(trust.max()),
            'arithmetic_tolerances':{'depth_mm':DEPTH_ARITHMETIC_TOL_MM,'linear_each_native_unit':LINEAR_ARITHMETIC_TOL_NATIVE,'coefficient_and_R_trust_rad':BOX_ARITHMETIC_TOL_RAD},
            'scope':'Original122 sampled generated-overlap/operator plus original linears/box; artificial R slack and solver status excluded. No positive-gap/dynamics qualification.'}


def gauss_newton_model(r,J):
    r=finite(r);J=sparse.csr_matrix(J)
    if r.shape!=(6596,) or J.shape!=(6596,340) or not np.isfinite(J.data).all():raise ValueError('Wrong unchanged fit residual/J')
    g=np.asarray(J.T@r).ravel();H=(J.T@J).toarray()+GN_MU*np.eye(340)
    return g,H


def qp_contract(x,depth_mm,C,contract):
    x=finite(x,(340,));depth=finite(depth_mm,(122,));C=sparse.csr_matrix(C)
    if C.shape!=(122,340) or not np.isfinite(C.data).all():raise ValueError('Wrong QP geometric Jacobian')
    A=contract['A_ub'];linear_rhs=contract['b_centered']-A@(x-contract['x0'])
    lower=np.maximum(contract['lower']-x,-F_STEP_RADIUS_RAD);upper=np.minimum(contract['upper']-x,F_STEP_RADIUS_RAD)
    if np.any(lower>upper):raise ValueError('Empty QP step box')
    return {'A_original':A,'b_original':np.asarray(linear_rhs).ravel(),'C_geometry':C,'b_geometry':DEPTH_LIMIT_MM-depth,'lower':lower,'upper':upper,'x0':np.zeros(340),
            'scope':'F has original coefficient box and per-major0.02rad step box; R seed trustbox is NOT imposed in F.'}


def qp_direction_audit(step,g,qp):
    p=finite(step,(340,));g=finite(g,(340,));lv=np.asarray(qp['A_original']@p-qp['b_original']).ravel();dv=np.asarray(qp['C_geometry']@p-qp['b_geometry']).ravel();bv=np.maximum(np.maximum(qp['lower']-p,p-qp['upper']),0.)
    if not np.isfinite(lv).all() or not np.isfinite(dv).all():raise ValueError('Nonfinite QP residual')
    slope=float(g@p);nonzero=bool(np.max(np.abs(p))>DIRECTION_ZERO_RAD)
    raw={'linear':bool(np.all(lv<=0)),'linearized_depth':bool(np.all(dv<=0)),'step_and_coefficient_bounds':bool(np.all(bv==0))}
    numeric={'linear':bool(np.all(lv<=LINEAR_ARITHMETIC_TOL_NATIVE)),'linearized_depth':bool(np.all(dv<=DEPTH_ARITHMETIC_TOL_MM)),'step_and_coefficient_bounds':bool(np.all(bv<=BOX_ARITHMETIC_TOL_RAD))}
    return {'eligible':all(numeric.values()) and nonzero and slope<0,'linearized_primal_pass':all(numeric.values()),'exact_raw':raw,'declared_arithmetic':numeric,'step_abs_max_rad':float(np.max(np.abs(p))),'zero_threshold_rad':DIRECTION_ZERO_RAD,'nonzero_direction':nonzero,'g_dot_step':slope,'strict_descent':slope<0,'max_original_linear_residual_native':float(lv.max()),'max_linearized_depth_residual_mm':float(dv.max()),'max_step_box_excess_rad':float(bv.max()),'scope':'Normally returned QP terminal step only; solver status and theoretical GN descent do not replace this independent audit.'}


def armijo_decision(current_cost,trial_cost,alpha,g_dot_step,raw_primal,*,trial_x_changed):
    if float(alpha) not in ALPHAS:raise ValueError('Only frozen six alphas')
    if not np.isfinite([current_cost,trial_cost,alpha,g_dot_step]).all():
        return {'accepted':False,'reason':'NONFINITE_COST_OR_SLOPE','primal_pass':bool(raw_primal['passed']),'trial_x_changed':bool(trial_x_changed),'actual_cost_strictly_decreases':False,'Armijo_rhs':None,'Armijo_pass':False,'g_dot_step':float(g_dot_step) if np.isfinite(g_dot_step) else None,'alpha':float(alpha),'c1':ARMIJO_C1}
    rhs=float(current_cost+ARMIJO_C1*alpha*g_dot_step)
    return {'accepted':bool(raw_primal['passed'] and trial_x_changed and g_dot_step<0 and trial_cost<current_cost and trial_cost<=rhs),'primal_pass':bool(raw_primal['passed']),'trial_x_changed':bool(trial_x_changed),'actual_cost_strictly_decreases':bool(trial_cost<current_cost),'Armijo_rhs':rhs,'Armijo_pass':bool(trial_cost<=rhs),'g_dot_step':float(g_dot_step),'alpha':float(alpha),'c1':ARMIJO_C1}
