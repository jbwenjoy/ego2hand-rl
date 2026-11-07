"""One strict force-hard ideal support-witness LP, no native/backend execution.

Consumes one complete frozen minimax LP operand context. This searches the
original force/moment acceptance box directly instead of trading force error
against moment error. Ideal fixed-contact support is not native holding.
"""
import numpy as np
from scipy.optimize import linprog


def _finite(value,shape,name):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError('Finite '+name+' '+str(shape))
    return a


def solve_strict_support(lp_output,*,pd_gain20,pd_offset20,ctrl_range20x2,
                         moment_tolerance_Nm=1e-4,force_tolerance_N=1e-8):
    """At most one solve; no fallback, force-law solve, forward or state change."""
    p=np.asarray(lp_output['points_world'],dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<1 or not np.isfinite(p).all():
        raise ValueError('Finite nonempty original native contact points')
    N=len(p);K=4*N
    rays=_finite(lp_output['rays_world_per_normal_N'],(N,4,3),'original force rays')
    J=_finite(lp_output['J_hand_3x20'],(N,3,20),'original hand Jacobians')
    com=_finite(lp_output['COM_world'],(3,),'original COM')
    g=_finite(lp_output['gravity_world'],(3,),'original gravity')
    mass=float(lp_output['mass_kg'])
    if not np.isfinite(mass) or mass<=0:raise ValueError('Positive original mass')
    labels=list(lp_output['finger_groups']);groups=list(lp_output['group_names'])
    if (len(labels)!=N or any(type(v) is not str or not v for v in labels)
            or groups!=list(dict.fromkeys(labels))):
        raise ValueError('Original exact per-contact/group ordering')
    caps=_finite(lp_output['group_cap_N'],(len(groups),),'original grouped normal caps')
    if np.any(caps<=0):raise ValueError('Positive original group caps')
    bias=_finite(lp_output['bias_minus_passive20'],(20,),'bias minus passive')
    lo=_finite(lp_output['PD_torque_lower20'],(20,),'original torque lower')
    hi=_finite(lp_output['PD_torque_upper20'],(20,),'original torque upper')
    gain=_finite(pd_gain20,(20,),'same-context PD gain')
    offset=_finite(pd_offset20,(20,),'same-context PD offset')
    cr=_finite(ctrl_range20x2,(20,2),'same-context ctrl range')
    ar=_finite(lp_output['raw_actuator_force_range20x2'],(20,2),'actuator force range')
    jr=_finite(lp_output['raw_joint_force_range20x2'],(20,2),'joint force range')
    if not np.array_equal(cr,_finite(lp_output['raw_ctrl_range20x2'],(20,2),'original ctrl range')):
        raise ValueError('Passed control range must match minimax operand context')
    if np.any(gain<=0) or np.any(lo>hi) or any(np.any(r[:,0]>r[:,1]) for r in (cr,ar,jr)):
        raise ValueError('Positive gains and ordered original bounds')
    expected_lo=np.maximum.reduce((gain*cr[:,0]+offset,ar[:,0],jr[:,0]))
    expected_hi=np.minimum.reduce((gain*cr[:,1]+offset,ar[:,1],jr[:,1]))
    if not np.array_equal(lo,expected_lo) or not np.array_equal(hi,expected_hi):
        raise ValueError('Same-context PD/gain/offset/force intersection required')
    mt=float(moment_tolerance_Nm);ft=float(force_tolerance_N)
    if not np.isfinite([mt,ft]).all() or mt<=0 or ft<=0:
        raise ValueError('Explicit finite positive force/moment gates')
    B=np.concatenate([r.T for r in rays],axis=1)
    M=np.concatenate([np.cross(p[i]-com,rays[i]).T for i in range(N)],axis=1)
    T=np.concatenate([J[i].T@rays[i].T for i in range(N)],axis=1)
    S=np.zeros((len(groups),K))
    for i,label in enumerate(labels):S[groups.index(label),4*i:4*i+4]=1.
    target=-mass*g
    Aub=np.vstack((M,-M,T,-T,S))
    bub=np.r_[np.full(6,mt),hi-bias,bias-lo,caps]
    options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
    sol=linprog(np.ones(K),A_ub=Aub,b_ub=bub,A_eq=B,b_eq=target,
                bounds=[(0,None)]*K,method='highs',options=options)
    out={'schema':'strict_native_point_support_PD_witness:7','LP_calls':1,
        'status':'INFEASIBLE_STRICT_IDEAL_POINT_WITNESS' if sol.status==2 else 'UNKNOWN_STRICT_IDEAL_POINT_WITNESS',
        'solver_status':int(sol.status),'solver_message':sol.message,
        'objective':'min sum of all ray normal-load coefficients, with force3 equality and moment3 acceptance box hard',
        'contact_count':N,'finger_groups':labels,'group_names':groups,'group_cap_N':caps,
        'force_tolerance_N':ft,'moment_tolerance_Nm':mt,'bound_roundoff_tolerance':1e-9,
        'solver_options':options,'strict_force_moment_pass':False,'hard_bounds_pass':False,
        'force_gate_pass':False,'moment_gate_pass':False,'proposed_ctrl20':None,
        'force_residual_world_N':None,'moment_residual_world_Nm':None,
        'lambda4_per_contact_N':None,'forces_on_object_world_N':None,'tau_required20':None,
        'raw_inverse_ctrl20':None,'group_normal_N':None,
        'holding_qualified':False,'native_soft_equilibrium_qualified':False,
        'scope':'Fixed original contacts/rays/J and legal unsaturated PD/force intersection only; no native soft law, frictionloss bonus, dynamic or global geometry qualification.'}
    if sol.status!=0 or sol.x is None:return out
    lam=np.asarray(sol.x,dtype=np.float64)
    forces=np.einsum('ik,ikj->ij',lam.reshape(N,4),rays)
    force_res=B@lam-target;moment_res=M@lam;tau=bias+T@lam
    u=(tau-offset)/gain;normal=S@lam
    finite=bool(np.isfinite(np.r_[lam,forces.ravel(),force_res,moment_res,tau,u,normal]).all())
    bt=1e-9
    hard=bool(finite and np.min(lam)>=-bt and np.min(caps-normal)>=-bt
        and np.min(tau-lo)>=-bt and np.min(hi-tau)>=-bt
        and np.min(u-cr[:,0])>=-bt and np.min(cr[:,1]-u)>=-bt)
    fg=bool(finite and np.max(np.abs(force_res))<=ft)
    mg=bool(finite and np.max(np.abs(moment_res))<=mt)
    accepted=bool(hard and fg and mg)
    out.update(status='STRICT_IDEAL_POINT_SUPPORT_WITNESS' if accepted else 'NUMERICAL_WITNESS_OUTSIDE_ORIGINAL_GATE',
        strict_force_moment_pass=accepted,hard_bounds_pass=hard,force_gate_pass=fg,moment_gate_pass=mg,
        proposed_ctrl20=u.copy() if accepted else None,raw_inverse_ctrl20=u,
        force_residual_world_N=force_res,moment_residual_world_Nm=moment_res,
        lambda4_per_contact_N=lam.reshape(N,4),forces_on_object_world_N=forces,
        tau_required20=tau,group_normal_N=normal,normal_cap_slack_N=caps-normal,
        torque_slack_lower20=tau-lo,torque_slack_upper20=hi-tau,
        ctrl_slack_lower20=u-cr[:,0],ctrl_slack_upper20=cr[:,1]-u,
        minimum_lambda_N=float(np.min(lam)),total_normal_load_N=float(np.sum(lam)),
        all_solution_values_finite=finite)
    return out
