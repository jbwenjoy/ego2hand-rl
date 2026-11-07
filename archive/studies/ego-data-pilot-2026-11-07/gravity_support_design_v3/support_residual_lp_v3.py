"""Conditional support proposal value: hard force/actuator bounds, soft moment.

Pure NumPy/SciPy. No model/geometry/FK, no pose changes, no integration.
This LP never labels nonzero moment residual as strict static support.
"""
import math
import numpy as np
from scipy.optimize import linprog


def array(x,shape,name):
    a=np.asarray(x,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite '+name+' '+str(shape)+' required')
    return a


def support_residual_lp(contact_points_world,normals_force_on_object,object_COM,
        mass,mu_actual,J_robot_contact_3x20,robot_gravity_bias_tau,torque_lower,torque_upper,
        *,normal_cap_N=3.0,gravity_world=(0.,0.,-9.81),
        tangent_reference_axis_world=(0.,1.,0.)):
    """Minimize max absolute COM contact-moment component, in Nm.

    Contacts must be current genuine paired witnesses; caller enrolls geometry.
    Points are OBJECT witnesses. J is at the paired HAND point, first20 DoFs.
    Fixed reference-axis projection gives a smooth tangent chart away from its
    pole. At pole/invalid input: raise, never silently switch chart or repair.
    Torque bounds are caller's actual PD/actuator/joint reachable intersection.
    """
    p=np.asarray(contact_points_world,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<1 or not np.isfinite(p).all():raise ValueError('Finite N by3 points required')
    N=len(p);n0=array(normals_force_on_object,(N,3),'normals')
    norm=np.linalg.norm(n0,axis=1)
    if np.any(norm<=0) or np.max(np.abs(norm-1))>1e-8:raise ValueError('Explicit unit normals required')
    n=n0/norm[:,None];com=array(object_COM,(3,),'COM');g=array(gravity_world,(3,),'gravity')
    if not np.isfinite(mass) or mass<=0:raise ValueError('Positive mass required')
    mu=np.broadcast_to(np.asarray(mu_actual,dtype=float),(N,)).copy()
    cap=np.broadcast_to(np.asarray(normal_cap_N,dtype=float),(N,)).copy()
    if not np.isfinite(mu).all() or np.any(mu<0) or not np.isfinite(cap).all() or np.any(cap<=0):raise ValueError('Valid friction and normal caps required')
    J=array(J_robot_contact_3x20,(N,3,20),'current hand-point Jacobians')
    bias=array(robot_gravity_bias_tau,(20,),'actual bias')
    lo=array(torque_lower,(20,),'torque lower');hi=array(torque_upper,(20,),'torque upper')
    if np.any(lo>hi):raise ValueError('Inverted torque bounds')
    ref=array(tangent_reference_axis_world,(3,),'fixed tangent reference axis')
    if np.linalg.norm(ref)<=0:raise ValueError('Zero reference axis')
    ref=ref/np.linalg.norm(ref);t1=ref-(n@ref)[:,None]*n;tn=np.linalg.norm(t1,axis=1)
    if np.any(tn<=1e-8):raise ValueError('Tangent chart pole: unknown, no per-pose chart switching')
    t1/=tn[:,None];t2=np.cross(n,t1)
    theta=np.arange(8)*math.pi/4;mu_d=mu/math.sqrt(2)
    rays=n[:,None,:]+mu_d[:,None,None]*(np.cos(theta)[None,:,None]*t1[:,None,:]+np.sin(theta)[None,:,None]*t2[:,None,:])
    K=N*8;F=np.zeros((3*N,K));S=np.zeros((N,K))
    for i in range(N):F[3*i:3*i+3,8*i:8*i+8]=rays[i].T;S[i,8*i:8*i+8]=1
    B=np.concatenate([rays[i].T for i in range(N)],axis=1)
    M=np.concatenate([np.cross(p[i]-com,rays[i]).T for i in range(N)],axis=1)
    Jtrans=J.transpose(2,0,1).reshape(20,3*N);T=Jtrans@F
    # Decision x=[lambda(N*8), t_Nm]. Only t enters the objective: no hidden
    # secondary squeeze objective or interpretation of a nonunique force split.
    Aeq=np.c_[B,np.zeros(3)];beq=-float(mass)*g
    Aub=np.vstack([np.c_[M,-np.ones(3)],np.c_[-M,-np.ones(3)],
                   np.c_[T,np.zeros(20)],np.c_[-T,np.zeros(20)],np.c_[S,np.zeros(N)]])
    bub=np.r_[np.zeros(6),hi-bias,bias-lo,cap]
    objective=np.r_[np.zeros(K),1.]
    result=linprog(objective,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,
        bounds=[(0,None)]*(K+1),method='highs',
        options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    out={'schema':'conditional_support_minimum_moment_residual_LP:3',
        'solver_status':int(result.status),'solver_message':result.message,
        'objective':'min max(abs(COM contact moment components)) [Nm]; force3 equality and torque/cap bounds hard',
        'contact_points_world_m':p.tolist(),'normals_force_on_object':n.tolist(),'input_normal_norms':norm.tolist(),
        'object_COM_world_m':com.tolist(),'mass_kg':float(mass),'gravity_world_m_s2':g.tolist(),
        'mu_actual':mu.tolist(),'mu_design':mu_d.tolist(),'normal_cap_N':cap.tolist(),
        'tangent_reference_axis_world':ref.tolist(),'tangent1_world':t1.tolist(),'tangent2_world':t2.tolist(),
        'rays_force_world_per_normal_N':rays.tolist(),'lambda_count':K,
        'cone_scope':'inscribed8 in circular mu_actual/sqrt(2) cone; L1<=sqrt(2)*L2 implies inclusion in any tangent-rotated native condim3 sliding4-ray cone of equal mu. No torsion/rolling.',
        'feasible_force_balance':False,'proposal_gate':False,'strict_static_support_qualified':False,'holding_qualified':False,
        'force_recovery':None,'LP_calls':1,
        'proposal_moment_limit_Nm':1e-4,'force_equality_tolerance_N':1e-8,
        'hard_bound_numerical_tolerance':1e-9,
        'lambda_normal_N':None,'force_world_N':None,'moment_residual_world_Nm':None,'moment_infinity_Nm':None,
        'force_balance_residual_world_N':None,'tau_act_required_Nm':None,
        'force_solution_uniqueness':'Not asserted; only the optimum residual value is the optimization objective',
        'scope':'Current geometry and limits are caller evidence. Positive gaps are only potential closed contacts. Small residual is an engineering pose proposal, not exact statics, stable holding or a2second5degree guarantee. Final exact LP and real finite-mesh rollout remain separate.'}
    if result.status!=0:
        out['status']='PRIMARY_FORCE_HARD_LP_INFEASIBLE' if result.status==2 else 'PRIMARY_SOLVER_ERROR_UNKNOWN'
        if result.status!=2:return out
        # Proposal guidance only: retain all torque/cap bounds, soften force
        # into3 component inequalities. Never substitute this for primary LP.
        AR=np.vstack([np.c_[B,-np.ones(3)],np.c_[-B,-np.ones(3)],
            np.c_[T,np.zeros(20)],np.c_[-T,np.zeros(20)],np.c_[S,np.zeros(N)]])
        br=np.r_[beq,-beq,hi-bias,bias-lo,cap]
        recovery=linprog(objective,A_ub=AR,b_ub=br,bounds=[(0,None)]*(K+1),method='highs',
            options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        out['LP_calls']=2
        R={'solver_status':int(recovery.status),'solver_message':recovery.message,
           'objective':'min force equality infinity residual [N]; torque/caps remain hard, moment not constrained',
           'status':'NO_TORQUE_FEASIBLE_CONE' if recovery.status==2 else 'RECOVERY_SOLVER_ERROR_UNKNOWN',
           'qualified':False,'feasible_force_balance':False,'force_infinity_N':None}
        if recovery.status==0:
            lam=np.asarray(recovery.x[:K]);forces=(F@lam).reshape(N,3);fr=B@lam-beq;tau=bias+T@lam
            R.update(status='FINITE_FORCE_RECOVERY_GUIDANCE_ONLY',lambda_normal_N=lam.reshape(N,8).tolist(),
                force_world_N=forces.tolist(),force_balance_residual_world_N=fr.tolist(),
                force_infinity_N=float(max(abs(fr))),solver_force_slack_t_N=float(recovery.x[-1]),
                moment_residual_world_Nm=(M@lam).tolist(),normal_force_N=(S@lam).tolist(),normal_cap_slack_N=(cap-S@lam).tolist(),
                tau_act_required_Nm=tau.tolist(),torque_slack_lower_Nm=(tau-lo).tolist(),torque_slack_upper_Nm=(hi-tau).tolist())
        out['force_recovery']=R
        return out
    x=np.asarray(result.x);lam=x[:K];forces=(F@lam).reshape(N,3);moment=M@lam
    fr=B@lam-beq;tau=bias+T@lam;fn=S@lam
    low=tau-lo;high=hi-tau;cap_slack=cap-fn
    valid=bool(lam.min()>=-1e-9 and max(abs(fr))<=1e-8 and low.min()>=-1e-9 and high.min()>=-1e-9 and cap_slack.min()>=-1e-9)
    out.update(status='OPTIMAL_PRIMARY_MOMENT_RESIDUAL' if valid else 'PRIMARY_NUMERICAL_HARD_BOUND_FAILURE',
        feasible_force_balance=valid,proposal_gate=bool(valid and max(abs(moment))<=1e-4),
        lambda_normal_N=lam.reshape(N,8).tolist(),force_world_N=forces.tolist(),
        normal_force_N=fn.tolist(),normal_cap_slack_N=cap_slack.tolist(),
        moment_residual_world_Nm=moment.tolist(),moment_infinity_Nm=float(max(abs(moment))),
        solver_slack_objective_t_Nm=float(x[-1]),moment_slack_component_Nm=(x[-1]-abs(moment)).tolist(),
        force_balance_residual_world_N=fr.tolist(),tau_act_required_Nm=tau.tolist(),
        torque_slack_lower_Nm=low.tolist(),torque_slack_upper_Nm=high.tolist(),
        torque_lower_Nm=lo.tolist(),torque_upper_Nm=hi.tolist(),actual_bias_Nm=bias.tolist())
    return out
