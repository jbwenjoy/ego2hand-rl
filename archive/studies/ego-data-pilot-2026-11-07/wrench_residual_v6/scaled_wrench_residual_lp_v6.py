"""Pure scaled six-D wrench proposal LP; no force-hard feasibility switch.

No model/FK/geometry/physics. Normal budgets group by explicit finger labels,
not by number of surface sample points. No torsion/rolling contact variables.
"""
import math
import numpy as np
from scipy.optimize import linprog


def array(x,shape,name):
    a=np.asarray(x,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite '+name+' '+str(shape)+' required')
    return a


def scaled_wrench_residual_lp(contact_points_world,normals_force_on_object,
        object_COM,mass,mu_actual,J_robot_contact_3x20,robot_gravity_bias_tau,
        torque_lower,torque_upper,*,finger_group_ids,characteristic_length_m=.077,
        normal_cap_per_finger_N=3.0,gravity_world=(0.,0.,-9.81),
        tangent_reference_axis_world=(0.,1.,0.),tangent1_world=None,tangent2_world=None):
    """Minimize max(|sumF+mg|/Fscale, |sum(r cross F)|/Mscale).

    Fscale=mass*9.81N; Mscale=Fscale*characteristic_length_m Nm. Hard constraints
    are nonnegative cone coefficients, grouped normal caps, and supplied true
    torque bounds. Caller must separately enroll actual current paired geometry.
    """
    p=np.asarray(contact_points_world,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<1 or not np.isfinite(p).all():raise ValueError('Finite N by3 points required')
    N=len(p);raw_n=array(normals_force_on_object,(N,3),'normals');norm=np.linalg.norm(raw_n,axis=1)
    if np.any(norm<=0) or max(abs(norm-1))>1e-8:raise ValueError('Explicit unit force-on-object normals required')
    n=raw_n/norm[:,None];COM=array(object_COM,(3,),'COM');g=array(gravity_world,(3,),'gravity')
    if not np.isfinite(mass) or mass<=0 or not np.isfinite(characteristic_length_m) or characteristic_length_m<=0:raise ValueError('Positive mass and explicit length required')
    mu=np.broadcast_to(np.asarray(mu_actual,dtype=float),(N,)).copy()
    if not np.isfinite(mu).all() or np.any(mu<0):raise ValueError('Nonnegative finite mu required')
    labels=list(finger_group_ids)
    if len(labels)!=N or any(type(x) is not str or not x for x in labels):raise ValueError('Explicit nonempty string finger label for every point required')
    groups=list(dict.fromkeys(labels));group_index=np.array([groups.index(x) for x in labels])
    if isinstance(normal_cap_per_finger_N,dict):
        if set(normal_cap_per_finger_N)!=set(groups):raise ValueError('Exact finger-cap mapping required')
        caps=np.array([normal_cap_per_finger_N[x] for x in groups],dtype=float)
    else:caps=np.full(len(groups),float(normal_cap_per_finger_N))
    if not np.isfinite(caps).all() or np.any(caps<=0):raise ValueError('Positive finite grouped normal caps required')
    J=array(J_robot_contact_3x20,(N,3,20),'current paired-hand point Jacobians')
    bias=array(robot_gravity_bias_tau,(20,),'actual robot bias');lo=array(torque_lower,(20,),'torque lower');hi=array(torque_upper,(20,),'torque upper')
    if np.any(lo>hi):raise ValueError('Inverted true torque interval')
    ref=array(tangent_reference_axis_world,(3,),'reference axis')
    if np.linalg.norm(ref)<=0:raise ValueError('Zero reference axis')
    ref=ref/np.linalg.norm(ref);frame_branches=[]
    if (tangent1_world is None)!=(tangent2_world is None):raise ValueError('Both supplied tangent arrays required')
    if tangent1_world is not None:
        t1=array(tangent1_world,(N,3),'tangent1');t2=array(tangent2_world,(N,3),'tangent2')
        for i in range(N):
            frame=np.stack((t1[i],t2[i],n[i]))
            if not np.allclose(frame@frame.T,np.eye(3),rtol=0,atol=1e-8) or np.linalg.det(frame)<0:raise ValueError('Supplied frame not right-handed orthonormal')
        frame_branches=['SUPPLIED_FROZEN_FRAME']*N
    else:
        t1=np.empty_like(n);t2=np.empty_like(n)
        for i,nn in enumerate(n):
            axis=ref;tt=axis-nn*(nn@axis)
            branch='REFERENCE_AXIS_PROJECTION'
            if np.linalg.norm(tt)<=1e-8:
                axis=np.eye(3)[int(np.argmin(abs(nn)))];tt=axis-nn*(nn@axis)
                branch='EXPLICIT_LEAST_ALIGNED_AXIS_POLE_BRANCH'
            t1[i]=tt/np.linalg.norm(tt);t2[i]=np.cross(nn,t1[i]);frame_branches.append(branch)
    theta=np.arange(8)*math.pi/4;mud=mu/math.sqrt(2)
    rays=n[:,None,:]+mud[:,None,None]*(np.cos(theta)[None,:,None]*t1[:,None,:]+np.sin(theta)[None,:,None]*t2[:,None,:])
    K=N*8;F=np.zeros((3*N,K));S=np.zeros((len(groups),K))
    for i in range(N):F[3*i:3*i+3,8*i:8*i+8]=rays[i].T;S[group_index[i],8*i:8*i+8]=1
    B=np.concatenate([rays[i].T for i in range(N)],axis=1)
    M=np.concatenate([np.cross(p[i]-COM,rays[i]).T for i in range(N)],axis=1)
    JT=J.transpose(2,0,1).reshape(20,3*N);T=JT@F
    Fscale=float(mass)*9.81;Mscale=Fscale*float(characteristic_length_m)
    W=np.vstack((B/Fscale,M/Mscale));target=np.r_[-float(mass)*g/Fscale,np.zeros(3)]
    Aub=np.vstack((np.c_[W,-np.ones(6)],np.c_[-W,-np.ones(6)],np.c_[T,np.zeros(20)],np.c_[-T,np.zeros(20)],np.c_[S,np.zeros(len(groups))]))
    bub=np.r_[target,-target,hi-bias,bias-lo,caps]
    objective=np.r_[np.zeros(K),1.]
    result=linprog(objective,A_ub=Aub,b_ub=bub,bounds=[(0,None)]*(K+1),method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    out={'schema':'scaled_six_D_wrench_residual_LP:6','solver_status':int(result.status),'solver_message':result.message,
      'objective':'min maximum absolute scaled force3 and moment3 components; t is dimensionless',
      'status':'UNKNOWN_NO_TORQUE_FEASIBLE_CONE' if result.status==2 else 'UNKNOWN_SOLVER_ERROR',
      'contact_points_world_m':p.tolist(),'normals_force_on_object':n.tolist(),'input_normal_norms':norm.tolist(),'object_COM_world_m':COM.tolist(),
      'mass_kg':float(mass),'gravity_world_m_s2':g.tolist(),'characteristic_length_m':float(characteristic_length_m),'Fscale_N':Fscale,'Mscale_Nm':Mscale,
      'mu_actual':mu.tolist(),'mu_design':mud.tolist(),'finger_group_ids':labels,'group_names':groups,'group_cap_N':caps.tolist(),
      'tangent1_world':t1.tolist(),'tangent2_world':t2.tolist(),'tangent_frame_branches':frame_branches,'rays_force_world_per_normal_N':rays.tolist(),
      'J_robot_contact_3x20':J.tolist(),'actual_bias_Nm':bias.tolist(),'torque_lower_Nm':lo.tolist(),'torque_upper_Nm':hi.tolist(),
      'lambda_count':K,'LP_calls':1,'hard_tau_group_cone_feasible':False,'scaled_loss':None,'solver_t':None,
      'force_residual_world_N':None,'moment_residual_world_Nm':None,'scaled_residual6':None,
      'force_residual_gate':False,'moment_residual_gate':False,'both_residual_gates_pass':False,
      'force_gate_N':1e-8,'moment_gate_Nm':1e-4,'strict_static_support_qualified':False,'holding_qualified':False,
      'scope':'Point forces only, no torsion/rolling. Inscribed8 in circular mu/sqrt(2) is inside any tangent-rotated native condim3 equal-mu4-ray cone. Explicit chart branches may be piecewise smooth. Low scaled value is a proposal value, not strict support, current-controller enrollment, physical calibration or holding. Geometry/J/bias/bounds are caller evidence.'}
    if result.status!=0:return out
    lam=np.asarray(result.x[:K]);force=(F@lam).reshape(N,3);fn=np.sum(lam.reshape(N,8),axis=1);group_fn=S@lam
    force_res=B@lam+float(mass)*g;moment=M@lam;scaled=np.r_[force_res/Fscale,moment/Mscale]
    tau=bias+T@lam;lower_slack=tau-lo;upper_slack=hi-tau;group_slack=caps-group_fn
    hard=bool(lam.min()>=-1e-9 and min(lower_slack)>=-1e-9 and min(upper_slack)>=-1e-9 and min(group_slack)>=-1e-9)
    loss=float(max(abs(scaled)));fg=bool(hard and max(abs(force_res))<=1e-8);mg=bool(hard and max(abs(moment))<=1e-4)
    out.update(status='OPTIMAL_SCALED_WRENCH_PROPOSAL' if hard else 'UNKNOWN_NUMERICAL_HARD_BOUND_FAILURE',
      hard_tau_group_cone_feasible=hard,lambda_normal_N=lam.reshape(N,8).tolist(),normal_force_per_point_N=fn.tolist(),normal_force_per_group_N=group_fn.tolist(),group_cap_slack_N=group_slack.tolist(),
      force_world_N=force.tolist(),contact_force_sum_world_N=force.sum(0).tolist(),gravity_force_world_N=(float(mass)*g).tolist(),
      force_residual_world_N=force_res.tolist(),moment_residual_world_Nm=moment.tolist(),force_infinity_N=float(max(abs(force_res))),moment_infinity_Nm=float(max(abs(moment))),
      scaled_residual6=scaled.tolist(),scaled_loss=loss,solver_t=float(result.x[-1]),six_residual_slack=(result.x[-1]-abs(scaled)).tolist(),
      tau_act_required_Nm=tau.tolist(),torque_slack_lower_Nm=lower_slack.tolist(),torque_slack_upper_Nm=upper_slack.tolist(),
      force_residual_gate=fg,moment_residual_gate=mg,both_residual_gates_pass=bool(fg and mg),
      inequality_dual_marginals=result.ineqlin.marginals.tolist(),lower_bound_dual_marginals=result.lower.marginals.tolist(),
      primal_dual_objective_gap=float(result.fun-bub@result.ineqlin.marginals))
    return out
