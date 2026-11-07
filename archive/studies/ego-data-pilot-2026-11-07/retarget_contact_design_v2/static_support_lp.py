"""Pure conditional point-contact support LP; no simulator/FK/optimizer model.

Force normals are ON THE OBJECT. With hand-gap normal +n pointing phone->hand,
pass -n here. Default four rays match condim3 pyramidal encoding in the supplied
tangent frame; an unmeasured deterministic frame is not a native frame proof.
"""
import math
import numpy as np
from scipy.optimize import linprog


def _array(value, shape, name):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(name+' finite shape '+str(shape)+' required')
    return a


def _frame(n,tangent1,tangent2):
    count=len(n)
    if (tangent1 is None)!=(tangent2 is None):raise ValueError('Both tangent arrays must be supplied together')
    if tangent1 is not None:
        t1=_array(tangent1,(count,3),'tangent1_world');t2=_array(tangent2,(count,3),'tangent2_world')
        for i in range(count):
            M=np.stack((t1[i],t2[i],n[i]))
            if not np.allclose(M@M.T,np.eye(3),rtol=0,atol=1e-9) or np.linalg.det(M)<0:
                raise ValueError('Caller tangent frame must already be right-handed orthonormal; no silent repair')
        return t1,t2,'caller_supplied_not_independently_verified_native'
    t1=np.empty_like(n);t2=np.empty_like(n)
    for i,normal in enumerate(n):
        axis=np.eye(3)[int(np.argmin(np.abs(normal)))]
        t1[i]=np.cross(normal,axis);t1[i]/=np.linalg.norm(t1[i]);t2[i]=np.cross(normal,t1[i])
    return t1,t2,'deterministic_hypothetical_tangent_frame'


def static_support_lp(contact_points_world,normals_force_on_object,object_COM,mass,mu,
                      J_robot_contact_3x20=None,robot_gravity_bias_tau=None,
                      torque_lower=None,torque_upper=None,*,Jtrans=None,
                      mode='native_pyramidal4',tangent1_world=None,tangent2_world=None,
                      gravity_world=(0.,0.,-9.81)):
    """Minimize total normal N subject to COM wrench and optional20-joint bounds.

    Caller MUST reject remote contacts, wrong pad patches, interpenetration and
    unknown geometry before this call; no gap/holding validity is inferred here.
    robot_gravity_bias_tau is actual caller-supplied zero-speed bias; this routine
    never invents it. Contact reaction ON HAND is -f_object, hence required
    tau_act=bias+sum(J^T f_object). Static dry-friction assistance is ignored.
    Jtrans optionally maps stacked object forces directly to20 joint torques.
    Bounds must already represent true actuator/joint/PD-command reachability.
    """
    points=np.asarray(contact_points_world,dtype=np.float64)
    if points.ndim!=2 or points.shape[1]!=3 or len(points)<1 or not np.isfinite(points).all():
        raise ValueError('Finite N x3 contact points required')
    count=len(points);raw_n=_array(normals_force_on_object,(count,3),'normals_force_on_object')
    norms=np.linalg.norm(raw_n,axis=1)
    if np.any(norms<=0) or np.max(np.abs(norms-1))>1e-8:
        raise ValueError('Input normals must be explicitly unit length within1e-8')
    n=raw_n/norms[:,None]  # explicit, returned normalization; never reflects hand/object
    com=_array(object_COM,(3,),'object_COM');gravity=_array(gravity_world,(3,),'gravity_world')
    if not np.isfinite(mass) or mass<=0:raise ValueError('Positive finite mass required')
    mus=np.broadcast_to(np.asarray(mu,dtype=np.float64),(count,)).copy()
    if not np.isfinite(mus).all() or np.any(mus<0):raise ValueError('Finite nonnegative mu required')
    t1,t2,frame_source=_frame(n,tangent1_world,tangent2_world)
    if mode=='native_pyramidal4':
        xy=np.array([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    elif mode=='ideal_inscribed8':
        theta=np.arange(8)*math.pi/4;xy=np.column_stack((np.cos(theta),np.sin(theta)))
    else:raise ValueError('mode must be native_pyramidal4 or ideal_inscribed8')
    rays=n[:,None,:]+mus[:,None,None]*(xy[None,:,0,None]*t1[:,None,:]+xy[None,:,1,None]*t2[:,None,:])
    K=len(xy);columns=count*K;F=np.zeros((3*count,columns))
    for i in range(count):F[3*i:3*i+3,K*i:K*i+K]=rays[i].T
    force_matrix=np.concatenate([rays[i].T for i in range(count)],axis=1)
    lever=points-com
    moment_matrix=np.concatenate([np.cross(lever[i],rays[i]).T for i in range(count)],axis=1)
    W=np.vstack((force_matrix,moment_matrix));required=np.r_[-float(mass)*gravity,np.zeros(3)]
    map_tau=None
    if J_robot_contact_3x20 is not None:
        J=_array(J_robot_contact_3x20,(count,3,20),'J_robot_contact_3x20')
        map_tau=J.transpose(2,0,1).reshape(20,3*count)
    if Jtrans is not None:
        supplied=_array(Jtrans,(20,3*count),'Jtrans')
        if map_tau is not None and not np.allclose(supplied,map_tau,rtol=0,atol=1e-12):raise ValueError('Jtrans disagrees with supplied contact Jacobians')
        map_tau=supplied
    torque_checked=map_tau is not None
    any_torque_fields=any(x is not None for x in [robot_gravity_bias_tau,torque_lower,torque_upper])
    if torque_checked:
        if any(x is None for x in [robot_gravity_bias_tau,torque_lower,torque_upper]):raise ValueError('Actual bias and both torque bounds are mandatory when J is supplied')
        bias=_array(robot_gravity_bias_tau,(20,),'robot_gravity_bias_tau')
        lower=_array(torque_lower,(20,),'torque_lower');upper=_array(torque_upper,(20,),'torque_upper')
        if np.any(lower>upper):raise ValueError('Inverted torque bounds')
        Tmap=map_tau@F;Aub=np.vstack((Tmap,-Tmap));bub=np.r_[upper-bias,bias-lower]
    else:
        if any_torque_fields:raise ValueError('Torque bounds/bias without J cannot establish actuator feasibility')
        bias=lower=upper=Tmap=Aub=bub=None
    result=linprog(np.ones(columns),A_ub=Aub,b_ub=bub,A_eq=W,b_eq=required,
                   bounds=(0,None),method='highs',
                   options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    output={'schema':'conditional_point_contact_static_support_LP:1','mode':mode,
       'cone_geometry_label':'specified_pyramidal4' if mode=='native_pyramidal4' else 'ideal_circular_Coulomb_inscribed8',
       'solver':'scipy.optimize.linprog/highs','solver_status':int(result.status),'solver_message':result.message,
       'mass_kg':float(mass),'mu':mus.tolist(),'contact_points_world_m':points.tolist(),'object_COM_world_m':com.tolist(),
       'gravity_world_m_s2':gravity.tolist(),'normal_input_norms':norms.tolist(),'normals_force_on_object_unit':n.tolist(),
       'tangent1_world':t1.tolist(),'tangent2_world':t2.tolist(),'tangent_frame_source':frame_source,
       'native_contact_frame_verified_by_module':False,'rays_force_world_per_normal_N':rays.tolist(),
       'normal_normalized_ray_count_per_contact':K,'direct_torsion_or_rolling_allowed':False,
       'actuator_bounds_checked':torque_checked,'actual_bias_synthesized':False,
       'robot_gravity_bias_tau_Nm':bias.tolist() if torque_checked else None,
       'torque_lower_Nm':lower.tolist() if torque_checked else None,'torque_upper_Nm':upper.tolist() if torque_checked else None,
       'Jtrans_object_force_to_joint_torque':map_tau.tolist() if torque_checked else None,
       'PD_command_reachability_independently_verified_by_module':False,
       'static_frictionloss_assistance_used':False,'required_contact_COM_wrench_N_Nm':required.tolist(),
       'force_residual_tolerance_N':1e-8,'moment_residual_tolerance_Nm':1e-9,'torque_bound_roundoff_tolerance_Nm':1e-9,
       'feasible':False,'force_world_N':None,'lambda_normal_N':None,'total_normal_force_N':None,
       'COM_wrench_residual_N_Nm':None,'tau_act_required_Nm':None,'torque_slack_lower_Nm':None,'torque_slack_upper_Nm':None,
       'holding_or_grip_qualified':False,'real_parameters_calibrated':False,
       'scope':'Conditional point-contact statics only. Caller must enroll near closed actual pad contacts and hard nonpenetration first. Four-ray encoding needs an explicitly matched native tangent frame to claim native representation; eight-ray mode is only an inner approximation of ideal circular Coulomb friction, not the native pyramid. No arbitrary torsional support. Bias/J/limits are caller evidence, not computed here. Infeasible point-model/frame/limit assumptions do not prove that a real finite mesh patch or a later explicitly modeled torsional patch cannot hold.'}
    if result.status==0:
        lam=np.asarray(result.x,dtype=np.float64);forces=(F@lam).reshape(count,3);residual=W@lam-required
        tau=bias+map_tau@forces.ravel() if torque_checked else None
        low_slack=tau-lower if torque_checked else None;up_slack=upper-tau if torque_checked else None
        good=(np.min(lam)>=-1e-9 and np.max(np.abs(residual[:3]))<=1e-8 and np.max(np.abs(residual[3:]))<=1e-9)
        if torque_checked:good=good and np.min(low_slack)>=-1e-9 and np.min(up_slack)>=-1e-9
        output.update(feasible=bool(good),force_world_N=forces.tolist(),lambda_normal_N=lam.reshape(count,K).tolist(),
          total_normal_force_N=float(np.sum(lam)),COM_wrench_residual_N_Nm=residual.tolist(),
          tau_act_required_Nm=tau.tolist() if torque_checked else None,
          torque_slack_lower_Nm=low_slack.tolist() if torque_checked else None,
          torque_slack_upper_Nm=up_slack.tolist() if torque_checked else None,
          minimum_raw_lambda_N=float(lam.min()),normal_force_per_contact_N=np.sum(lam.reshape(count,K),axis=1).tolist())
    return output
