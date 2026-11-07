"""Ideal point-force proposal on a saved/fresh native contact design probe.

No model loading or physical advance. Native condim3 four-ray frames and
per-finger normal budgets are retained. Robot bias/passive, affine PD and true
force limits constrain the actuator torque needed for the proposed forces.
This ideal static LP is guidance, never native soft-equilibrium/holding proof.
"""
import numpy as np
from scipy.optimize import linprog


def finite(x,shape,name):
    a=np.asarray(x,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite '+name+' '+str(shape))
    return a


def solve_native_wrench(*,points_world,frames_rows,force_on_object_sign,contact_dims,
        friction5,J_hand_3x20,finger_groups,COM_world,mass_kg,
        robot_bias_minus_passive20,pd_gain20,pd_offset20,ctrl_range20x2,
        actuator_force_range20x2,joint_force_range20x2,
        normal_cap_per_finger_N=3.,gravity_world=(0.,0.,-9.81),length_m=.077):
    p=np.asarray(points_world,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or len(p)<1 or not np.isfinite(p).all():
        raise ValueError('At least one explicit finite native contact point')
    N=len(p);C=finite(frames_rows,(N,3,3),'native frames')
    if not np.all(finite(contact_dims,(N,),'native dimensions')==3):raise ValueError('Native condim3 only')
    sign=finite(force_on_object_sign,(N,),'object force signs')
    if not np.all(np.isin(sign,[-1.,1.])):raise ValueError('Explicit geom-order signs')
    if np.max(abs(C@C.transpose(0,2,1)-np.eye(3)))>1e-8 or np.any(np.linalg.det(C)<.99999999):raise ValueError('Right-handed orthonormal native frames')
    mu=finite(friction5,(N,5),'native friction5')
    if np.any(mu[:,:2]<0):raise ValueError('Nonnegative native tangential friction')
    J=finite(J_hand_3x20,(N,3,20),'hand contact Jacobians')
    com=finite(COM_world,(3,),'COM');g=finite(gravity_world,(3,),'gravity')
    if not np.isfinite(mass_kg) or mass_kg<=0 or not np.isfinite(length_m) or length_m<=0:raise ValueError('Positive scales')
    labels=list(finger_groups)
    if len(labels)!=N or any(type(v) is not str or not v for v in labels):raise ValueError('Explicit finger groups')
    groups=list(dict.fromkeys(labels))
    if isinstance(normal_cap_per_finger_N,dict):
        if set(normal_cap_per_finger_N)!=set(groups):raise ValueError('Exact cap mapping')
        caps=np.asarray([normal_cap_per_finger_N[k] for k in groups],dtype=float)
    else:caps=np.full(len(groups),float(normal_cap_per_finger_N))
    if np.any(caps<=0) or not np.isfinite(caps).all():raise ValueError('Finite positive design caps')
    bias=finite(robot_bias_minus_passive20,(20,),'robot bias minus passive')
    gain=finite(pd_gain20,(20,),'PD gain');offset=finite(pd_offset20,(20,),'PD offset')
    cr=finite(ctrl_range20x2,(20,2),'ctrl range')
    ar=finite(actuator_force_range20x2,(20,2),'actuator force range')
    jr=finite(joint_force_range20x2,(20,2),'joint force range')
    if np.any(gain<=0) or any(np.any(r[:,0]>r[:,1]) for r in (cr,ar,jr)):raise ValueError('Positive gain and true ordered limits')
    lo=np.maximum.reduce((gain*cr[:,0]+offset,ar[:,0],jr[:,0]))
    hi=np.minimum.reduce((gain*cr[:,1]+offset,ar[:,1],jr[:,1]))
    if np.any(lo>hi):raise ValueError('Empty true PD/force intersection')
    rays=np.empty((N,4,3))
    for i in range(N):
        rays[i]=sign[i]*np.array([C[i,0]-mu[i,0]*C[i,1],C[i,0]+mu[i,0]*C[i,1],
                                 C[i,0]-mu[i,1]*C[i,2],C[i,0]+mu[i,1]*C[i,2]])
    K=N*4
    F=np.zeros((3*N,K));S=np.zeros((len(groups),K))
    for i in range(N):F[3*i:3*i+3,4*i:4*i+4]=rays[i].T;S[groups.index(labels[i]),4*i:4*i+4]=1
    B=np.concatenate([r.T for r in rays],axis=1)
    M=np.concatenate([np.cross(p[i]-com,rays[i]).T for i in range(N)],axis=1)
    T=J.transpose(2,0,1).reshape(20,3*N)@F
    fs=float(mass_kg)*float(np.linalg.norm(g));ms=fs*float(length_m)
    if fs<=0:raise ValueError('Nonzero gravity scale')
    scale=np.r_[np.full(3,fs),np.full(3,ms)]
    W=np.vstack((B,M));target=np.r_[-float(mass_kg)*g,np.zeros(3)]
    A=W/scale[:,None];b=target/scale
    Aub=np.vstack((np.c_[A,-np.ones(6)],np.c_[-A,-np.ones(6)],
                   np.c_[T,np.zeros(20)],np.c_[-T,np.zeros(20)],np.c_[S,np.zeros(len(groups))]))
    bub=np.r_[b,-b,hi-bias,bias-lo,caps]
    objective=np.r_[np.zeros(K),1.]
    sol=linprog(objective,A_ub=Aub,b_ub=bub,bounds=[(0,None)]*(K+1),method='highs',
      options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
    out={'schema':'native_contact_ideal_wrench_PD_LP:7','LP_calls':1,
      'status':'UNKNOWN_LP','solver_status':int(sol.status),'solver_message':sol.message,
      'contact_count':N,'finger_groups':labels,'group_names':groups,'group_cap_N':caps,
      'points_world':p,'frames_rows':C,'force_on_object_sign':sign,'friction5':mu,
      'J_hand_3x20':J,'rays_world_per_normal_N':rays,'COM_world':com,'mass_kg':mass_kg,
      'gravity_world':g,'scale6':scale,'bias_minus_passive20':bias,
      'PD_torque_lower20':lo,'PD_torque_upper20':hi,
      'torque_interval_scope':'Raw affine-PD interval intersected with actuator/joint caps; legal unsaturated inverse-control witnesses. Not a general proof of all clamped native reachable torques.',
      'raw_ctrl_range20x2':cr,'raw_actuator_force_range20x2':ar,'raw_joint_force_range20x2':jr,
      'hard_bounds_pass':False,'strict_force_moment_pass':False,'scaled_loss':None,
      'holding_qualified':False,'native_soft_equilibrium_qualified':False,
      'scope':'Ideal fixed native point geometry/4ray force proposal, no direct torsion; real static PD/force bounds and grouped Fn design caps. No frictionloss bonus, native soft law, dynamic stability or actual holding proof.'}
    if sol.status!=0:return out
    lam=sol.x[:K];forces=(F@lam).reshape(N,3);tau=bias+T@lam
    residual=W@lam-target;u=(tau-offset)/gain
    violation=float(max(0.,np.max(Aub@sol.x-bub),-np.min(lam)))
    hard=bool(np.isfinite(sol.x).all() and violation<=1e-8
              and np.min(u-cr[:,0])>=-1e-9 and np.min(cr[:,1]-u)>=-1e-9)
    fg=bool(hard and max(abs(residual[:3]))<=1e-8);mg=bool(hard and max(abs(residual[3:]))<=1e-4)
    out.update(status='OPTIMAL_IDEAL_POINT_WRENCH_PROPOSAL' if hard else 'UNKNOWN_NUMERICAL_BOUNDS',
      lambda4_per_contact_N=lam.reshape(N,4),forces_on_object_world_N=forces,
      group_normal_N=S@lam,tau_required20=tau,proposed_ctrl20=u,
      force_residual_world_N=residual[:3],moment_residual_world_Nm=residual[3:],
      scaled_residual6=residual/scale,scaled_loss=float(max(abs(residual/scale))),
      solver_t=float(sol.x[-1]),max_inequality_violation=violation,
      torque_slack_lower20=tau-lo,torque_slack_upper20=hi-tau,
      hard_bounds_pass=hard,force_gate_pass=fg,moment_gate_pass=mg,strict_force_moment_pass=bool(fg and mg))
    return out
