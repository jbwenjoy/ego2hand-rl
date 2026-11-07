"""Frozen native contact/actuator support LPs, never applied-force witnesses.

These omit compliance and contact evolution. A feasible primal only describes
this fixed geometry with freely chosen normal coefficients and stated torque
bounds; infeasibility does not prove dynamic or global grasp impossibility.
"""
import numpy as np
from scipy.optimize import linprog


def pyramid3(mu):
    """Unit-normal rays of the condim3 native pyramidal friction convention."""
    mu=np.asarray(mu,float)
    if mu.shape!=(2,) or not np.isfinite(mu).all() or np.any(mu<0):
        raise ValueError('Two finite nonnegative tangent friction values')
    return np.array([[1.,mu[0],0],[1.,-mu[0],0],[1.,0,mu[1]],[1.,0,-mu[1]]])


def torque_box(model,data,control_lower,control_upper):
    lower,upper=np.asarray(control_lower),np.asarray(control_upper)
    if lower.shape!=(20,) or upper.shape!=(20,) or np.any(lower>upper) or np.any(lower<model.actuator_ctrlrange[:,0]) or np.any(upper>model.actuator_ctrlrange[:,1]):
        raise ValueError('Declared legal command interval, no implicit projection')
    gain=model.actuator_gainprm[:,0]
    bias=model.actuator_biasprm[:,0]+model.actuator_biasprm[:,1]*data.actuator_length+model.actuator_biasprm[:,2]*data.actuator_velocity
    cap_lo=np.maximum(model.actuator_forcerange[:,0],model.jnt_actfrcrange[:20,0])
    cap_hi=np.minimum(model.actuator_forcerange[:,1],model.jnt_actfrcrange[:20,1])
    if np.any(gain<=0):raise ValueError('Fixed positive affine PD gain required')
    return np.clip(gain*lower+bias,cap_lo,cap_hi),np.clip(gain*upper+bias,cap_lo,cap_hi)


def contact_basis(model,data,parts,body,allowed):
    import mujoco
    parts,allowed=set(map(int,parts)),set(map(int,allowed));body=int(body)
    if int(model.opt.cone)!=int(mujoco.mjtCone.mjCONE_PYRAMIDAL):raise ValueError('Pyramidal model required')
    columns=[];loads=[];records=[];actual_hand=np.zeros(20)
    size=mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION)
    before=np.empty(size);after=np.empty(size)
    mujoco.mj_getState(model,data,before,mujoco.mjtState.mjSTATE_INTEGRATION)
    for ci in range(data.ncon):
        con=data.contact[ci];a,b=map(int,con.geom);oa,ob=a in parts,b in parts
        if oa==ob or con.efc_address<0:continue
        other=a if ob else b
        if other not in allowed:raise ValueError('Nonintent/table contact cannot silently provide support')
        if con.dim!=3:raise ValueError('Native condim3 point forces only')
        sign=1 if ob else -1;frame=con.frame.reshape(3,3)
        jac=np.zeros((3,model.nv));mujoco.mj_jac(model,data,jac,None,con.pos.copy(),int(model.geom_bodyid[other]))
        force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
        world=sign*(frame.T@force[:3]);actual_hand-=jac[:,:20].T@world
        for ray in pyramid3(con.friction[:2]):
            F=sign*(frame.T@ray);columns.append(np.r_[F,np.cross(con.pos-data.xipos[body],F)])
            loads.append(jac[:,:20].T@F)
        records.append(dict(contact_index=ci,geom_ids=[a,b],other_geom=other,distance_m=float(con.dist),
            position_world_m=con.pos.copy(),normal_on_object_world=sign*frame[0],friction_tangents=con.friction[:2].copy(),
            local_force6=force,hand_jacobian3x20=jac[:,:20].copy()))
    mujoco.mj_getState(model,data,after,mujoco.mjtState.mjSTATE_INTEGRATION)
    if before.tobytes()!=after.tobytes():raise ValueError('Read-only contact/Jacobian query changed integration state')
    return dict(wrench=np.array(columns).reshape(-1,6).T,loads=np.array(loads).reshape(-1,20).T,
        contacts=records,actual_hand_contact_torque20=actual_hand,
        other_constraint_torque20=data.qfrc_constraint[:20]-actual_hand,
        jacobian_queries_preserve_integration_state=True)


def static_feasibility(wrench,target,*,loads=None,offset=None,lower=None,upper=None,friction=None,force_only=False,radius=.1):
    W=np.asarray(wrench,float);target=np.asarray(target,float);n=W.shape[1]
    if not n:return dict(success=False,status='EMPTY_ACTIVE_CONTACT_SET',witness=None)
    Aeq=W[:3].copy() if force_only else W.copy();rhs=target[:3].copy() if force_only else target.copy()
    if not force_only:Aeq[3:]/=radius;rhs[3:]/=radius
    A=b=None
    if loads is not None:
        f=np.zeros(20) if friction is None else np.asarray(friction)
        A=np.r_[loads,-loads];b=np.r_[upper+f-offset,offset-lower+f]
    result=linprog(np.ones(n),A_eq=Aeq,b_eq=rhs,A_ub=A,b_ub=b,bounds=(0,None),method='highs')
    answer=dict(success=bool(result.success),status=int(result.status),message=result.message,witness=None)
    if result.success:
        residual=W@result.x-target;moment=residual[3:]
        if np.max(np.abs(residual[:3]))>1e-8 or (not force_only and np.max(np.abs(moment))>1e-9):
            raise ArithmeticError('LP primal force/moment residual')
        tau=None;margin=None
        if loads is not None:
            tau=offset+loads@result.x;f=np.zeros(20) if friction is None else np.asarray(friction)
            margin=float(np.minimum(tau-(lower-f),(upper+f)-tau).min())
            if margin < -1e-8:raise ArithmeticError('LP primal torque bound')
        answer['witness']=dict(ray_coefficients_N=result.x,total_normal_coefficients_N=float(result.fun),
            force_residual_N=residual[:3],moment_residual_Nm=moment,required_torque20_Nm=tau,minimum_torque_margin_Nm=margin)
    return answer


def maximum_support(wrench,loads,offset,lower,upper,friction,*,direction=(0,0,1),radius=.1):
    W=np.asarray(wrench);n=W.shape[1]
    if not n:return dict(success=False,status='EMPTY_ACTIVE_CONTACT_SET',support_weight_N=None,witness=None)
    direction=np.asarray(direction,float)
    if direction.shape!=(3,) or abs(np.linalg.norm(direction)-1)>1e-10:raise ValueError('Unit force direction')
    target=np.r_[direction,np.zeros(3)];eq=np.c_[W,-target];eq[3:]/=radius
    f=np.asarray(friction);A=np.r_[np.c_[loads,np.zeros(20)],np.c_[-loads,np.zeros(20)]]
    b=np.r_[upper+f-offset,offset-lower+f];cost=np.zeros(n+1);cost[-1]=-1
    result=linprog(cost,A_eq=eq,b_eq=np.zeros(6),A_ub=A,b_ub=b,bounds=(0,None),method='highs')
    answer=dict(success=bool(result.success),status=int(result.status),message=result.message,support_weight_N=None,witness=None)
    if result.success:
        weight=float(result.x[-1]);lam=result.x[:-1];tau=offset+loads@lam;error=W@lam-target*weight
        margin=float(np.minimum(tau-(lower-f),(upper+f)-tau).min())
        if np.max(np.abs(error[:3]))>1e-8 or np.max(np.abs(error[3:]))>1e-9 or margin < -1e-8:
            raise ArithmeticError('Maximum support primal failed')
        answer.update(support_weight_N=weight,witness=dict(ray_coefficients_N=lam,required_torque20_Nm=tau,
            force_residual_N=error[:3],moment_residual_Nm=error[3:],minimum_torque_margin_Nm=margin))
    return answer
