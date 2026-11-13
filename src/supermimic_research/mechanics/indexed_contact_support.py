"""Indexed full-wrench static support witnesses, including torsional friction."""
import numpy as np
from .support import static_feasibility


def friction_rays(dim,mu):
    if dim not in [1,3,4,6]:raise ValueError('Rigid contact dimension')
    if dim==1:return np.array([[1.,0,0,0,0,0]])
    result=np.zeros((2*(dim-1),6));result[:,0]=1.
    for j in range(dim-1):result[2*j,j+1]=mu[j];result[2*j+1,j+1]=-mu[j]
    return result


def indexed_static_support(model,data,body,parts,allowed,dofs,actuators,target_lower,target_upper):
    import mujoco
    parts=set(map(int,parts));allowed=set(map(int,allowed));dofs=np.array(dofs,int);actuators=np.array(actuators,int)
    if len(dofs)!=20 or len(actuators)!=20 or int(model.opt.cone)!=int(mujoco.mjtCone.mjCONE_PYRAMIDAL):raise ValueError('Explicit20 controlled joints, original pyramidal cone')
    W=[];loads=[];actual_hand=np.zeros(20);records=[];R=data.xmat[body].reshape(3,3)
    for ci,con in enumerate(data.contact):
        a,b=map(int,con.geom);oa,ob=a in parts,b in parts
        if oa==ob or con.efc_address<0:continue
        other=a if ob else b
        if other not in allowed:raise ValueError('Nonintent support cannot be silently added')
        sign=1. if ob else -1.;frame=con.frame.reshape(3,3);jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jac(model,data,jp,jr,con.pos,int(model.geom_bodyid[other]));force=np.zeros(6);mujoco.mj_contactForce(model,data,ci,force)
        F=sign*frame.T@force[:3];T=sign*frame.T@force[3:];actual_hand-=jp[:,dofs].T@F+jr[:,dofs].T@T
        for ray in friction_rays(int(con.dim),con.friction):
            F=sign*frame.T@ray[:3];couple=sign*frame.T@ray[3:];torque=couple+np.cross(con.pos-data.xipos[body],F)
            W.append(np.r_[R.T@F,R.T@torque]);loads.append(jp[:,dofs].T@F+jr[:,dofs].T@couple)
        records.append(dict(geoms=[a,b],dim=int(con.dim),distance_m=float(con.dist),local_force6=force.tolist()))
    W=np.array(W).reshape(-1,6).T;loads=np.array(loads).reshape(-1,20).T
    other=data.qfrc_constraint[dofs]-actual_hand;offset=data.qfrc_bias[dofs]-data.qfrc_passive[dofs]-other
    gain=model.actuator_gainprm[actuators,0];bias=model.actuator_biasprm[actuators,0]+model.actuator_biasprm[actuators,1]*data.actuator_length[actuators]+model.actuator_biasprm[actuators,2]*data.actuator_velocity[actuators]
    jid=model.actuator_trnid[actuators,0]
    if not np.all(gain>0) or not np.array_equal(model.actuator_gear[actuators],np.c_[np.ones(20),np.zeros((20,5))]):raise ValueError('Direct unit-gear positive affine PD')
    cap_lower=np.maximum(model.actuator_forcerange[actuators,0],model.jnt_actfrcrange[jid,0]);cap_upper=np.minimum(model.actuator_forcerange[actuators,1],model.jnt_actfrcrange[jid,1])
    lower=np.clip(gain*target_lower+bias,cap_lower,cap_upper);upper=np.clip(gain*target_upper+bias,cap_lower,cap_upper)
    target=np.r_[R.T@(-model.body_mass[body]*model.opt.gravity),np.zeros(3)]
    lp=static_feasibility(W,target,loads=loads,offset=offset,lower=lower,upper=upper,radius=.125)
    result=dict(lp=lp,wrench_BODY=W,loads20=loads,target_BODY=target,contacts=records,torque_lower=lower,torque_upper=upper,offset=offset,
                full_wrench_rank=int(np.linalg.matrix_rank(W)) if W.size else 0,control=None)
    if lp['success']:
        tau=lp['witness']['required_torque20_Nm'];control=(tau-bias)/gain
        if np.any(control<target_lower-1e-10) or np.any(control>target_upper+1e-10):raise ValueError('PD witness inversion outside declared control interval')
        result['control']=control
    return result
