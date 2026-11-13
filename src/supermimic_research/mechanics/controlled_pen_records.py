"""Indexed object-first pen records in the maintained state55/H20 chart."""
import numpy as np
from scipy.spatial.transform import Rotation
from .transition_records import state_rotation, transition_xy
from .response_decode import momentum_defect


def native_state55(qpos,qvel,qids,dofs,qa,da):
    q,v=np.asarray(qpos,float),np.asarray(qvel,float)
    quat=q[qa+3:qa+7]
    if len(qids)!=20 or len(dofs)!=20 or abs(np.linalg.norm(quat)-1)>1e-5:
        raise ValueError('Explicit twenty hand indices and unit free-object quaternion')
    R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
    return np.r_[q[qids],v[dofs],q[qa:qa+3],R[:,0],R[:,1],v[da:da+3],R@v[da+3:da+6]]


def pack_case(model,case,z,wrench,qids,dofs,aids,initial_warm,physics):
    body=case['model_layout']['object_body'];qa=case['model_layout']['object_qposadr'];da=case['model_layout']['object_dofadr']
    if np.any(case['model_layout']['COM_BODY']) or model.body_parentid[body]!=0:raise ValueError('Explicit world free body with COM at origin')
    n=round(.02/case['dt']);N=len(z['post_qpos']);K=len(z['actions20'])
    if N!=K*n:raise ValueError('Whole original20ms control intervals')
    Fn=np.zeros((N,6));nonhand=np.zeros(N);raw=z['raw_contact_tick_plus23']
    for row in raw:
        tick,a,b=int(row[0]),int(row[1]),int(row[2]);oa,ob=model.geom_bodyid[a]==body,model.geom_bodyid[b]==body
        if oa==ob:continue
        name=model.geom(b if oa else a).name or '';normal=max(0.,float(row[18]))
        if name.startswith('robot/'):
            digit=next((i-1 for i in range(1,6) if name.startswith(f'robot/right_finger{i}_')),5)
            Fn[tick,digit]+=normal
        else:nonhand[tick]+=normal
    tick_valid=(np.linalg.norm(wrench['force_residual_N'],axis=1)<=1e-4)&(np.linalg.norm(wrench['torque_residual_Nm'],axis=1)<=1e-5)&(np.linalg.norm(wrench['linear_impulse_residual_N_s'],axis=1)<=5e-7)
    tick_valid&=~z['overflow'].astype(bool)
    tick_valid&=nonhand<=1e-9
    tick_valid&=~np.any(z['post_qfrc_applied']!=0,axis=1)
    tick_valid&=~np.any(z['post_xfrc_applied']!=0,axis=(1,2))
    xs,ys,heads,valid,geometry=[],[],[],[],[]
    for k in range(K):
        a,b=k*n,(k+1)*n-1
        s=native_state55(z['pre_qpos'][a],z['pre_qvel'][a],qids,dofs,qa,da)
        t=native_state55(z['post_qpos'][b],z['post_qvel'][b],qids,dofs,qa,da)
        target=z['post_ctrl'][a,aids]
        if not np.array_equal(z['post_ctrl'][a:b+1][:,aids],np.tile(target,(n,1))):raise ValueError('Held actual PD target per20ms interval')
        mean_world=wrench['world_COM_wrench'][a:b+1].mean(0);R=state_rotation(s)
        mean_BODY=np.r_[R.T@mean_world[:3],R.T@mean_world[3:]]
        transition=dict(source55=s,next55=t,warmstart26_before=initial_warm if a==0 else z['post_qacc'][a-1],
            warmstart26_after=z['post_qacc'][b],decision=dict(actual_control20=target),
            mean_native_wrench_source_BODY_COM=mean_BODY,mean_native_group_Fn_N=Fn[a:b+1].mean(0))
        x,y=transition_xy(transition,case['dt']);head=y.copy()
        head[46:52]=momentum_defect(s,t,y,physics['mass'],np.asarray(physics['inertia_BODY']),np.asarray(physics['gravity']),.02)
        xs.append(x);ys.append(y);heads.append(head)
        valid.append(bool(np.all(tick_valid[a:b+1]) and np.linalg.norm(y[43:46])<np.pi-.001))
        geometry.append([float(z['pre_penetration_m'][a:b+1].max()),float(z['pre_self_penetration_m'][a:b+1].max()),float(z['post_joint_limit_excursion_rad'][a:b+1].max())])
    return dict(x=np.array(xs),y=np.array(ys),heads=np.array(heads),valid=np.array(valid),geometry_flags=np.array(geometry),control_index=np.arange(K))
