"""CPU-only independent pose/EFC checks on saved fixed-wrist pen executions."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def diagnose(model, trace, raw, windows, gates, dt=.0025):
    z=trace;n=len(z['post_qpos']);controls=n//8;t=np.arange(1,n+1)*dt
    body=model.body('object/cube').id;qa=int(model.jnt_qposadr[model.body_jntadr[body]])
    if np.any(model.body_ipos[body]):raise ValueError('This reader requires declared COM-at-free-origin pen')
    q=z['post_qpos'];pre=np.concatenate([z['initial_qpos'][None],q[:-1]])
    center=q[:,qa:qa+3];actual=Rotation.from_quat(q[:,qa+np.array([4,5,6,3])])
    reference=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    target=Slerp(z['time_s'],reference)(t)
    targetp=np.stack([np.interp(t,z['time_s'],z['COM_world_m'][:,j]) for j in range(3)],axis=-1)
    distance=np.linalg.norm(center-targetp,axis=-1);angle=(target.inv()*actual).magnitude()
    force=np.zeros((len(raw['dim']),6));addresses_valid=True
    # Independent EFC pyramid decoding; do not call the producing decoder.
    for i,dim in enumerate(raw['dim']):
        tick=int(raw['tick'][i]);dim=int(dim);addr=raw['efc_address'][i]
        if addr[0]<0:continue
        width=1 if dim==1 else (2*(dim-1) if model.opt.cone==0 else dim)
        offset=int(raw['efc_offsets'][tick]);count=int(raw['efc_offsets'][tick+1])-offset;ids=addr[:width].astype(int)
        if dim not in [1,3,4,6] or len(ids)!=width or np.any(ids<0) or np.any(ids>=count):
            addresses_valid=False;continue
        values=raw['efc_force'][offset+ids].astype(np.float64)
        if model.opt.cone==0 and dim>1:
            force[i,0]=sum(values)
            for j in range(dim-1):force[i,j+1]=(values[2*j]-values[2*j+1])*float(raw['friction'][i,j])
        else:force[i,:dim]=values
    pen=np.zeros(n);selfpen=np.zeros(n);hand=np.zeros(n);nonhand=np.zeros(n);wrench=np.zeros((n,6))
    for i,(g,h) in enumerate(raw['geom']):
        g,h=int(g),int(h);tick=int(raw['tick'][i]);ob=model.geom_bodyid[[g,h]]==body
        gn,hn=model.geom(g).name or '',model.geom(h).name or '';normal=max(0.,force[i,0])
        if ob.sum()==1:
            pen[tick]=max(pen[tick],-float(raw['dist'][i]));other=hn if ob[0] else gn
            if other.startswith('robot/'):
                hand[tick]+=normal
                sign=-1 if ob[0] else 1;frame=raw['frame'][i].astype(np.float64)
                F=sign*frame.T@force[i,:3];M=sign*frame.T@force[i,3:]+np.cross(raw['pos'][i].astype(np.float64)-pre[tick,qa:qa+3],F)
                wrench[tick,:3]+=F;wrench[tick,3:]+=M
            else:nonhand[tick]+=normal
        elif gn.startswith('robot/') and hn.startswith('robot/'):
            selfpen[tick]=max(selfpen[tick],-float(raw['dist'][i]))
    jids=[model.joint(f'robot/right_finger{i}_joint{j}').id for i in range(1,6) for j in range(1,5)]
    jq=q[:,model.jnt_qposadr[jids]]
    joint=np.maximum(np.maximum(model.jnt_range[jids,0]-jq,jq-model.jnt_range[jids,1]),0).max(-1)
    metrics=dict(actual_duration_s=float(t[-1]),COM_RMS_m=float(np.sqrt(np.mean(distance**2))),COM_max_m=float(distance.max()),
        angle_RMS_rad=float(np.sqrt(np.mean(angle**2))),angle_max_rad=float(angle.max()),max_penetration_m=float(pen.max()),
        max_self_penetration_m=float(selfpen.max()),max_joint_limit_excursion_rad=float(joint.max()),
        max_object_nonhand_normal_force_N=float(nonhand.max()),hand_contact_coverage_over_0_1N=float(np.mean(hand>.1)))
    mean_world=wrench.reshape(controls,8,6).mean(1)
    # Resident moving-COM hand WORLD slot1 is independent of source-frame timing.
    saved_world=windows['wrench_sum'][:,1]/8
    R=windows['source_R'].astype(np.float64)
    mean_source=np.concatenate([np.einsum('nji,nj->ni',R,mean_world[:,:3]),np.einsum('nji,nj->ni',R,mean_world[:,3:])],-1)
    support_loss=((mean_source[:,:3]-windows['objective_target_source'][:,:3])/.3)**2
    support_torque=((mean_source[:,3:]-windows['objective_target_source'][:,3:])/.003)**2
    loss=support_loss.mean(-1)+support_torque.mean(-1)
    checks=dict(raw_addresses=addresses_valid,raw_forces=np.allclose(force,raw['local_force6'],atol=1e-10,rtol=0),
        COM_origin=np.array_equal(center,z['post_COM']),object_penetration=np.array_equal(pen,z['pre_penetration_m']),
        self_penetration=np.array_equal(selfpen,z['pre_self_penetration_m']),joint_excursion=np.array_equal(joint,z['post_joint_limit_excursion_rad']),
        hand_normal=np.allclose(hand,z['pre_hand_normal_N'],atol=1e-10,rtol=0),nonhand_normal=np.allclose(nonhand,z['pre_object_nonhand_normal_N'],atol=1e-10,rtol=0),
        mean_WORLD_wrench=np.allclose(mean_world,saved_world,atol=1e-9,rtol=1e-9),
        mean_SOURCE_wrench=np.allclose(mean_source,windows['objective_measured_source'],atol=1e-9,rtol=1e-9),
        support_loss=np.allclose(loss,windows['objective_support_loss'],atol=1e-7,rtol=1e-7),
        used_phase=np.array_equal(z['pre_action_phase'],np.arange(1,controls+1)),eight_samples=bool((windows['counts'][:,0]==8).all()))
    first={}
    for key,value in [('COM_max_m',distance),('angle_max_rad',angle),('max_penetration_m',pen),('max_self_penetration_m',selfpen)]:
        ids=np.flatnonzero(value>gates[key]);first[key]=None if not len(ids) else dict(native_index=int(ids[0]),time_s=float(t[ids[0]]),value=float(value[ids[0]]),limit=gates[key])
    contact_loss=np.flatnonzero(hand<=.1)
    common=t<=.2+1e-12
    by_control=[]
    for k in range(controls):
        sl=slice(8*k,8*(k+1));by_control.append(dict(end_s=.02*(k+1),COM_end_m=float(distance[8*k+7]),angle_end_rad=float(angle[8*k+7]),
            hand_normal_mean_N=float(hand[sl].mean()),hand_contact_fraction=float(np.mean(hand[sl]>.1)),
            mean_hand_WORLD_COM_wrench=mean_world[k].tolist(),target_SOURCE_COM_wrench=windows['objective_target_source'][k].tolist(),
            support_loss=float(loss[k]),support_bonus=float(windows['objective_support_bonus'][k]),
            max_penetration_m=float(pen[sl].max())))
    return dict(metrics=metrics,checks={k:bool(v) for k,v in checks.items()},first_breaches=first,
        first_low_hand_load=None if not len(contact_loss) else dict(time_s=float(t[contact_loss[0]]),normal_N=float(hand[contact_loss[0]])),
        common_0_20s=dict(COM_RMS_m=float(np.sqrt(np.mean(distance[common]**2))),angle_RMS_rad=float(np.sqrt(np.mean(angle[common]**2)))),
        mean_wrench_max_abs_difference=float(abs(mean_world-saved_world).max()),by_control=by_control)
