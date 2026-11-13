"""Native diagnostic of source-following bounded initialization corrections."""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .controlled_snapshot import native_controlled_snapshot
from .compound_contacts import CompoundContacts
from .compound_holding import object_state
from ..runtime.native_evaluation import model_digest_except_dt
from ..runtime.controlled_evaluation import summarize_trace


def evaluate_projected_pen(case,reference_q,delta,qpos,dt,config,folder):
    import mujoco
    m,d,source,index=native_controlled_snapshot(case);m.opt.timestep=dt
    if len(reference_q)!=369 or abs(round(.02/dt)*dt-.02)>1e-15:raise ValueError('Original finite 50Hz reference clock')
    if np.max(abs(delta))>.10+1e-12 or np.any(abs(2*delta)>1):raise ValueError('Bounded physical offset / original raw action box')
    before=model_digest_except_dt(m,np);d.qpos[:]=qpos
    if not np.array_equal(d.qvel,source['initial_qvel']):raise ValueError('Source initial velocities preserved')
    d.qacc_warmstart[:]=0;d.warning.number[:]=0
    body=case['model_layout']['object_body'];qa=case['model_layout']['object_qposadr'];da=case['model_layout']['object_dofadr'];COM_BODY=np.array(case['model_layout']['COM_BODY'])
    hand=[g for g in range(m.ngeom) if (m.geom(g).name or '').startswith('robot/') and (m.geom_contype[g] or m.geom_conaffinity[g])]
    parts=[g for g in range(m.ngeom) if m.geom_bodyid[g]==body and (m.geom_contype[g] or m.geom_conaffinity[g])]
    plane=next(g for g in range(m.ngeom) if int(m.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_PLANE))
    contacts=CompoundContacts(m,parts,body,{1:hand},plane)
    # Group1 intentionally means ALL hand collision geoms in this diagnostic.
    # It must not be interpreted as anatomical finger1 or a tripod label.
    lo=np.maximum(m.jnt_range[index['joints'],0],m.actuator_ctrlrange[index['actuators'],0]);hi=np.minimum(m.jnt_range[index['joints'],1],m.actuator_ctrlrange[index['actuators'],1])
    memory=np.array(delta,float).copy();raw_action=2*np.array(delta,float)
    d.ctrl[index['actuators']]=np.clip(reference_q[0]+memory,lo,hi)
    mujoco.mj_forward(m,d);initial_qpos=d.qpos.copy();initial_qvel=d.qvel.copy();mocap=np.r_[d.mocap_pos.ravel(),d.mocap_quat.ravel()]
    reference={k:source[k].copy() for k in ['time_s','COM_world_m','quaternion_wxyz']}
    keys=['pre_qpos','pre_qvel','post_qpos','post_qvel','post_ctrl','post_qfrc_applied','post_xfrc_applied',
          'post_COM','post_quaternion_wxyz','post_COM_velocity','post_omega_world','pre_object_COM',
          'pre_hand_normal_N','pre_object_nonhand_normal_N','pre_penetration_m','pre_self_penetration_m',
          'post_joint_limit_excursion_rad','overflow','post_native_time','pre_world_COM_wrench','post_qfrc_constraint']
    trace={key:[] for key in keys};raw=[];controls=[];memories=[];clip_amounts=[];phase=[];terminated=False;reason='REFERENCE_END';start=time.perf_counter()
    targetR=Rotation.from_quat(reference['quaternion_wxyz'][:,[1,2,3,0]]).as_matrix();ticks=0;n=round(.02/dt)
    for k in range(368):
        prior=memory.copy();memory=.5*np.clip(.5*raw_action,-.6,.6)+.5*memory
        requested=reference_q[k+1]+memory;target=np.clip(requested,lo,hi);d.ctrl[index['actuators']]=target
        controls.append(target.copy());memories.append(np.r_[prior,memory]);clip_amounts.append(abs(requested-target));phase.append(k+1)
        for _ in range(n):
            preq,prev=d.qpos.copy(),d.qvel.copy();mujoco.mj_step(m,d);contact=contacts.read(d,raw=True)
            if len(contact['raw23']):raw.extend(np.c_[np.full(len(contact['raw23']),ticks),contact['raw23']])
            if np.any(d.qfrc_applied) or np.any(d.xfrc_applied) or not np.array_equal(np.r_[d.mocap_pos.ravel(),d.mocap_quat.ravel()],mocap):raise ValueError('Object aid/wrist mutation')
            center,quat,vel,omega=object_state(m,d.qpos,d.qvel,qa,da,COM_BODY)
            q=d.qpos[index['qpos']];ranges=m.jnt_range[index['joints']]
            excursion=max(0.,float((ranges[:,0]-q).max()),float((q-ranges[:,1]).max()))
            values=[preq,prev,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.qfrc_applied.copy(),d.xfrc_applied.copy(),
                center,quat,vel,omega,contact['object_COM_world_m'],contact['normal_by_finger_N'][0],contact['other_normal_N']+contact['table_normal_N'],
                contact['max_native_penetration_m'],contact['max_self_penetration_m'],excursion,False,float(d.time),
                np.r_[contact['force_world_N'],contact['torque_about_COM_world_Nm']],d.qfrc_constraint.copy()]
            for key,value in zip(trace,values):trace[key].append(value)
            ticks+=1
            if np.any(d.warning.number) or not np.isfinite(np.r_[d.qpos,d.qvel,d.qacc]).all():
                terminated=True;reason='NATIVE_NUMERICAL';break
        if terminated:break
        # Native POST endpoint check; original GPU manager sees one substep-stale
        # derived kinematics. This timing distinction is explicitly retained.
        pose_error=np.linalg.norm(center-reference['COM_world_m'][k+1])
        R=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix();shaft=np.arccos(np.clip(R[:,0]@targetR[k+1,:,0],-1,1))
        if pose_error>.05:terminated=True;reason='POSITION_ERROR_OVER_50MM'
        elif shaft>=np.pi/3:terminated=True;reason='SHAFT_ERROR_OVER_60DEG'
        if terminated:break
    if before!=model_digest_except_dt(m,np):raise ValueError('Physical model changed during native diagnostic')
    z={key:np.array(value) for key,value in trace.items()};complete=len(controls)==368 and not terminated
    summary=summarize_trace(z,reference,dt,complete,config);summary['warning_count']=int(d.warning.number.sum());summary['warning_scope']='Actual native counters, unlike copied Warp warnings.'
    folder.mkdir();np.savez_compressed(folder/'trace.npz',**z,**reference,raw_contact_tick_plus23=np.array(raw).reshape(-1,24),
        raw_action20=raw_action,actual_targets20=np.array(controls),EMA_before_after40=np.array(memories),absolute_target_clip20=np.array(clip_amounts),
        pre_action_phase=np.array(phase),initial_qpos=initial_qpos,initial_qvel=initial_qvel,delta20=delta)
    return dict(dt=dt,termination=reason,complete=complete,summary=summary,native_steps=ticks,control_intervals=len(controls),
        clipped_target_coordinates=int(np.count_nonzero(clip_amounts)),max_target_clipping_rad=float(np.max(clip_amounts)),
        wall_s=time.perf_counter()-start,model_digest_except_dt=before,source_velocities_preserved=True,
        object_body=body,object_qposadr=qa,object_dofadr=da,
        scope='Native CPU original/half dt diagnostic, source joint reference plus fixed residual delta and declared initial EMA. No aid, no repeated pose injection, not GPU policy or initial-state-only causal experiment.')
