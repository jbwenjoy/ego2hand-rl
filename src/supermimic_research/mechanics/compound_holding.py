"""No-aid native static-hold diagnostics for an explicitly qualified compound.

Same numerical gates as the frozen phone evaluator, with declared mug-specific
canonical contact regions. This is a new mechanical feasibility experiment,
not a policy comparison, source trajectory replay or source-gravity transfer.
"""
from pathlib import Path
import json
import numpy as np
from supermimic_research.runtime.native_evaluation import GATES, DTS, model_digest_except_dt
from .compound_contacts import CompoundContacts


def object_state(model, qpos, qvel, qadr, dadr, com):
    q=qpos[qadr+3:qadr+7]
    if abs(np.linalg.norm(q)-1)>1e-6:
        raise ValueError('Native nonunit quaternion, no repair')
    w,x,y,z=q
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    arm=R@com;omega=R@qvel[dadr+3:dadr+6]
    return qpos[qadr:qadr+3]+arm,q.copy(),qvel[dadr:dadr+3]+np.cross(omega,arm),omega


def evaluate_hold(model, qpos, ctrl, binding, pad_groups, dt, output):
    import mujoco
    output=Path(output);output.mkdir()
    if dt not in DTS:raise ValueError('Declared double timestep check only')
    qpos,ctrl=np.array(qpos),np.array(ctrl)
    if qpos.shape!=(27,) or ctrl.shape!=(20,) or np.any(ctrl<model.actuator_ctrlrange[:,0]) or np.any(ctrl>model.actuator_ctrlrange[:,1]):
        raise ValueError('Raw legal physical position command')
    digest=model_digest_except_dt(model,np);model.opt.timestep=dt
    d=mujoco.MjData(model);mujoco.mj_resetData(model,d)
    d.qpos[:]=qpos;d.qvel[:]=0;d.ctrl[:]=ctrl;d.qacc_warmstart[:]=0
    mujoco.mj_forward(model,d)
    oid=binding['mug_body_id'];qadr=binding['mug_free_qpos'];dadr=binding['mug_free_dof'];com=model.body_ipos[oid].copy()
    reader=CompoundContacts(model,binding['mug_part_geom_ids'],oid,pad_groups,model.geom('mug_task_table').id)
    initial_com,initial_quat,_,_=object_state(model,d.qpos,d.qvel,qadr,dadr,com)
    names=['pre_qpos','pre_qvel','post_qpos','post_qvel','post_COM','post_quat','post_COM_velocity',
        'post_omega_world','pre_group_normal_N','pre_other_normal_N','pre_table_normal_N','pre_penetration_m',
        'pre_self_penetration_m','pre_wrench_BODY_COM','pre_object_COM','post_limit_excursion_rad']
    trace={n:[] for n in names};raw=[];sample_ticks=[];rne_errors=[]
    counts=dict(step1=0,step2=0,contact_force_reads=0,rne_postconstraint=0,integration_steps=0)
    reason=None;steps=round(2./dt);stride=round(.01/dt)
    for tick in range(steps):
        mujoco.mj_step1(model,d);counts['step1']+=1
        preq,prev=d.qpos.copy(),d.qvel.copy()
        d.ctrl[:]=ctrl
        if np.any(d.qfrc_applied) or np.any(d.xfrc_applied) or np.any(model.body_gravcomp) or model.neq or model.nmocap:
            raise ValueError('Object aid or floating/mocap support prohibited')
        mujoco.mj_step2(model,d);counts['step2']+=1;counts['integration_steps']+=1
        snapshot=reader.read(d,raw=(tick%stride==0));counts['contact_force_reads']+=d.ncon
        if tick%stride==0:
            raw.extend(np.c_[np.full(len(snapshot['raw23']),tick),snapshot['raw23']]);sample_ticks.append(tick)
            mujoco.mj_rnePostConstraint(model,d);counts['rne_postconstraint']+=1
            world=np.r_[snapshot['torque_about_COM_world_Nm'],snapshot['force_world_N']]
            error=float(np.max(np.abs(world-d.cfrc_ext[oid])));rne_errors.append(error)
            if error>1e-8:raise ValueError('Compound moving-COM wrench disagrees with native external wrench')
        center,quat,velocity,omega=object_state(model,d.qpos,d.qvel,qadr,dadr,com)
        violation=float(max(0.,np.max(model.jnt_range[:20,0]-d.qpos[:20]),np.max(d.qpos[:20]-model.jnt_range[:20,1])))
        values=[preq,prev,d.qpos.copy(),d.qvel.copy(),center,quat,velocity,omega,snapshot['normal_by_finger_N'][:3],
            snapshot['other_normal_N'],snapshot['table_normal_N'],snapshot['max_native_penetration_m'],snapshot['max_self_penetration_m'],
            snapshot['wrench_BODY_about_COM'],snapshot['object_COM_world_m'],violation]
        for name,value in zip(names,values):trace[name].append(value)
        if not np.isfinite(np.r_[d.qpos,d.qvel,d.qacc]).all():reason='NONFINITE';break
        if np.any(d.warning.number):reason='NATIVE_WARNING';break
        if np.linalg.norm(center-initial_com)>.05:reason='COM_DRIFT_OVER_50MM';break
        angle=2*np.arccos(np.clip(abs(quat@initial_quat),0,1))
        if angle>np.pi/4:reason='ORIENTATION_OVER_45DEG';break
    trace={k:np.array(v) for k,v in trace.items()};completed=len(trace['post_qpos']);time=completed*dt
    tail=(np.arange(completed)+1)*dt>=1.5;pre_tail=np.arange(completed)*dt>=1.5
    metrics=dict(final_COM_drift_m=float(np.linalg.norm(trace['post_COM'][-1]-initial_com)),
        final_angle_rad=float(2*np.arccos(np.clip(abs(trace['post_quat'][-1]@initial_quat),0,1))),
        tail_linear_velocity_RMS_m_s=float(np.sqrt(np.mean(np.sum(trace['post_COM_velocity'][tail]**2,axis=1)))) if np.any(tail) else None,
        tail_angular_velocity_RMS_rad_s=float(np.sqrt(np.mean(np.sum(trace['post_omega_world'][tail]**2,axis=1)))) if np.any(tail) else None,
        tail_triple_pad_force_coverage=float(np.mean(np.all(trace['pre_group_normal_N'][pre_tail]>.1,axis=1))) if np.any(pre_tail) else None,
        max_native_penetration_m=float(trace['pre_penetration_m'].max()),max_self_penetration_m=float(trace['pre_self_penetration_m'].max()),
        max_phone_nonpad_normal_force_N=float(trace['pre_other_normal_N'].max()),max_table_normal_force_N=float(trace['pre_table_normal_N'].max()),
        max_joint_limit_excursion_rad=float(trace['post_limit_excursion_rad'].max()),warning_count=int(np.sum(d.warning.number)))
    gates={k:bool(v is not None and (v>=GATES[k] if k=='tail_triple_pad_force_coverage' else v<=GATES[k])) for k,v in metrics.items()}
    gates['duration_complete']=completed==steps and reason is None
    np.savez_compressed(output/'trace.npz',**trace,raw_contact_tick_plus23=np.array(raw).reshape(-1,24),
        raw_sample_ticks=np.array(sample_ticks),rne_external_wrench_errors=np.array(rne_errors),ctrl20=ctrl,initial_qpos=qpos,
        initial_COM=initial_com,initial_quat=initial_quat,native_dt_s=np.array(dt))
    result=dict(schema='compound_mug_noaid_static_hold_case:1',dt_s=dt,completed_steps=completed,actual_duration_s=time,
        termination=reason,metrics=metrics,gates=gates,native_holding_gates_pass=all(gates.values()),counts=counts,
        max_sampled_native_external_wrench_error=float(max(rne_errors)),raw_contact_sampling_interval_s=stride*dt,
        initial_qpos_is_static_geometry_qualified=True,controller='constant legal absolute PD target; not an Actor or frozen phone residual controller',
        source_trajectory_replayed=False,source_gravity_preserved=False,
        scope='New mug mechanical static-hold diagnostic using unchanged numeric phone thresholds but explicitly expanded distal contact regions. Native penetration and force gates every integration step; complete raw contacts and native external-wrench crosscheck at10ms samples. No all-state independent convex audit, calibrated dynamics, policy/ego benefit claim.')
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    if model_digest_except_dt(model,np)!=digest:raise ValueError('Physical model changed except declared timestep')
    return result
