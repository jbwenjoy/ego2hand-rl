"""Scheduled-control native loop derived from frozen v3; no backend on import."""
from pathlib import Path
import hashlib,importlib.util
BASE=Path(__file__).resolve().parents[1]/'functional_tripod_dynamics_v1/run_functional_tripod_dynamics_v3.py'
if hashlib.sha256(BASE.read_bytes()).hexdigest()!='d433a51df307bf45b40e70bc2c74776654eb2ece65bd2de67ae8163c201fd960':raise ValueError('Frozen evaluator changed')
_sp=importlib.util.spec_from_file_location('policy_eval_frozen_native_gates',BASE);frozen=importlib.util.module_from_spec(_sp);_sp.loader.exec_module(frozen)
DURATION,GATES,OLD_GATES=frozen.DURATION,frozen.GATES,frozen.OLD_GATES
pin,write,contact_snapshot=frozen.pin,frozen.write,frozen.contact_snapshot

def evaluate_case_controlled(m,variant,dt,case_dir,phone,pg,table,pads,mujoco,np,counts,controller):
    """Original 2s native loop and old gates; three-pad classification additive."""
    q0=np.asarray(variant['qpos27'],dtype=float);ctrl=np.asarray(variant['ctrl20'],dtype=float);m.opt.timestep=dt
    d=mujoco.MjData(m);counts['MjData']+=1;d.qpos[:]=q0;d.qvel[:]=0;d.ctrl[:]=ctrl;d.qacc_warmstart[:]=0
    mujoco.mj_forward(m,d);counts['initial_forward']+=1
    if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or np.any(d.warning.number):raise ValueError('Initial aid/warning')
    R=np.empty(9)
    def COM(q):
        mujoco.mju_quat2Mat(R,q[23:27]);return q[20:23]+R.reshape(3,3)@m.body_ipos[phone]
    COM0=COM(q0).copy();initial=contact_snapshot(m,d,pg,table,pads,mujoco,np,counts,-1)
    initial_record=dict(phase='RESET_FORWARD',pad_normal_N3=initial[0].tolist(),phone_nonpad_normal_N=initial[1],
        old_two_pad_phone_nonpad_normal_N=initial[1]+float(initial[0][2]),table_normal_N=initial[2],
        contact_rows24=np.asarray(initial[5],dtype=float).reshape(-1,24).tolist(),qacc26_descriptor=d.qacc.tolist(),
        actuator_force20=d.actuator_force.tolist(),qfrc_actuator26=d.qfrc_actuator.tolist())
    steps=round(DURATION/dt);sample_stride=round(.01/dt)
    keys=['post_time','pre_time_from_integer_tick','post_COM','post_angle','post_qpos','post_qvel',
        'pre_native_energy','pre_ncon','pre_pad_normal_N','pre_phone_nonpad_normal_N','pre_table_normal_N',
        'pre_penetration_m','pre_self_penetration_m','pre_qfrc_actuator','pre_actuator_force','pre_solver_niter',
        'pre_actuator_length20','pre_actuator_velocity20','pre_actuator_force_margin20x2','pre_joint_actuator_force_margin20x2']
    rows={k:[] for k in keys};contact_rows=[];geometry_samples=[];terminate='COMPLETE';attempted_steps=0
    # A declared control hook is the sole new mutation of d.ctrl. No state/force resets.
    stride=round(.02/dt)
    if stride not in (64,128) or abs(stride*dt-.02)>1e-15:raise ValueError('Integer20ms control stride')
    control_records=[];pre_ctrl=[]
    sig=mujoco.mjtState.mjSTATE_INTEGRATION;size=mujoco.mj_stateSize(m,sig)
    if int(sig)!=16383 or size!=300:raise ValueError('Frozen integration receipt shape')
    for tick in range(steps):
        if tick%stride==0:
            before=np.empty(size);after=np.empty(size);mujoco.mj_getState(m,d,before,sig)
            requested=np.asarray(controller(tick,dt,m,d),dtype=np.float64)
            mujoco.mj_getState(m,d,after,sig)
            counts['controller_receipt_reads']=counts.get('controller_receipt_reads',0)+2
            if before.tobytes()!=after.tobytes():raise ValueError('Controller mutated native integration state')
            if requested.shape!=(20,) or not np.isfinite(requested).all() or np.any(requested<m.actuator_ctrlrange[:,0]) or np.any(requested>m.actuator_ctrlrange[:,1]):raise ValueError('Finite legal absolute command, no clamp')
            ctrl=requested.copy();d.ctrl[:]=ctrl
            control_records.append(dict(pre_tick=tick,time_s=tick*dt,ctrl20_F64=ctrl.tolist()))
            counts['controller_calls']=counts.get('controller_calls',0)+1
        pre_ctrl.append(ctrl.copy())
        mujoco.mj_step(m,d);counts['integration_steps']+=1;attempted_steps+=1
        values=[d.time,d.qpos,d.qvel,d.qacc,d.energy,d.qfrc_actuator,d.actuator_force]
        if not all(np.isfinite(x).all() for x in values):terminate='ABORT_NONFINITE';break
        if np.any(d.xfrc_applied) or np.any(d.qfrc_applied) or not np.array_equal(d.ctrl,ctrl):raise ValueError('Object aid/controller mutation')
        pad,other,table_force,penetration,self_pen,cr=contact_snapshot(m,d,pg,table,pads,mujoco,np,counts,tick);contact_rows.extend(cr)
        com=COM(d.qpos).copy();angle=float(2*np.arccos(np.clip(abs(d.qpos[23:27]@q0[23:27]),0,1)))
        am=np.c_[d.actuator_force-m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]-d.actuator_force]
        jm=np.c_[d.qfrc_actuator[:20]-m.jnt_actfrcrange[:20,0],m.jnt_actfrcrange[:20,1]-d.qfrc_actuator[:20]]
        for k,v in zip(keys,[float(d.time),tick*dt,com,angle,d.qpos.copy(),d.qvel.copy(),d.energy.copy(),int(d.ncon),
            pad,other,table_force,penetration,self_pen,d.qfrc_actuator.copy(),d.actuator_force.copy(),np.asarray(d.solver_niter).copy(),
            d.actuator_length.copy(),d.actuator_velocity.copy(),am,jm]):rows[k].append(v)
        if tick%sample_stride==0:geometry_samples.append(dict(pre_tick=tick,geom_xpos=d.geom_xpos.copy(),geom_xmat=d.geom_xmat.copy(),pre_qpos=q0.copy() if tick==0 else rows['post_qpos'][-2].copy()))
        if np.any(d.warning.number):terminate='ABORT_NATIVE_WARNING';break
        if np.linalg.norm(com-COM0)>GATES['abort_COM_drift_m'] or angle>GATES['abort_angle_rad']:terminate='ABORT_LOSS_OF_HOLD';break
    r={k:np.asarray(v) for k,v in rows.items()};completed=len(r['post_time'])
    if not completed:raise ValueError('No finite completed record')
    tail_start_tick=round((DURATION-GATES['tail_duration_s'])/dt);tail=np.arange(1,completed+1)>=tail_start_tick;pre_tail=np.arange(completed)>=tail_start_tick
    qlimits=m.jnt_range[:20];qvio=np.maximum(qlimits[:,0]-r['post_qpos'][:,:20],0)+np.maximum(r['post_qpos'][:,:20]-qlimits[:,1],0)
    lin=r['post_qvel'][tail,20:23];omega=r['post_qvel'][tail,23:26]
    metrics=dict(final_COM_drift_m=float(np.linalg.norm(r['post_COM'][-1]-COM0)),final_angle_rad=float(r['post_angle'][-1]),
        tail_linear_velocity_RMS_m_s=float(np.sqrt(np.mean(np.sum(lin*lin,axis=1)))) if len(lin) else None,
        tail_angular_velocity_RMS_rad_s=float(np.sqrt(np.mean(np.sum(omega*omega,axis=1)))) if len(omega) else None,
        tail_triple_pad_force_coverage=float(np.mean(np.all(r['pre_pad_normal_N'][pre_tail]>.1,axis=1))) if np.any(pre_tail) else None,
        max_native_penetration_m=float(r['pre_penetration_m'].max()),max_self_penetration_m=float(r['pre_self_penetration_m'].max()),
        max_phone_nonpad_normal_force_N=float(r['pre_phone_nonpad_normal_N'].max()),max_table_normal_force_N=float(r['pre_table_normal_N'].max()),
        max_joint_limit_excursion_rad=float(qvio.max()),warning_count=int(np.sum(d.warning.number)))
    gates=dict(duration_complete=terminate=='COMPLETE' and completed==steps)
    for k,v in metrics.items():gates[k]=v is not None and (v>=GATES[k] if k=='tail_triple_pad_force_coverage' else v<=GATES[k])
    oldmetrics={k:v for k,v in metrics.items() if k!='tail_triple_pad_force_coverage'}
    oldmetrics['tail_dual_pad_force_coverage']=float(np.mean(np.all(r['pre_pad_normal_N'][pre_tail,:2]>.1,axis=1))) if np.any(pre_tail) else None
    oldnonpad=r['pre_phone_nonpad_normal_N']+r['pre_pad_normal_N'][:,2]
    oldmetrics['max_phone_nonpad_normal_force_N']=float(oldnonpad.max());oldgates=dict(duration_complete=gates['duration_complete'])
    for k,v in oldmetrics.items():oldgates[k]=v is not None and (v>=OLD_GATES[k] if k=='tail_dual_pad_force_coverage' else v<=OLD_GATES[k])
    write(case_dir/'control_schedule.json',control_records)
    np.savez_compressed(case_dir/'trace.npz',**r,pre_executed_ctrl20=np.asarray(pre_ctrl[:completed]),ctrl20=ctrl,ctrl_margin20x2=np.asarray(variant['ctrl_margin20x2']),initial_qpos27=q0,initial_COM_world3=COM0,
        pre_old_two_pad_phone_nonpad_normal_N=oldnonpad,contact_rows=np.asarray(contact_rows,dtype=float).reshape(-1,24),
        sample_pre_tick=np.asarray([v['pre_tick'] for v in geometry_samples]),sample_pre_qpos=np.asarray([v['pre_qpos'] for v in geometry_samples]),
        sample_pre_geom_xpos=np.asarray([v['geom_xpos'] for v in geometry_samples]),sample_pre_geom_xmat=np.asarray([v['geom_xmat'] for v in geometry_samples]))
    summary=dict(name=variant['name'],candidate_id=variant['candidate_id'],route=variant['route'],dt_s=dt,termination=terminate,
        integration_steps=attempted_steps,finite_steps_saved=completed,actual_final_time_s=float(d.time),metrics=metrics,gates=gates,
        nominal_holding_dynamics_pass=all(gates.values()),old_two_pad_metrics=oldmetrics,old_two_pad_gates=oldgates,old_two_pad_pass=all(oldgates.values()),
        trace=pin(case_dir/'trace.npz'),warnings=np.asarray(d.warning.number).tolist(),initial_native=initial_record,
        source_candidate=variant['candidate'],source_state_npz=variant['state_npz'],command=variant,
        energy_force_geometry_phase='PRE(tick*dt); qpos/qvel/COM POST((tick+1)*dt)',
        velocity_metric_semantics='Preserved root translational freejoint qvel RMS; angular body qvel norm is frame invariant, not a new COM-velocity gate.',
        contact_row_columns=['pre_tick','geom1','geom2','dim','efc_address','distance_m','position3','frame9','local_wrench6'],
        nonpad_metric_definition='Maximum across PRE ticks of the SUM of positive Fn over every phone contact with non-designated hand geoms; same aggregation as original two-pad gate. old_two_pad also includes all positive Fn on geom41.',
        no_external_forces_equality_mocap_or_pose_reset=True,static_zero_acceleration_entry_required=False,
        control_schedule=pin(case_dir/'control_schedule.json'),control_scope='Planned50Hz controller only; trace pre_executed_ctrl20 is actual perPRE-tick command. Legacy trace ctrl20 is last command and ctrl_margin20x2 describes initial source only. Raw integration receipts protect callback state; no full-MjData certificate.',
        native_loop_and_physical_success_gates_from='frozen functional_tripod_dynamics_v3, unchanged after insertion of explicit control hook and schedule fields')
    write(case_dir/'summary.json',summary)
    endpoint=dict(dt_s=dt,final_COM_world_m=r['post_COM'][-1].tolist(),final_quaternion_wxyz=r['post_qpos'][-1,23:27].tolist(),
        actual_final_time_s=float(d.time),termination=terminate,finite_steps_saved=completed,original_case_pass=summary['nominal_holding_dynamics_pass'])
    return summary,endpoint
