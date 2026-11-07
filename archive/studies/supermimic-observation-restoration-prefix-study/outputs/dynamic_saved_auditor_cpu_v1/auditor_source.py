"""Independent saved-only engineering audit for the new H0/T0 short prefix.

Reads a terminal attempt once. No subprocess, CIM query, Warp, Torch, dynamics,
collision/force query, retry, physics run or change to any input is performed.
Worker gate booleans are compared only after independent raw reconstruction.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,re,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
VARIANT='H123_cubic_minimax_restoration_filtered_prefix16_v1'
CONDITIONS=('H0_MATCHED_DYNAMIC_INIT_HOLD','T0_MATCHED_DYNAMIC_INIT_TRACK')
WORKER_SHA='8834a386f001cc7a138fba3906774bc11d6a2a996c0695ddfc772eff11a90db7'
DRIVER_SHA='f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def loadnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def same32(a,b):return np.array_equal(np.asarray(a,np.float32).view(np.uint32),np.asarray(b,np.float32).view(np.uint32))


def quat_matrix(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def check_schedule(raw,solver,events,schedule,aids):
    """No producer scheduler import; stepk uses targetfloor((k-1)/4)."""
    t=np.asarray(raw['raw_time_s']);idx=np.r_[0,(np.arange(1,61)-1)//4];ctrl=np.asarray(raw['ctrl_rad'])
    if len(t)!=61 or len(solver)!=60 or len(events)!=14 or schedule.shape!=(15,20):raise ValueError('Incomplete raw/schedule coverage')
    errors=[]
    def require(ok,message):
        if not ok:errors.append(message)
    require(same32(ctrl,schedule[idx]),'Raw61 ctrl does not equal the preintegration interval schedule')
    require(np.array_equal(raw['control_interval_used'],idx),'Raw control interval indices differ')
    require(np.array_equal(raw['force_prestate_index'],np.r_[0,np.arange(60)]),'Raw force prestate indices differ')
    require(np.max(np.abs(raw['force_solver_time_s']-np.r_[0,t[:-1]]))<=2e-6,'Raw force timestamps differ')
    for j,s in enumerate(solver):
        k=j+1;ci=j//4
        require(s['step']==k and s['prestate_index']==j and s['poststate_index']==k and s['control_interval_used']==ci,'Solver step/pre/post/control phase differs')
        require(s['phase']=='just_completed_solver_preintegration','Solver force phase mislabeled')
        require(abs(s['solver_t_s']-t[j])<=2e-6 and abs(s['poststate_t_s']-t[k])<=2e-6,'Solver timestamps differ')
        require(same32(s['ctrl_solver_native'],schedule[ci]) and same32(s['ctrl_solver_native'],ctrl[k]),'Solver actual ctrl differs')
        require(np.array_equal(np.asarray(s['actuator_force_solver_native']),raw['actuator_force_native'][k]),'Solver/raw measured actuator force differs')
    for j,e in enumerate(events):
        k=5+4*j;ci=j+1
        require(e['before_step']==k and e['prestate_index']==k-1 and e['previous_raw_state_ctrl_interval']==ci-1 and e['new_solver_ctrl_interval']==ci,'Control event boundary differs')
        require(abs(e['raw_prestate_time_s']-t[k-1])<=2e-6,'Event clock differs from saved prestate')
        require(np.array_equal(e['position_ctrl_ids'],aids) and same32(e['selected_planned_FP32_rad'],schedule[ci]),'Event target/mapping differs')
        require(same32(np.asarray(e['actual_ctrl_readback_native'])[aids],schedule[ci]),'Event actual ctrl readback differs')
    return {'passed':not errors,'errors':errors,'events':14,'raw_state4_interval':int(idx[4]),'step5_solver_interval':1,'step5_force_prestate':4}


def audit(plan_path,raw_path,watchdog_path,output):
    start=time.perf_counter();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'auditor_source.py').write_bytes(Path(__file__).read_bytes());checks=[];inputs={};metrics={};trace_rows=[]
    def check(name,ok,detail=None):checks.append({'name':name,'passed':bool(ok),**({'detail':detail} if detail is not None else {})})
    def capture(name,path):
        inputs[name]=pin(path);return path
    def json_file(name,path):return read(capture(name,path))
    try:
        plan_path=Path(plan_path).resolve(strict=True);raw_path=Path(raw_path).resolve(strict=True);watchdog_path=Path(watchdog_path).resolve(strict=True)
        p=json_file('authorized_plan',plan_path);runtime=json_file('runtime',raw_path/'runtime.json');state=json_file('watchdog_state',watchdog_path/'state.json');post=json_file('post_terminal_CIM',watchdog_path/'post_terminal_CIM.json');identity=json_file('supervisor_identity',watchdog_path/'engineering_identity.json');evaluation=json_file('worker_evaluation_record_only',raw_path/'evaluation.json')
        check('new_authorized_plan_condition_variant',p['schema']=='restoration_dynamic_prefix_plan:1' and p['retarget_variant_id']==VARIANT and p['control_condition'] in CONDITIONS and p.get('GPU_run_authorized') is True)
        check('raw_output_equals_authorized_plan',Path(p['output']).resolve()==raw_path)
        check('runtime_terminal_complete_source_plan_identity',runtime['status']=='completed_engineering_new_dynamic_diagnostic' and runtime['plan_sha256']==sha(plan_path) and runtime['source_sha256']==WORKER_SHA and runtime['control_condition']==p['control_condition'] and runtime['retarget_variant_id']==VARIANT)
        check('runtime_fixed_operations',all(runtime[k]==v for k,v in {'physics_steps':60,'step_calls':60,'startup_Warp_forward_calls':1,'postinit_state_writes':0,'postinit_ctrl_writes':14,'overflow_reads':1,'resets_after_initialization':0,'actor_EMA_training_used':False}.items()))
        check('watchdog_plan_snapshot_identical',sha(watchdog_path/'plan_snapshot.json')==sha(plan_path));capture('watchdog_plan_snapshot',watchdog_path/'plan_snapshot.json')
        for name,q in p['source_pins'].items():check('pin__'+name,sha(q['path'])==q['sha256']);inputs['plan_source__'+name]=q
        check('execution_worker_driver_fixed',p['source_pins']['worker']['sha256']==WORKER_SHA and p['source_pins']['supervisor']['sha256']==DRIVER_SHA)
        observed=state['actual_process_OS_observation'];pid=runtime['pid'];launcher=state['launcher_pid'];expected_parent=state['supervisor_pid'] if pid==launcher else launcher
        check('exact_OS_PID_parent_chain',type(pid) is int and pid>0 and state['actual_pid']==pid and observed['status']=='exact_OS_identity_observed' and observed['pid']==pid and observed['parent_pid']==runtime['parent_pid']==expected_parent)
        stamp=lambda s:datetime.fromisoformat(s.replace('Z','+00:00'))
        stamps=[stamp(state['Popen_prelaunch_utc']),stamp(observed['creation_date_utc']),stamp(runtime['started_utc']),stamp(observed['observed_utc'])]
        check('OS_creation_time_bound',all(s.utcoffset() is not None for s in stamps) and stamps==sorted(stamps))
        command=state['command'];line=observed['command_line'].replace('/','\\').lower();exe=observed['executable_path'];py=identity['selected_python_identity']
        # Python's generated executable token is quoted iff needed; parse the
        # first token independently of the reviewed driver's implementation.
        first=observed['command_line'].split('"',2)[1] if observed['command_line'].startswith('"') else observed['command_line'].split(None,1)[0]
        check('OS_executable_actual_base_samefile',os.path.samefile(first,exe) and os.path.samefile(exe,py['base_executable_raw']) and os.path.samefile(exe,py['base_executable_resolved']) and sha(exe)==py['base_executable_sha256'] and os.path.samefile(py['chosen_python'],py['supervisor_sys_executable']) and os.path.samefile(py['chosen_python'],command[0]))
        norm=lambda s:str(Path(s).resolve()).replace('/','\\').lower()
        check('OS_command_worker_plan_output_binding',all(norm(q) in line for q in (p['source_pins']['worker']['path'],plan_path,raw_path)) and '--run' in line and command[-1]=='--run' and Path(command[2]).resolve()==Path(p['source_pins']['worker']['path']).resolve() and Path(command[4]).resolve()==plan_path and Path(command[6]).resolve()==raw_path)
        check('watchdog_exact_success_and_no_remaining_owned_PID',state['status']=='completed' and state['exit_code']==0 and state['timeout'] is False and post['status']=='NO_OWNED_PID_REMAINING' and post['queried_pids']==sorted(set([launcher,pid])) and post['actual_CIM_rows']==[])
        check('captured_runtime_identity_matches_final',all(state['runtime_metadata'][k]==runtime[k] for k in ('pid','parent_pid','started_utc','plan_sha256','source_sha256')))
        manifests=[json_file(k,p[k]['path']) for k in ('reference_manifest','init_manifest','control_manifest')]
        for kind,m in zip(('reference','init','control'),manifests):
            check('bundle_'+kind,m['schema']=='restored_prefix_separate_bundles:1' and m['bundle_kind']==kind and m['retarget_variant_id']==VARIANT and m['protocol']==p['protocol'] and m['constrained_protocol']==p['constrained_protocol'] and m['input_parent_protocol']==p['input_parent_protocol'] and sha(m['artifact']['path'])==m['artifact']['sha256']);inputs[kind+'_artifact']=m['artifact']
        reference,initial,control=[loadnp(m['artifact']['path']) for m in manifests]
        raw=loadnp(capture('raw_states',raw_path/'raw_states.npz'));requested=loadnp(capture('requested_initialization_and_control',raw_path/'requested_initialization_and_control.npz'));solver=json_file('solver_samples',raw_path/'solver_samples.json');events=[json.loads(s) for s in Path(capture('control_events',raw_path/'control_events.jsonl')).read_text().splitlines() if s];contacts=[json.loads(s) for s in Path(capture('raw_contacts',raw_path/'raw_contacts.jsonl')).read_text().splitlines() if s]
        check('61_full_numeric_states_finite',raw['full_qpos'].shape[0]==61 and raw['full_qvel'].shape[0]==61 and all(np.isfinite(raw[k]).all() for k in ('full_qpos','full_qvel','object_pose_w','object_twist_native','ctrl_rad','actuator_force_native','mocap_pos_w','mocap_quat_wxyz')))
        check('60_solver14event_coverage',len(solver)==60 and len(events)==14 and np.array_equal(raw['index'],np.arange(61)))
        check('clock61_absolute2us',np.max(np.abs(raw['raw_time_s']-np.arange(61)*.005))<=2e-6)
        import mujoco
        model=mujoco.MjModel.from_binary_path(p['model']['path']);host=mujoco.MjData(model);names=[f'robot/right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)];jids=np.array([model.joint(n).id for n in names]);qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids];aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids]);obj=model.body('object/cube').id;oj=model.joint('object/cube_freejoint').id;oq=model.jnt_qposadr[oj];ov=model.jnt_dofadr[oj]
        check('index_maps_exact_MJB',np.array_equal(requested['joint_qpos_ids'],qa) and np.array_equal(requested['joint_dof_ids'],va) and np.array_equal(requested['position_ctrl_ids'],aids) and np.array_equal(raw['position_ctrl_ids'],aids))
        lower=np.maximum(model.jnt_range[jids,0],model.actuator_ctrlrange[aids,0]);upper=np.minimum(model.jnt_range[jids,1],model.actuator_ctrlrange[aids,1]);innerlo=lower.astype(np.float32)+np.float32(.020);innerhi=upper.astype(np.float32)-np.float32(.020)
        track=control['control__fp32_old_export__tracking_u_exec_15_FP32_rad'];hold=control['control__fp32_old_export__hold_u_exec_15_FP32_rad'];schedule=hold if p['control_condition']==CONDITIONS[0] else track
        check('planned15_C2_exact_sameu0_and_condition',same32(track,np.minimum(np.maximum(track,innerlo),innerhi)) and same32(hold,np.broadcast_to(track[0],(15,20))) and same32(schedule,requested['selected_planned15_FP32_rad']) and same32(schedule,raw['selected_planned15_FP32_rad']) and same32(initial['u_planned0_FP32_rad'],track[0]))
        rate=float(np.max(np.abs(np.diff(schedule.astype(float),axis=0)/.02)));check('planned14_rate_le15',rate<=15)
        schedule_result=check_schedule(raw,solver,events,schedule,aids);check('independent_raw_schedule_phase',schedule_result['passed'],schedule_result)
        check('all61_fullctrl_matches_mapping',same32(raw['full_ctrl_native'][:,aids],raw['ctrl_rad']))
        check('all61_no_exogenous_no_mocap_write',np.max(raw['external_force_max_abs'])==0 and same32(raw['mocap_pos_w'],np.broadcast_to(raw['mocap_pos_w'][0],raw['mocap_pos_w'].shape)) and same32(raw['mocap_quat_wxyz'],np.broadcast_to(raw['mocap_quat_wxyz'][0],raw['mocap_quat_wxyz'].shape)))
        host.qpos[qa]=initial['q_init_requested_FP32_rad'];host.qpos[oq:oq+7]=initial['object_pose_init_requested_FP32_w'];host.qvel[:]=0;host.qvel[va]=initial['qd_init_requested_FP32_rad_s'];host.qvel[ov:ov+3]=initial['object_bodyorigin_v_requested_FP32_world_m_s'];host.qvel[ov+3:ov+6]=initial['object_angular_requested_FP32_body_rad_s'];host.ctrl[aids]=schedule[0]
        check('full_initial_q_qd_object_twist_FP32_exact',same32(raw['full_qpos'][0],host.qpos) and same32(raw['full_qvel'][0],host.qvel) and same32(requested['full_qpos_requested_FP32'],host.qpos) and same32(requested['full_qvel_requested_FP32'],host.qvel))
        check('initial_q0_never_projected_nonzero_velocity',same32(raw['joint_rad'][0],reference['q_goal_rad'][0]) and same32(raw['joint_velocity_rad_s'][0],reference['analytic_qd16_rad_s'][0]) and np.any(raw['full_qvel'][0]!=0))
        check('initial_mocap_modeldefault_unchanged',same32(raw['mocap_pos_w'][0],host.mocap_pos) and same32(raw['mocap_quat_wxyz'][0],host.mocap_quat) and same32(raw['ctrl_rad'][0],schedule[0]))
        R0=quat_matrix(raw['object_pose_w'][0,3:]);check('initial_native_bodyangular_and_worldlinear_frames',same32(raw['object_twist_native'][0,3:],R0.T@reference['object_omega_world_rad_s'][0]) and same32(raw['object_twist_native'][0,:3],reference['object_v_world_m_s'][0]) and np.all(model.body_ipos[obj]==0))
        # CPU kinematics from measured and requested fullqpos, not a physics
        # refresh or an independently saved GPU handpoint measurement.
        sys.path.insert(0,str(Path(p['robot_kinematics']['path']).parent));from retarget_palmroot_cpu import RobotKinematics
        robot=RobotKinematics(p['model']['path']);robot.data.qpos[:]=raw['full_qpos'][0];robot.data.mocap_pos[:]=raw['mocap_pos_w'][0];robot.data.mocap_quat[:]=raw['mocap_quat_wxyz'][0];robot.last_q=None;measured_fk=robot.forward(raw['joint_rad'][0])[0]
        robot.data.qpos[:]=host.qpos;robot.data.mocap_pos[:]=host.mocap_pos.astype(np.float32);robot.data.mocap_quat[:]=host.mocap_quat.astype(np.float32);robot.last_q=None;expected_fk=robot.forward(host.qpos[qa])[0]
        fkerr=float(np.max(np.abs(measured_fk-expected_fk)));check('CPU_FK_from_measured_full_initial_state',fkerr<=1e-12)
        startup=json_file('startup_initialization',raw_path/'startup_initialization_checks.json');check('logged_GPU_FK_tolerance_record_only',startup['startup_FK_vs_requested_FP32_max_abs_m']<=1e-6)
        context=loadnp(p['context']['path']);expected=loadnp(capture('expected_context_fields',raw_path/'expected_context_fields.npz'));applied=loadnp(capture('actual_context_applied',raw_path/'actual_context_applied.npz'));readbacks=loadnp(capture('actual_context_readbacks',raw_path/'actual_context_readbacks.npz'))
        check('context_key_coverage_8_and16',set(expected)==set(applied)==set(FIELDS) and set(readbacks)=={stage+'__'+f for stage in ('after_copy','after_set_const_0') for f in FIELDS})
        for field in FIELDS:
            check('context__'+field,np.array_equal(expected[field],context[field][0]) and same32(applied[field],context[field][0]) and same32(readbacks['after_copy__'+field],context[field][0]) and same32(readbacks['after_set_const_0__'+field],context[field][0]) and np.array_equal(applied[field],readbacks['after_set_const_0__'+field]))
        capacity=json_file('capacity_audit',raw_path/'capacity_audit.json');check('final_sticky_oneworld_zero_once_at60',capacity['status']=='OK' and capacity['overflow_bitmask']==0 and capacity['flags_by_world']==[0] and capacity['worlds']==1 and capacity['read_count']==1 and capacity['read_after_step']==60 and capacity['resets_after_initialization']==0 and capacity['backend_field']=='wp_data.overflow')
        check('capacity_saved_raw_counts_below_bound',np.all(raw['nacon']<capacity['contact_capacity']) and np.all(raw['nefc']<capacity['constraint_capacity']))
        check('measured_contacts_fullphase_finite',all(0<=c['step']<=60 and c['control_interval_used']==(0 if c['step']==0 else (c['step']-1)//4) and c['phase']==('startup_forward_only' if c['step']==0 else 'just_completed_solver_preintegration') and abs(c['solver_t_s']-raw['raw_time_s'][max(0,c['step']-1)])<=2e-6 and abs(c['poststate_t_s']-raw['raw_time_s'][c['step']])<=2e-6 and np.isfinite(c['force6_contact_frame']).all() and np.isfinite(c['pos_w']).all() and np.isfinite(c['distance_m']) for c in contacts))
        counts=np.bincount([c['step'] for c in contacts],minlength=61);check('raw_contact_rows_match_global_oneworld_nacon',np.array_equal(counts,raw['nacon']))
        # Independently reconstruct operational overlap categories and measured
        # HO wrench from the native saved contactframe and its geom side.
        object_geoms=set(np.flatnonzero(model.geom_bodyid==obj));hand_geoms={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])};contact_ok=True;depths=[];nonhand=[]
        for k in range(1,61):
            rows=[c for c in contacts if c['step']==k];category={'hand_object':0.,'hand_self':0.,'other':0.};env=0;force=np.zeros(3);moment=np.zeros(3);com=np.asarray(solver[k-1]['solver_object_COM_w'])
            for c in rows:
                a,b=c['geom'];hasobject=a in object_geoms or b in object_geoms;hashand=a in hand_geoms or b in hand_geoms;kind='hand_object' if hasobject and hashand else 'hand_self' if a in hand_geoms and b in hand_geoms else 'other';category[kind]=max(category[kind],max(0,-c['distance_m']));env+=int(hasobject and not hashand)
                if hasobject and hashand:
                    sign=1 if b in object_geoms else -1;frame=np.asarray(c['frame_rows_w']);f6=np.asarray(c['force6_contact_frame']);f=sign*frame.T@f6[:3];direct=sign*frame.T@f6[3:];force+=f;moment+=np.cross(np.asarray(c['pos_w'])-com,f)+direct
            s=solver[k-1];contact_ok &= all(abs(category[a]-s['penet_category_m'][a])<=1e-12 for a in category) and abs(max(category.values())-s['penetration_m'])<=1e-12 and env==s['object_environment_contacts'];depths.append(max(category.values()));nonhand.append(env)
            measured=s['measured_hand_object'];savedF=np.asarray(measured['force_w_N']).reshape(-1,3).sum(axis=0);contact_ok &= np.max(np.abs(savedF-force))<=1e-12
            trace_rows.append({'step':k,'solver_t_s':s['solver_t_s'],'poststate_t_s':s['poststate_t_s'],'control_interval':(k-1)//4,'contact_count':len(rows),'hand_object_depth_m':category['hand_object'],'hand_self_depth_m':category['hand_self'],'other_depth_m':category['other'],'HO_force_x_N':force[0],'HO_force_y_N':force[1],'HO_force_z_N':force[2],'HO_COM_moment_x_Nm':moment[0],'HO_COM_moment_y_Nm':moment[1],'HO_COM_moment_z_Nm':moment[2]})
        check('independent_contact_category_and_force_reconstruction',contact_ok)
        position=np.linalg.norm(raw['object_pose_w'][:,:3]-reference['object_COM_world_m'],axis=1);R=quat_matrix(raw['object_pose_w'][:,3:]);shaft=np.degrees(np.arccos(np.clip(np.sum(R[:,:,0]*reference['object_rotation_w'][:,:,0],axis=1),-1,1)));excess=np.maximum(np.maximum(model.jnt_range[jids,0]-raw['joint_rad'],raw['joint_rad']-model.jnt_range[jids,1]),0)
        capslo=np.where(model.actuator_forcelimited[aids],model.actuator_forcerange[aids,0],-np.inf);capshi=np.where(model.actuator_forcelimited[aids],model.actuator_forcerange[aids,1],np.inf)
        capslo=np.maximum(capslo,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,0],-np.inf));capshi=np.minimum(capshi,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,1],np.inf));forces=raw['actuator_force_native'][1:];ratio=np.where(forces>=0,forces/capshi,forces/capslo);charge=np.zeros(20);maxcharge=0.
        for j,dt in enumerate(np.diff(raw['raw_time_s'])):charge=np.clip(charge+dt*(np.minimum(ratio[j],1)-charge/2),0,1);maxcharge=max(maxcharge,float(charge.max()))
        g=p['gates'];primary=np.arange(0,61,4);quality={'moving_object_position_16':bool(position[primary].max()<=.02),'moving_shaft_angle_16':bool(shaft[primary].max()<=20),'actual_joint_state_all61':bool(excess.max()<=.001),'penetration_solver60':bool(max(depths)<=.003),'nonhand_object_contacts':sum(nonhand)==0,'no_exogenous':bool(np.max(raw['external_force_max_abs'])==0),'actual_ctrl_schedule_exact':schedule_result['passed'],'clock':bool(np.max(np.abs(raw['raw_time_s']-np.arange(61)*.005))<=2e-6),'raw_coverage':len(solver)==60 and raw['full_qpos'].shape[0]==61,'finite_raw':bool(all(np.isfinite(raw[k]).all() for k in ('full_qpos','full_qvel','ctrl_rad','actuator_force_native'))),'load_ratio':bool(ratio.max()<=1.01),'load_charge':maxcharge<=.8,'drop':bool(raw['object_pose_w'][:,2].min()>=host.mocap_pos[0,2]-.15),'capacity':capacity['status']=='OK'}
        metrics={'recomputed_quality_gates':quality,'recomputed_short_dynamic_pass':all(quality.values()),'moving_reference16_max_position_m':float(position[primary].max()),'moving_reference61_diagnostic_max_position_m':float(position.max()),'moving_reference16_max_shaft_deg':float(shaft[primary].max()),'moving_reference61_diagnostic_max_shaft_deg':float(shaft.max()),'all61_max_joint_excess_rad':float(excess.max()),'all60_max_generated_overlap_m':max(depths),'all60_max_load_ratio':float(ratio.max()),'load_charge_max':maxcharge,'planned14_target_rate_max_rad_s':rate,'CPU_FK_initial_raw_vs_requested_max_abs_m':fkerr,'GPU_hand_FK_independent_raw_coverage':'NOT_SAVED; CPU FK from full measured initial state is checked, worker GPU FK error log is separately retained','worker_quality_record':evaluation.get('short_dynamic_pass'),'quality_failure_does_not_veto_engineering_gate':True,'cost':{'worker_wall_seconds':runtime.get('wall_seconds'),'watchdog_elapsed_seconds':state.get('elapsed_s')},'actual_pid':pid,'OSCreationDate':observed['creation_date_utc']}
        check('worker_grade_matches_independent_raw_reconstruction',evaluation.get('short_dynamic_pass')==all(quality.values()) and evaluation.get('gates')==quality)
    except Exception as exc:check('saved_audit_complete_without_harness_error',False,repr(exc))
    passed=bool(checks) and all(c['passed'] for c in checks)
    report={'schema':'restoration_dynamic_prefix_saved_audit:1','status':'PASS_ENGINEERING' if passed else 'FAIL_OR_UNKNOWN_ENGINEERING','engineering_passed':passed,'can_release_next_condition_engineering_only':passed,'check_count':len(checks),'passed_checks':sum(c['passed'] for c in checks),'checks':checks,'metrics':metrics,'auditor':pin(__file__),'input_pins':inputs,'physics_steps_executed_by_auditor':0,'child_processes_started':0,'Torch_imported':'torch' in sys.modules,'Warp_imported':any(k in sys.modules for k in ('warp','mujoco_warp')),'wall_seconds':time.perf_counter()-start,'scope':'Saved raw engineering evidence only; sequential known-historyN1, no retry/selection/newphysics/learning/full3stask claim. Quality is independently reported and does not stop a separately authorized T0 if H0 engineering passes.'}
    if trace_rows:
        with (output/'solver_trace.csv').open('w',newline='',encoding='utf-8') as f:writer=csv.DictWriter(f,fieldnames=list(trace_rows[0]));writer.writeheader();writer.writerows(trace_rows)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');return report


def self_test(output):
    """Small schedule negatives, all synthetic; no artifact read/physics."""
    import copy
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'auditor_source.py').write_bytes(Path(__file__).read_bytes());t=np.arange(61)*.005;ci=np.r_[0,(np.arange(1,61)-1)//4];schedule=np.arange(15)[:,None]*np.ones((15,20))*.001;aids=np.arange(20)
    raw={'raw_time_s':t,'ctrl_rad':schedule[ci],'control_interval_used':ci,'force_prestate_index':np.r_[0,np.arange(60)],'force_solver_time_s':np.r_[0,t[:-1]],'actuator_force_native':np.zeros((61,20))}
    solver=[{'step':k,'prestate_index':k-1,'poststate_index':k,'control_interval_used':(k-1)//4,'phase':'just_completed_solver_preintegration','solver_t_s':t[k-1],'poststate_t_s':t[k],'ctrl_solver_native':schedule[(k-1)//4],'actuator_force_solver_native':np.zeros(20)} for k in range(1,61)]
    events=[{'before_step':k,'prestate_index':k-1,'previous_raw_state_ctrl_interval':(k-1)//4-1,'new_solver_ctrl_interval':(k-1)//4,'raw_prestate_time_s':t[k-1],'selected_planned_FP32_rad':schedule[(k-1)//4],'actual_ctrl_readback_native':schedule[(k-1)//4],'position_ctrl_ids':aids} for k in range(5,58,4)];checks=[]
    def test(name,call):
        try:
            if not call():raise AssertionError('Unexpected acceptance/classification')
            checks.append({'name':name,'passed':True})
        except Exception as exc:checks.append({'name':name,'passed':False,'error':repr(exc)})
    test('correct_state4u0_before5u1_step5pre4',lambda:check_schedule(raw,solver,events,schedule,aids)['passed'])
    def negative(name,change):
        rr,ss,ee=copy.deepcopy((raw,solver,events));change(rr,ss,ee)
        def rejected():
            try:return not check_schedule(rr,ss,ee,schedule,aids)['passed']
            except (ValueError,KeyError,IndexError):return True
        test(name,rejected)
    negative('anticipating_state4u1_rejected',lambda r,s,e:r['ctrl_rad'].__setitem__(4,schedule[1]))
    negative('wrong_event_beforestep4_rejected',lambda r,s,e:e[0].__setitem__('before_step',4))
    negative('force_mislabeled_poststate_rejected',lambda r,s,e:s[4].__setitem__('prestate_index',5))
    negative('force_timestamp_mislabeled_posttime_rejected',lambda r,s,e:s[4].__setitem__('solver_t_s',t[5]))
    negative('solver_old_target_after_update_rejected',lambda r,s,e:s[4].__setitem__('ctrl_solver_native',schedule[0]))
    negative('actual_event_readback_mismatch_rejected',lambda r,s,e:e[0].__setitem__('actual_ctrl_readback_native',schedule[0]))
    negative('event_target_mapping_mismatch_rejected',lambda r,s,e:e[0].__setitem__('position_ctrl_ids',aids[::-1]))
    negative('missing_event_rejected',lambda r,s,e:e.pop())
    negative('missing_solver_rejected',lambda r,s,e:s.pop())
    negative('missing_raw_state_rejected',lambda r,s,e:r.__setitem__('raw_time_s',t[:-1]))
    test('no_Torch_Warp_dynamics_import',lambda:not any(k in sys.modules for k in ('torch','warp','mujoco_warp','mujoco')))
    report={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','checks':checks,'check_count':len(checks),'auditor':pin(__file__),'future_raw_read':False,'physics_steps':0,'processes_started':0,'GPU_used':False};(output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path);ap.add_argument('--raw',type=Path);ap.add_argument('--watchdog',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:
        r=self_test(a.output);print(json.dumps(r,indent=2));raise SystemExit(r['status']!='PASS')
    if any(v is None for v in (a.plan,a.raw,a.watchdog)):ap.error('--plan --raw --watchdog are required for terminal saved audit')
    r=audit(a.plan,a.raw,a.watchdog,a.output);print(json.dumps({'status':r['status'],'checks':r['check_count'],'passed':r['passed_checks'],'metrics':r['metrics']},indent=2));raise SystemExit(not r['engineering_passed'])
if __name__=='__main__':main()
