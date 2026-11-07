"""Saved-only independent audit of one Manager/PenSimReader native A0.

Does not import/call native factory, probe.grade, Runner, Torch, Warp, a reset,
physics, native force/collision query or a subprocess. CPU FK is kinematics only.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,re,sys,time,zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
GATES={'object_position_max_m':.02,'directed_shaft_error_max_deg':20.,'actual_joint_hard_excess_max_rad':.001,'penetration_any_category_max_m':.003,'object_nonhand_candidate_contact_count_max':0,'external_force_max_abs':0.,'clock_absolute_tolerance_s':2e-6,'target_rate_max_rad_s':15.,'actuator_load_ratio_max':1.01,'load_charge_max':.8,'charge_decay_time_s':2.,'drop_center_below_default_wrist_m':.15}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def same32(a,b):return np.array_equal(np.asarray(a,np.float32).view(np.uint32),np.asarray(b,np.float32).view(np.uint32))


def rotation(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def phase_audit(raw,solver,events,baseline):
    problems=[]
    def require(ok,msg):
        if not ok:problems.append(msg)
    if len(raw['raw_time_s'])!=61 or len(solver)!=60 or len(events)!=15:raise ValueError('Native A0 requires61raw/60solver/15control records')
    indices=np.r_[0,(np.arange(1,61)-1)//4];require(same32(raw['ctrl_rad'],baseline[indices]),'Actual61ctrl schedule mismatch')
    for k,e in enumerate(events):
        require(e['phase']==k and np.array_equal(e['preaction_phase'],[k]) and np.array_equal(e['postaction_phase'],[k+1]),'Manager pre/post command phase mismatch')
        require(same32(e['actualu'],baseline[k:k+1]) and np.asarray(e['residual']).shape==(1,20) and not np.any(e['residual']),'Zeroresidual event actual target mismatch')
        require(np.asarray(e['terminated']).shape==(1,) and not bool(e['terminated'][0]) and bool(e['truncated'][0])==(k==14),'Natural end must be timeout only on interval14')
        require(np.isfinite(e['reward']).all(),'Nonfinite recorded native reward')
    for j,s in enumerate(solver):
        k=j+1;require(s['step']==k and s['prestate_index']==j and s['poststate_index']==k and s['control_interval_used']==j//4,'Solver pre/post index mismatch')
        require(s['phase']=='just_completed_solver_preintegration_before_Manager_forward','Native force phase mismatch')
        require(abs(s['solver_t_s']-raw['raw_time_s'][j])<=2e-6 and abs(s['poststate_t_s']-raw['raw_time_s'][k])<=2e-6,'Force/native state timestamp mismatch')
        require(same32(s['ctrl_solver_native'],baseline[j//4]) and np.array_equal(s['actuator_force_solver_native'],raw['actuator_force_native'][k]),'Native solver actual ctrl/force differs from raw')
    return {'passed':not problems,'problems':problems,'preaction_intervals':[0,15],'terminal_observation':15,'raw_state4_target':0,'solver_step5_target':1,'solver_step5_prestate':4,'no_u15':True}


def audit(config_path,raw_path,watchdog_path,output):
    started=time.perf_counter();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'auditor_source.py').write_bytes(Path(__file__).read_bytes());checks=[];inputs={};metrics={};trace=[]
    def check(name,ok,detail=None):checks.append({'name':name,'passed':bool(ok),**({'detail':detail} if detail is not None else {})})
    def load(name,p):inputs[name]=pin(p);return read(p)
    def npz(name,p):inputs[name]=pin(p);return arrays(p)
    try:
        config_path=Path(config_path).resolve(strict=True);raw_path=Path(raw_path).resolve(strict=True);watchdog_path=Path(watchdog_path).resolve(strict=True);p=load('authorized_config',config_path);runtime=load('runtime',raw_path/'runtime.json');state=load('watchdog_state',watchdog_path/'state.json');post=load('postterminal_CIM',watchdog_path/'post_terminal_CIM.json');identity=load('identity_record',watchdog_path/'engineering_identity.json');result=load('native_result',raw_path/'result.json');evaluation=load('worker_evaluation_record_only',raw_path/'evaluation.json')
        check('new_A0_no_training_config',p['schema']=='rough_prefix_native_run:1' and p['role']=='native_a0' and p['method']=='A0' and p['training']=={'num_envs':1} and p['GPU_run_authorized'] is True and p['training_authorized'] is False and p['contact_binding'] is None and p['evaluation_gates']==GATES)
        check('bounded_single_attempt_no_retry',p['hard_process_seconds']==120 and p['soft_worker_seconds']==110 and p['automatic_retry'] is False and p['compile_fallback'] is False)
        check('owned_exact_output',Path(p['output']).resolve()==raw_path)
        check('config_and_watchdog_snapshots_exact',sha(raw_path/'config.json')==sha(config_path) and sha(watchdog_path/'config_snapshot.json')==sha(config_path))
        for k,q in p['source_pins'].items():check('sourcepin__'+k,sha(q['path'])==q['sha256']);inputs['source__'+k]=q
        inventory=load('archived_inventory',raw_path/'source/inventory.json');check('archived_inventory_matches_config',inventory==p['source_pins']);inputs['archive']=pin(raw_path/'source/execution_sources.zip')
        with zipfile.ZipFile(raw_path/'source/execution_sources.zip') as z:
            check('archived_source_bytes_all_match',all(hashlib.sha256(z.read(f'{i:04d}_{Path(q["path"]).name}')).hexdigest()==q['sha256'] for i,(key,q) in enumerate(p['source_pins'].items())))
        entry=p['source_pins']['study__scripts/train_rough_prefix_pen_v2.py'];check('runtime_terminal_entry_config_identity',runtime['status']=='completed_native_A0' and runtime['source_sha256']==entry['sha256'] and runtime['plan_sha256']==sha(config_path) and runtime['role']=='native_a0' and runtime['method']=='A0' and runtime['optimizer_updates']==0 and runtime['physics_steps']==60)
        observed=state['actual_process_OS_observation'];pid=runtime['pid'];launcher=state['launcher_pid'];expected_parent=state['supervisor_pid'] if pid==launcher else launcher
        check('OS_PID_parent_exact_binding',observed['status']=='exact_OS_identity_observed' and state['actual_pid']==pid==observed['pid'] and runtime['parent_pid']==observed['parent_pid']==expected_parent)
        stamp=lambda s:datetime.fromisoformat(s.replace('Z','+00:00'));dates=[stamp(state['Popen_prelaunch_utc']),stamp(observed['creation_date_utc']),stamp(runtime['started_utc']),stamp(observed['observed_utc'])];check('OS_creation_time_bound',dates==sorted(dates) and all(d.utcoffset() is not None for d in dates))
        line=observed['command_line'];first=line.split('"',2)[1] if line.startswith('"') else line.split(None,1)[0];py=identity['selected_python_identity'];check('observed_executable_actual_base_samefile',os.path.samefile(first,observed['executable_path']) and os.path.samefile(observed['executable_path'],py['base_executable_raw']) and os.path.samefile(observed['executable_path'],py['base_executable_resolved']) and sha(observed['executable_path'])==py['base_executable_sha256'] and os.path.samefile(state['command'][0],py['chosen_python']))
        norm=lambda q:str(Path(q).resolve()).replace('/','\\').lower();check('OS_command_config_output_entry_bound',all(norm(q) in line.replace('/','\\').lower() for q in (entry['path'],config_path,raw_path)) and '--run' in line and Path(state['command'][4]).resolve()==config_path and Path(state['command'][6]).resolve()==raw_path)
        check('completed_exit0_no_owned_PID_remaining',state['status']=='completed' and state['exit_code']==0 and state['timeout'] is False and post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[] and post['queried_pids']==sorted(set([pid,launcher])))
        check('captured_runtime_identity_unchanged',all(state['runtime_metadata'][k]==runtime[k] for k in ('pid','parent_pid','started_utc','source_sha256','plan_sha256')))
        check('native_A0_fullcoverage_cadence_zero_optimizer',result['status']=='COMPLETED_NATIVE_A0_ENGINEERING_QUALITY_RECORDED' and result['optimizer_updates']==0 and result['cleanup_errors']==[] and evaluation['complete'] is True and evaluation['physical_steps']==60 and evaluation['native_postinterval_forward_calls']==15 and evaluation['control_intervals']==15 and evaluation['resets_after_start']==0)
        effective=load('effective_probe_config',raw_path/'effective_probe_config.json');check('no_Runner_or_optimizer_branch',effective['mode']=='one_world_native_Manager_PenSimReader_no_training_runner' and effective['num_envs']==1 and effective['optimizer_objects_created']==0 and not (raw_path/'train').exists())
        bundle=load('input_bundle',p['input_bundle']['path']);check('new_bundle_hash_window_unqualified',sha(p['input_bundle']['path'])==p['input_bundle']['sha256'] and bundle['raw_source_identity']['primary_pose_source_frame_range_half_open']==[200,216] and bundle['shared10_enrollment'] is False and bundle['physical_reference_qualified'] is False)
        for k,q in bundle['source_pins'].items():check('bundlepin__'+k,sha(q['path'])==q['sha256'])
        reference=npz('frozen_reference_arrays',bundle['source_pins']['reference_arrays']['path']);initial=npz('frozen_init_arrays',bundle['source_pins']['init_arrays']['path']);controls=npz('frozen_control_arrays',bundle['source_pins']['control_arrays']['path']);physical=npz('frozen_context',bundle['context']['path']);baseline=controls['control__fp32_old_export__tracking_u_exec_15_FP32_rad']
        raw=npz('raw_states',raw_path/'raw_states.npz');requested=npz('requested_init_control',raw_path/'requested_initialization_and_control.npz');solver=load('solver_samples',raw_path/'solver_samples.json');events=load('control_events',raw_path/'control_events.json');contacts=[json.loads(s) for s in Path(raw_path/'raw_contacts.jsonl').read_text().splitlines() if s];inputs['raw_contacts']=pin(raw_path/'raw_contacts.jsonl')
        check('raw61_fullfinite_expectedclock',np.array_equal(raw['index'],np.arange(61)) and np.max(np.abs(raw['raw_time_s']-np.arange(61)*.005))<=2e-6 and all(np.isfinite(raw[k]).all() for k in ('full_qpos','full_qvel','ctrl_rad','actuator_force_native','object_pose_w','object_twist_native','mocap_pos_w','mocap_quat_wxyz')))
        phases=phase_audit(raw,solver,events,baseline);check('native_15interval_60solver_61state_phase',phases['passed'],phases)
        import mujoco
        model=mujoco.MjModel.from_binary_path(bundle['model']['path']);check('nominal_MJB_exact',sha(raw_path/'nominal_model.mjb')==bundle['model']['sha256']);inputs['nominal_model']=pin(raw_path/'nominal_model.mjb');default=mujoco.MjData(model);jids=np.array([model.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)]);qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids];aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids]);obj=model.body('object/cube').id;oj=model.joint('object/cube_freejoint').id;oq=model.jnt_qposadr[oj];ov=model.jnt_dofadr[oj]
        check('index_maps_caps_actual_MJB',np.array_equal(requested['joint_qpos_ids'],qa) and np.array_equal(requested['joint_dof_ids'],va) and np.array_equal(requested['position_ctrl_ids'],aids) and np.array_equal(raw['joint_lower_rad'],model.jnt_range[jids,0]) and np.array_equal(raw['joint_upper_rad'],model.jnt_range[jids,1]))
        wanted_q=default.qpos.astype(np.float32);wanted_q[qa]=initial['q_init_requested_FP32_rad'];wanted_q[oq:oq+7]=initial['object_pose_init_requested_FP32_w'];wanted_v=np.zeros(model.nv,np.float32);wanted_v[va]=initial['qd_init_requested_FP32_rad_s'];wanted_v[ov:ov+3]=initial['object_bodyorigin_v_requested_FP32_world_m_s'];wanted_v[ov+3:ov+6]=initial['object_angular_requested_FP32_body_rad_s']
        check('full_requested_init_binding',np.array_equal(requested['full_qpos_requested_FP32'],wanted_q) and np.array_equal(requested['full_qvel_requested_FP32'],wanted_v) and same32(raw['full_qpos'][0],wanted_q))
        check('actual_nonzero_fullvelocity_and_world_body_frames',same32(raw['full_qvel'][0,va],wanted_v[va]) and same32(raw['full_qvel'][0,ov:ov+3],wanted_v[ov:ov+3]) and np.max(np.abs(raw['full_qvel'][0,ov+3:ov+6]-wanted_v[ov+3:ov+6]))<=1e-6 and np.any(raw['full_qvel'][0]!=0) and np.all(model.body_ipos[obj]==0))
        check('all_other_velocity_components_match_full_requested',np.max(np.abs(raw['full_qvel'][0]-wanted_v))<=1e-6)
        check('default_mocap_initial_and_all61_constant',same32(raw['mocap_pos_w'][0],default.mocap_pos) and same32(raw['mocap_quat_wxyz'][0],default.mocap_quat) and np.array_equal(raw['mocap_pos_w'],np.broadcast_to(raw['mocap_pos_w'][0],raw['mocap_pos_w'].shape)) and np.array_equal(raw['mocap_quat_wxyz'],np.broadcast_to(raw['mocap_quat_wxyz'][0],raw['mocap_quat_wxyz'].shape)))
        check('baseline_and_same_initial_u0_exact',same32(raw['selected_planned15_FP32_rad'],baseline) and same32(requested['selected_planned15_FP32_rad'],baseline) and same32(raw['ctrl_rad'][0],initial['u_planned0_FP32_rad']))
        lower=np.maximum(model.jnt_range[jids,0],model.actuator_ctrlrange[aids,0]).astype(np.float32);upper=np.minimum(model.jnt_range[jids,1],model.actuator_ctrlrange[aids,1]).astype(np.float32);check('all15_C2_domain_no_repair',same32(baseline,np.minimum(np.maximum(baseline,lower+np.float32(.02)),upper-np.float32(.02))))
        # Actual GPU handpoint arrays are not saved. Independently reconstruct
        # BODY FK from measured fullq0 and compare to requested full state.
        aa=mujoco.MjData(model);bb=mujoco.MjData(model);aa.qpos[:]=raw['full_qpos'][0];aa.mocap_pos[:]=raw['mocap_pos_w'][0];aa.mocap_quat[:]=raw['mocap_quat_wxyz'][0];bb.qpos[:]=wanted_q;bb.mocap_pos[:]=default.mocap_pos.astype(np.float32);bb.mocap_quat[:]=default.mocap_quat.astype(np.float32);mujoco.mj_kinematics(model,aa);mujoco.mj_kinematics(model,bb);bodies=np.array([model.body('robot/right_palm_link').id]+[model.body(f'robot/right_finger{f}_link{j}').id for f in range(1,6) for j in range(1,5)]);fkerr=float(np.max(np.abs(aa.xpos[bodies]-bb.xpos[bodies])));check('CPU_initial_BODY_FK_from_actual_fullq',fkerr<=1e-12)
        expected=npz('expected_actual8',raw_path/'expected_context_fields.npz');readbacks=npz('actual8_readbacks',raw_path/'actual_context_readbacks.npz');fixed=npz('postreset_fixed_physics_context',raw_path/'fixed_physics_context.npz');check('eight_fields_and16_realreadback_keys',set(expected)==set(FIELDS) and set(readbacks)=={s+'__'+f for s in ('after_copy','after_set_const_0') for f in FIELDS})
        for field in FIELDS:
            check('actual8__'+field,np.array_equal(expected[field],physical[field][0]) and all(np.array_equal(value,np.broadcast_to(physical[field][0].astype(value.dtype),value.shape)) for value in (readbacks['after_copy__'+field],readbacks['after_set_const_0__'+field],fixed[field])))
        check('actual_reader_context46_matches_source',fixed['reader_context'].shape==(1,46) and np.array_equal(fixed['reader_context'],physical['reader_context'][:1]))
        capacity=load('capacity',raw_path/'capacity_audit.json');check('final_sticky0_once_at60_noreset',capacity['status']=='OK' and capacity['flags_by_world']==[0] and capacity['read_count']==1 and capacity['read_after_step']==60 and capacity['resets_during_probe']==0)
        check('all_actual_counts_below_allocated_cap',np.max(raw['nacon'])<capacity['contact_capacity'] and np.max(raw['nefc'])<capacity['constraint_capacity'])
        check('no_external_force_all61',np.max(raw['external_force_max_abs'])==0)
        check('contact_row_coverage_matches_nacon',np.array_equal(np.bincount([c['step'] for c in contacts],minlength=61),raw['nacon']))
        check('contact_phase_and_values_finite',all(c['phase']==('startup_after_native_reset' if c['step']==0 else 'preintegration_solver') and np.isfinite(c['distance_m']) and np.isfinite(c['pos_w']).all() and np.isfinite(c['frame_rows_w']).all() and np.isfinite(c['force6_contact_frame']).all() for c in contacts))
        object_geoms=set(np.flatnonzero(model.geom_bodyid==obj));hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])};nativecontacts_ok=True;depth=[];nonhand=[]
        for k in range(1,61):
            category={'hand_object':0.,'hand_self':0.,'other':0.};envcontacts=0;force=np.zeros(3);moment=np.zeros(3);com=np.asarray(solver[k-1]['solver_object_COM_w'])
            for c in (c for c in contacts if c['step']==k):
                a,b=c['geom'];ho=a in object_geoms or b in object_geoms;hh=a in hand or b in hand;kind='hand_object' if ho and hh else 'hand_self' if a in hand and b in hand else 'other';category[kind]=max(category[kind],max(0,-c['distance_m']));envcontacts+=int(ho and not hh)
                if ho and hh:
                    sign=1 if b in object_geoms else -1;frame=np.asarray(c['frame_rows_w']);f6=np.asarray(c['force6_contact_frame']);f=sign*frame.T@f6[:3];force+=f;moment+=np.cross(np.asarray(c['pos_w'])-com,f)+sign*frame.T@f6[3:]
            s=solver[k-1];nativecontacts_ok &= all(abs(category[name]-s['penet_category_m'][name])<=1e-12 for name in category) and abs(max(category.values())-s['penetration_m'])<=1e-12 and envcontacts==s['object_environment_contacts'];savedF=np.asarray(s['measured_hand_object']['force_w_N']).reshape(-1,3).sum(0);nativecontacts_ok &= np.max(np.abs(savedF-force))<=1e-12;depth.append(max(category.values()));nonhand.append(envcontacts)
            trace.append({'step':k,'solver_t_s':s['solver_t_s'],'poststate_t_s':s['poststate_t_s'],'target_interval':(k-1)//4,'hand_object_depth_m':category['hand_object'],'hand_self_depth_m':category['hand_self'],'other_depth_m':category['other'],'HO_Fz_N':force[2],'HO_COM_Mx_Nm':moment[0],'HO_COM_My_Nm':moment[1],'HO_COM_Mz_Nm':moment[2]})
        check('independent_contact_depth_and_wrench_reconstruction',nativecontacts_ok)
        caplo=model.actuator_forcerange[aids,0].copy();caphi=model.actuator_forcerange[aids,1].copy();caplo=np.maximum(caplo,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,0],-np.inf));caphi=np.minimum(caphi,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,1],np.inf));check('force_bounds_from_actual_transmissions',np.array_equal(caplo,raw['force_lower_Nm']) and np.array_equal(caphi,raw['force_upper_Nm']) and np.array_equal(caplo,requested['force_lower_Nm']) and np.array_equal(caphi,requested['force_upper_Nm']) and np.array_equal(model.actuator_gear[aids],np.broadcast_to([1.,0,0,0,0,0],(20,6))))
        R=rotation(raw['object_pose_w'][:,3:]);pos=np.linalg.norm(raw['object_pose_w'][:,:3]-reference['object_COM_world_m'],axis=1);angle=np.degrees(np.arccos(np.clip(np.sum(R[:,:,0]*reference['object_rotation_w'][:,:,0],axis=1),-1,1)));primary=np.arange(0,61,4);excess=np.maximum(np.maximum(model.jnt_range[jids,0]-raw['joint_rad'],raw['joint_rad']-model.jnt_range[jids,1]),0);forces=np.asarray([s['actuator_force_solver_native'] for s in solver]);ratio=np.where(forces>=0,forces/caphi,forces/caplo);charge=np.zeros(20);maxcharge=0
        for dt,row in zip(np.diff(raw['raw_time_s']),ratio):charge=np.clip(charge+dt*(np.minimum(row,1)-charge/2),0,1);maxcharge=max(maxcharge,float(charge.max()))
        actual15=raw['ctrl_rad'][[0,*range(5,58,4)]];rate=float(np.abs(np.diff(actual15,axis=0)).max()/.02);quality={'object_position_16':bool(pos[primary].max()<=.02),'directed_shaft_16':bool(angle[primary].max()<=20),'actual_q_all61':bool(excess.max()<=.001),'penetration_solver60':bool(max(depth)<=.003),'no_nonhand_object_contacts':sum(nonhand)==0,'no_exogenous':bool(raw['external_force_max_abs'].max()==0),'all_actual_u_exact':same32(raw['ctrl_rad'],baseline[np.r_[0,(np.arange(1,61)-1)//4]]),'actual_target_rate':rate<=15,'clock':bool(np.max(np.abs(raw['raw_time_s']-np.arange(61)*.005))<=2e-6),'load_ratio':bool(ratio.max()<=1.01),'load_charge':maxcharge<=.8,'drop':bool(raw['object_pose_w'][:,2].min()>=raw['mocap_pos_w'][0,0,2]-.15),'capacity':capacity['status']=='OK'}
        metrics={'independent_quality_gates':quality,'native_A0_quality_pass':all(quality.values()),'max_object_position_error16_m':float(pos[primary].max()),'max_object_position_error61_m':float(pos.max()),'max_shaft_error16_deg':float(angle[primary].max()),'max_shaft_error61_deg':float(angle.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_native_overlap_m':max(depth),'max_actual_load_ratio':float(ratio.max()),'max_charge':maxcharge,'actual_planned_target_rate_rad_s':rate,'initial_BODY_FK_CPU_error_m':fkerr,'initial_native_bodyomega_API_error_rad_s':float(np.max(np.abs(raw['full_qvel'][0,ov+3:ov+6]-wanted_v[ov+3:ov+6]))),'GPU_handpoint_independent_FK':'NOT_SAVED; CPU from raw fullinitial state only','cost':{'whole_worker_seconds':runtime['wall_seconds'],'native_call_seconds_nested':result['native_call_wall_seconds_excluding_imports'],'watchdog_slot_seconds':state['elapsed_s']},'actual_pid':pid,'OSCreationDate':observed['creation_date_utc'],'full3s_task':'NOT_ASSESSED','historical_directWarp_T0':'separate cadence and condition, not a pooled baseline'}
        check('worker_grade_matches_raw_recomputation',evaluation['native_A0_pass']==all(quality.values()) and evaluation['gates']==quality)
    except Exception as exc:check('audit_harness_complete',False,repr(exc))
    passed=bool(checks) and all(x['passed'] for x in checks);report={'schema':'rough_prefix_native_A0_saved_independent_audit:1','status':'PASS_ENGINEERING' if passed else 'FAIL_OR_UNKNOWN_ENGINEERING','engineering_passed':passed,'check_count':len(checks),'passed_checks':sum(x['passed'] for x in checks),'checks':checks,'metrics':metrics,'input_pins':inputs,'auditor':pin(__file__),'physics_steps_executed':0,'Runner_or_optimizer_constructed':False,'probe_grade_or_native_factory_called':False,'Torch_imported':'torch' in sys.modules,'Warp_imported':any(k in sys.modules for k in ('warp','mujoco_warp')),'wall_seconds':time.perf_counter()-started,'scope':'One saved native Manager A0, zero residual and no research training. Full raw engineering/capacity is separate from quality; no method/CHORD/ego benefit or force prediction claim.'}
    if trace:
        with (output/'solver_trace.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');return report


def self_test(output):
    import copy
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'auditor_source.py').write_bytes(Path(__file__).read_bytes());baseline=np.arange(15)[:,None]*np.ones((15,20))*.001;t=np.arange(61)*.005;idx=np.r_[0,(np.arange(1,61)-1)//4];raw={'raw_time_s':t,'ctrl_rad':baseline[idx],'actuator_force_native':np.zeros((61,20))};solver=[{'step':k,'prestate_index':k-1,'poststate_index':k,'control_interval_used':(k-1)//4,'solver_t_s':t[k-1],'poststate_t_s':t[k],'phase':'just_completed_solver_preintegration_before_Manager_forward','ctrl_solver_native':baseline[(k-1)//4],'actuator_force_solver_native':np.zeros(20)} for k in range(1,61)];events=[{'phase':k,'preaction_phase':[k],'postaction_phase':[k+1],'actualu':baseline[k:k+1],'residual':np.zeros((1,20)),'terminated':[False],'truncated':[k==14],'reward':[1.]} for k in range(15)];checks=[]
    def probe(name,edit=None):
        r,s,e=copy.deepcopy((raw,solver,events))
        if edit:edit(r,s,e)
        try:actual=phase_audit(r,s,e,baseline)['passed'];passed=actual==(edit is None)
        except (ValueError,KeyError):passed=edit is not None
        checks.append({'name':name,'passed':passed})
    probe('native_15events_61beforeforward_states_60solver_correct');probe('old14_event_assumption_rejected',lambda r,s,e:e.pop());probe('state4_cannot_anticipate_u1',lambda r,s,e:r['ctrl_rad'].__setitem__(4,baseline[1]));probe('solverstep5_usesu1',lambda r,s,e:s[4].__setitem__('ctrl_solver_native',baseline[0]));probe('force_phase_cannot_be_post_Manager_forward',lambda r,s,e:s[3].__setitem__('phase','after_Manager_forward'));probe('forceprestate_not_poststate',lambda r,s,e:s[3].__setitem__('prestate_index',4));probe('terminal_timeout_only_at_interval14',lambda r,s,e:e[13].__setitem__('truncated',[True]));probe('no_final_timeout_rejected',lambda r,s,e:e[14].__setitem__('truncated',[False]));probe('no_u15_event',lambda r,s,e:e[14].__setitem__('postaction_phase',[16]));probe('nonzero_residual_not_A0',lambda r,s,e:e[4].__setitem__('residual',np.ones((1,20))));checks.append({'name':'no_Torch_Warp_MuJoCo_import','passed':not any(k in sys.modules for k in ('torch','warp','mujoco_warp','mujoco'))});r={'status':'PASS' if all(c['passed'] for c in checks) else 'FAIL','check_count':len(checks),'checks':checks,'source':pin(__file__),'future_raw_read':False,'physics_steps':0,'GPU_used':False};(output/'report.json').write_text(json.dumps(r,indent=2)+'\n');return r


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--config',type=Path);ap.add_argument('--raw',type=Path);ap.add_argument('--watchdog',type=Path);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:r=self_test(a.output);print(json.dumps(r,indent=2));raise SystemExit(r['status']!='PASS')
    if any(x is None for x in (a.config,a.raw,a.watchdog)):ap.error('--config --raw --watchdog are required after true terminal')
    r=audit(a.config,a.raw,a.watchdog,a.output);print(json.dumps({'status':r['status'],'checks':r['check_count'],'passed':r['passed_checks'],'metrics':r['metrics']},indent=2));raise SystemExit(not r['engineering_passed'])
if __name__=='__main__':main()
