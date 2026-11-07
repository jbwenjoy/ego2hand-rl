"""Artifact-only native151 A0 lifecycle/control audit; imports no production code."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sys
import traceback
import zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151_native_A0_v1'
WATCH=ROOT/'outputs/full151_native_A0_v1_watchdog'
OUT=ROOT/'outputs/full151_native_A0_lifecycle_independent_v1'
if OUT.exists():raise FileExistsError('Keep every saved-only audit attempt')
OUT.mkdir(parents=True,exist_ok=False)
(OUT/'audit_source.py').write_bytes(Path(__file__).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def array_sha(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def utc(v):return dt.datetime.fromisoformat(v.replace('Z','+00:00'))
checks={};details={}
def check(name,condition):
    try:
        assert bool(condition() if callable(condition) else condition)
        checks[name]=True
    except BaseException:
        checks[name]=False;details[name]=traceback.format_exc()

config=read(RUN/'config.json');cfgpath=Path(WATCH/'config_snapshot.json')
cfgsha=sha(cfgpath);inventory=read(RUN/'source/inventory.json')
watch=read(WATCH/'state.json');identity=read(WATCH/'engineering_identity.json')
post=read(WATCH/'post_terminal_CIM.json');runtime=read(RUN/'runtime.json')
completion=read(RUN/'completion.json');result=read(RUN/'result.json');evaluation=read(RUN/'evaluation.json')
check('136_source_table_identical_to_actual_config',len(inventory)==136 and inventory==config['source_pins'])
check('136_current_source_input_hashes_unchanged',lambda:all(sha(v['path'])==v['sha256'] for v in inventory.values()))
zip_rows=[]
with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
    expected=[f'{k:04d}_{Path(v["path"]).name}' for k,v in enumerate(inventory.values())]
    check('ZIP136_members_exact_order_no_extra',z.namelist()==expected)
    for (name,value),member in zip(inventory.items(),expected):
        got=hashlib.sha256(z.read(member)).hexdigest()
        zip_rows.append({'logical_name':name,'member':member,'matches_declared':got==value['sha256']})
    check('ZIP136_all_archived_bytes_match_declared',all(v['matches_declared'] for v in zip_rows))
check('actual_config_three_copies_byteequal',sha(RUN/'config.json')==cfgsha==sha(watch['command'][4]))
check('config_actual_bundle_and_protocol_aliases',config['input_bundle']==inventory['input_bundle'] and config['protocol']==inventory['protocol'])
proto=read(inventory['protocol']['path'])
check('new_full151_A0_only_no_training_or_prior',config['role']=='native_a0' and config['method']=='A0' and config['training']=={'num_envs':1} and config['contact_binding'] is None and config['training_authorized'] is False and proto['training_authorized'] is False)
check('single_fixed_run_no_retry_no_fallback',config['seed']==0 and config['GPU_run_authorized'] is True and config['automatic_retry'] is False and config['compile_fallback'] is False and watch['automatic_retry'] is False)
check('hard240_soft180_source_contract',config['hard_process_seconds']==watch['hard_seconds']==240 and config['soft_worker_seconds']==180)
check('watch_completed_exit0_no_timeout_or_failure',watch['status']=='completed' and watch['exit_code']==0 and watch['timeout'] is False and not (RUN/'failure.json').exists())
check('identity12_flags_true_raw_scope_preserved',identity['engineering_identity_qualified'] is True and len(identity['checks'])==12 and all(identity['checks'].values()) and identity['physical_outcome_qualified_by_supervisor'] is False)
osrec=watch['actual_process_OS_observation'];cmd=watch['command'];chosen=identity['selected_python_identity']
check('OS_actualPID_PPID_runtime_initial_terminal',watch['actual_pid']==osrec['pid']==runtime['pid']==watch['runtime_metadata']['pid']==40120 and watch['launcher_pid']==osrec['parent_pid']==runtime['parent_pid']==41436)
check('OS_CreationDate_within_predeclared_Popen_bounds',utc(watch['Popen_prelaunch_utc'])<=utc(osrec['creation_date_utc'])<=utc(watch['Popen_postlaunch_utc'])+dt.timedelta(seconds=1) and utc(watch['actual_creation_conservative_bounds']['lower_utc'])<=utc(osrec['creation_date_utc'])<=utc(watch['actual_creation_conservative_bounds']['upper_utc']))
expected_cmd=f'"{chosen["base_executable_raw"]}" '+' '.join(cmd[1:])
check('OS_command_exact_worker_config_output_run',osrec['command_line']==expected_cmd and Path(cmd[2]).samefile(inventory['worker']['path']) and Path(cmd[4]).samefile(ROOT/'configs/full151-a0-authorized-v1.json') and Path(cmd[6]).samefile(RUN) and cmd[7]=='--run')
check('venv_base_junction_samefile_actualOS',os.path.samefile(chosen['chosen_python'],chosen['supervisor_sys_executable']) and os.path.samefile(chosen['base_executable_raw'],chosen['base_executable_resolved']) and os.path.samefile(osrec['executable_path'],chosen['base_executable_raw']) and sha(osrec['executable_path'])==chosen['base_executable_sha256'])
check('runtime_source_and_config_are_actual_launch_bytes',runtime['source_sha256']==inventory['worker']['sha256']==watch['runtime_metadata']['source_sha256'] and runtime['plan_sha256']==cfgsha==completion['config_sha256']==watch['runtime_metadata']['plan_sha256'])
check('postterminal_saved_CIM_ownedPIDempty_only',post['status']=='NO_OWNED_PID_REMAINING' and post['queried_pids']==[40120,41436] and post['actual_CIM_rows']==[])
check('supervisor_and_identity_library_sourcebound',identity['supervisor_sha256']==inventory['supervisor']['sha256']==sha(WATCH/'supervisor_source.py') and identity['identity_source_sha256']==inventory['qualified_identity_v3']['sha256'])
check('terminal_runtime_completion_result_scopes_match',runtime['status']=='completed_native_A0' and completion['result']==result and result['evaluation']==evaluation and completion['quality_is_not_engineering_gate'] is True and result['cleanup_errors']==[])
check('zero_Actor_Runner_optimizer_training_old_migration',result['Actor_instances']==result['Runner_instances']==result['optimizer_objects_created']==result['training_updates']==result['optimizer_updates']==0 and result['old_checkpoint_migration'] is False)

raw=load(RUN/'raw_states.npz');requested=load(RUN/'requested_initialization_and_control.npz')
solver=read(RUN/'solver_samples.json');events=read(RUN/'control_events.json');transitions=read(RUN/'native_reward_transitions.json')
spec=read(inventory['input_bundle']['path']);ref=load(inventory['input__reference_arrays']['path']);init=load(inventory['input__init_arrays']['path']);control=load(inventory['input__control_arrays']['path'])
base=np.asarray(control['u_postC2_150_FP32_rad']);qa=requested['joint_qpos_ids'];va=requested['joint_dof_ids']
check('raw601_solver600_event150_reward150_no_padding',raw['joint_rad'].shape==(601,20) and raw['full_qpos'].shape==(601,27) and raw['full_qvel'].shape==(601,26) and len(solver)==600 and len(events)==150 and len(transitions)==150)
check('all_raw_saved_arrays_numeric_finite',all(np.isfinite(v).all() for v in raw.values()))
check('reported_API_attempts_returns600_600_150_150',evaluation['episode_API_counts']=={'step_attempted':600,'step_returned':600,'forward_attempted':150,'forward_returned':150}==runtime['episode_API_counts'] and evaluation['sampled_state_rows']==601 and evaluation['sampled_solver_rows']==600)
check('startup_work_explicitly_UNKNOWN_notwholeprocess',evaluation['construction_reset_setup_API_counts']=='UNKNOWN_NOT_WHOLEPROCESS_OBSERVED' and 'UNKNOWN_NOT_WHOLEPROCESS_OBSERVED' in result['counter_scope'])
check('planned150_tables_byteexact_all_saved_aliases',base.dtype==np.float32 and base.shape==(150,20) and np.array_equal(base,raw['selected_planned150_FP32_rad']) and np.array_equal(base,requested['selected_planned150_FP32_rad']))
check('raw_joint_columns_exact_fullstate_mapping',np.array_equal(raw['joint_rad'],raw['full_qpos'][:,qa]) and np.array_equal(raw['joint_velocity_rad_s'],raw['full_qvel'][:,va]))
check('requested_startup_fullq_FP32_exact',np.array_equal(raw['full_qpos'][0].astype(np.float32),requested['full_qpos_requested_FP32']) and np.array_equal(raw['joint_rad'][0].astype(np.float32),init['q_init_requested_FP32_rad']))
check('requested_startup_jointqd_analytic_not_nativeFD',np.array_equal(raw['joint_velocity_rad_s'][0].astype(np.float32),init['qd_init_requested_FP32_rad_s']) and np.array_equal(ref['analytic_qd151_rad_s'][0].astype(np.float32),init['qd_init_requested_FP32_rad_s']))
check('requested_startup_object_pose_worldlinear_exact',np.array_equal(raw['object_pose_w'][0].astype(np.float32),init['object_pose_init_requested_FP32_w']) and np.array_equal(raw['object_twist_native'][0,:3].astype(np.float32),init['object_bodyorigin_v_requested_FP32_world_m_s']))
omega_error=float(np.max(np.abs(raw['object_twist_native'][0,3:]-init['object_angular_requested_FP32_body_rad_s'])))
check('requested_startup_bodyomega_API_arithmetic_tolerance',omega_error<=1e-6 and omega_error==read(RUN/'startup_initialization_checks.json')['body_omega_max_error_rad_s'])
check('fixed_defaultmocap_all601_exact_requested',np.array_equal(raw['mocap_pos_w'],np.broadcast_to(requested['mocap_pos_requested_FP32'],raw['mocap_pos_w'].shape)) and np.array_equal(raw['mocap_quat_wxyz'],np.broadcast_to(requested['mocap_quat_requested_FP32'],raw['mocap_quat_wxyz'].shape)))
reset=read(RUN/'runtime_reset_readback.json')
check('reset_generation1_phase0_time0_selected0',reset['env_ids']==[0] and reset['generation']==[1] and reset['phase']==[0] and reset['time']==[0.])
check('reset_readback_matches_startupraw_fullq_qd',np.array_equal(np.asarray(reset['full_qpos'])[0],raw['full_qpos'][0]) and np.array_equal(np.asarray(reset['full_qvel'])[0],raw['full_qvel'][0]))
check('reset_control_cache_actual_u0_and_residual0',np.array_equal(np.asarray(reset['cached_target'])[0],base[0]) and np.array_equal(np.asarray(reset['actual_ctrl'])[0],base[0]) and all(np.count_nonzero(reset[k])==0 for k in ('EMA_residual','action','previous_action')))
check('all_actualctrl600_and_startup_matches150schedule',np.array_equal(raw['ctrl_rad'].astype(np.float32),base[np.r_[0,(np.arange(1,601)-1)//4]]) and raw['ctrl_exact'].all())
check('phase150events150transitions0_through149',all(e['phase']==k and e['preaction_phase']==[k] and e['postaction_phase']==[k+1] and t['phase']==k for k,(e,t) in enumerate(zip(events,transitions))))
check('only_last149_refend_and_timeout_terminalobs150',all(e['terminated']==[False] for e in events) and all(e['truncated']==[False] and e['reference_end']==[False] for e in events[:-1]) and events[-1]['truncated']==events[-1]['reference_end']==[True] and events[-1]['postaction_phase']==[150])
check('all150_zeroresidual_and_executedu_shared',all(np.count_nonzero(e['residual'])==0 and np.array_equal(np.asarray(e['actualu'])[0],base[k]) and np.array_equal(t['actualu'],base[k]) for k,(e,t) in enumerate(zip(events,transitions))))
check('preaction_memory_phase_start_end_age_noepisodecrossing',all(t['memory']['time']==[[float(k)]] and t['memory']['start']==[[0.]] and t['memory']['end']==[[151.]] and t['memory']['age']==[[float(k)]] for k,t in enumerate(transitions)))
check('preaction_memory_EMA_action_encoder_masks_allzero',all(np.count_nonzero(t['memory'][name])==0 for t in transitions for name in ('c1','c2','a1','a2','encoder_bias','mask','mask_remaining','mask_active')))
check('frozen_target_preaction_is_prior_actual_not_goalq',all(np.array_equal(np.asarray(t['memory']['frozen_target'])[0],base[max(k-1,0)]) for k,t in enumerate(transitions)))
check('native_reward_capture58_and_self15_shapes_finite',all(np.asarray(t['x0']).shape==np.asarray(t['x1_reward_snapshot']).shape==np.asarray(t['x1_after_Manager']).shape==(58,) and np.asarray(t['native_selfcollision15']).shape==(15,) and np.isfinite(t['native_selfcollision15']).all() for t in transitions))
check('state58_q_qd_pre4k_post4kplus4_exact_FP32',all(np.array_equal(np.asarray(t['x0'][:20]),raw['joint_rad'][4*k].astype(np.float32)) and np.array_equal(np.asarray(t['x0'][20:40]),raw['joint_velocity_rad_s'][4*k].astype(np.float32)) and np.array_equal(np.asarray(t['x1_reward_snapshot'][:20]),raw['joint_rad'][4*k+4].astype(np.float32)) and np.array_equal(np.asarray(t['x1_reward_snapshot'][20:40]),raw['joint_velocity_rad_s'][4*k+4].astype(np.float32)) for k,t in enumerate(transitions)))
check('native_torque_snapshot_matches_lastpreintegration_solver',all(np.array_equal(t['torque_reward_snapshot'],solver[4*k+3]['actuator_force_solver_native']) for k,t in enumerate(transitions)))
check('solver600_prepostindices_times_and_controlphase',all(s['step']==n and s['prestate_index']==n-1 and s['poststate_index']==n and s['control_interval_used']==(n-1)//4 and s['solver_t_s']==raw['raw_time_s'][n-1] and s['poststate_t_s']==raw['raw_time_s'][n] and np.array_equal(s['ctrl_solver_native'],base[(n-1)//4]) for n,s in enumerate(solver,1)))
check('reward_snapshot_force_preintegration_phase_explicit',all(t['phase_note']=='x1raw state at20ms; native self15/torque lastsolver15ms before originalManagerforward; reference reward uses preactionk' for t in transitions) and all(s['phase']=='just_completed_solver_preintegration_before_original_Manager_forward' for s in solver))

expected=np.zeros(601,np.float32)
for k in range(600):expected[k+1]=np.float32(expected[k]+np.float32(.005))
clock_drift=float(np.max(np.abs(raw['raw_time_s']-np.arange(601)*.005)))
check('clock601_raw_exact_FP32_expectedbits_and_canonicaltick',np.array_equal(raw['raw_time_s'],expected.astype(float)) and np.array_equal(raw['raw_time_s'].astype(np.float32).view(np.uint32),requested['expected_clock601_FP32_s'].view(np.uint32)) and np.array_equal(raw['canonical_time_s'],np.arange(601)*.005))
check('full15150us_AND_bitcontract_legacy2us_diagnostic',clock_drift<=5e-5 and clock_drift==evaluation['clock_audit']['max_abs_nominal_drift_s'] and evaluation['clock_audit']['legacy_2us_diagnostic_pass'] is False and clock_drift>2e-6)
cap=read(RUN/'capacity_audit.json');capfinal=read(RUN/'capacity_observer_final.json')
check('capacity3once_records_constructor_prereset_final',cap['record_count']==3 and [x['event'] for x in cap['records']]==['after_constructor_before_actual8','before_reset','final'] and cap['before_reset_read_count']==cap['final_read_count']==1)
check('capacity3actual_rawint32_nonempty_zero_OR_unknown0',all(x['raw_flags_dtype']=='int32' and x['raw_flags_shape']==[1] and x['raw_flags_values']==[0] and x['raw_bitmask']==0 and x['read_attempts_for_record']==1 and x['read_status']=='OK' and x['flags_cleared_by_observer'] is False for x in cap['records']) and cap['cumulative_OR_bitmask']==cap['unknown_read_count']==0 and cap['engineering_capacity_qualified'] is True)
check('capacity_final_hostsnapshot_same_rawrecords_no_additionalread',capfinal['records']==cap['records'] and capfinal['record_count']==3 and capfinal['finalized'] is True and cap['final_read_after_returned_episode_step']==cap['final_read_after_sampled_solver_row']==600)
check('capacity_no_reset_during_episode_actualallocbounds',cap['resets_during_probe']==evaluation['resets_after_start']==0 and cap['contact_capacity']==192 and cap['constraint_capacity']==4096 and cap['maximum_recorded_nacon']==int(raw['nacon'].max())<192 and cap['maximum_recorded_nefc']==int(raw['nefc'].max())<4096 and cap['flag_shape_is_world_count'] is False)
check('capacity_read_timestamps_order_before_terminalmetadata',all(utc(cap['records'][i]['observed_UTC'])<utc(cap['records'][i+1]['observed_UTC']) for i in (0,1)) and utc(cap['records'][-1]['observed_UTC'])<utc(runtime['finished_utc'])<utc(watch['finished_utc']))

actual=load(RUN/'fixed_physics_context.npz');before=load(RUN/'expected_context_fields.npz');reads=load(RUN/'actual_context_readbacks.npz');source_context=load(inventory['input__actual_context']['path'])
fields=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
check('actual8_expected_fields_from_frozen_sharedsource',all(np.array_equal(before[name],source_context[name][0]) for name in fields))
check('actual8_both_readbacks_and_reader_final_all_exact',all(np.array_equal(reads[stage+'__'+name],actual[name]) and np.array_equal(actual[name],np.broadcast_to(source_context[name][0].astype(np.float32),actual[name].shape)) for name in fields for stage in ('after_copy','after_set_const_0')))
check('xi46_actual_fixed_context_and_source_exact',actual['reader_context'].shape==(1,46) and np.array_equal(actual['reader_context'],source_context['reader_context'][:1]))
check('nominal_hostmodel_unchanged_sourceMJB',sha(RUN/'nominal_model.mjb')==inventory['input__model']['sha256'])

descriptor=read(RUN/'controller_descriptor.json');base_descriptor=read(ROOT/'outputs/full151_native_input_v1/descriptor.json')
root_descriptor=descriptor['root_descriptor'];addon={'trained_method':'A0','training_seed':0,'run_role':'native_a0','config_sha256':cfgsha,'execution_source_pins':inventory}
check('controller_rootdescriptor_is_exact_frozenbundle_plus_execidentity',root_descriptor=={**base_descriptor,**addon} and descriptor['root_descriptor_sha256']==digest(root_descriptor))
check('controller_signature_canonical_digest_not_relabelled104',descriptor['reference_controller_signature']==digest({k:v for k,v in descriptor.items() if k!='reference_controller_signature'}) and descriptor['format_version']==105 and descriptor['kind']=='FULL151_KNOWNROUGH_BASELINE_EMA_C2' and descriptor['exact_resume_supported'] is False)
check('controller150_baseline_source_and_C2identity_signature',descriptor['baseline_shape']==[150,20] and descriptor['baseline_u150_FP32_sha256']==array_sha(base) and descriptor['controller_source_sha256']==inventory['full151__controller.py']['sha256'] and descriptor['pure_projection_sha256']==inventory['actual_projection']['sha256'] and descriptor['margin_rad']==.020)
check('controller_real_intervals_terminal_inactive_u149_metadata',descriptor['active_intervals']==[0,150] and descriptor['terminal_observation']==150 and 'u149' in descriptor['inactive_imagination'] and 'residual EMA only' in descriptor['memory'])
reader_arrays=load(inventory['input__native_reader_arrays']['path'])
h=hashlib.sha256()
for name in ('joint_pos','object_pose','link_pos'):h.update(reader_arrays[name].tobytes())
h.update(np.array([151],dtype=np.int64).tobytes())
check('native_reference_signature_recomputed_saved_BODY21_frames',descriptor['native_reference_signature']==h.hexdigest() and reader_arrays['link_pos'].shape==(151,21,3))
check('sourcewindow_legacy200_351_no_shared_enrollment_no_egoN',base_descriptor['source_identity']['primary_pose_source_frame_range_half_open']==[200,351] and base_descriptor['reference_states']==151 and base_descriptor['control_intervals']==150 and base_descriptor['shared10_enrollment'] is False and result['source_lineage_N']==1 and result['independent_ego_N']==0)
check('knownrough_parentFAIL_and_physicalfalse_not_changed_by_A0',result['physical_reference_qualified'] is False and base_descriptor['physical_reference_qualified'] is False and base_descriptor['geometric_candidate_qualified'] is False and base_descriptor['parent_candidate_status']=='CPU_COUPLED_CANDIDATE_FAIL_RETAINED' and spec['training_authorized'] is False)
check('quality3failure_preserved_engineering_completion_not_taskpass',evaluation['complete'] is True and evaluation['native_A0_pass'] is False and len(evaluation['gates'])==13 and [k for k,v in evaluation['gates'].items() if not v]==['object_position_151','directed_shaft_151','penetration_solver600'])
check('bounded_nested_wallscopes_not_added',0<result['native_call_wall_seconds_excluding_imports']<runtime['wall_seconds']<watch['elapsed_s']<240)
check('saved_only_auditor_no_device_framework_imports',not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')))

source_pins={str(p.relative_to(ROOT)):pin(p) for p in [RUN/'config.json',RUN/'source/inventory.json',RUN/'source/execution_sources.zip',RUN/'raw_states.npz',RUN/'requested_initialization_and_control.npz',RUN/'runtime_reset_readback.json',RUN/'controller_descriptor.json',RUN/'runtime.json',RUN/'result.json',RUN/'evaluation.json',RUN/'completion.json',RUN/'control_events.json',RUN/'native_reward_transitions.json',RUN/'solver_samples.json',RUN/'capacity_audit.json',RUN/'capacity_observer_final.json',RUN/'actual_context_readbacks.npz',RUN/'fixed_physics_context.npz',WATCH/'state.json',WATCH/'engineering_identity.json',WATCH/'post_terminal_CIM.json']}
report={'schema':'full151_native_A0_saved_lifecycle_independent:1','status':'PASS_SAVED_ENGINEERING_SCOPE_QUALITY_FAIL_PRESERVED' if all(checks.values()) else 'INDEPENDENT_AUDIT_FAILURE_PRESERVED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'details':details,'inputs':source_pins,'auditor':pin(Path(__file__)),'ZIP_members':zip_rows,
 'measured_summary':{'actual_pid':40120,'launcher_pid':41436,'supervisor_pid':watch['supervisor_pid'],'OS_creation_date_utc':osrec['creation_date_utc'],'rawstates':601,'solver_records':600,'control_events':150,'native_reward_captures':150,'clock_max_drift_s':clock_drift,'bodyomega_API_init_error_rad_s':omega_error,'selfcollision15_positive_components':int(np.count_nonzero([t['native_selfcollision15'] for t in transitions])),'max_snapshot_vs_postManager_state58_difference':float(np.max(np.abs(np.array([t['x1_after_Manager'] for t in transitions])-np.array([t['x1_reward_snapshot'] for t in transitions])))),'native_wall_excluding_imports_s':result['native_call_wall_seconds_excluding_imports'],'entry_wall_s':runtime['wall_seconds'],'watch_wall_s':watch['elapsed_s']},
 'limitations':['No new Env, physics, FK, collision, Reader, WM, Actor, CUDA or process was executed. OS/CIM evidence is independently rebound from saved observations; no fresh CIM sampling is claimed.','136 current/archive hashes are checked now. The entry does not emit an independent terminal source-postcheck artifact; this audit does not invent one.','Constructor/reset/graph API counts remain UNKNOWN_NOT_WHOLEPROCESS_OBSERVED. runtime.physics_step_scope contains inherited wording saying separately recorded; result/evaluation explicitly retain the stronger UNKNOWN scope.','Cached actualu is source-backed by the controller equality assertions and reset cached readback; all150 actualu and rawctrl independently match saved baseline. The probe does not serialize each pending cache field.','selfcollision15 is the saved native reward snapshot before the original Manager forward. Raw sensor found/sensordata are not separately stored, so no independent reconstruction of sensor labels is claimed. It is not a hand-object contact label.','This one seen legacy trajectory A0 has N=1 source, not 151 or601 independent trials. Three original quality gates fail. Engineering completion is neither physical reference qualification nor a format105 trained checkpoint/loader qualification.'],
 'scope':{'saved_only':True,'production_functions_called':False,'quality_grader_reexecuted':False,'new_model_or_physics_work':False,'training_or_A2_objective_qualified':False}}
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':pin(OUT/'report.json'),'passed':report['passed'],'total':report['total'],'status':report['status']}))
if not all(checks.values()):raise SystemExit(2)
