"""Independent saved-only v2 audit; only tuple/list JSON representation correction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import ctypes, datetime, hashlib, json, math, sys, zipfile
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/smoke_a1l_v2'
WATCH=ROOT/'outputs/smoke_a1l_v2_watchdog'
OUT=ROOT/'outputs/independent_smoke_a1l_v2'
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def rows(p):return [json.loads(line) for line in Path(p).read_text().splitlines()]
def finite(v):
    if torch.is_tensor(v):return bool(torch.isfinite(v).all())
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
    return math.isfinite(v) if isinstance(v,float) else True
def argv(s):
    n=ctypes.c_int();fn=ctypes.windll.shell32.CommandLineToArgvW;fn.argtypes=[ctypes.c_wchar_p,ctypes.POINTER(ctypes.c_int)];fn.restype=ctypes.POINTER(ctypes.c_wchar_p)
    p=fn(s,ctypes.byref(n))
    try:return [p[i] for i in range(n.value)]
    finally:ctypes.windll.kernel32.LocalFree(p)
def utc(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))

def main():
    OUT.mkdir(exist_ok=False);torch.set_num_threads(1);checks={};detail={}
    def ck(k,v):checks[k]=bool(v)
    cfg=read(RUN/'config.json');runtime=read(RUN/'runtime.json');result=read(RUN/'result.json');completion=read(RUN/'training_completion.json');watch=read(WATCH/'state.json');identity=read(WATCH/'engineering_identity.json');post=read(WATCH/'post_terminal_CIM.json')
    args=watch['command'];config=Path(args[args.index('--config')+1]);protocol=read(cfg['protocol']['path']);count={'wm':20,'fail_head':20,'actor':8,'critic':32}
    ck('config_copied_exact_and_GPU_authorized_smoke_only',sha(config)==sha(RUN/'config.json')==runtime['plan_sha256'] and cfg['GPU_run_authorized'] is True and cfg['role']=='engineering_smoke' and cfg['method']=='A1L' and cfg['seed']==0 and cfg['training_authorized'] is True)
    ck('protocol3_role_seed_budget_and_physical_false',sha(cfg['protocol']['path'])==cfg['protocol']['sha256'] and protocol['protocol_revision']==3 and protocol['allowed_roles']==['engineering_smoke'] and protocol['allowed_seeds']==[0] and cfg['training']==protocol['engineering_smoke_training'] and protocol['physical_reference_qualified'] is False)
    osr=watch['actual_process_OS_observation'];observed=argv(osr['command_line']);py=identity['selected_python_identity']
    ck('watch_completed_no_timeout_no_retry',watch['status']=='completed' and watch['exit_code']==0 and watch['timeout'] is False and watch['automatic_retry'] is False and watch['hard_seconds']==cfg['hard_process_seconds']==300)
    ck('OS_actual_PID_parent_runtime_exact',osr['status']=='exact_OS_identity_observed' and osr['pid']==runtime['pid']==watch['actual_pid']==identity['actual_pid'] and osr['parent_pid']==runtime['parent_pid']==watch['launcher_pid']==identity['launcher_pid'] and runtime['pid']!=runtime['parent_pid'])
    ck('OS_command_plan_output_worker_and_samefile_base',observed[1:]==args[1:] and Path(args[0]).samefile(py['chosen_python']) and Path(observed[0]).samefile(osr['executable_path']) and Path(observed[0]).samefile(py['base_executable_raw']) and Path(py['base_executable_raw']).samefile(py['base_executable_resolved']) and sha(observed[0])==py['base_executable_sha256'] and Path(args[args.index('--output')+1]).resolve()==RUN.resolve())
    ck('OS_creation_time_order_and_runtime_source',utc(watch['Popen_prelaunch_utc'])<=utc(osr['creation_date_utc'])<=utc(runtime['started_utc'])<=utc(osr['observed_utc'])<=utc(watch['finished_utc']) and sha(args[2])==runtime['source_sha256'] and watch['runtime_metadata']['pid']==runtime['pid'] and watch['runtime_metadata']['plan_sha256']==runtime['plan_sha256'])
    ck('postterminal_saved_owned_PIDs_absent',post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[] and set(post['queried_pids'])=={runtime['pid'],runtime['parent_pid']})
    ck('supervisor_and_config_snapshots_exact',sha(WATCH/'supervisor_source.py')==identity['supervisor_sha256'] and read(WATCH/'config_snapshot.json')==cfg and identity['engineering_identity_qualified'] is True)
    inv=read(RUN/'source/inventory.json');ck('source_inventory_matches_config',inv==cfg['source_pins'])
    with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
        ck('all_source_archive_bytes_exact',len(z.namelist())==len(inv) and all(hashlib.sha256(z.read(f'{i:04d}_{Path(v["path"]).name}')).hexdigest()==v['sha256'] for i,v in enumerate(inv.values())))
    ck('all_source_current_bytes_preserved',all(sha(v['path'])==v['sha256'] for v in inv.values()))
    bundle=read(cfg['input_bundle']['path']);ck('bundle_and_all_input_sourcepins_preserved',sha(cfg['input_bundle']['path'])==cfg['input_bundle']['sha256'] and all(sha(v['path'])==v['sha256'] for v in bundle['source_pins'].values()) and bundle['physical_reference_qualified'] is False and bundle['training_authorized'] is False and bundle['raw_source_identity']['primary_pose_source_frame_range_half_open']==[200,216])
    ref=load(bundle['source_pins']['reference_arrays']['path']);ini=load(bundle['source_pins']['init_arrays']['path']);ctl=load(bundle['source_pins']['control_arrays']['path']);baseline=ctl['control__fp32_old_export__tracking_u_exec_15_FP32_rad'];bridge=load(bundle['source_pins']['contact_reference']['path'])
    ck('analytic_reference_init_not_native_FD_and_baseline0',np.array_equal(ini['q_init_requested_FP32_rad'],ref['q_goal_rad'][0].astype(np.float32)) and np.array_equal(ini['qd_init_requested_FP32_rad_s'],ref['analytic_qd16_rad_s'][0].astype(np.float32)) and np.array_equal(ini['u_planned0_FP32_rad'],baseline[0]) and np.array_equal(ref['q_goal_rad'],bridge['joint_pos_rad']) and np.array_equal(ref['mapped_object_pose_w'],bridge['object_pose_w']) and not np.array_equal(ini['qd_init_requested_FP32_rad_s'],ref['native_FD32_rad_s'][0]))
    ck('world_to_body_initial_omega_and_object_linear_velocity',np.allclose(ini['object_angular_requested_FP32_body_rad_s'],ref['object_rotation_w'][0].T@ref['object_omega_world_rad_s'][0],atol=2e-7,rtol=0) and np.array_equal(ini['object_bodyorigin_v_requested_FP32_world_m_s'],ref['object_v_world_m_s'][0].astype(np.float32)))
    expected=load(bundle['context']['path']);got=load(RUN/'actual_context_readbacks.npz');fixed=load(RUN/'fixed_physics_context.npz');applied=load(RUN/'expected_context_fields.npz')
    ck('actual8_after_copy_and_after_setconst_raw_exact',all(np.array_equal(applied[k],expected[k][0]) and np.array_equal(fixed[k],np.broadcast_to(expected[k][0].astype(fixed[k].dtype),fixed[k].shape)) and all(np.array_equal(got[s+'__'+k],np.broadcast_to(expected[k][0].astype(got[s+'__'+k].dtype),got[s+'__'+k].shape)) for s in ('after_copy','after_set_const_0')) for k in FIELDS))
    ck('nominal_MJB_and_reader_context_exact',sha(RUN/'nominal_model.mjb')==bundle['model']['sha256'] and sha(bundle['context']['path'])==bundle['context']['sha256'] and np.array_equal(fixed['reader_context'],np.broadcast_to(expected['reader_context'][0],fixed['reader_context'].shape)))
    s=torch.load(RUN/'train/model_final.pt',map_location='cpu',weights_only=False);s10=torch.load(RUN/'train/model_10.pt',map_location='cpu',weights_only=False);effective=read(RUN/'effective_training_config.json');meta=read(RUN/'train/model_final.meta.json');c=s['rough_controller'];root=c['root_descriptor']
    ck('final104_10iter_effective_fromscratch',s['format_version']==104 and s['iteration']==10 and canon(s['config'])==canon(effective) and effective['checkpoint'] is None and effective['model_checkpoint'] is None and all(effective[k]==v for k,v in cfg['training'].items()) and effective['control_hz']==50 and effective['task']=='hand2-pen-spin')
    ck('all_saved_model_optimizer_tensors_finite',finite(s) and set(s['modules'])==set(s['optimizers'])==set(count))
    ck('actor155to20_and_WM_native_output67',s['modules']['actor']['net.0.weight'].shape==torch.Size([512,155]) and s['modules']['actor']['net.6.weight'].shape==torch.Size([20,128]) and s['modules']['wm']['net.9.weight'].shape==torch.Size([67,512]))
    ck('final_checkpoint_SHA_index_and_declared_iteration',sha(RUN/'train/model_final.pt')==meta['sha256']==result['checkpoint']['sha256'] and meta in rows(RUN/'checkpoint_index.jsonl') and meta['iteration']==10 and meta['actual_optimizer_steps']==count and 'no quality selection' in meta['selection'])
    ck('actual_hook_and_adam_update_counts',result['actual_optimizer_steps']==count and runtime['actual_optimizer_steps']==count and runtime['optimizer_updates']==sum(count.values()) and all({float(p['step']) for p in s['optimizers'][n]['state'].values() if 'step' in p}=={float(count[n])} for n in count))
    ev=rows(RUN/'rough_actualu_evidence.jsonl');expect_steps=[16,2,3,2,3,2,3,2,3,2,3]
    ck('11collect_41steps_5248_actual_transitions',len(ev)==11 and [x['collect_call'] for x in ev]==list(range(1,12)) and [x['steps'] for x in ev]==expect_steps and all(x['worlds']==128 and x['real_transitions']==x['steps']*128 for x in ev) and sum(x['real_transitions'] for x in ev)==s['sim_steps']==meta['sim_steps']==result['sim_steps']==5248 and result['phase_calls']['collect']==11)
    ck('allbatch_cache_actual_replay_residual_exact_and_active15',all(x['all_batch_exact'] is True and x['all_finite'] is True and x['inactive_real_actions']==0 and 0<=x['phase_min']<=x['phase_max']<15 and len(x['phase_histogram'])==15 and sum(x['phase_histogram'])==x['real_transitions'] and all(y==0 for y in x['max_abs_errors'].values()) for x in ev) and c['latest_actualu_evidence']==ev[-1])
    low=np.array(c['target_lower_rad'],np.float32);high=np.array(c['target_upper_rad'],np.float32)
    for call in (1,2,3):
        t=load(RUN/f'rough_actualu_trace_{call:05d}.npz');u=np.clip(np.clip(t['proposed'],low,high),low+np.float32(.02),high-np.float32(.02))
        ck(f'trace{call}_independent_projection_baseline_EMA_replay',np.array_equal(t['baseline'],baseline[t['phase']]) and np.array_equal(t['proposed'],t['baseline']+t['residual_before_filter']) and np.array_equal(u,t['cached']) and np.array_equal(t['cached'],t['actual']) and np.array_equal(t['actual'],t['replay_u']) and np.array_equal(t['residual_before_filter'],t['residual']) and np.array_equal(t['residual'],t['replay_c']))
    basec={k:v for k,v in c.items() if k not in ('latest_actualu_evidence','reference_controller_signature')}
    ck('controller_signature_and_descriptor_snapshot',hashlib.sha256(canon(basec).encode()).hexdigest()==c['reference_controller_signature']==s['reference_signature'] and hashlib.sha256(canon(root).encode()).hexdigest()==c['root_descriptor_sha256'] and read(RUN/'controller_descriptor.json')==basec|{'reference_controller_signature':s['reference_signature']})
    ck('controller_source_input_arm_seed_signature_exact',root['config_sha256']==sha(config) and root['execution_source_pins']==inv and root['source_pins']==bundle['source_pins'] and root['trained_method']=='A1L' and root['training_seed']==0 and root['run_role']=='engineering_smoke' and c['format_version']==104 and c['exact_resume_supported'] is False and c['margin_rad']==.02 and c['active_intervals']==[0,15] and c['terminal_observation']==15 and c['baseline_u15_FP32_sha256']==hashlib.sha256(baseline.tobytes()).hexdigest())
    raw=torch.load(RUN/'initial_parameter_tensors.pt',map_location='cpu',weights_only=False);finger=read(RUN/'initial_parameters.json');deltas={};recomputed={}
    for n,m in raw['modules'].items():
        h=hashlib.sha256();sq=0
        for key,v in zip(m['parameter_names'],m['parameters']):
            h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.contiguous().view(torch.uint8).numpy().tobytes());sq+=float((s['modules'][n][key]-v).double().square().sum())
        recomputed[n]={'sha256':h.hexdigest(),'tensors':len(m['parameters']),'parameters':sum(v.numel() for v in m['parameters'])};deltas[n]=sq**.5
    ck('initial_named_parameter_hashes_and_final_deltas_independent',recomputed==finger==s['rough_initial_parameter_fingerprints'] and all(math.isclose(v,result['parameter_delta_l2'][k],rel_tol=1e-10,abs_tol=1e-12) and v>0 for k,v in deltas.items()) and raw['capture_point']=='after_native_DexRunner_init_before_first_collect_or_optimizer' and raw['source_pins']==s['rough_capacity_evidence']['source_pins'])
    p=s['rough_runtime_evidence']['policy'];ph=np.array(p['start_phases']).reshape(-1);mask=ph[None,:]+np.arange(8)[:,None]<15;last=ph+8<15
    ck('H8_independent_endmask_critic_norm_and_active_losses',np.array_equal(mask,np.array(p['active_mask_H_B'])) and p['critic_norm_eligible']==int(mask.sum()+last.sum()) and sum(p['critic_loss_minibatch_active_counts'])==2*int(mask.sum()) and len(p['critic_loss_minibatch_active_counts'])==4 and p['optimizer_actor_calls']==1 and p['optimizer_critic_calls']==4)
    replay=s['rough_runtime_evidence']['replay_counts'];window=s['rough_runtime_evidence']['latest_window'];ck('replay_and_reset_suffix_counts',replay['transitions']==5248 and 0<=replay['natural_end_rows']<=5248 and 0<=replay['physical_terminated_rows']<=5248 and window['valid_window_entries']+window['reset_suffix_entries_masked']==window['candidate_window_entries'] and result['runtime_evidence']==s['rough_runtime_evidence'])
    final=read(RUN/'capacity_observation.json');prefix=s['rough_capacity_evidence']
    for label,cap in [('prefix',prefix),('final',final)]:
        flags=[np.array(x['raw_flags_values'],dtype=np.dtype(x['raw_flags_dtype'])) for x in cap['records']]
        ck('capacity_'+label+'_raw_OR_no_unknown_no_clear',all(x.size and x.dtype.kind in 'iu' and not x.any() for x in flags) and all(x['read_status']=='OK' and x['raw_bitmask']==0 and x['cumulative_OR_bitmask']==0 and x['cumulative_unknown_reads']==0 and x['read_attempts_for_record']==1 and x['flags_cleared_by_observer'] is False for x in cap['records']) and cap['cumulative_OR_bitmask']==0 and cap['unknown_read_count']==0 and cap['status']=='OK' and cap['configured_fixed_capacities']=={'nconmax':192,'njmax':4096})
    ck('checkpoint_capacity_prefix_not_final',prefix['records']==final['records'][:len(prefix['records'])] and prefix['record_count']==7 and prefix['before_reset_read_count']==5 and prefix['final_read_count']==0 and prefix['finalized'] is False and prefix['engineering_capacity_qualified'] is False and prefix['prefix_at_checkpoint_capacity_qualified'] is True and prefix['records'][-1]['event']=='checkpoint' and prefix['checkpoint_iteration']==10 and prefix['checkpoint_sim_steps']==5248)
    ck('final_capacity_once_not_world_attribution',final['record_count']==8 and final['before_reset_read_count']==5 and final['final_read_count']==1 and final['finalized'] is True and final['engineering_capacity_qualified'] is True and final['entire_run_finalized'] is True and final['records'][-1]['event']=='final' and final['flag_shape_is_world_count'] is False and sum(r['event']=='final' for r in final['records'])==1)
    ck('completion_schema_final_capacity_and_native_result_separate',set(completion)=={'schema','config_sha256','result'} and completion['schema']=='rough_prefix_training_completion:1' and completion['config_sha256']==sha(config) and completion['result']=={**result,'rough_capacity_observation':pin(RUN/'capacity_observation.json'),'rough_capacity_qualified_final':True} and 'rough_capacity_qualified_final' not in result)
    ck('completed_engineering_only_not_quality_or_reference_upgrade',runtime['status']=='completed_rough_training' and result['status']=='COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY' and result['cleanup_errors']==[] and result['physical_reference_qualified'] is False and result['source_lineage_N']==1 and result['independent_ego_N']==0)
    metrics=rows(RUN/'metrics.jsonl');ck('10finite_metrics_no_prior_in_A1L',len(metrics)==10 and [x['iteration'] for x in metrics]==list(range(10)) and finite(metrics) and not any(any(k.startswith('prior/') for k in x) for x in metrics) and cfg['contact_binding'] is None and not (RUN/'contact_objective_metadata.json').exists())
    ck('runtime_batched_and_world_substeps_scope',runtime['physics_steps']==164 and runtime['physical_world_substeps']==20992 and runtime['collected_sim_transitions']==5248 and 'excludes startup graph capture/forwards' in runtime['physics_step_scope'])
    ck('module_weights_equal_iteration10_and_final_no_extra_learning',all(torch.equal(v,s10['modules'][n][k]) for n,m in s['modules'].items() for k,v in m.items()) and s10['iteration']==10 and s10['sim_steps']==5248)
    ck('CPU_audit_no_CUDA_or_environment_modules',not torch.cuda.is_initialized() and not any(n.split('.')[0] in ('mujoco','mujoco_warp','warp','mjlab','dex_supermimic') for n in sys.modules))
    detail={'actual_pid':runtime['pid'],'launcher_pid':runtime['parent_pid'],'OS_creation_utc':osr['creation_date_utc'],'iterations':10,'collect_calls':11,'transitions':5248,'optimizer_steps':count,'parameter_delta_l2_independent':deltas,'capacity_records':{'prefix':7,'before_reset':5,'checkpoint':2,'final':1,'entire_run':8},'replay':replay,'last_window':window,'H8_active_rows':int(mask.sum()),'H8_norm_eligible':int(mask.sum()+last.sum()),'sources':len(inv),'cost_seconds_nested_not_additive':{'watch':watch['elapsed_s'],'worker':runtime['wall_seconds'],'native_excluding_imports':result['native_call_wall_seconds_excluding_imports']},'initial_actual_state_scope':'Runtime source performs strict reset readback and failed resets raise. Training artifact retains source/init arrays, not independent full raw startup qpos/qvel for all resets; no claim to reconstruct missing physical states.'}
    watched_files=['config.json','runtime.json','result.json','training_completion.json','capacity_observation.json','initial_parameters.json','initial_parameter_tensors.pt','source/inventory.json','source/execution_sources.zip','rough_actualu_evidence.jsonl','train/model_final.pt']
    report={'schema':'rough_A1L_smoke_independent_saved_review:1','status':'PASS_SAVED_ENGINEERING_HEALTH' if all(checks.values()) else 'BLOCKING_OR_FAILED_CHECK','checks':checks,'passed':sum(checks.values()),'total':len(checks),'details':detail,'artifact_pins':{name:pin(RUN/name) for name in watched_files},'auditor':pin(__file__),'scope':'Independent CPU saved-array/torch-tensor/sourceZIP/Windows identity review, not production auditor reuse. No Env, reset, step, FK, collision query or GPU. Confirms engineering smoke, not physical quality, learned contact dynamics or full trajectory success.','GPU_used':False,'physics_steps_run_by_audit':0,'optimizer_steps_run_by_audit':0,'task_quality':'NOT_ASSESSED','physical_reference_qualified':False,'harness_correction':{'old_report':pin(ROOT/'outputs/independent_smoke_a1l_v1/report.json'),'old_source':pin(ROOT/'scripts/audit_smoke_a1l_independent_saved_cpu.py'),'change':'Compare effective config using canonical JSON: native tuple wm_hidden versus serialized list; all numeric/string/bool/null values remain strict. No training/source/threshold/artifact changes.'}}
    (OUT/'auditor_snapshot.py').write_bytes(Path(__file__).read_bytes());(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')},indent=2));return 0 if all(checks.values()) else 1
if __name__=='__main__':raise SystemExit(main())
