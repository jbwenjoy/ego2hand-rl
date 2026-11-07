"""Saved-only schema2 three-arm105 policy audit; no native imports or model execution."""
from pathlib import Path
import argparse
import datetime as dt
import hashlib,json,os,re,sys,traceback,zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--health',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--expected-method',choices=['A1L','P_ONLY','A2'],required=True);args=ap.parse_args()
RUN=args.run_dir.resolve(strict=True);WATCH=RUN.with_name(RUN.name+'_watchdog');OUT=args.output.resolve()
if not OUT.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Study-owned independent output required')
if OUT.exists():raise FileExistsError('Keep every saved audit attempt')
OUT.mkdir(exist_ok=False);(OUT/'audit_source.py').write_bytes(Path(__file__).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def utc(v):return dt.datetime.fromisoformat(v.replace('Z','+00:00'))
def same(a,b):return np.array_equal(np.asarray(a,np.float32).view(np.uint32),np.asarray(b,np.float32).view(np.uint32))
checks={};errors={}
def check(name,fn):
    try:assert bool(fn() if callable(fn) else fn);checks[name]=True
    except BaseException:checks[name]=False;errors[name]=traceback.format_exc()
c=read(RUN/'config.json');inv=read(RUN/'source/inventory.json');raw=load(RUN/'raw_states.npz');req=load(RUN/'requested_initialization_and_control.npz')
check('new_schema2_arm_selection',c.get('schema')=='full151_native_policy_evaluation_run:2' and c.get('method')==args.expected_method and c.get('role')=='native_policy_eval')
ev=read(RUN/'evaluation.json');result=read(RUN/'result.json');done=read(RUN/'completion.json');runtime=read(RUN/'runtime.json')
solver=read(RUN/'solver_samples.json');events=read(RUN/'control_events.json');tr=read(RUN/'native_reward_transitions.json')
watch=read(WATCH/'state.json');ident=read(WATCH/'engineering_identity.json');post=read(WATCH/'post_terminal_CIM.json')
health=read(c['health']['path']);train=read(c['training_config']['path']);cd=read(RUN/'controller_descriptor.json')
cap=read(RUN/'capacity_audit.json');capfinal=read(RUN/'capacity_observer_final.json')
ref=load(inv['base__input__reference_arrays']['path']);control=load(inv['base__input__control_arrays']['path']);init=load(inv['base__input__init_arrays']['path'])
base=control['u_postC2_150_FP32_rad'];lower=req['target_lower_rad'];upper=req['target_upper_rad']
check('complete601_600_150_150_finite',raw['joint_rad'].shape==(601,20) and len(solver)==600 and len(events)==len(tr)==150 and all(np.isfinite(v).all() for v in raw.values()))
check('source_table_config_current_exact',bool(inv) and inv==c['source_pins'] and all(sha(v['path'])==v['sha256'] for v in inv.values()))
with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
    names=[f'{i:04d}_{Path(v["path"]).name}' for i,v in enumerate(inv.values())]
    check('sourceZIP_members_and_content_exact',z.namelist()==names and all(hashlib.sha256(z.read(n)).hexdigest()==v['sha256'] for n,v in zip(names,inv.values())))
check('actualconfig_snapshots_byteexact',sha(RUN/'config.json')==sha(WATCH/'config_snapshot.json')==sha(watch['command'][4])==runtime['plan_sha256']==done['config_sha256'])
check('OS12_and_exit0_noTimeout_savedPIDempty',watch['status']=='completed' and watch['exit_code']==0 and watch['timeout'] is False and len(ident['checks'])==12 and all(ident['checks'].values()) and post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[])
osrec=watch['actual_process_OS_observation'];py=ident['selected_python_identity'];cmd=watch['command']
check('actualOS_PID_PPID_command_source_basealias',osrec['pid']==runtime['pid']==watch['actual_pid'] and osrec['parent_pid']==runtime['parent_pid']==watch['launcher_pid'] and osrec['command_line']==f'"{py["base_executable_raw"]}" '+' '.join(cmd[1:]) and os.path.samefile(osrec['executable_path'],py['base_executable_raw']) and os.path.samefile(py['base_executable_raw'],py['base_executable_resolved']) and runtime['source_sha256']==sha(cmd[2]))
check('OScreation_original_bounded_observation',utc(watch['actual_creation_conservative_bounds']['lower_utc'])<=utc(osrec['creation_date_utc'])<=utc(watch['actual_creation_conservative_bounds']['upper_utc']))
check('samefinal_health_config_auditor_checkpoint_source',health['schema']=='full151105_checkpoint_CPU_health:1' and health['status']=='PASS_SAVED_ENGINEERING_HEALTH' and health['passed']==health['total']==len(health['checks']) and all(v is True for v in health['checks'].values()) and health['checkpoint']==c['checkpoint'] and health['config']==c['training_config'] and health['auditor']==c['expected_health_auditor'] and health['source_pins']==train['source_pins'] and health['method']==c['method']==train['method'] and health['seed']==c['seed']==train['seed'] and health['iteration']==c['checkpoint_iteration']==train['training']['max_iterations'] and health['role']==train['role'] and train['role'] in ('engineering_smoke','primary_matched') and health['final_iteration_scope'] is True and sha(c['checkpoint']['path'])==c['checkpoint']['sha256'])
expected_root={**read(ROOT/'outputs/full151_native_input_v1/descriptor.json'),'trained_method':c['method'],'training_seed':c['seed'],'run_role':train['role'],'config_sha256':c['training_config']['sha256'],'execution_source_pins':train['source_pins']}
check('controller_retains_training_root_not_eval_resigning',cd['root_descriptor']==expected_root and cd['root_descriptor_sha256']==digest(expected_root) and cd['reference_controller_signature']==digest({k:v for k,v in cd.items() if k!='reference_controller_signature'}) and cd['format_version']==105)
traincd=read(Path(train['output'])/'controller_descriptor.json')
check('actual_eval_controller_exact_training_descriptor',cd=={k:v for k,v in traincd.items() if k!='latest_actualu_evidence'} and same(lower,cd['target_lower_rad']) and same(upper,cd['target_upper_rad']) and np.array_equal(base,raw['selected_planned150_FP32_rad']))

# Independent arithmetic uses NumPy FP32 operations, not the production oracle
# or torch/pure target_inset. All control-history rows are retained.
predictions=[];prior=np.zeros((1,20),np.float32);clamped=0
for k,e in enumerate(events):
    a=np.asarray(e['actor_raw_action'],np.float32);action=np.clip(a,np.float32(-1),np.float32(1))
    residual=np.float32(.5)*np.clip(np.float32(.5)*action,np.float32(-.6),np.float32(.6))+np.float32(.5)*prior
    proposed=base[k][None]+residual
    bounded=np.minimum(np.maximum(proposed,lower),upper)
    actual=np.minimum(np.maximum(bounded,lower+np.float32(.02)),upper-np.float32(.02))
    clamped+=int(np.count_nonzero(actual!=proposed))
    check(f'control{k:03d}_FP32_independent_action_EMA_C2_cache_actual',same(action,e['action']) and same(prior,e['prior_residual']) and same(residual,e['residual']) and same(actual,e['cached']) and same(actual,e['actualu']) and all(same(value,e['independent_expected'][name]) for name,value in {'residual':residual,'proposed':proposed,'hard_bounded':bounded,'actual':actual}.items()))
    predictions.append(actual[0]);prior=residual
applied=np.asarray(predictions,np.float32)
expectedraw=np.vstack((base[0],np.repeat(applied,4,axis=0)))
check('all601_rawctrl_matches_independent150_targets',same(raw['ctrl_rad'],expectedraw) and same(raw['learned_applied_FP32_rad'],applied) and raw['ctrl_exact'].all())
check('phase_no_reset_terminal149_to_obs150',all(e['phase']==k and e['preaction_phase']==[k] and e['postaction_phase']==[k+1] and e['terminated']==[False] for k,e in enumerate(events)) and all(e['reference_end']==e['truncated']==[False] for e in events[:-1]) and events[-1]['reference_end']==events[-1]['truncated']==[True])
check('memory_phase_residual_frozen_prev_target',all(t['phase']==k and t['memory']['time']==[[float(k)]] and t['memory']['start']==[[0.]] and t['memory']['end']==[[151.]] and same(t['memory']['frozen_target'],base[0][None] if k==0 else applied[k-1][None]) and same(t['memory']['c1'],events[k]['prior_residual']) and same(t['memory']['a1'],np.zeros((1,20),np.float32) if k==0 else events[k-1]['action']) for k,t in enumerate(tr)))
policy155=np.asarray([e['policy_input155'][0] for e in events],np.float32)
reader_saved=load(inv['base__input__native_reader_arrays']['path'])
policy_map={
 'reference_q_0_20':(policy155[:,:20],reader_saved['joint_pos'][:150]),
 'reference_BODY21_26_89':(policy155[:,26:89],reader_saved['link_pos'][:150].reshape(150,63)),
 'current_q_89_109':(policy155[:,89:109],raw['joint_rad'][4*np.arange(150)].astype(np.float32)),
 'current_quarter_qd_109_129':(policy155[:,109:129],np.float32(.25)*raw['joint_velocity_rad_s'][4*np.arange(150)].astype(np.float32)),
 'previous_action_135_155':(policy155[:,135:155],np.asarray([np.zeros(20)]+[e['action'][0] for e in events[:-1]],np.float32))}
policy_metrics={name:{'exact_equal':bool(np.array_equal(a,b)),'max_abs_error':float(np.max(np.abs(a-b)))} for name,(a,b) in policy_map.items()}
check('policy155_finite_and_five_correct_feature_slices_exact',policy155.shape==(150,155) and np.isfinite(policy155).all() and all(v['exact_equal'] for v in policy_metrics.values()))
# The two bounded-point blocks are independent geometry arithmetic, not a
# physics gate. Eight float32 epsilons cover the source's quaternion/rotation
# and tanh roundoff in these bounded(<.15m) features; raw errors are retained.
rp=reader_saved['object_pose'][:150].astype(float);rq=rp[:,3:];rq=rq/np.linalg.norm(rq,axis=1,keepdims=True)
rw,rx,ry,rz=rq.T;r_axis=np.stack((1-2*(ry*ry+rz*rz),2*(rx*ry+rz*rw),2*(rx*rz-ry*rw)),axis=-1)
ref_points=.15*np.tanh((rp[:,:3,None]+.1*r_axis[:,:,None]*np.array([1.,-1.]))/.15)
ref_points=ref_points.transpose(0,2,1).reshape(150,6)
x0=np.asarray([t['x0'] for t in tr],float);rot=x0[:,43:52].reshape(150,3,3)
current_points=.15*np.tanh((x0[:,40:43,None]+.1*rot[:,:,0,None]*np.array([1.,-1.]))/.15)
current_points=current_points.transpose(0,2,1).reshape(150,6)
roundoff_bound=8*np.finfo(np.float32).eps
for name,a,b in [('reference_axis_20_26',policy155[:,20:26],ref_points),('current_axis_129_135',policy155[:,129:135],current_points)]:
 err=float(np.max(np.abs(a-b)));policy_metrics[name]={'max_abs_error_m':err,'arithmetic_atol_m':float(roundoff_bound),'physical_threshold':False}
 check('policy155_'+name+'_source_formula_roundoff',err<=roundoff_bound)
check('CLI_health_exact_bound_input',args.health.resolve()==Path(c['health']['path']).resolve() and sha(args.health)==c['health']['sha256'])
check('native_reward_x58_and_self15_phase',all(np.asarray(t['native_selfcollision15']).shape==(15,) and same(t['x0'][:20],raw['joint_rad'][4*k]) and same(t['x0'][20:40],raw['joint_velocity_rad_s'][4*k]) and same(t['x1_reward_snapshot'][:20],raw['joint_rad'][4*k+4]) and same(t['x1_reward_snapshot'][20:40],raw['joint_velocity_rad_s'][4*k+4]) and np.array_equal(t['torque_reward_snapshot'],solver[4*k+3]['actuator_force_solver_native']) for k,t in enumerate(tr)))
check('solver_prepost_clock_and_control_phase',all(s['step']==n and s['prestate_index']==n-1 and s['poststate_index']==n and s['solver_t_s']==raw['raw_time_s'][n-1] and s['poststate_t_s']==raw['raw_time_s'][n] and s['control_interval_used']==(n-1)//4 and same(s['ctrl_solver_native'],applied[(n-1)//4]) for n,s in enumerate(solver,1)))
check('startup_analytic_nonzero_q_qd_worldv_bodyomega',same(raw['joint_rad'][0],init['q_init_requested_FP32_rad']) and same(raw['joint_velocity_rad_s'][0],init['qd_init_requested_FP32_rad_s']) and same(raw['object_pose_w'][0],init['object_pose_init_requested_FP32_w']) and same(raw['object_twist_native'][0,:3],init['object_bodyorigin_v_requested_FP32_world_m_s']) and np.max(np.abs(raw['object_twist_native'][0,3:]-init['object_angular_requested_FP32_body_rad_s']))<=1e-6)
check('full_initial_qpos_and_joint_mappings_exact',same(raw['full_qpos'][0],req['full_qpos_requested_FP32']) and same(raw['joint_velocity_rad_s'][0],req['full_qvel_requested_FP32'][req['joint_dof_ids']]) and np.array_equal(raw['joint_rad'],raw['full_qpos'][:,req['joint_qpos_ids']]) and np.array_equal(raw['joint_velocity_rad_s'],raw['full_qvel'][:,req['joint_dof_ids']]))
check('fixed_mocap_all601_noassist',np.array_equal(raw['mocap_pos_w'],np.broadcast_to(req['mocap_pos_requested_FP32'],raw['mocap_pos_w'].shape)) and np.array_equal(raw['mocap_quat_wxyz'],np.broadcast_to(req['mocap_quat_requested_FP32'],raw['mocap_quat_wxyz'].shape)) and np.max(raw['external_force_max_abs'])==0.)
check('capacity_actual3raw_preReset_final_OR0',cap['status']=='OK' and cap['engineering_capacity_qualified'] is True and [v['event'] for v in cap['records']]==['after_constructor_before_actual8','before_reset','final'] and all(v['raw_flags_values']==[0] and v['raw_flags_dtype']=='int32' and v['read_status']=='OK' and v['read_attempts_for_record']==1 for v in cap['records']) and cap['cumulative_OR_bitmask']==cap['unknown_read_count']==0 and cap['records']==capfinal['records'] and cap['final_read_count']==1 and cap['before_reset_read_count']==1 and cap['resets_during_probe']==0)
check('capacity_final_raw_allocations_and_step600_boundary',cap['final_read_after_returned_episode_step']==cap['final_read_after_sampled_solver_row']==600 and cap['maximum_recorded_nacon']==int(raw['nacon'].max()) and cap['maximum_recorded_nefc']==int(raw['nefc'].max()) and int(raw['nacon'].max())<cap['contact_capacity'] and int(raw['nefc'].max())<cap['constraint_capacity'])
expected_ctx=load(RUN/'expected_context_fields.npz');actual_ctx=load(RUN/'actual_context_readbacks.npz');fixed=load(RUN/'fixed_physics_context.npz');srcctx=load(inv['base__input__actual_context']['path'])
check('actual8_double_readbacks_fixedxi46_source',all(np.array_equal(fixed[k],actual_ctx[stage+'__'+k]) and np.array_equal(fixed[k],np.broadcast_to(expected_ctx[k],fixed[k].shape)) and np.array_equal(expected_ctx[k],srcctx[k][0]) for k in expected_ctx for stage in ('after_copy','after_set_const_0')) and np.array_equal(fixed['reader_context'],srcctx['reader_context'][:1]))

# Reuse the frozen passive-MJB material inventory as saved topology only.
# No MuJoCo import, MjModel/MjData or new native geometry operation is needed.
material_path=ROOT/'outputs/full151_actual_material_cone_contract_v1/material_cone_contract.json'
material=read(material_path);topology_path=Path(material['material_table']['path']);topology=read(topology_path)
check('saved_passive_model_topology_identity',sha(topology_path)==material['material_table']['sha256'] and sha(RUN/'nominal_model.mjb')==material['inputs']['model']['sha256']=='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd' and material['passed']==material['total']==17 and all(v is True for v in material['checks'].values()))
hand={int(g['geom_id']) for g in topology['rows'] if g['name'].startswith('robot/') and g['enabled']}
object_geoms={int(g['geom_id']) for g in topology['rows'] if not g['name'].startswith('robot/')}
check('saved21hand3object_category_members',len(hand)==21 and len(object_geoms)==3 and not hand.intersection(object_geoms))
depth=np.zeros(600);category=np.zeros((600,3));nonhand=np.zeros(600,np.int64);count=np.zeros(601,np.int64)
for line in (RUN/'raw_contacts.jsonl').read_text().splitlines():
    row=json.loads(line);step=row['step'];count[step]+=1
    if step==0:continue
    a,b=row['geom'];ho=(a in object_geoms or b in object_geoms) and (a in hand or b in hand);hs=a in hand and b in hand
    dep=max(0.,-row['distance_m']);depth[step-1]=max(depth[step-1],dep);cat=0 if ho else 1 if hs else 2
    category[step-1,cat]=max(category[step-1,cat],dep)
    if (a in object_geoms or b in object_geoms) and not (a in hand or b in hand):nonhand[step-1]+=1
check('rawcontact_count_eachstate_and_all_depth_categories_reconstructed',np.array_equal(count,raw['nacon']) and np.array_equal(depth,np.array([s['penetration_m'] for s in solver])) and np.array_equal(nonhand,np.array([s['object_environment_contacts'] for s in solver])) and all(np.array_equal(category[:,j],[s['penet_category_m'][name] for s in solver]) for j,name in enumerate(('hand_object','hand_self','other'))))
check('all600_solver_force_preintegration_saved_poststep_phase',np.array_equal(np.asarray([s['actuator_force_solver_native'] for s in solver]),raw['actuator_force_native'][1:]))
q=raw['object_pose_w'][:,3:];norm=np.linalg.norm(q,axis=1,keepdims=True);check('objectquatfinite_unit_API_roundoff',np.isfinite(q).all() and np.max(np.abs(norm-1))<1e-6)
w,x,y,z=(q/norm).T
localx=np.stack((1-2*(y*y+z*z),2*(x*y+z*w),2*(x*z-y*w)),axis=1)
shaft=np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',localx,ref['object_rotation_w'][:,:,0]),-1,1)))
position=np.linalg.norm(raw['object_pose_w'][:,:3]-ref['object_COM_world_m'],axis=1);knots=np.arange(0,601,4)
excess=np.maximum(np.maximum(raw['joint_lower_rad']-raw['joint_rad'],raw['joint_rad']-raw['joint_upper_rad']),0.)
delta=np.diff(np.vstack((raw['ctrl_rad'][0],applied.astype(float))),axis=0);rate=np.abs(delta)/.02
expected_clock=np.zeros(601,np.float32)
for k in range(600):expected_clock[k+1]=np.float32(expected_clock[k]+np.float32(.005))
clock_ok=np.array_equal(raw['raw_time_s'],expected_clock.astype(float)) and np.max(np.abs(raw['raw_time_s']-np.arange(601)*.005))<=5e-5
forces=np.asarray([s['actuator_force_solver_native'] for s in solver]);ratio=np.where(forces>=0,forces/raw['force_upper_Nm'],forces/raw['force_lower_Nm'])
charge=np.zeros(20);charge_max=np.zeros(20)
for elapsed,row in zip(np.diff(raw['raw_time_s']),ratio):charge=np.clip(charge+elapsed*(np.minimum(row,1)-charge/2.),0,1);charge_max=np.maximum(charge_max,charge)
gates={'object_position_151':bool(position[knots].max()<=.02),'directed_shaft_151':bool(shaft[knots].max()<=20),'actual_q_all601':bool(excess.max()<=.001),'penetration_solver600':bool(depth.max()<=.003),'no_nonhand_object_contacts':int(nonhand.sum())==0,'no_exogenous':bool(raw['external_force_max_abs'].max()==0),'all_actual_u_exact':bool(same(raw['ctrl_rad'],expectedraw)),'actual_target_rate':bool(rate.max()<=15),'clock':bool(clock_ok),'load_ratio':bool(ratio.max()<=1.01),'load_charge':bool(charge_max.max()<=.8),'drop':bool(raw['object_pose_w'][:,2].min()>=raw['mocap_pos_w'][0,0,2]-.15),'capacity':bool(cap['status']=='OK' and cap['engineering_capacity_qualified'])}
metrics={'max_object_position_error_151_m':float(position[knots].max()),'max_object_position_error_601_m':float(position.max()),'max_shaft_angle_151_deg':float(shaft[knots].max()),'max_shaft_angle_601_deg':float(shaft.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':float(depth.max()),'max_actuator_load_ratio':float(ratio.max()),'max_load_charge':float(charge_max.max()),'max_actual_target_rate_rad_s':float(rate.max())}
check('all13_original_gates_recomputed_equal',gates==ev['gates'] and len(gates)==13 and bool(all(gates.values()))==ev['native_policy_pass'])
check('all9_continuous_metrics_recomputed_roundoff_equal',all(abs(value-ev[key])<=1e-12*max(1.,abs(value)) for key,value in metrics.items()))
check('all150_rate_differences_startup_and_149_updates',ev['startup_and_update_rate']['rate_difference_count']==150 and float(rate[0].max())==ev['startup_and_update_rate']['startup_rate_rad_s'] and float(rate[1:].max())==ev['startup_and_update_rate']['subsequent_updates_max_rate_rad_s'])
check('no_quality_censor_and_API_counts_complete',ev['complete'] is True and ev['episode_API_counts']=={'step_attempted':600,'step_returned':600,'forward_attempted':150,'forward_returned':150} and result['evaluation']==ev and done['result']==result and done['quality_is_not_engineering_gate'] is True and result['cleanup_errors']==[])
al=read(RUN/'actor_load.json');ae=read(RUN/'actor_state_evidence.json');au=read(RUN/'actor_unchanged.json')
check('Actor105load_155_20_selectedfinal_and_checkpoint_bindings',al['status']=='PASS_EVALUATION_ACTOR_LOAD' and al['format_version']==105 and al['checkpoint']==c['checkpoint'] and al['iteration']==c['checkpoint_iteration'] and al['actor_observation_dim']==155 and al['actor_action_dim']==20 and al['reference_signature']==cd['reference_controller_signature'] and al['bundle']==c['input_bundle'] and sha(al['loader']['path'])==al['loader']['sha256'])
check('Actor_normalizer_weight_sourcebacked_saved_digests_unchanged',ae['all_weights_and_normalizer_unchanged'] is True and ae['actor_state_before_sha256']==ae['actor_state_after_sha256']==al['actor_state_sha256']==au['actor_state_sha256'] and ae['state_copy_before_final_capacity_read'] is True and ae['norm_update_calls']==ae['optimizer_objects_created']==ae['action_noise_draws']==0 and au['normalization_updated'] is False)
check('clock_scope_and_quality_not_upgraded',ev['construction_reset_setup_API_counts']=='UNKNOWN_NOT_WHOLEPROCESS_OBSERVED' and ev['clock_audit']['legacy_2us_diagnostic_pass'] is False and ev['physical_reference_qualified'] is False and result['physical_reference_qualified'] is False and not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp','mjlab')))
np.savez_compressed(OUT/'independent_numeric_arrays.npz',applied150_FP32=applied,position_error601_m=position,shaft_error601_deg=shaft,joint_excess601_rad=excess,penetration600_m=depth,category_depth600_m=category,nonhand_contacts600=nonhand,target_rate150_rad_s=rate,load_ratio600=ratio)
report={'schema':'full151105_policy_saved_independent:4','method':c['method'],'engineering_qualified':bool(checks) and all(checks.values()),'status':'PASS_SAVED_AUDIT_ORIGINAL_QUALITY_PRESERVED' if all(checks.values()) else 'SAVED_AUDIT_HARNESS_OR_EVIDENCE_FAILURE_PRESERVED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'errors':errors,'auditor':pin(Path(__file__)),'inputs':{str(p.relative_to(ROOT)):pin(p) for p in [RUN/'config.json',RUN/'raw_states.npz',RUN/'solver_samples.json',RUN/'control_events.json',RUN/'raw_contacts.jsonl',RUN/'native_reward_transitions.json',RUN/'source/execution_sources.zip',RUN/'source/inventory.json',RUN/'evaluation.json',RUN/'controller_descriptor.json',RUN/'actor_load.json',RUN/'actor_state_evidence.json',WATCH/'state.json',WATCH/'engineering_identity.json',WATCH/'post_terminal_CIM.json']},'saved_topology_contract':pin(material_path),'saved_topology_table':pin(topology_path),'actual_checkpoint':c['checkpoint'],'health':c['health'],'recomputed_gates':gates,'recomputed_metrics':metrics,'failures':[k for k,v in gates.items() if not v],'source_members':len(inv),'checkpoint_iteration':c['checkpoint_iteration'],'training_role':train['role'],'policy155_feature_metrics':policy_metrics,'replaces_old_extra_assert':'Only in this new auditor: first20 are reference q; actual q is89:109. Old178/179 report/source remain unchanged.','contacts_including_startup':int(count.sum()),'contact_or_target_details':{'nonhand_object_candidate_contacts':int(nonhand.sum()),'HO_steps_over3mm':int(np.count_nonzero(category[:,0]>.003)),'self_steps_over3mm':int(np.count_nonzero(category[:,1]>.003)),'other_steps_over3mm':int(np.count_nonzero(category[:,2]>.003)),'C2_or_hard_clamped_components':clamped,'rate_peak_interval_and_joint':list(map(int,np.unravel_index(rate.argmax(),rate.shape))),'startup_rate_rad_s':float(rate[0].max())},'scope':['Independent NumPy reconstruction plus saved source-bound passive topology inventory; no new model/native import. No production function/Actor/WM/Reader/Env/FK/collision/contactForce/step/forward/GPU was invoked.','Checkpoint file hash, loaded signature and saved Actor-state digests are rebound. Runtime in-memory Actor weights/normalizer comparisons are source-backed saved evidence; this audit does not independently reconstruct their raw tensors or rerun inference.','Saved exact OS/CIM observations are rebound, with no new process query. No wholeprocess construction/reset/graph count or independent sensorfound labels are invented.','No task-quality stop: all150 actual intervals remain. All recomputed original gate outcomes are preserved. No improvement or method-benefit conclusion is made from this single evaluation, and no old104/prefix/shared cohort is pooled.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'report':pin(OUT/'report.json'),'passed':report['passed'],'total':report['total'],'status':report['status']}))
if not all(checks.values()):raise SystemExit(2)
