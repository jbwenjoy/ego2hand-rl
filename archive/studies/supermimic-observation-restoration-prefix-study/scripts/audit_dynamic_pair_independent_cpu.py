"""Saved-only independent H0/T0 metrics, phases and immutable input review."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def shaft_x(q):
    """Rotate local +x directly using normalized wxyz (no axis sign symmetry)."""
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y+w*z),2*(x*z-w*y)],axis=-1)
def same32(a,b):return np.array_equal(np.asarray(a,np.float32).view(np.uint32),np.asarray(b,np.float32).view(np.uint32))

out=ROOT/'outputs/dynamic_pair_independent_saved_review_v1'
out.mkdir(parents=True,exist_ok=False)
checks={};pins={'reviewer':pin(__file__)}
def check(name,value):checks[name]=bool(value)
summary=read(ROOT/'outputs/dynamic_pair_summary_v1/report.json');pins['pair_summary']=pin(ROOT/'outputs/dynamic_pair_summary_v1/report.json')
plans={k:read(ROOT/f'configs/dynamic-prefix-{k}-v1.json') for k in ('hold','track')}
manifests=[read(plans['hold'][k]['path']) for k in ('reference_manifest','init_manifest','control_manifest')]
ref,init,controls=[load(m['artifact']['path']) for m in manifests]
canonical=load(manifests[0]['candidate']['path'])
pipeline=load(Path(manifests[0]['candidate']['path']).with_name('pipeline_return.npz'))
check('canonical_is_R_lastaccepted_not_best_history',np.array_equal(pipeline['final_x'],canonical['optimizer_x']) and np.array_equal(pipeline['last_accepted_x'],canonical['optimizer_x']))
check('reference_matches_canonical_q_pose_controls',np.array_equal(ref['q_goal_rad'],canonical['q_goal_rad']) and np.array_equal(ref['mapped_object_pose_w'],canonical['mapped_object_pose_w']) and np.array_equal(ref['control_points_rad'],canonical['control_points_rad']))
check('reference16_and61_phase_0through300ms',np.array_equal(ref['time_s'],np.arange(16)*.02) and np.array_equal(ref['planned_time61_s'],np.arange(61)*.005) and np.allclose(shaft_x(ref['mapped_object_pose_w'][:,3:]),ref['object_rotation_w'][::4,:,0],atol=1e-13,rtol=0) and np.max(np.abs(ref['mapped_object_pose_w'][:,:3]-ref['object_COM_world_m'][::4]))<1e-13)
check('three_bundles_same_candidate_protocol_inputs',all(m['candidate']==manifests[0]['candidate'] and m['source_plan']==manifests[0]['source_plan'] and m['source_pins']==manifests[0]['source_pins'] and sha(m['artifact']['path'])==m['artifact']['sha256'] for m in manifests))
check('candidate_source_plan_and_pin_current',manifests[0]['candidate']['sha256']=='6364a6e262f7fbedac049d2d9e789b2c9077cbd6ef7656ef2e9b394161bab81a' and sha(manifests[0]['candidate']['path'])==manifests[0]['candidate']['sha256'] and sha(manifests[0]['source_plan']['path'])==manifests[0]['source_plan']['sha256'])
for i,m in enumerate(manifests):pins[f'bundle{i}']=m['artifact']
pins['candidate']=manifests[0]['candidate'];pins['producer_plan']=manifests[0]['source_plan']
plan_diffs={k for k in plans['hold'].keys()|plans['track'].keys() if plans['hold'].get(k)!=plans['track'].get(k)}
check('paired_plan_only_condition_and_output_differ',plan_diffs=={'control_condition','output'})
FIXED={'object_position_max_m':.02,'directed_shaft_error_max_deg':20.,'actual_joint_hard_excess_max_rad':.001,'penetration_any_category_max_m':.003,'object_nonhand_candidate_contact_count_max':0,'external_force_max_abs':0.,'clock_absolute_tolerance_s':2e-6,'target_rate_max_rad_s':15.,'actuator_load_ratio_max':1.01,'load_charge_max':.8,'charge_decay_time_s':2.,'drop_center_below_default_wrist_m':.15}
rows=[];raws={};requested={};alltraces={}
import mujoco
model=mujoco.MjModel.from_binary_path(plans['hold']['model']['path'])
names=[f'robot/right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]
jids=np.array([model.joint(n).id for n in names]);aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids]);obj=model.body('object/cube').id
object_geoms=set(np.flatnonzero(model.geom_bodyid==obj));hand_geoms={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
for kind in ('hold','track'):
    p=plans[kind];d=ROOT/f'outputs/dynamic_matched_{kind}_v1';w=d.with_name(d.name+'_watchdog')
    ev=read(d/'evaluation.json');runtime=read(d/'runtime.json');cap=read(d/'capacity_audit.json');identity=read(w/'engineering_identity.json');watch=read(w/'state.json')
    raw=load(d/'raw_states.npz');req=load(d/'requested_initialization_and_control.npz');raws[kind]=raw;requested[kind]=req
    solver=read(d/'solver_samples.json');events=[json.loads(s) for s in (d/'control_events.jsonl').read_text().splitlines() if s];contacts=[json.loads(s) for s in (d/'raw_contacts.jsonl').read_text().splitlines() if s]
    check(kind+'_all_plan_pins_match',all(sha(v['path'])==v['sha256'] for v in p['source_pins'].values()))
    check(kind+'_original_criteria_unchanged',p['gates']==FIXED and p['steps']==60 and p['dt_s']==.005 and p['duration_s']==.3 and not p['automatic_retry'] and p['GPU_run_authorized'] and not p['training_authorized'])
    check(kind+'_runtime_ev_plan_source_fixed',runtime['plan_sha256']==ev['plan_sha256']==sha(ROOT/f'configs/dynamic-prefix-{kind}-v1.json') and runtime['source_sha256']==ev['worker_sha256']=='8834a386f001cc7a138fba3906774bc11d6a2a996c0695ddfc772eff11a90db7')
    check(kind+'_full_horizon_no_state_assistance',len(raw['index'])==61 and np.array_equal(raw['index'],np.arange(61)) and len(solver)==60 and len(events)==14 and all(runtime[k]==v for k,v in {'physics_steps':60,'step_calls':60,'startup_Warp_forward_calls':1,'postinit_state_writes':0,'postinit_ctrl_writes':14,'overflow_reads':1,'resets_after_initialization':0}.items()))
    check(kind+'_strict_identity_existing_auditor_pass',identity['engineering_identity_qualified'] and watch['status']=='completed' and watch['exit_code']==0 and read(ROOT/f'outputs/dynamic_{kind}_saved_audit_v1/report.json')['engineering_passed'])
    schedule=controls['control__fp32_old_export__'+('hold' if kind=='hold' else 'tracking')+'_u_exec_15_FP32_rad'];interval=np.r_[0,np.arange(60)//4]
    check(kind+'_actual_raw_schedule_preintegration_mapping',same32(raw['ctrl_rad'],schedule[interval]) and np.array_equal(raw['control_interval_used'],interval) and same32(raw['selected_planned15_FP32_rad'],schedule) and same32(req['selected_planned15_FP32_rad'],schedule))
    event_ok=True;solver_ok=True
    for j,e in enumerate(events):
        k=5+4*j
        event_ok &= e['before_step']==k and e['prestate_index']==k-1 and e['previous_raw_state_ctrl_interval']==j and e['new_solver_ctrl_interval']==j+1 and abs(e['raw_prestate_time_s']-raw['raw_time_s'][k-1])<2e-6 and same32(np.asarray(e['actual_ctrl_readback_native'])[aids],schedule[j+1])
    for j,s in enumerate(solver):
        solver_ok &= s['step']==j+1 and s['prestate_index']==j and s['poststate_index']==j+1 and s['control_interval_used']==j//4 and s['phase']=='just_completed_solver_preintegration' and abs(s['solver_t_s']-raw['raw_time_s'][j])<2e-6 and abs(s['poststate_t_s']-raw['raw_time_s'][j+1])<2e-6 and same32(s['ctrl_solver_native'],schedule[j//4]) and np.array_equal(s['actuator_force_solver_native'],raw['actuator_force_native'][j+1])
    check(kind+'_14_events_state4u0_before5u1',event_ok and same32(raw['ctrl_rad'][4],schedule[0]))
    check(kind+'_60_force_solver_prestate_poststate_clocks',solver_ok and np.array_equal(raw['force_prestate_index'],np.r_[0,np.arange(60)]) and np.max(np.abs(raw['force_solver_time_s']-np.r_[0,raw['raw_time_s'][:-1]]))<2e-6)
    pos=np.linalg.norm(raw['object_pose_w'][:,:3]-ref['object_COM_world_m'],axis=1)
    actual_x=shaft_x(raw['object_pose_w'][:,3:]);expected_x=ref['object_rotation_w'][:,:,0]
    axis=np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',actual_x,expected_x),-1,1)))
    excess=np.maximum(np.maximum(model.jnt_range[jids,0]-raw['joint_rad'],raw['joint_rad']-model.jnt_range[jids,1]),0)
    depths=np.zeros(60);envcount=0
    for c in contacts:
        k=c['step']
        if k==0:continue
        depths[k-1]=max(depths[k-1],max(0,-c['distance_m']));a,b=c['geom'];envcount += int((a in object_geoms or b in object_geoms) and not (a in hand_geoms or b in hand_geoms))
    check(kind+'_raw_contacts_reproduce_depths_and_counts',np.max(np.abs(depths-np.array([s['penetration_m'] for s in solver])))<=1e-12 and np.array_equal(np.bincount([c['step'] for c in contacts],minlength=61),raw['nacon']))
    caplo=np.maximum(np.where(model.actuator_forcelimited[aids],model.actuator_forcerange[aids,0],-np.inf),np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,0],-np.inf))
    caphi=np.minimum(np.where(model.actuator_forcelimited[aids],model.actuator_forcerange[aids,1],np.inf),np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,1],np.inf))
    force=raw['actuator_force_native'][1:];ratio=np.where(force>=0,force/caphi,force/caplo);charge=np.zeros(20);peakcharge=0.
    for dt,rho in zip(np.diff(raw['raw_time_s']),ratio):charge=np.clip(charge+dt*(np.minimum(rho,1)-charge/2.),0,1);peakcharge=max(peakcharge,float(charge.max()))
    primary=np.arange(0,61,4);quality={'moving_object_position_16':pos[primary].max()<=.02,'moving_shaft_angle_16':axis[primary].max()<=20,'actual_joint_state_all61':excess.max()<=.001,'penetration_solver60':depths.max()<=.003,'nonhand_object_contacts':envcount==0,'no_exogenous':raw['external_force_max_abs'].max()==0,'actual_ctrl_schedule_exact':same32(raw['ctrl_rad'],schedule[interval]),'clock':np.max(np.abs(raw['raw_time_s']-np.arange(61)*.005))<=2e-6,'raw_coverage':len(solver)==60 and len(raw['index'])==61,'finite_raw':all(np.isfinite(raw[k]).all() for k in ('full_qpos','full_qvel','ctrl_rad','actuator_force_native')),'load_ratio':ratio.max()<=1.01,'load_charge':peakcharge<=.8,'drop':raw['object_pose_w'][:,2].min()>=raw['mocap_pos_w'][0,0,2]-.15,'capacity':cap['status']=='OK'}
    quality={k:bool(v) for k,v in quality.items()}
    metrics={'max_object_position_error_16_m':float(pos[primary].max()),'max_object_position_error_61_m':float(pos.max()),'max_shaft_angle_16_deg':float(axis[primary].max()),'max_shaft_angle_61_deg':float(axis.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':float(depths.max()),'max_actuator_load_ratio':float(ratio.max()),'max_load_charge':peakcharge}
    check(kind+'_independent_metrics_match_worker_and_summary',all(abs(metrics[k]-ev[k])<=1e-12 and abs(metrics[k]-summary['conditions'][0 if kind=='hold' else 1][k])<=1e-12 for k in metrics))
    check(kind+'_raw_original_quality_gates_exactly_match',quality==ev['gates'] and ev['short_dynamic_pass']==all(quality.values()) and [k for k,v in quality.items() if not v]==['moving_shaft_angle_16'])
    check(kind+'_sticky0_onceat60_noreset',cap['status']=='OK' and cap['overflow_bitmask']==0 and cap['flags_by_world']==[0] and cap['read_count']==1 and cap['read_after_step']==60 and cap['resets_after_initialization']==0)
    fail=np.flatnonzero(axis[primary]>20)
    rows.append({'condition':p['control_condition'],'original_worker_status':ev['status'],'original_short_dynamic_pass':ev['short_dynamic_pass'],'independent_gates':quality,**metrics,'first_primary_shaft_failure_state':int(primary[fail[0]]) if len(fail) else None,'shaft_peak_state61':int(axis.argmax()),'position_peak_state61':int(pos.argmax()),'directed_shaft_metric':'angle(R_actual @ [1,0,0], R_reference @ [1,0,0]); no absolute dot, no axial roll equivalence','raw_final_shaft_direction_w':actual_x[-1].tolist(),'reference_final_shaft_direction_w':expected_x[-1].tolist()})
    alltraces[kind]={'position_m':pos,'shaft_deg':axis,'joint_excess_rad':excess,'solver_depth_m':depths}
    pins[kind+'_plan']=pin(ROOT/f'configs/dynamic-prefix-{kind}-v1.json')
    for f in d.iterdir():
        if f.is_file():pins[kind+'_'+f.name]=pin(f)

check('same_requested_initialization_except_selected_future_schedule',all(np.array_equal(requested['hold'][k],requested['track'][k]) for k in requested['hold'] if k!='selected_planned15_FP32_rad'))
check('same_actual_full_initial_state_ctrl_mocap',all(np.array_equal(raws['hold'][k][0],raws['track'][k][0]) for k in ('full_qpos','full_qvel','full_ctrl_native','mocap_pos_w','mocap_quat_wxyz')))
check('same_raw_preintervention_states_0through4',all(np.array_equal(raws['hold'][k][:5],raws['track'][k][:5]) for k in ('full_qpos','full_qvel','ctrl_rad')))
check('source_window_legacy200_216_not_shared0_151',manifests[0]['source_identity']['primary_pose_source_frame_range_half_open']==[200,216] and manifests[0]['source_identity']['parent_observation_source_frame_range_half_open']==[200,351])
check('summary_status_and_scope_preserved',summary['status']=='PAIRED_COMPLETE_METRICS_RECOMPUTED' and summary['attempts_per_condition']==1 and summary['same_source_lineage_N']==1 and not summary['learning_started'])
check('summary_all_recorded_hashes_still_match',all(sha(v['path'])==v['sha256'] for v in summary['source_pins'].values()))
report={'status':'PASS_SAVED_INDEPENDENT_REVIEW' if all(checks.values()) else 'FAIL_SAVED_INDEPENDENT_REVIEW','checks':checks,'check_count':len(checks),'passed':sum(checks.values()),'original_pair_summary_status':summary['status'],'conditions':rows,'track_minus_hold':{k:rows[1][k]-rows[0][k] for k in ('max_object_position_error_16_m','max_shaft_angle_16_deg','max_actual_joint_excess_rad','max_actuator_load_ratio','max_load_charge')},'source_pins':pins,'reviewed_utc':datetime.now(timezone.utc).isoformat(),'limits':'Sequential known-history N1 per condition. Both violate the original directed shaft gate; lower T0 position error and modest shaft difference do not establish overall task benefit, full3s feasibility, CHORD/learning benefit or independent-ego generalization. No threshold, controller, source, physical state or worker gate changed.','review_execution':{'new_physics_steps':0,'solver_calls':0,'geometry_queries':0,'child_processes':0,'GPU':0,'production_functions_imported':False,'Torch_imported':'torch' in sys.modules,'Warp_imported':'warp' in sys.modules or 'mujoco_warp' in sys.modules}}
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
np.savez_compressed(out/'independent_traces.npz',**{condition+'__'+k:v for condition,d in alltraces.items() for k,v in d.items()})
print(json.dumps({'status':report['status'],'passed':report['passed'],'count':len(checks),'report_sha256':sha(out/'report.json'),'conditions':rows},indent=2))
raise SystemExit(not all(checks.values()))
