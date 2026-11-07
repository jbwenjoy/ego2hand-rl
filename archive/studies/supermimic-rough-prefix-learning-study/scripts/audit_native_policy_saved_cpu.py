"""Independent saved-only native Actor physical audit; no Env or physics calls."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse,hashlib,json,math,sys,zipfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
GEOMETRY=ROOT/'outputs/contact_binding_cpu_v1/report.json'
GEOMETRY_SHA='6ae482017180e4feff97a9d4f1250b50748ad552e7684589ebe5b18429783b0b'
MODEL_SHA='24fd7bf2658e1a6795e636eff1f1bc0a6f65f14baf4ed88a85dd2ffe125c33dd'
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
GATES={'object_position_max_m':.02,'directed_shaft_error_max_deg':20.,'actual_joint_hard_excess_max_rad':.001,'penetration_any_category_max_m':.003,'object_nonhand_candidate_contact_count_max':0,'external_force_max_abs':0.,'clock_absolute_tolerance_s':2e-6,'target_rate_max_rad_s':15.,'actuator_load_ratio_max':1.01,'load_charge_max':.8,'charge_decay_time_s':2.,'drop_center_below_default_wrist_m':.15}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):p=Path(p).resolve();return {'path':str(p),'sha256':digest(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def checked(p):
    if set(p)!={'path','sha256'} or not Path(p['path']).is_absolute() or digest(p['path'])!=p['sha256']:raise ValueError('Missing/changed pinned file')
    return Path(p['path'])
def loadnp(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)
def finite(v):
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
    if isinstance(v,np.ndarray):return bool(np.isfinite(v).all()) if v.dtype.kind in 'fiubc' else True
    if torch.is_tensor(v):return bool(torch.isfinite(v).all())
    return math.isfinite(v) if isinstance(v,(float,np.floating)) else True
def same32(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    return a.shape==b.shape and a.tobytes()==b.tobytes()
def quatmat(q):
    q=np.asarray(q,float);norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or np.any(norm<1e-9):raise ValueError('Invalid quaternion')
    w,x,y,z=np.moveaxis(q/norm,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))
def actor_digest(state):
    h=hashlib.sha256()
    for key,v in sorted(state.items()):h.update(key.encode());h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
def predict(action,previous,baseline,lo,hi):
    action=np.asarray(action,np.float32);previous=np.asarray(previous,np.float32);baseline=np.asarray(baseline,np.float32);lo=np.asarray(lo,np.float32);hi=np.asarray(hi,np.float32)
    residual=np.float32(.5)*np.clip(np.float32(.5)*np.clip(action,np.float32(-1),np.float32(1)),np.float32(-.6),np.float32(.6))+np.float32(.5)*previous
    proposed=baseline+residual;hard=np.minimum(np.maximum(proposed,lo),hi);actual=np.minimum(np.maximum(hard,lo+np.float32(.02)),hi-np.float32(.02))
    return {'residual':residual,'proposed':proposed,'hard_bounded':hard,'actual':actual}
def policy_rates(initial,events):
    initial=np.asarray(initial,float).reshape(20);u=np.asarray([e['actualu'] for e in events],float).reshape(-1,20)
    if len(u)!=15 or not finite([initial,u]):raise ValueError('All15 actualtargets required')
    delta=np.diff(np.vstack((initial,u)),axis=0)
    return {'max_rate_rad_s':float(np.max(np.abs(delta))/.02),'startup_jump_max_rad':float(np.max(np.abs(delta[0]))),'startup_rate_rad_s':float(np.max(np.abs(delta[0]))/.02),'subsequent_updates_max_rate_rad_s':float(np.max(np.abs(delta[1:]))/.02),'rate_difference_count':15}
def capacity_status(value,states):
    try:
        raw=np.asarray(value['flags_by_world']);dtype=np.dtype(value['raw_flags_dtype']);flags=raw.astype(dtype)
        if raw.shape!=(1,) or raw.dtype.kind not in 'iu' or dtype.kind not in 'iu' or value['raw_flags_shape']!=[1] or int(raw[0])!=int(flags[0]):return 'UNKNOWN'
        if int(flags[0])!=0:return 'OVERFLOW'
        if any(np.asarray(states[k]).dtype.kind not in 'iu' or np.any(np.asarray(states[k])<0) for k in ('nacon','nefc')):return 'UNKNOWN'
        return 'OK' if max(states['nacon'])<value['contact_capacity'] and max(states['nefc'])<value['constraint_capacity'] else 'UNKNOWN'
    except (ValueError,KeyError,TypeError):return 'UNKNOWN'
def grade(states,solver,events,capacity,reference,wrist_z):
    if len(states.get('index',[]))!=61 or len(solver)!=60 or len(events)!=15:return {'qualified':False,'native_policy_pass':None,'reason':'PARTIAL_UNKNOWN'}
    if not finite([states,solver,events]):return {'qualified':False,'native_policy_pass':None,'reason':'NONFINITE_UNKNOWN'}
    pose=np.asarray(states['object_pose_w']);R=quatmat(pose[:,3:]);Rref=reference['object_rotation_w'];p_error=np.linalg.norm(pose[:,:3]-reference['object_COM_world_m'],axis=-1);axis_error=np.degrees(np.arccos(np.clip(np.einsum('ni,ni->n',R[:,:,0],Rref[:,:,0]),-1,1)))
    fullSO3=np.degrees(np.arccos(np.clip((np.einsum('nij,nij->n',R,Rref)-1)/2,-1,1)))
    excess=np.maximum(np.maximum(states['joint_lower_rad']-states['joint_rad'],states['joint_rad']-states['joint_upper_rad']),0);f=np.array([s['actuator_force_solver_native'] for s in solver]);load=np.where(f>=0,f/states['force_upper_Nm'],f/states['force_lower_Nm']);charge=np.zeros(20);peak=np.zeros(20)
    for dt,row in zip(np.diff(states['raw_time_s']),load):charge=np.clip(charge+dt*(np.minimum(row,1)-charge/2),0,1);peak=np.maximum(peak,charge)
    rate=policy_rates(states['ctrl_rad'][0],events);cap=capacity_status(capacity,states)
    gates={'object_position_16':bool(p_error[::4].max()<=.02),'directed_shaft_16':bool(axis_error[::4].max()<=20),'actual_q_all61':bool(excess.max()<=.001),'penetration_solver60':max(s['penetration_m'] for s in solver)<=.003,'no_nonhand_object_contacts':sum(s['object_environment_contacts'] for s in solver)==0,'no_exogenous':bool(np.max(states['external_force_max_abs'])==0),'all_actual_u_exact':bool(np.asarray(states['ctrl_exact']).all()),'actual_target_rate':rate['max_rate_rad_s']<=15.,'clock':bool(np.max(np.abs(states['raw_time_s']-np.arange(61)*.005))<=2e-6),'load_ratio':bool(load.max()<=1.01),'load_charge':bool(peak.max()<=.8),'drop':bool(pose[:,2].min()>=wrist_z-.15),'capacity':cap=='OK'}
    return {'qualified':cap=='OK','native_policy_pass':all(gates.values()) if cap=='OK' else None,'gates':gates,'criteria':GATES,'max_object_position_error_16_m':float(p_error[::4].max()),'max_object_position_error_61_m':float(p_error.max()),'max_shaft_angle_16_deg':float(axis_error[::4].max()),'max_shaft_angle_61_deg':float(axis_error.max()),'max_full_SO3_error_61_deg':float(fullSO3.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':float(max(s['penetration_m'] for s in solver)),'max_actuator_load_ratio':float(load.max()),'max_load_charge':float(peak.max()),'max_actual_target_rate_rad_s':rate['max_rate_rad_s'],'startup_and_update_rate':rate,'_arrays':{'position_error61_m':p_error,'directed_shaft_error61_deg':axis_error,'full_SO3_error61_deg':fullSO3,'actual_joint_excess61_rad':excess,'load_ratio60':load}}
def contact_rows(raw,object_ids,hand_ids,pose,com):
    R=quatmat(np.asarray(pose)[3:]);pos=np.asarray(pose)[:3];depth={'hand_object':0.,'hand_self':0.,'other':0.};out=[];environment=0;net=np.zeros(6)
    for c in raw:
        g0,g1=c['geom'];hasobj=g0 in object_ids or g1 in object_ids;hashand=g0 in hand_ids or g1 in hand_ids;category='hand_object' if hasobj and hashand else 'hand_self' if g0 in hand_ids and g1 in hand_ids else 'other';depth[category]=max(depth[category],max(0.,-c['distance_m']))
        if hasobj and not hashand:environment+=1
        if category!='hand_object':continue
        sign=1. if g1 in object_ids else -1.;frame=np.asarray(c['frame_rows_w']);wrench=np.asarray(c['force6_contact_frame']);point=np.asarray(c['pos_w']);force=sign*(frame.T@wrench[:3]);direct=sign*(frame.T@wrench[3:]);net[:3]+=force;net[3:]+=np.cross(point-com,force)+direct
        out.append({'pos_o_m':R.T@(point-pos),'normal_o':R.T@(sign*frame[0]),'force_w_N':force,'torque_at_contact_w_Nm':direct,'geom_pair':[g0,g1],'active':bool(wrench[0]>1e-6)})
    return depth,environment,out,net

def audit(config_path,run,a0):
    if not run.exists():return {'schema':'rough_native_policy_saved_CPU_audit:1','status':'NOT_RUN','engineering_qualified':False,'native_policy_pass':None,'passed':0,'total':0,'checks':{},'config':pin(config_path) if config_path.exists() else None,'checkpoint':read(config_path).get('checkpoint') if config_path.exists() else None,'evaluation':None,'run_dir':str(run),'scope':'Missing run is not a physical failure','GPU_used':False},None
    config=read(config_path);checks={};details={};arrays={};issues=[]
    def check(k,v):checks[k]=bool(v)
    def section(k,fn):
        try:return fn()
        except Exception as exc:check(k+'_readable_and_valid',False);issues.append({'stage':k,'error':repr(exc)});return None
    state=section('raw_states',lambda:loadnp(run/'raw_states.npz'));solver=section('solver_samples',lambda:read(run/'solver_samples.json'));events=section('control_events',lambda:read(run/'control_events.json'));capacity=section('capacity',lambda:read(run/'capacity_audit.json'));saved_grade=section('evaluation',lambda:read(run/'evaluation.json'))
    full=state is not None and solver is not None and events is not None and len(state.get('index',[]))==61 and len(solver)==60 and len(events)==15
    check('61states_60solver_15events',full)
    if not full:return {'schema':'rough_native_policy_saved_CPU_audit:1','status':'PARTIAL_OR_INVALID_ENGINEERING_UNKNOWN','engineering_qualified':False,'native_policy_pass':None,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'errors':issues,'config':pin(config_path),'checkpoint':config['checkpoint'],'evaluation':pin(run/'evaluation.json') if (run/'evaluation.json').exists() else None,'run_dir':str(run),'auditor':pin(__file__),'GPU_used':False},None
    check('all_raw_finite',finite([state,solver,events]));check('indices_clock_expected',np.array_equal(state['index'],np.arange(61)) and np.array_equal(state['expected_time_s'],np.arange(61)*.005))
    b=read(checked(config['input_bundle']));ref=loadnp(checked(b['source_pins']['reference_arrays']));init=loadnp(checked(b['source_pins']['init_arrays']));control=loadnp(checked(b['source_pins']['control_arrays']));baseline=control['control__fp32_old_export__tracking_u_exec_15_FP32_rad'];controller=read(run/'controller_descriptor.json');lo=np.array(controller['target_lower_rad'],np.float32);hi=np.array(controller['target_upper_rad'],np.float32)
    def source():
        for p in config['source_pins'].values():checked(p)
        for p in b['source_pins'].values():checked(p)
        inventory=read(run/'source/inventory.json');check('source_inventory_config_exact',inventory==config['source_pins'])
        with zipfile.ZipFile(run/'source/execution_sources.zip') as z:check('source_ZIP_all_pins_exact',all(hashlib.sha256(z.read(f'{i:04d}_{Path(p["path"]).name}')).hexdigest()==p['sha256'] for i,p in enumerate(inventory.values())))
        check('config_copy_exact',digest(config_path)==digest(run/'config.json'));protocol=read(checked(config['protocol']));check('protocol_gates_original',config['evaluation_gates']==protocol['evaluation_gates']==GATES and protocol['input_bundle']==config['input_bundle'])
        check('model_input_unchanged',digest(run/'nominal_model.mjb')==b['model']['sha256']==MODEL_SHA)
        expected=loadnp(checked(b['context']));applied=loadnp(run/'expected_context_fields.npz');got=loadnp(run/'actual_context_readbacks.npz')
        check('actual8_twice_raw_exact',all(np.array_equal(applied[k],expected[k][0]) and all(np.array_equal(got[phase+'__'+k],np.broadcast_to(expected[k][0].astype(got[phase+'__'+k].dtype),got[phase+'__'+k].shape)) for phase in ('after_copy','after_set_const_0')) for k in FIELDS))
        a0got=loadnp(a0/'actual_context_readbacks.npz');check('actual8_twice_same_saved_A0',set(got)==set(a0got) and all(np.array_equal(got[k],a0got[k]) and got[k].dtype==a0got[k].dtype for k in got))
        xi=loadnp(run/'reader_context.npz')['reader_context'];check('reader_xi_raw_exact',np.array_equal(xi,np.broadcast_to(expected['reader_context'][0],xi.shape)))
        check('source_identity_variant_window',b['retarget_variant_id']=='H123_cubic_minimax_restoration_filtered_prefix16_v1' and b['raw_source_identity']['primary_pose_source_frame_range_half_open']==[200,216])
    section('sources_context',source)
    def actor():
        path=checked(config['checkpoint']);s=torch.load(path,map_location='cpu',weights_only=False);loaded=read(run/'actor_load.json');unchanged=read(run/'actor_unchanged.json');evidence=read(run/'actor_state_evidence.json');expected=actor_digest(s['modules']['actor']);details['checkpoint']=pin(path);details['actor_state_sha256']=expected
        check('actor104_selected_final_identity',s['format_version']==104 and s['iteration']==config['checkpoint_iteration'] and s['rough_controller']['root_descriptor']['trained_method']==config['method'] and s['rough_controller']['root_descriptor']['training_seed']==config['seed'])
        check('actor_load_unchanged_digest_checkpoint',loaded['status']=='PASS_EVALUATION_ACTOR_LOAD' and loaded['checkpoint']==unchanged['checkpoint']==config['checkpoint'] and loaded['actor_state_sha256']==unchanged['actor_state_sha256']==expected and unchanged['status']=='PASS' and unchanged['normalization_updated'] is False)
        check('actor_no_update_noise_optimizer',evidence['all_weights_and_normalizer_unchanged'] is True and evidence['norm_update_calls']==evidence['optimizer_objects_created']==evidence['action_noise_draws']==0 and evidence['Actor_mode']=='eval' and unchanged['optimizer_objects_created']==0)
        trained={k:v for k,v in s['rough_controller'].items() if k!='latest_actualu_evidence'};check('same_trained_controller_descriptor',canon(trained)==canon(controller))
        check('training_health_actorload_receipts_bound',read(checked(config['health_report']))['checkpoint']==config['checkpoint'] and read(checked(config['health_report']))['status']=='PASS_SAVED_ENGINEERING_HEALTH')
        details['actor_evidence_strength']='Checkpoint Actor digest independently recomputed; post-eval digest and unchanged-all-tensors assertions are source-enforced saved evidence. No post-eval state_dict or interval policy155 observations were saved for independent inference replay.'
    section('actor',actor)
    def startup():
        s0=loadnp(a0/'raw_states.npz');request=loadnp(run/'requested_initialization_and_control.npz');a0request=loadnp(a0/'requested_initialization_and_control.npz');qa=request['joint_qpos_ids'];va=request['joint_dof_ids'];oq=np.array(sorted(set(range(state['full_qpos'].shape[1]))-set(qa)));ov=np.array(sorted(set(range(state['full_qvel'].shape[1]))-set(va)))
        check('full_initial_state_same_actual_A0',all(same32(state[k][0],s0[k][0]) for k in ('full_qpos','full_qvel','mocap_pos_w','mocap_quat_wxyz','ctrl_rad','object_pose_w','object_twist_native')))
        check('requested_init_control_same_A0',set(request)==set(a0request) and all(np.array_equal(request[k],a0request[k]) and request[k].dtype==a0request[k].dtype for k in request))
        check('requested_nonzero_init_exact',same32(state['full_qpos'][0],request['full_qpos_requested_FP32']) and same32(state['joint_rad'][0],init['q_init_requested_FP32_rad']) and same32(state['joint_velocity_rad_s'][0],init['qd_init_requested_FP32_rad_s']) and same32(state['object_twist_native'][0,:3],init['object_bodyorigin_v_requested_FP32_world_m_s']) and np.allclose(state['object_twist_native'][0,3:],init['object_angular_requested_FP32_body_rad_s'],rtol=0,atol=1e-6) and same32(state['ctrl_rad'][0],init['u_planned0_FP32_rad']))
        check('full_q_qd_object_mapping_all61',len(oq)==7 and len(ov)==6 and np.array_equal(state['full_qpos'][:,qa],state['joint_rad']) and np.array_equal(state['full_qvel'][:,va],state['joint_velocity_rad_s']) and np.array_equal(state['full_qpos'][:,oq],state['object_pose_w']) and np.array_equal(state['full_qvel'][:,ov],state['object_twist_native']))
        check('no_mocap_change_all61',all(same32(state[k],np.broadcast_to(state[k][0],state[k].shape)) for k in ('mocap_pos_w','mocap_quat_wxyz')))
        arrays['joint_qd_reference16_rad_s']=ref['analytic_qd16_rad_s'];arrays['joint_reference16_rad']=ref['q_goal_rad']
    section('startup',startup)
    def controls():
        previous=np.zeros((1,20),np.float32);allok=True;phases=True;substeps=True;preds=[]
        for k,e in enumerate(events):
            prediction=predict(e['action'],previous,baseline[k][None],lo,hi);preds.append(prediction['actual'][0]);phases=phases and e['phase']==k and np.array_equal(e['preaction_phase'],[k]) and np.array_equal(e['postaction_phase'],[k+1]) and not any(e['terminated']) and bool(e['truncated'][0])==(k==14)
            allok=allok and same32(e['action'],np.clip(np.asarray(e['actor_raw_action'],np.float32),-1,1)) and same32(e['prior_residual'],previous) and same32(e['baseline'],baseline[k]) and all(same32(e['independent_expected'][key],v) for key,v in prediction.items()) and same32(e['residual'],prediction['residual']) and same32(e['cached'],prediction['actual']) and same32(e['actualu'],prediction['actual'])
            substeps=substeps and same32(state['ctrl_rad'][4*k+1:4*k+5],np.broadcast_to(prediction['actual'],(4,20)))
            previous=np.asarray(e['residual'],np.float32)
        check('independent_original_EMA_baseline_hard_C2_all15',allok);check('all60_actual_ctrl_match_interval_prediction',substeps);check('prephase_k_postphase_kplus1_terminal15',phases)
        check('saved_planned_and_learned_targets_exact',same32(state['selected_planned15_FP32_rad'],baseline) and same32(state['learned_applied15_FP32_rad'],preds))
        check('actual_ctrl_and_forces_solver_alignment',all(s['step']==i+1 and s['prestate_index']==i and s['poststate_index']==i+1 and s['control_interval_used']==i//4 and s['solver_t_s']==state['raw_time_s'][i] and s['poststate_t_s']==state['raw_time_s'][i+1] and same32(s['ctrl_solver_native'],state['ctrl_rad'][i+1]) and np.array_equal(s['actuator_force_solver_native'],state['actuator_force_native'][i+1]) and s['phase']=='just_completed_solver_preintegration_before_Manager_forward' for i,s in enumerate(solver)))
        arrays['learned_applied15_FP32_rad']=np.asarray(preds,np.float32);details['startup_and_14updates_rate']=policy_rates(state['ctrl_rad'][0],events)
    section('controls_phase',controls)
    def contact():
        if digest(GEOMETRY)!=GEOMETRY_SHA:raise ValueError('Known model geometry ID manifest changed')
        m=read(GEOMETRY)['geometry_metadata'];check('known_geom_catalog_same_MJB',m['model_sha256']==MODEL_SHA);ids=m['geometry'];hands={g['id'] for g in ids['hand_collision_geoms']};objects={g['id'] for g in ids['object_geoms']};raw=[json.loads(line) for line in (run/'raw_contacts.jsonl').read_text().splitlines() if line];check('raw_contacts_finite_valid_step_phase',finite(raw) and all(type(c['step']) is int and 0<=c['step']<=60 and c['phase']==('startup_after_native_reset' if c['step']==0 else 'preintegration_solver') for c in raw));net=[];counts=[];valid=True
        for i,s in enumerate(solver,1):
            rr=[c for c in raw if c['step']==i];depth,n,ho,w=contact_rows(rr,objects,hands,s['solver_object_body_pose_w'],s['solver_object_COM_w']);measured=s['measured_hand_object'];net.append(w);counts.append(len(ho));valid=valid and max(depth.values())==s['penetration_m'] and depth==s['penet_category_m'] and n==s['object_environment_contacts']
            for field in ('pos_o_m','normal_o','force_w_N','torque_at_contact_w_Nm','geom_pair'):
                expected=np.array([r[field] for r in ho]).reshape(-1,2 if field=='geom_pair' else 3);got=np.asarray(measured[field]).reshape(expected.shape);valid=valid and np.allclose(expected,got,rtol=1e-9,atol=1e-10)
            valid=valid and len(measured['active'])==len(ho) and all(bool(a)==r['active'] for a,r in zip(measured['active'],ho))
        check('raw_contact_force_sign_COM_directtorsion_geometry_recomputed',valid);arrays['actual_hand_object_wrench60_world_COM']=np.array(net);arrays['hand_object_generated_count60']=np.array(counts);details['contact_raw_rows_including_startup']=len(raw);details['contact_phase']='Solver[k] is just-completed preintegration force/geometry at state[k-1]; poststate q/qd is state[k]. Direct contact torque included in COM wrench. No continuous dynamics closure claimed.'
    section('contacts',contact)
    independent=section('physical_grade',lambda:grade(state,solver,events,capacity,ref,float(state['mocap_pos_w'][0,0,2])))
    def final():
        runtime=read(run/'runtime.json');ev=saved_grade;result=read(run/'result.json');watchdog=run.with_name(run.name+'_watchdog');identity=read(watchdog/'engineering_identity.json');post=read(watchdog/'post_terminal_CIM.json');check('supervisor_OS_identity_qualified_no_owned_PIDs',identity['engineering_identity_qualified'] is True and all(identity['checks'].values()) and post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[])
        check('native_full_horizon_15forward_noreset_nooptimizer',ev['complete'] is True and ev['physical_steps']==60 and ev['native_postinterval_forward_calls']==15 and ev['control_intervals']==15 and ev['optimizer_updates']==ev['resets_after_start']==0)
        check('runtime_source_config_completed',runtime['status']=='completed_native_policy_evaluation' and runtime['plan_sha256']==digest(config_path) and any(v['sha256']==runtime['source_sha256'] and Path(v['path']).name.startswith('evaluate_rough_prefix_pen') for v in config['source_pins'].values()) and runtime['automatic_retry'] is False)
        check('final_capacity_sticky_once60_noreset',capacity_status(capacity,state)=='OK' and capacity['status']=='OK' and capacity['read_count']==1 and capacity['read_after_step']==60 and capacity['resets_during_probe']==0)
        check('saved_grade_matches_independent_original_gates',independent is not None and independent['gates']==ev['gates'] and independent['native_policy_pass']==ev['native_policy_pass'] and ev['criteria']==GATES)
        if independent is not None:
            keys=('max_object_position_error_16_m','max_object_position_error_61_m','max_shaft_angle_16_deg','max_shaft_angle_61_deg','max_actual_joint_excess_rad','max_penetration_m','max_actuator_load_ratio','max_load_charge','max_actual_target_rate_rad_s')
            check('saved_grade_scalar_values_recomputed',all(math.isclose(independent[k],ev[k],rel_tol=1e-10,abs_tol=1e-10) for k in keys))
        check('result_completed_no_cleanup_errors',result['status']=='COMPLETED_NATIVE_POLICY_ENGINEERING_QUALITY_RECORDED' and result['cleanup_errors']==[])
        details['supervisor_identity']=pin(watchdog/'engineering_identity.json');details['runtime']=pin(run/'runtime.json')
    section('final_identity',final)
    check('CPU_no_environment_device',not torch.cuda.is_initialized() and not any(k.split('.')[0] in ('warp','mujoco','mujoco_warp','mjlab','dex_supermimic') for k in sys.modules))
    qualified=all(checks.values()) and independent is not None and independent['qualified'];quality=independent['native_policy_pass'] if qualified else None
    if independent is not None:arrays.update(independent.pop('_arrays',{}));details['independent_physical_grade']=independent
    return {'schema':'rough_native_policy_saved_CPU_audit:1','status':'PASS_SAVED_ENGINEERING_AUDIT' if qualified else 'INCOMPLETE_OR_INVALID_ENGINEERING_UNKNOWN','engineering_qualified':qualified,'native_policy_pass':quality,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'errors':issues,'details':details,'config':pin(config_path),'checkpoint':config['checkpoint'],'evaluation':pin(run/'evaluation.json'),'run_dir':str(run),'A0_startup_reference':str(a0),'auditor':pin(__file__),'known_geometry_id_source':pin(GEOMETRY),'GPU_used':False,'created_environment':False,'new_geometry_queries':0,'physics_steps':0,'scope':'Independent saved-only physical gate/control/phase/contact reconstruction under the declared native prefix protocol. A single engineering checkpoint/process doesnot establish method benefit, full-motion success or real ego-data generalization.'},arrays

def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not math.isfinite(v):return repr(v)
    return v
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--a0-dir',type=Path,default=ROOT/'outputs/native_a0_v2');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    try:report,arrays=audit(a.config.resolve(),a.run_dir.resolve(),a.a0_dir.resolve())
    except Exception as exc:report={'status':'INVALID_OR_MISSING_SAVED_ARTIFACT_UNKNOWN','engineering_qualified':False,'native_policy_pass':None,'error':repr(exc),'auditor':pin(__file__),'GPU_used':False};arrays=None
    if arrays:np.savez_compressed(a.output/'derived_diagnostics.npz',**arrays)
    out=a.output/'report.json';out.write_text(json.dumps(clean(report),indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'engineering_qualified':report['engineering_qualified'],'native_policy_pass':report['native_policy_pass'],'passed':report.get('passed'),'total':report.get('total'),'failed':[k for k,v in report.get('checks',{}).items() if not v],'error':report.get('error'),'report':pin(out)},indent=2));return 0 if report['engineering_qualified'] or report['status']=='NOT_RUN' else 1
if __name__=='__main__':raise SystemExit(main())
