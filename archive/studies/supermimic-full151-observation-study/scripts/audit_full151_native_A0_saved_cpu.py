"""Independent raw arithmetic audit of the one completed native full151 A0.

No producer grader/reader/helper is imported. MjModel is read for constant IDs,
names/limits only; no MjData, FK, collision, contactForce, Env, Torch or WM calls.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re,time,traceback
from pathlib import Path
import numpy as np

FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def pin(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def exact(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def rot(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))
def finite_tree(x):
    if isinstance(x,dict):return all(finite_tree(v) for v in x.values())
    if isinstance(x,list):return all(finite_tree(v) for v in x)
    return not isinstance(x,float) or math.isfinite(x)


def main(run,output):
    tick=time.perf_counter();run=Path(run).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    checks={};facts={};metrics={};gates={};exceptions=[];rawpins={}
    def check(k,v):checks[k]=bool(v)
    def close(name,got,want,atol=1e-12):
        err=float(np.max(np.abs(np.asarray(got)-np.asarray(want)))) if np.asarray(got).size else 0.
        facts[name+'_max_abs_error']=err;check(name,err<=atol)
    try:
        files=['config.json','evaluation.json','runtime.json','result.json','completion.json','startup_initialization_checks.json',
               'requested_initialization_and_control.npz','raw_states.npz','solver_samples.json','control_events.json','native_reward_transitions.json','raw_contacts.jsonl',
               'actual_context_readbacks.npz','expected_context_fields.npz','fixed_physics_context.npz','capacity_audit.json','capacity_observer_final.json','controller_descriptor.json','runtime_reset_readback.json']
        rawpins={k:pin(run/k) for k in files}
        config=read(run/'config.json');ev=read(run/'evaluation.json');raw=load(run/'raw_states.npz');sol=read(run/'solver_samples.json');events=read(run/'control_events.json');trans=read(run/'native_reward_transitions.json')
        req=load(run/'requested_initialization_and_control.npz');cap=read(run/'capacity_audit.json');context=load(run/'fixed_physics_context.npz');expected=load(run/'expected_context_fields.npz');readbacks=load(run/'actual_context_readbacks.npz')
        spec=read(config['input_bundle']['path']);ref=load(spec['source_pins']['reference_arrays']['path']);initial=load(spec['source_pins']['init_arrays']['path']);control=load(spec['source_pins']['control_arrays']['path']);frozen_context=load(spec['context']['path']);baseline=control['u_postC2_150_FP32_rad']
        check('complete_raw_601_600_150_150',raw['index'].shape==(601,) and len(sol)==600 and len(events)==150 and len(trans)==150 and np.array_equal(raw['index'],np.arange(601)))
        check('all_raw_arrays_and_JSON_finite',all(v.dtype.kind not in 'fc' or np.isfinite(v).all() for v in raw.values()) and finite_tree(sol) and finite_tree(events) and finite_tree(trans))
        check('saved_baseline_and_initial_requested_aliases',exact(raw['selected_planned150_FP32_rad'],baseline) and exact(req['selected_planned150_FP32_rad'],baseline) and exact(initial['u_planned0_FP32_rad'],baseline[0]))
        # Read immutable model constants, prohibit every native query path.
        import mujoco
        forbidden={}
        def deny(name):
            def call(*a,**kw):forbidden[name]+=1;raise AssertionError('Saved-only audit forbids '+name)
            return call
        for name in ('MjData','mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_kinematics','mj_comPos','mj_collision','mj_geomDistance','mj_contactForce'):
            forbidden[name]=0;setattr(mujoco,name,deny(name))
        m=mujoco.MjModel.from_binary_path(str(run/'nominal_model.mjb'))
        check('actual_nominal_MJB_source_hash',pin(run/'nominal_model.mjb')['sha256']==spec['model']['sha256'])
        joints=np.array([m.joint(f'robot/right_finger{f}_joint{j}').id for f in range(1,6) for j in range(1,5)])
        qa=m.jnt_qposadr[joints];va=m.jnt_dofadr[joints]
        aids=np.array([np.flatnonzero((m.actuator_trnid[:,0]==j)&(m.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)).item() for j in joints])
        obj=m.body('object/cube').id;oj=m.joint('object/cube_freejoint').id;oq=int(m.jnt_qposadr[oj]);ov=int(m.jnt_dofadr[oj])
        check('saved_id_mappings_match_model_constants',np.array_equal(req['joint_qpos_ids'],qa) and np.array_equal(req['joint_dof_ids'],va) and np.array_equal(req['position_ctrl_ids'],aids))
        check('raw_full_state_selected_channels_exact',exact(raw['full_qpos'][:,qa],raw['joint_rad']) and exact(raw['full_qvel'][:,va],raw['joint_velocity_rad_s']) and exact(raw['full_qpos'][:,oq:oq+7],raw['object_pose_w']) and exact(raw['full_qvel'][:,ov:ov+6],raw['object_twist_native']))
        check('joint_limits_from_actual_fixed_MJB',exact(raw['joint_lower_rad'],m.jnt_range[joints,0]) and exact(raw['joint_upper_rad'],m.jnt_range[joints,1]))
        flo=np.maximum(m.actuator_forcerange[aids,0],np.where(m.jnt_actfrclimited[joints],m.jnt_actfrcrange[joints,0],-np.inf));fhi=np.minimum(m.actuator_forcerange[aids,1],np.where(m.jnt_actfrclimited[joints],m.jnt_actfrcrange[joints,1],np.inf))
        check('effective_force_caps_from_MJB_actuator_joint_intersection',exact(flo,raw['force_lower_Nm']) and exact(fhi,raw['force_upper_Nm']) and exact(flo,req['force_lower_Nm']) and exact(fhi,req['force_upper_Nm']))
        check('initial_full_qpos_unprojected_exact_FP32',exact(raw['full_qpos'][0].astype(np.float32),req['full_qpos_requested_FP32']) and exact(raw['joint_rad'][0].astype(np.float32),initial['q_init_requested_FP32_rad']) and exact(raw['object_pose_w'][0].astype(np.float32),initial['object_pose_init_requested_FP32_w']))
        check('initial_joint_and_world_linear_velocity_exact_FP32',exact(raw['joint_velocity_rad_s'][0].astype(np.float32),initial['qd_init_requested_FP32_rad_s']) and exact(raw['object_twist_native'][0,:3].astype(np.float32),initial['object_bodyorigin_v_requested_FP32_world_m_s']))
        body_err=float(np.max(np.abs(raw['object_twist_native'][0,3:]-initial['object_angular_requested_FP32_body_rad_s'])))
        facts['initial_native_body_omega_API_arithmetic_error_rad_s']=body_err
        check('initial_body_omega_original_API_arithmetic_1e6',body_err<=1e-6)
        requested=initial['object_angular_requested_FP32_body_rad_s'];world=(rot(initial['object_pose_init_requested_FP32_w'][3:])@requested).astype(np.float32)
        facts['FP32_roundtrip_body_to_world_omega_error_rad_s']=float(np.max(np.abs(world-initial['object_angular_requested_FP32_world_rad_s'])))
        check('object_COM_offset_zero',np.array_equal(m.body_ipos[obj],np.zeros(3)))
        check('fixed_default_mocap_all601',exact(raw['mocap_pos_w'],np.broadcast_to(raw['mocap_pos_w'][:1],raw['mocap_pos_w'].shape)) and exact(raw['mocap_quat_wxyz'],np.broadcast_to(raw['mocap_quat_wxyz'][:1],raw['mocap_quat_wxyz'].shape)) and exact(raw['mocap_pos_w'][0].astype(np.float32),req['mocap_pos_requested_FP32']) and exact(raw['mocap_quat_wxyz'][0].astype(np.float32),req['mocap_quat_requested_FP32']))
        for k in FIELDS:
            check('actual8_expected__'+k,exact(expected[k],frozen_context[k][0]))
            for stage in ('after_copy','after_set_const_0'):
                v=readbacks[stage+'__'+k];target=np.ascontiguousarray(np.broadcast_to(expected[k],v.shape))
                check('actual8_raw_readback__'+stage+'__'+k,exact(v,target))
            check('actual8_final_fixed_context__'+k,exact(context[k],np.ascontiguousarray(np.broadcast_to(expected[k],context[k].shape))))
        check('reader_xi46_frozen_exact',exact(context['reader_context'],np.ascontiguousarray(np.broadcast_to(frozen_context['reader_context'][:1],context['reader_context'].shape))))
        # Actual episode input/control order, not just event self-assertions.
        used=np.maximum((np.arange(601)-1)//4,0)
        check('all601_actual_u_matches_declared_interval_FP32',exact(raw['ctrl_rad'].astype(np.float32),baseline[used]))
        check('all600_solver_u_matches_same_held_interval',exact(np.asarray([s['ctrl_solver_native'] for s in sol],np.float32),baseline[np.arange(600)//4]))
        check('150events_actual_u_and_zero_residual',all(e['phase']==k and e['preaction_phase']==[k] and e['postaction_phase']==[k+1] and exact(np.asarray(e['actualu'],np.float32)[0],baseline[k]) and np.count_nonzero(e['residual'])==0 for k,e in enumerate(events)))
        check('natural_end_only_interval149_observation150',all(e['terminated']==[False] and e['truncated']==[k==149] and e['reference_end']==[k==149] for k,e in enumerate(events)))
        check('all600_solver_prepost_ids_times_phases',all(s['step']==k+1 and s['prestate_index']==k and s['poststate_index']==k+1 and s['control_interval_used']==k//4 and s['solver_t_s']==raw['raw_time_s'][k] and s['poststate_t_s']==raw['raw_time_s'][k+1] and s['phase']=='just_completed_solver_preintegration_before_original_Manager_forward' for k,s in enumerate(sol)))
        close('raw_actuator_force_and_solver_same_readback',np.asarray([s['actuator_force_solver_native'] for s in sol]),raw['actuator_force_native'][1:],0.)
        close('solver_COM_matches_prestate_position',np.asarray([s['solver_object_COM_w'] for s in sol]),raw['object_pose_w'][:-1,:3],0.)
        solver_pose=np.asarray([s['solver_object_body_pose_w'] for s in sol]);Rpre=rot(raw['object_pose_w'][:-1,3:]);Rsolver=rot(solver_pose[:,3:])
        facts['solver_vs_prestate_rotation_matrix_max_diff']=float(np.max(np.abs(Rsolver-Rpre)))
        check('solver_prestate_rotation_matches_with_normalization_roundoff',facts['solver_vs_prestate_rotation_matrix_max_diff']<=1e-6)
        fpclock=np.empty(601,np.float32);fpclock[0]=0
        for k in range(600):fpclock[k+1]=np.float32(fpclock[k]+np.float32(.005))
        clockexact=exact(raw['raw_time_s'].astype(np.float32),fpclock) and exact(raw['raw_time_s'].astype(np.float32).astype(np.float64),raw['raw_time_s'])
        drift=float(np.max(np.abs(raw['raw_time_s']-np.arange(601)*.005)))
        check('601_clock_FP32_recurrence_exact',clockexact)
        check('declared_50us_3s_clock_gate_and_canonical',drift<=5e-5 and exact(raw['canonical_time_s'],np.arange(601)*.005))
        check('actual_wp_clock_timestep_FP32',exact(req['actual_wp_timestep_FP32_s'],np.array([.005],np.float32)) and exact(req['expected_clock601_FP32_s'],fpclock))
        facts['clock_drift_s']=drift;facts['legacy_2us_diagnostic_pass']=drift<=2e-6
        # Contact geometry/force is a just-completed-solver measurement. No query.
        hand={g for g in range(m.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',m.geom(g).name) and (m.geom_contype[g] or m.geom_conaffinity[g])}
        object_geoms=set(np.flatnonzero(m.geom_bodyid==obj).tolist());by_step=[[] for _ in range(601)]
        raw_contacts=[read_line for read_line in (json.loads(line) for line in (run/'raw_contacts.jsonl').read_text().splitlines())]
        for c in raw_contacts:by_step[c['step']].append(c)
        check('all_saved_contacts_finite_and_phase_scoped',finite_tree(raw_contacts) and all(c['phase']==('startup_after_native_reset' if c['step']==0 else 'just_completed_preintegration_solver') and len(c['geom'])==2 for c in raw_contacts))
        facts['raw_contact_rows']=len(raw_contacts);facts['contact_counts_per601step']=[len(g) for g in by_step]
        facts['nacon_minus_saved_contact_count_minmax']=[int(np.min(raw['nacon']-np.array(facts['contact_counts_per601step']))),int(np.max(raw['nacon']-np.array(facts['contact_counts_per601step'])))]
        categories=('hand_object','hand_self','other');depths=np.zeros((600,3));counts=np.zeros((600,3),int);nonhand=np.zeros(600,int);wrench=np.zeros((600,6));force_errors=[];normal_errors=[];point_errors=[];torque_errors=[];pair_ok=True;active_consistency=True;normal_norm_error=[]
        for k,s in enumerate(sol):
            group=by_step[k+1];ho=[]
            for c in group:
                a,b=c['geom'];has_o=a in object_geoms or b in object_geoms;has_h=a in hand or b in hand
                cat='hand_object' if has_o and has_h else 'hand_self' if a in hand and b in hand else 'other';j=categories.index(cat)
                counts[k,j]+=1;depths[k,j]=max(depths[k,j],-c['distance_m'],0.)
                if has_o and not has_h:nonhand[k]+=1
                if cat=='hand_object':
                    sign=1. if b in object_geoms else -1.;frame=np.asarray(c['frame_rows_w']);f6=np.asarray(c['force6_contact_frame']);f=sign*frame.T@f6[:3];tau=sign*frame.T@f6[3:];point=np.asarray(c['pos_w']);normal=sign*frame[0]
                    normal_norm_error.append(abs(np.linalg.norm(normal)-1.));wrench[k,:3]+=f;wrench[k,3:]+=np.cross(point-np.asarray(s['solver_object_COM_w']),f)+tau
                    ho.append((c,f,tau,(Rsolver[k].T@(point-solver_pose[k,:3])),Rsolver[k].T@normal))
            close('contact_category_depth_step'+str(k+1),depths[k],np.array([s['penet_category_m'][name] for name in categories]),0.)
            close('contact_max_depth_step'+str(k+1),depths[k].max(),s['penetration_m'],0.)
            measured=s['measured_hand_object'];pair_ok &= measured['geom_pair']==[h[0]['geom'] for h in ho]
            if ho:
                force_errors.append(float(np.max(np.abs(np.array([h[1] for h in ho])-measured['force_w_N']))))
                torque_errors.append(float(np.max(np.abs(np.array([h[2] for h in ho])-measured['torque_at_contact_w_Nm']))))
                point_errors.append(float(np.max(np.abs(np.array([h[3] for h in ho])-measured['pos_o_m']))))
                normal_errors.append(float(np.max(np.abs(np.array([h[4] for h in ho])-measured['normal_o']))))
                # efc_address was not saved in raw_contacts: only check that a
                # claimed active row has positive fn. Do not fabricate that field.
                active_consistency &= all(not active or h[0]['force6_contact_frame'][0]>1e-6 for h,active in zip(ho,measured['active']))
            check('object_nonhand_count_step'+str(k+1),nonhand[k]==s['object_environment_contacts'])
        check('all600_measured_handobject_pair_order_matches_raw',pair_ok)
        for name,values in [('force_on_object',force_errors),('direct_torque_on_object',torque_errors),('object_local_contact_point',point_errors),('object_local_normal',normal_errors)]:
            facts[name+'_max_abs_reconstruction_error']=max(values,default=0.);check(name+'_reconstructed_from_saved_frame_force6',max(values,default=0.)<=1e-12)
        facts['normal_world_max_unit_length_error']=max(normal_norm_error,default=0.);check('contact_normals_unit_FP32_roundoff',facts['normal_world_max_unit_length_error']<=1e-6)
        check('saved_active_claims_have_positive_normal_force',active_consistency)
        facts['active_efc_address_independent_reconstruction']='NOT_AVAILABLE_RAW_FIELD_NOT_SAVED; no inferred efc_address or force-based contact label substitution'
        # Source58 snapshots: verify unambiguous native channels exactly. Float
        # frame transforms use saved poses only, report their reconstruction error.
        x0=np.asarray([t['x0'] for t in trans]);x1=np.asarray([t['x1_reward_snapshot'] for t in trans]);xpost=np.asarray([t['x1_after_Manager'] for t in trans]);labels=np.asarray([t['native_selfcollision15'] for t in trans])
        check('native150_reward_x58_self15_shapes_finite',x0.shape==x1.shape==xpost.shape==(150,58) and labels.shape==(150,15) and np.isfinite(x0).all() and np.isfinite(x1).all() and np.isfinite(labels).all())
        check('native_reward_x1_unchanged_by_original_forward',exact(x1,xpost))
        check('native_reward_q_qd_start_end_exact',np.array_equal(x0[:,:20],raw['joint_rad'][0:600:4]) and np.array_equal(x0[:,20:40],raw['joint_velocity_rad_s'][0:600:4]) and np.array_equal(x1[:,:20],raw['joint_rad'][4::4]) and np.array_equal(x1[:,20:40],raw['joint_velocity_rad_s'][4::4]))
        check('native150_transitions_memory_phases_and_end',all(t['phase']==k and t['memory']['time']==[[float(k)]] and t['memory']['start']==[[0.]] and t['memory']['end']==[[151.]] and exact(np.asarray(t['actualu'],np.float32),baseline[k]) for k,t in enumerate(trans)))
        check('native150_memory_actions_residuals_encoder_masks_zero',all(np.count_nonzero(t['memory'][key])==0 for t in trans for key in ('c1','c2','a1','a2','encoder_bias','mask','mask_remaining','mask_active')))
        check('native_self15_saved_boolean_measurements',np.isin(labels,[0.,1.]).all())
        check('native150_torque_reward_matches_last_preintegration_solver',np.array_equal(np.asarray([t['torque_reward_snapshot'] for t in trans]),np.asarray([sol[k]['actuator_force_solver_native'] for k in range(3,600,4)])))
        rewards=np.asarray([e['reward'][0] for e in events]);term_sums=np.array([sum(float(np.asarray(v).reshape(-1)[0]) for v in t['native_terms'].values()) for t in trans]);facts['reward_term_sum_F64_vs_native_F32_max_error']=float(np.max(np.abs(term_sums-rewards)))
        check('native_dt_scaled_term_sum_matches_reward_roundoff',facts['reward_term_sum_F64_vs_native_F32_max_error']<=1e-6)
        facts['native_self15_positive_rows']=int(np.count_nonzero(np.any(labels>0,axis=1)));facts['native_self15_positive_components']=int(np.count_nonzero(labels))
        # Independent original13 gates and continuous scalar outputs.
        R=rot(raw['object_pose_w'][:,3:]);primary=np.arange(0,601,4)
        pos=np.linalg.norm(raw['object_pose_w'][:,:3]-ref['object_COM_world_m'],axis=1);angle=np.rad2deg(np.arccos(np.clip(np.sum(R[:,:,0]*ref['object_rotation_w'][:,:,0],axis=1),-1,1)))
        excess=np.maximum(np.maximum(raw['joint_lower_rad']-raw['joint_rad'],raw['joint_rad']-raw['joint_upper_rad']),0.)
        force=np.asarray([s['actuator_force_solver_native'] for s in sol]);ratio=np.where(force>=0,force/fhi,force/flo)
        charge=np.zeros(20);charge_max=np.zeros(20);charge_trace=[]
        for dt,row in zip(np.diff(raw['raw_time_s']),ratio):charge=np.clip(charge+dt*(np.minimum(row,1)-charge/2.),0,1);charge_max=np.maximum(charge_max,charge);charge_trace.append(charge.copy())
        rate=float(np.abs(np.diff(np.asarray([raw['ctrl_rad'][k] for k in [0,*range(5,598,4)]]),axis=0)).max()/.02)
        gates={'object_position_151':bool(pos[primary].max()<=.02),'directed_shaft_151':bool(angle[primary].max()<=20),'actual_q_all601':bool(excess.max()<=.001),'penetration_solver600':bool(depths.max()<=.003),'no_nonhand_object_contacts':int(nonhand.sum())==0,'no_exogenous':float(raw['external_force_max_abs'].max())==0.,'all_actual_u_exact':bool(np.array_equal(raw['ctrl_rad'].astype(np.float32),baseline[used])),'actual_target_rate':rate<=15.,'clock':bool(clockexact and drift<=5e-5),'load_ratio':bool(ratio.max()<=1.01),'load_charge':bool(charge_max.max()<=.8),'drop':bool(raw['object_pose_w'][:,2].min()>=raw['mocap_pos_w'][0,0,2]-.15),'capacity':cap['status']=='OK'}
        metrics={'max_object_position_error_151_m':float(pos[primary].max()),'max_object_position_error_601_m':float(pos.max()),'max_shaft_angle_151_deg':float(angle[primary].max()),'max_shaft_angle_601_deg':float(angle.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':float(depths.max()),'max_actuator_load_ratio':float(ratio.max()),'max_load_charge':float(charge_max.max()),'max_actual_target_rate_rad_s':rate}
        check('all13_original_gate_values_independently_equal_saved',gates==ev['gates'])
        for k,v in metrics.items():close('original_metric__'+k,v,ev[k],1e-12)
        check('original_quality_FAIL_three_reasons_not_regraded',ev['native_A0_pass'] is False and {k for k,v in gates.items() if not v}=={'object_position_151','directed_shaft_151','penetration_solver600'})
        # Capacity arithmetic from the actual saved raw flags; no inferred zeros.
        OR=0;unknown=0
        for seq,record in enumerate(cap['records'],1):
            flags=np.asarray(record['raw_flags_values']);mask=0
            for value in flags.flat:mask|=int(value)
            OR|=mask;unknown+=record['read_status']=='UNKNOWN'
            check('capacity_record_rawflags_'+str(seq),record['sequence']==seq and flags.dtype.kind in 'iu' and flags.size>0 and record['raw_bitmask']==mask and record['cumulative_OR_bitmask']==OR and record['cumulative_unknown_reads']==unknown and record['flags_cleared_by_observer'] is False and record['read_attempts_for_record']==1)
        check('capacity_constructor_prereset_final_real3records',len(cap['records'])==3 and [v['event'] for v in cap['records']]==['after_constructor_before_actual8','before_reset','final'] and cap['before_reset_read_count']==1 and cap['final_read_count']==1 and cap['finalized'] is True and OR==0 and unknown==0)
        check('capacity_actual_allocation_counts_no_reset_after_start',cap['resets_during_probe']==0 and cap['final_read_after_returned_episode_step']==600 and cap['final_read_after_sampled_solver_row']==600 and cap['maximum_recorded_nacon']==int(raw['nacon'].max()) and cap['maximum_recorded_nefc']==int(raw['nefc'].max()) and raw['nacon'].max()<cap['contact_capacity'] and raw['nefc'].max()<cap['constraint_capacity'])
        check('actual_returned_step_and_forward_counts',ev['episode_API_counts']=={'step_attempted':600,'step_returned':600,'forward_attempted':150,'forward_returned':150} and ev['physical_steps']==ev['sampled_solver_rows']==600 and ev['sampled_state_rows']==601 and ev['native_postinterval_forward_calls']==150 and ev['construction_reset_setup_API_counts']=='UNKNOWN_NOT_WHOLEPROCESS_OBSERVED')
        # Report saturation indicators; passing1.01 gate does not imply no cap.
        facts['near_cap_threshold_counts_diagnostic']={str(th):{'components':int(np.count_nonzero(ratio>=th)),'solver_steps':int(np.count_nonzero(np.any(ratio>=th,axis=1))),'by_joint':np.count_nonzero(ratio>=th,axis=0).tolist()} for th in (.95,.99,.999,.9999,.999999)}
        facts['contact_category_summary']={name:{'max_generated_depth_m':float(depths[:,j].max()),'steps_over3mm':int(np.count_nonzero(depths[:,j]>.003)),'generated_contact_rows':int(counts[:,j].sum()),'steps_with_generated_contacts':int(np.count_nonzero(counts[:,j]))} for j,name in enumerate(categories)}
        # PD arithmetic is diagnostic, separate from reference-predicted caps.
        g=context['actuator_gainprm'][0,aids,0].astype(float);b=context['actuator_biasprm'][0,aids].astype(float)
        check('diagnostic_PD_fixed_affine_joint_unitgear_ABI',np.all(m.actuator_gaintype[aids]==mujoco.mjtGain.mjGAIN_FIXED) and np.all(m.actuator_biastype[aids]==mujoco.mjtBias.mjBIAS_AFFINE) and np.array_equal(m.actuator_gear[aids,0],np.ones(20)) and np.count_nonzero(m.actuator_gear[aids,1:])==0)
        rawpd=g[None]*raw['ctrl_rad'][1:]+b[None,:,0]+b[None,:,1]*raw['joint_rad'][:-1]+b[None,:,2]*raw['joint_velocity_rad_s'][:-1]
        clipped=np.minimum(np.maximum(rawpd,flo),fhi);postpd=g[None]*raw['ctrl_rad'][1:]+b[None,:,0]+b[None,:,1]*raw['joint_rad'][1:]+b[None,:,2]*raw['joint_velocity_rad_s'][1:]
        facts['PD_prestate_capped_vs_solver_force_max_abs_Nm']=float(np.max(np.abs(clipped-force)))
        facts['PD_poststate_capped_wrongphase_diagnostic_max_abs_Nm']=float(np.max(np.abs(np.minimum(np.maximum(postpd,flo),fhi)-force)))
        facts['PD_prestate_uncapped_outside_effective_caps_components']=int(np.count_nonzero((rawpd<flo)|(rawpd>fhi)))
        facts['PD_phase_scope']='F64 affine arithmetic from saved prestate q/qd, actual held ctrl and actual8gain/bias; difference from FP32 solver recorded, not a new gate or a continuous force proof. Poststate arithmetic is separately labelled wrongphase diagnostic.'
        check('all_forbidden_native_queries_unused',not any(forbidden.values()));facts['forbidden_calls']=forbidden
        check('all_input_and_raw_files_unchanged',all(pin(run/k)==v for k,v in rawpins.items()))
        np.savez_compressed(out/'independent_recomputed_arrays.npz',position_error601_m=pos,shaft_error601_deg=angle,joint_excess601_rad=excess,category_depth600_m=depths,category_count600=counts,raw_actuator_load_ratio600=ratio,load_charge600=np.asarray(charge_trace),measured_object_COM_wrench600=wrench,PD_prestate_uncapped600_Nm=rawpd,PD_prestate_capped600_Nm=clipped)
    except Exception as e:exceptions.append({'exception':repr(e),'traceback':traceback.format_exc()})
    status='PASS_SAVED_ENGINEERING_AND_ORIGINAL_QUALITY_RECOMPUTATION' if checks and all(checks.values()) and not exceptions else 'SAVED_AUDIT_FAILURE_RETAINED'
    sibling=run.parent/'full151_native_A0_lifecycle_independent_v1/report.json'
    report={'schema':'full151_native_A0_saved_independent:1','status':status,'reviewer':pin(__file__),'run_dir':str(run),'raw_pins':rawpins,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'engineering_qualified':status.startswith('PASS_'),'original_quality_gates':gates,'native_A0_pass':all(gates.values()) if gates and gates.get('capacity') is True else None,'metrics':metrics,'facts':facts,'exceptions':exceptions,'wall_s':time.perf_counter()-tick,'sibling_source_lifecycle_scope':pin(sibling) if sibling.is_file() else None,
      'qualification_scope':['Same one legacy[200,351) known-rough reference. Original sampled-geometry/SCP FAIL stays retained; actualA0 has no Actor/WM/optimizer or105 checkpoint.','600 just-completed solver contact/force samples are preintegration.601 raw q/qvel samples are postintegration;150 rewards precede originalManagerforward. Direct torque at contact is included when reconstructing object COM wrench.','Self15 is the actual saved native finger-collision sensor truth captured in reward_snapshot, never hand-object labels. Raw primary sensor found values/efc_address are not separately saved, so the label construction itself remains source-bound rather than independently rederived from hand-object geometry.','Absent generated contact is not a positive-gap certificate. Force caps and charge are independent of the3 quality failures. Passing1.01 load gate does not establish no actuator saturation.','Startup/setup wholeprocess API counts UNKNOWN; only episode600/600/150/150 and sampled601/600 are observed.','No production grader/reader/WM/helper called. Only immutable MjModel constants read. Source ZIP/process lifecycle are separately audited by the sibling independent lifecycle review.'],
      'physics_steps_executed_by_this_reviewer':0,'new_FK_collision_or_Env':False,'Torch_or_WM_used':False,'GPU_used':False}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print({'status':status,'passed':report['passed'],'total':report['total'],'native_A0_pass':report['native_A0_pass'],'report':pin(out/'report.json')},flush=True)
    return 0 if status.startswith('PASS_') else 1


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args();raise SystemExit(main(a.run_dir,a.output))
