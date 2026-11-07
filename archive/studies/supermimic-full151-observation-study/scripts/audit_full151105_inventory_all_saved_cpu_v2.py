"""Read-only condition04 inventory/donor audit. No production module import."""
from pathlib import Path
import collections
import datetime
import hashlib
import importlib.util
import json
import os
import sys
import time
import zipfile
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'outputs/full151105_gradient_full_inventory04_v2'
WATCH=RUN.with_name(RUN.name+'_watchdog')
OUT=ROOT/'outputs/full151105_inventory_discovery_all_saved_review_v2'
HELPER=ROOT/'scripts/audit_full151105_phase0_empirical_saved_cpu_v1.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()=='f2df7dbda2f2a8ff0826a612e881786550db0df0037602350c116207514d045f'
spec=importlib.util.spec_from_file_location('_saved_numpy_helpers',HELPER);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
pin,read,load,same,f32,fp32same,maxabs,terms64,pose_terms64=h.pin,h.read,h.load,h.same,h.f32,h.fp32same,h.maxabs,h.terms64,h.pose_terms64
PHASES=(0,49,99,149)

def main():
    OUT.mkdir(exist_ok=False);start=time.perf_counter();checks={};details={};evidence={}
    def check(k,v):checks[k]=bool(v)
    def keep(p):evidence[str(Path(p).relative_to(ROOT))]=pin(p)
    c=read(RUN/'config.json');inv=read(RUN/'source/inventory.json');r=read(RUN/'physical_report.json');a=load(RUN/'physical_arrays.npz')
    rows=read(RUN/'branch_index.json');snaps=read(RUN/'snapshot_records.json');donors=read(RUN/'donor_records.json');ledgers=read(RUN/'capacity_ledgers.json')
    samples=h.lines(RUN/'physical_samples.jsonl');rewards=h.lines(RUN/'reward_snapshots.jsonl')
    rt=read(RUN/'runtime.json');watch=read(WATCH/'state.json');identity=read(WATCH/'engineering_identity.json');post=read(WATCH/'post_terminal_CIM.json')
    rootconfig=pin(watch['command'][4]);bundle=read(c['input_bundle']['path']);context=load(bundle['context']['path']);control=load(bundle['source_pins']['control_arrays']['path']);baseline=control['u_postC2_150_FP32_rad'];reference=load(bundle['source_pins']['reference_arrays']['path'])
    check('582source_current_config_inventory_exact',len(inv)==582 and inv==c['source_pins'] and all(pin(p['path'])['sha256']==p['sha256'] for p in inv.values()))
    with zipfile.ZipFile(RUN/'source/execution_sources.zip') as z:
        names=[f'{i:04d}_{Path(p["path"]).name}' for i,p in enumerate(inv.values())]
        check('582source_ZIP_members_and_bytes_exact',z.namelist()==names and all(hashlib.sha256(z.read(n)).hexdigest()==p['sha256'] for n,p in zip(names,inv.values())))
    check('rootconfig31c28_actualcommand_savedcopy_bound',rootconfig['sha256']==pin(RUN/'config.json')['sha256']==pin(WATCH/'config_snapshot.json')['sha256']==rt['plan_sha256']=='31c28ce071f54965b15dc43c4a4695336595621d750c093b071a6b8ece1617e4' and Path(rootconfig['path']).parent==ROOT/'configs')
    check('actual_worker_a24c_entry517_sourcepinned',any(p['sha256']=='a24c89da47f4aad7acfc9d9dfd1f053518cdc878230716bb2c857127ec13e592' for p in inv.values()) and rt['source_sha256']=='517c102bf3415c0a34523556bceab5255137c76daedc4f60ae9027101f6375a8' and pin(watch['command'][2])['sha256']==rt['source_sha256'])
    check('terminal_exit0_notimeout_completion_real',watch['status']=='completed' and watch['exit_code']==0 and watch['timeout'] is False and rt['status']=='completed_full151_gradient_probe' and (RUN/'completion.json').is_file() and not (RUN/'failure.json').exists())
    check('savedOS12all_true',identity['engineering_identity_qualified'] is True and len(identity['checks'])==12 and all(v is True for v in identity['checks'].values()))
    osr=watch['actual_process_OS_observation'];py=identity['selected_python_identity'];cmd=watch['command']
    check('raw_OS_PID_parent_exe_command_identity',osr['pid']==rt['pid']==watch['actual_pid']==9072 and osr['parent_pid']==rt['parent_pid']==watch['launcher_pid']==21656 and osr['command_line']==f'"{py["base_executable_raw"]}" '+' '.join(cmd[1:]) and os.path.samefile(osr['executable_path'],py['base_executable_raw']))
    check('OScreation_bounds_and_owned_CIM_empty',h.stamp(watch['actual_creation_conservative_bounds']['lower_utc'])<=h.stamp(osr['creation_date_utc'])<=h.stamp(watch['actual_creation_conservative_bounds']['upper_utc']) and post['status']=='NO_OWNED_PID_REMAINING' and post['actual_CIM_rows']==[])
    check('inventory_mode_reviewNone_no_train_retry',c['schema']=='full151105_same_warp_physical_probe_run:2' and c['probe_mode']=='inventory_discovery_all' and c['foreign_inventory_review'] is None and c['training_authorized'] is False and c['automatic_retry'] is False and c['compile_fallback'] is False)
    check('inventory_completed_not_baseline_restore_gradient',r['status']=='COMPLETED_INVENTORY_DISCOVERY' and r['inventory_discovery_only'] is True and r['engineering_qualified'] is True and all(r[k] is False for k in ('baseline_all_passed','full_serial_qualified','foreign_inventory_qualified','restore_inventory_qualified','longclock_qualified','gradient_qualified','whole_graph_hidden_state_qualified','physical_reference_qualified')) and r['empirical_visible_state_baseline_scope']=='NOT_RUN_INVENTORY_ONLY')
    check('zero_begin_finish_native_restore_actual_counters',all(r[k]==0 for k in ('transaction_begin_attempted','transaction_begin_returned','transaction_finish_attempted','transaction_finish_returned','native_snapshot_restore_attempted','native_snapshot_restore_returned','completed_branch_count','baseline_gate_count')))
    check('no_baseline_or_restore_receipts',read(RUN/'baseline_gates.json')==[] and read(RUN/'restore_receipts.json')==[])
    check('integration16get8samevalueSet_separate_counter_scope',r['integration_get_attempted']==r['integration_get_returned']==16 and r['integration_set_attempted']==r['integration_set_returned']==8 and 'same-value' in r['state_write_scope'])
    check('all152branchslots_NOT_RUN_no_restore_masks',len(rows)==152 and [x['row_id'] for x in rows]==list(range(152)) and all(x['scheduled'] is False and x['status']=='NOT_RUN' and x['restorequalified'] is False and x['actualu'] is None and x['r_k'] is None for x in rows) and all(not a[k].any() for k in ('completed','restore_inventory_bitexact','restore_qualified')))
    check('all_branch_response_arrays_NaN',all(np.isnan(a[k]).all() for k in a if k.startswith('branch_')))
    check('8captures_donor_masks_not_restoration',len(snaps)==8 and a['snapshot_completed'].all() and r['completed_snapshot_count']==8 and all(x['scheduled'] is True and x['status']=='REVIEW_PENDING' and x['inventory_construction_completed'] is True and x['integration_roundtrip_bitexact'] is True and x['foreign_review_status']=='REVIEW_PENDING' and x['native_restore_qualification_scope']=='NOT_RUN_INVENTORY_ONLY' and all(x[k] is False for k in ('foreign_review_qualified','full_native_restore_qualified','baseline3_qualified','longclock_qualified')) for x in snaps))
    check('1200steps300intervals_no_model_or_optimizer',r['scheduled_substeps']==r['episode_step_attempted']==r['episode_step_returned']==1200 and r['control_interval_attempted']==r['control_interval_returned']==300 and len(samples)==1200 and len(rewards)==300 and all(r[k]==0 for k in ('optimizer_updates','WM_forwards','Actor_forwards','Runner_instances','model_loads')))
    material=read(ROOT/'outputs/full151_actual_material_cone_contract_v1/material_cone_contract.json');topology=read(material['material_table']['path']);objects={g['geom_id'] for g in topology['rows'] if not g['name'].startswith('robot/')};hand={g['geom_id'] for g in topology['rows'] if g['name'].startswith('robot/') and g['enabled']}
    check('saved_passive_topology_pin_no_native_model_read',pin(material['material_table']['path'])==material['material_table'] and len(objects)==3 and len(hand)==21 and material['inputs']['model']['sha256']==bundle['model']['sha256'] and os.path.samefile(material['inputs']['model']['path'],bundle['model']['path']))
    inventory_pins={};inventory_stats=[];num_arrays={};all_counts=collections.Counter();force_errors=[];moment_errors=[];decode_errors=[];primary_errors=[];native_errors=[];native_phase_quat_diff=[];source_goal_secondary_errors=[]
    for di,name in enumerate(('A0','A2')):
        prefix=name+'__';directory=RUN/('donor_'+name);initial=read(directory/'initial_readback.json');reset=read(directory/'runtime_reset_readback.json');inputs=c['donor_inputs'][name]['input_pins'];old=load(inputs['raw_states.npz']['path']);events_old=read(inputs['control_events.json']['path']);record=donors[di];ss=[s for s in samples if s['path']==name];rr=[q for q in rewards if q['path']==name]
        expected=load(directory/'expected_context_fields.npz');actual=load(directory/'actual_context_readbacks.npz');meta=read(directory/'control_metadata.json');lo=f32(meta['lower']);hi=f32(meta['upper']);bias=f32(meta['encoder_bias']);innerlo=lo+np.float32(.02);innerhi=hi-np.float32(.02)
        check(prefix+'7input_pins_exact',len(inputs)==7 and all(pin(p['path'])==p for p in inputs.values()))
        check(prefix+'actual8_expected_two_measured_readbacks_exact',set(expected)==set(h.FIELDS) and len(actual)==16 and all(same(expected[k],context[k][0]) and all(same(actual[s+'__'+k],np.broadcast_to(expected[k],actual[s+'__'+k].shape)) for s in ('after_copy','after_set_const_0')) for k in h.FIELDS))
        check(prefix+'xi46_model_descriptor_actualsource_exact',same(load(directory/'reader_context.npz')['reader_context'],context['reader_context'][:1]) and pin(directory/'nominal_model.mjb')['sha256']==bundle['model']['sha256'] and read(directory/'controller_descriptor.json')==read(inputs['controller_descriptor.json']['path']))
        check(prefix+'initial_q_qd_ctrl_mocap_fullFP32_exact',initial['time']==0. and all(fp32same(initial[k],old[q][0]) for k,q in (('qpos','full_qpos'),('qvel','full_qvel'),('ctrl','ctrl_rad'),('mocap_pos','mocap_pos_w'),('mocap_quat','mocap_quat_wxyz'))))
        check(prefix+'resetphase0_residualaction_zero',reset['phase']==[0] and reset['time']==[0.] and not np.asarray(reset['EMA_residual']).any() and not np.asarray(reset['action']).any())
        check(prefix+'601_states150_events600samples150rewardrows',record['path']==name and record['fresh_state_count']==601 and len(record['events'])==150 and len(ss)==600 and len(rr)==150 and [q['k'] for q in rr]==list(range(150)) and [q['phase'] for q in record['events']]==list(range(150)))
        rawq=np.vstack((initial['qpos'],[s['post_qpos'] for s in ss]));rawv=np.vstack((initial['qvel'],[s['post_qvel'] for s in ss]));clock=[np.float32(0.)]
        for _ in range(600):clock.append(np.float32(clock[-1]+np.float32(.005)))
        check(prefix+'600poststates_contiguous_prechain',all(fp32same(s['pre_qpos'],rawq[i]) and fp32same(s['pre_qvel'],rawv[i]) for i,s in enumerate(ss)))
        check(prefix+'all_states_and_controls_finite',np.isfinite(rawq).all() and np.isfinite(rawv).all() and all(np.isfinite(np.asarray(s['ctrl'])).all() and np.isfinite(np.asarray(s['actuator_force'])).all() for s in ss))
        check(prefix+'600clock_bits_phase_and_substeps_exact',all(s['phase']==i//4 and s['substep']==i%4+1 and s['canonical_pre_tick']==i and s['canonical_post_tick']==i+1 and s['source_frame']==200+i//4 and fp32same(s['pre_time'],clock[i]) and fp32same(s['post_time'],clock[i+1]) and abs(s['post_time']-(i+1)*.005)<=5e-5 and s['mode']=='donor' and s['branch_id'] is None for i,s in enumerate(ss)))
        check(prefix+'terminal149_to150_no_u150',rr[-1]['k']==149 and rr[-1]['terminal_observation']==150 and rr[-1]['valid_next_policy_decision'] is False and all(q['no_u150'] is True and q['terminal_observation']==q['k']+1 for q in rr))
        actions=np.zeros((150,1,20),np.float32) if name=='A0' else f32([q['action'] for q in events_old]).reshape(150,1,20)
        check(prefix+'fixed_source_actions_not_teacher_forced',fp32same(record['actions'],actions) and np.isfinite(actions).all() and (abs(actions)<=1).all() and (name!='A0' or not any('action' in e for e in events_old)))
        previous=np.zeros((1,20),np.float32);previous_rows=[];actualu=[]
        for k in range(150):
            previous_rows.append(previous.copy());previous=np.float32(.5)*np.clip(np.float32(.5)*np.clip(actions[k],-1.,1.),np.float32(-.6),np.float32(.6))+np.float32(.5)*previous
            u=np.minimum(np.maximum(baseline[k][None]+previous-bias,lo),hi);u=np.minimum(np.maximum(u,innerlo),innerhi);actualu.append(u[0])
        actualu=np.asarray(actualu)
        check(prefix+'150actualu_originalEMA_hard_C2_numpy_exact',all(fp32same(record['events'][k]['actualu'],actualu[k]) and np.asarray(events_old[k]['actualu']).shape==(1,20) and np.asarray(record['events'][k]['actualu']).shape==(20,) and fp32same(f32(events_old[k]['actualu']).reshape(20),actualu[k]) for k in range(150)))
        check(prefix+'all600solver_controls_held_expected_actualu',all(fp32same(s['ctrl'],actualu[i//4]) for i,s in enumerate(ss)) and np.all(actualu>=innerlo) and np.all(actualu<=innerhi))
        smap=read(directory/'sensor_label_mapping.json');all_primary=[];all_native=[];all_netF=[];all_netT=[];all_nacon=[];phase_position=[]
        for k,(q,e) in enumerate(zip(rr,record['events'])):
            last=ss[4*k+3];x=f32(q['physics']);primary=f32(q['matched_post20_GPU_terms_FP32']);native15=f32(q['native15_GPU_object_rate_terms_secondary']);found=np.asarray(q['raw_found'])>0
            label=np.r_[found[0,smap['middle']],found[0,smap['distal']]|found[0,smap['tips']]].astype(np.float32)
            # Exact saved state58 is audited as a capture, not rerun through SimReader.
            check(prefix+'row'+str(k)+'_state_control_label_phase',fp32same(x,e['x']) and fp32same(x[:20],last['post_qpos'][7:]) and fp32same(x[20:40],last['post_qvel'][6:]) and fp32same(q['contact'],e['contact']) and fp32same(label,q['contact']) and fp32same(q['torque'],last['actuator_force']) and q['label_time']=='lastsolver_preintegration~kdt+.015' and q['state_time']=='postintegration_kdt+.020')
            # This all-row secondary uses source pose32 and saved palm32 in F64.
            # Its goal transform can differ from the actual GPU FP32 matmul by
            # roundoff. The exact eight saved gp_GPU/gR_GPU pairs are gated below.
            pose=f32(reference['mapped_object_pose_w'][k]).astype(float);palm=np.asarray(q['state20_reader_palm_rot'])[0];pp=np.asarray(q['state20_reader_palm_pos'])[0]
            source_goal_terms=terms64(x,palm.T@(pose[:3]-pp),palm.T@h.quatmat(pose[3:]));source_goal_secondary_errors.append(maxabs(source_goal_terms-primary))
            all_primary.append(primary);all_native.append(native15)
            check(prefix+'row'+str(k)+'_primary_and_native_sums_distinct',fp32same(np.sum(primary,dtype=np.float32),q['matched_post20_GPU_score_FP32']) and fp32same(np.sum(native15,dtype=np.float32),q['native15_GPU_object_rate_score_secondary']) and fp32same(primary,e['terms_k']) and q['matched_post20_GPU_score_FP32']==e['r_primary'] and fp32same(native15,e['native15_terms']) and q['native_GPU_capture_metadata']['compute_dt_s']==.02 and q['native_GPU_capture_metadata']['scale_by_dt'] is True and q['scalar_contract']==r['scalar_contract'])
            phase_position.append(maxabs(np.asarray(q['native15_object_pose_derived_world'])[0,:3]-np.asarray(last['pre_qpos'])[:3]))
            native_phase_quat_diff.append(maxabs(np.asarray(q['native15_object_pose_derived_world'])[0,3:]-np.asarray(last['pre_qpos'])[3:7]))
            nt=pose_terms64(q['native15_current_pose_wristlocal'][0],q['native15_goal_pose_wristlocal'][0]);native_errors.append(maxabs(nt-native15))
        check(prefix+'RM15_derived_position_equals_last_prestate',max(phase_position)==0.)
        check(prefix+'150event_mocap_fixed_to_initial',all(fp32same(e['rawstate']['mocap_pos'],initial['mocap_pos']) and fp32same(e['rawstate']['mocap_quat'],initial['mocap_quat']) for e in record['events']))
        # Reuse exact captured native goal pose, but avoid a quaternion conversion
        # roundtrip for the primary20 formula at the eight predeclared snapshots.
        for pi,k in enumerate(PHASES):
            sid=di*4+pi;q=rr[k];t=terms64(q['physics'],a['goal_p'][sid],a['goal_R'][sid]);primary=f32(q['matched_post20_GPU_terms_FP32']);bound=8*np.finfo(np.float32).eps*np.maximum(1.,abs(t));primary_errors.append(maxabs(primary.astype(float)-t))
            check(prefix+'snapshot'+str(k)+'_post20_F64_bound',np.all(abs(primary.astype(float)-t)<=bound) and np.array_equal(np.asarray(q['F64_object_rate_terms_diagnostic']),t) and np.array_equal(np.asarray(q['F64_arithmetic_bound']),bound) and fp32same(primary,a['donor_object_terms_k'][sid]) and fp32same(q['physics'],a['donor_next_x'][sid]) and fp32same(q['contact'],a['donor_self15'][sid]))
        check(prefix+'all300_F64_diagnostic_saved_bounds_finite',all(np.isfinite(np.asarray(q['F64_object_rate_terms_diagnostic'])).all() and np.all(abs(f32(q['matched_post20_GPU_terms_FP32']).astype(float)-q['F64_object_rate_terms_diagnostic'])<=q['F64_arithmetic_bound']) and np.array_equal(np.asarray(q['F64_arithmetic_bound']),8*np.finfo(np.float32).eps*np.maximum(1.,abs(np.asarray(q['F64_object_rate_terms_diagnostic'])))) for q in rr))
        # Full-row independent primary rescore uses reference pose transformed with
        # the exact saved palm. Native command goal pose is already palm-local;
        # compare its source-bound command quaternion arithmetic as secondary only.
        for s in ss:
            contacts=s['contacts'];ho=[q for q in contacts if (q['geom'][0] in objects or q['geom'][1] in objects) and (q['geom'][0] in hand or q['geom'][1] in hand)];m=s['measured_hand_object'];F=[];T=[]
            for j,q in enumerate(ho):
                frame=np.asarray(q['frame']);f=np.asarray(q['force6']);sign=1 if q['geom'][1] in objects else -1;F.append(sign*(f[:3]@frame));T.append(sign*(f[3:]@frame));adr=q['efc_address'];mu=np.asarray(m['friction'][j])
                if adr<0:decoded=np.zeros(6)
                else:
                    lam=np.asarray(s['raw_efc_force'])[adr:adr+6];decoded=np.r_[lam.sum(),mu[0]*(lam[0]-lam[1]),mu[1]*(lam[2]-lam[3]),mu[2]*(lam[4]-lam[5]),0.,0.]
                decode_errors.append(maxabs(f-decoded))
            F=np.asarray(F).reshape(-1,3);T=np.asarray(T).reshape(-1,3);pos=np.asarray([q['pos'] for q in ho]).reshape(-1,3);netF=F.sum(0);netT=(np.cross(pos-np.asarray(s['object_solver_COM']),F)+T).sum(0)
            force_errors.extend((maxabs(F-np.asarray(m['force_w_N']).reshape(-1,3)),maxabs(T-np.asarray(m['torque_at_contact_w_Nm']).reshape(-1,3)),maxabs(netF-s['hand_object_force_on_object_w_N'])))
            moment_errors.append(maxabs(netT-s['hand_object_moment_about_COM_w_Nm']));all_netF.append(netF);all_netT.append(netT);all_nacon.append(s['nacon'])
            all_counts[name+'HO_contacts']+=len(ho)
        check(prefix+'all_contacts_counts_order_finite',all(s['nacon']==len(s['contacts'])==len(s['raw_wp_contact']['geom']) and [q['geom'] for q in s['contacts']]==s['raw_wp_contact']['geom'] and len(s['raw_efc_force'])==s['nefc'] and all(np.isfinite(np.asarray(q[k])).all() for q in s['contacts'] for k in ('pos','frame','force6','dist')) for s in ss))
        check(prefix+'all_solver_COM_prestate_noassist_capcounts',all(maxabs(np.asarray(s['object_solver_COM'])-np.asarray(s['pre_qpos'])[:3])==0. and s['external_force_max']==0. and s['nacon']<192 and s['nefc']<4096 and s['phase_contract']=='solver quantities preintegration; q/qvel/time postintegration; before scene.update and Manager forward' for s in ss))
        check(prefix+'all_saved_measured_HO_order_matches_raw',all([q['geom'] for q in s['contacts'] if (q['geom'][0] in objects or q['geom'][1] in objects) and (q['geom'][0] in hand or q['geom'][1] in hand)]==s['measured_hand_object']['geom_pair'] and s['direct_torque_included_once'] is True for s in ss))
        ledger=ledgers[di]['ledger'];records=ledger['records'];events=['after_constructor_before_actual8','after_actual8_before_reset','before_reset']+['after_substep']*600+['donor_terminal','final']
        check(prefix+'external_capacity605_reads_no_restore_clear',ledgers[di]['path']==name and ledger['record_count']==len(records)==605 and [x['event'] for x in records]==events and [x['sequence'] for x in records]==list(range(1,606)) and all(x['flags_cleared_by_observer'] is False and x['read_status']=='OK' and x['raw_flags_values']==[0] and x['cumulative_OR_bitmask']==x['cumulative_unknown_reads']==0 for x in records))
        check(prefix+'capacity_OR0_unknown0_final_once_aftersteps',ledger['status']=='OK' and ledger['engineering_capacity_qualified'] is True and ledger['cumulative_OR_bitmask']==ledger['unknown_read_count']==0 and ledger['final_read_count']==ledger['before_reset_read_count']==1 and ledger['finalized'] is True)
        counts=r['backend_call_counts'];check(prefix+'API600step150forward_returned_donor',counts[name+'/donor']=={'Simulation.forward':150,'Simulation.step':600,'Simulation.step_returned':600,'Simulation.forward_returned':150})
        check(prefix+'constructor_reset_API_separate',counts[name+'/constructor']=={'CPU_mj_forward_API':1,'CPU_mj_forward_API_returned':1,'mjwarp.step_API':2,'mjwarp.step_API_returned':2,'mjwarp.forward_API':2,'mjwarp.forward_API_returned':2} and counts[name+'/checked_reset']=={'Simulation.forward':2,'Simulation.forward_returned':2})
        check(prefix+'archive_drift_descriptive_not_qualityregrade',record['archive_quality_not_regraded'] is True and np.isfinite(list(record['archive_observed_horizon_drift'].values())).all() and float(np.max(np.abs(rawq-old['full_qpos'])))==record['legacy_mixed_q_maxabs_descriptive_only'] and float(np.max(np.abs(rawv-old['full_qvel'])))==record['legacy_mixed_qd_maxabs_descriptive_only'])
        num_arrays[name+'__qpos601']=rawq;num_arrays[name+'__qvel601']=rawv;num_arrays[name+'__actualu150']=actualu;num_arrays[name+'__primary_terms150']=np.asarray(all_primary);num_arrays[name+'__native15_terms150']=np.asarray(all_native);num_arrays[name+'__HO_netforce600']=np.asarray(all_netF);num_arrays[name+'__HO_COMmoment600']=np.asarray(all_netT)
        for pi,k in enumerate(PHASES):
            sid=di*4+pi;s=snaps[sid];label=name+'/'+str(k);base=directory/('snapshot_'+str(k)+'_inventory');desc=read(base/'transaction_description.json');top=read(base/'inventory_topology.json');values=load(base/'inventory_values.npz');profile=read(directory/('profile_'+str(k)+'.json'))
            inventory_pins[label]={'description':pin(base/'transaction_description.json'),'topology':pin(base/'inventory_topology.json')}
            check(label+'__archive3pins_source_worker_tx_exact',all(pin(p['path'])==p for p in s['inventory_archive'].values()) and top['worker_source']['sha256']=='a24c89da47f4aad7acfc9d9dfd1f053518cdc878230716bb2c857127ec13e592' and top['worker_source']==pin(top['worker_source']['path']) and top['transaction_source']==pin(top['transaction_source']['path']))
            check(label+'__label_profile_native_preaction105',s['snapshot_id']==sid and s['k']==k and s['source_frame']==200+k and desc['label']==label and desc['profile']==profile and profile['format_version']==105 and profile['boundary']=='pre_action' and profile['phase']==k and profile['strict_native_types_checked'] is True)
            bytes_valid=[];keys=[];empty=0;total=0;counts=collections.Counter()
            for item in top['array_values']:
                key=item['key'];z=values[key];meta=item['target_meta'];n=int(np.prod(meta[2]));counts[meta[0]]+=1;keys.append(key);total+=len(z)
                valid=z.dtype==np.uint8 and z.ndim==1 and len(z)==(n*h.itemsize(meta) if n else 0) and hashlib.sha256(z.tobytes()).hexdigest()==item['value_sha256']
                if meta[0]=='warp' and meta[-1] is None:empty+=1;valid=valid and n==0 and len(z)==0
                bytes_valid.append(valid)
            check(label+'__all_arraybytes_meta_dtype_shape_SHA',all(bytes_valid) and len(keys)==len(set(keys))==desc['native_Warp_arrays']+desc['native_Torch_tensors']+desc['explicit_arrays'] and len(top['native_entries'])==desc['native_entries'] and len(top['closure_cells'])==desc['closure_cells'] and len(top['explicit_container_entries'])==desc['explicit_containers'])
            check(label+'__574WP_localcounts_emptyNone_no_unknown_dtypes',counts['warp']==desc['native_Warp_arrays']==574 and counts['torch']==desc['native_Torch_tensors'] and counts['numpy']==desc['explicit_arrays'] and empty==212)
            memory={n:np.frombuffer(values[item['key']].tobytes(),np.float32).reshape(item['meta'][2]) for n,item in top['PEN_memory'].items()}
            check(label+'__31PEN_phase_age_c1_action_source',len(memory)==31 and all(np.isfinite(z).all() for z in memory.values()) and np.array_equal(memory['time'],[[k]]) and np.array_equal(memory['end'],[[151]]) and fp32same(memory['c1'],previous_rows[k]) and fp32same(values['source_action_FP32'],actions[k]) and hashlib.sha256(values['source_action_FP32'].tobytes()).hexdigest()==desc['source_action_sha256'])
            check(label+'__integration_raw319_and_clone_only_flags',s['integration_sig']==16383 and s['integration_size']==319 and values['INTEGRATION_FP32'].shape==(1,319) and values['INTEGRATION_FP32'].dtype==np.float32 and np.isfinite(values['INTEGRATION_FP32']).all() and desc['native_physics_qualification']=='NOT_GRANTED_BY_HELPER' and desc['capacity_ledger_inside_snapshot'] is False)
            rng=read(base/'RNG_python_numpy.json');expected_keys=set(keys)|{v['key'] for v in top['PEN_memory'].values()}|{'INTEGRATION_FP32','source_action_FP32','RNG_numpy_MT','torch_cpu_0','torch_cuda_0'}
            check(label+'__RNG_allNPZkeys_unique_exactcoverage',set(values)==expected_keys and rng['numpy_header'][0]=='MT19937' and values['RNG_numpy_MT'].shape==(624,) and values['RNG_numpy_MT'].dtype==np.uint32 and values['torch_cpu_0'].dtype==values['torch_cuda_0'].dtype==np.uint8 and len(rng['python'])==3)
            check(label+'__opaque_explicit_unique_pending_not_qualified',len({z['path'] for z in desc['opaque_identity_only']})==len(desc['opaque_identity_only']) and all(z=='PENDING' for z in desc['qualification'].values()) and s['foreign_review'] is None and s['foreign_review_status']=='REVIEW_PENDING')
            check(label+'__snapshotx_u_rawphase_control_bind',fp32same(a['snapshot_x'][sid][:20],rawq[k*4,7:]) and fp32same(a['snapshot_x'][sid][20:40],rawv[k*4,6:]) and fp32same(a['snapshot_u'][sid],actualu[k]) and fp32same(a['donor_next_x'][sid],rr[k]['physics']))
            inventory_stats.append({'label':label,'native_entries':desc['native_entries'],'WP_arrays':desc['native_Warp_arrays'],'Torch_tensors':desc['native_Torch_tensors'],'sidecar_arrays':desc['explicit_arrays'],'array_bytes_total':total,'empty_Warp_ptr_None':empty,'closures':desc['closure_cells'],'opaque':len(desc['opaque_identity_only']),'NPZ_members':len(values)})
            for filename in ('inventory_values.npz','inventory_topology.json','transaction_description.json','RNG_python_numpy.json'):keep(base/filename)
        for filename in ('initial_readback.json','runtime_reset_readback.json','actual_context_readbacks.npz','expected_context_fields.npz','reader_context.npz'):keep(directory/filename)
    check('all1200_HO_F_world_sign_direct_torque_match',max(force_errors+[0.])==0.)
    check('all1200_COMmoment_cross_plus_direct_match',max(moment_errors+[0.])==0.)
    check('all1200_pyramid6decode_localforce_roundoff',max(decode_errors+[0.])<=1e-12)
    check('native_imports_Env_WM_Actor_GPU_none',not any(n.split('.')[0] in ('mujoco','mujoco_warp','mjlab','warp','torch','scipy','dex_supermimic') for n in sys.modules))
    np.savez(OUT/'independent_numeric_arrays.npz',**num_arrays)
    for filename in ('config.json','source/inventory.json','source/execution_sources.zip','physical_report.json','physical_arrays.npz','snapshot_records.json','branch_index.json','donor_records.json','physical_samples.jsonl','reward_snapshots.jsonl','capacity_ledgers.json'):keep(RUN/filename)
    for filename in ('state.json','engineering_identity.json','post_terminal_CIM.json'):keep(WATCH/filename)
    details.update(inventory_statistics=inventory_stats,primary20_F64_snapshot_term_maxerror=max(primary_errors+[0.]),all300_source_pose_F64_goal_transform_secondary_maxerror=max(source_goal_secondary_errors+[0.]),native15_samepose_F64_secondary_maxerror=max(native_errors+[0.]),native15_vs_rawpre_quaternion_maxdifference_diagnostic=max(native_phase_quat_diff+[0.]),HO_force_directtorque_maxerror=max(force_errors+[0.]),HO_COMmoment_maxerror=max(moment_errors+[0.]),pyramid_decode_maxerror=max(decode_errors+[0.]),contact_counts=dict(all_counts),cost={'native_excluding_imports_s':r['native_wall_seconds_excluding_imports'],'worker_s':rt['wall_seconds'],'watch_s':watch['elapsed_s']},archive_quality_not_regraded=True)
    report={'schema':'full151105_inventory_discovery_all_saved_independent:1','status':'PASS_SAVED_DONOR_AND_VISIBLE_INVENTORY_COVERAGE_ONLY' if all(checks.values()) else 'SAVED_AUDIT_FAILURE_RETAINED','engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,'auditor':pin(__file__),'pure_saved_helper':pin(HELPER),'probe_mode':c['probe_mode'],'completed_inventory_snapshots':int(a['snapshot_completed'].sum()),'episode_step_returned':r['episode_step_returned'],'restored_branches':int(a['restore_qualified'].sum()),'physical_gradient_qualified':False,'whole_hidden_state_qualified':False,'restored_baseline_qualified':False,'input_bundle':c['input_bundle'],'run_dir':str(RUN.resolve()),'config':rootconfig,'producer_report':pin(RUN/'physical_report.json'),'source_inventory':pin(RUN/'source/inventory.json'),'inventory_snapshot_pins':inventory_pins,'evidence':evidence,'details':details,'numeric_arrays':pin(OUT/'independent_numeric_arrays.npz'),'scope':['Two fresh fixed-action donor paths, 1200 episode substeps and eight inventory captures were independently checked from saved data. No restored baseline, perturbation, WM/Actor inference or derivative is present. These are not closed-loop Actor reruns or quality reassessments.','Every saved clone byte payload, dtype/shape, archive SHA and per-label topology/PEN/RNG/integration artifact is checked. Live pointer and target-vs-clone comparisons and same-value INTEGRATION set/get occurred inside the pinned runtime. Only one saved integration/clone payload is available per snapshot; this audit does not rerun those native operations or grant foreign internal-byte restoration.','The eight inventories vary in visited Python containers, tensors, lazy sidecars and opaque paths; no phase0 flat-list review is reused. New per-snapshot source/foreign review remains required. Hidden graph executable caches and scratch are UNKNOWN, irrespective of donor/inventory completion flags.','Primary controlled score uses captured post20 state and goal_k. Original RM derived15 score and its derived pose are separate secondary records. Saved x58 is a native capture; no SimReader/feature replay was executed here. Full-row saved F64 bounds are checked; independent arithmetic rescore initially uses the eight exact saved goal matrices, without a lossy quaternion roundtrip.','HO force and direct torque are reconstructed from saved six-component local contact forces/frames and summed about saved solver COM using raw world contact points. Solver data is preintegration; post q/qvel/time is sampled after each step. Self15 comes from saved raw_found, not HO/human contact.','Constructor CPU forward and Warp graph capture API calls plus checked-reset forwards are separate from actual1200 episode steps. Capacity is an external two-ledger605+605 all-zero observation scope, not a solver-accuracy guarantee.','No new production import, model/Env construction, FK/query, solver, step/forward, WM or GPU execution occurred. No old artifact or source was modified.'],'new_native_Env_model_WM_Actor_GPU':False,'wall_seconds':time.perf_counter()-start}
    report['reviewer_harness_v1_retained']={'source':pin(ROOT/'scripts/audit_full151105_inventory_all_saved_cpu_v1.py'),'report':pin(ROOT/'outputs/full151105_inventory_discovery_all_saved_review_v1/report.json'),'correction':'Only path spelling via samefile plus exact SHA, and explicit old[1,20]/new[20] shape binding before byte comparison; no numeric threshold or physical data change.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'passed':report['passed'],'total':report['total'],'failed':[k for k,v in checks.items() if not v],'report':pin(OUT/'report.json')},indent=2))
    return 0 if all(checks.values()) else 1

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as e:
        if OUT.exists():(OUT/'review_harness_failure.json').write_text(json.dumps({'error':repr(e),'auditor':pin(__file__),'new_native_execution':False},indent=2)+'\n')
        raise
