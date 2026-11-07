"""Saved-only two-source16case review; independent old saved math, no physics."""
from pathlib import Path
import hashlib,importlib.util,json,math,sys
import numpy as np
H=Path(__file__).resolve().parent;D=H.parent;RUN=D/'completed_run1'
UTIL=D.parent/'functional_tripod_dynamics_v1/projection_saved_review_v1/audit_saved_v2.py'
s=importlib.util.spec_from_file_location('previous_independent_saved_math',UTIL);u=importlib.util.module_from_spec(s);s.loader.exec_module(u)
def read(p):return json.loads(Path(p).read_text())
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    if pin(r['path'])!=r:raise ValueError('Changed pin '+r['path'])
    return Path(r['path'])
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def main():
    r=read(RUN/'report.json');p=read(checked(r['protocol']));t=read(D/'process_run1/terminal.json')
    for ref in [*p['inputs'].values(),*p['source_pins'].values()]:checked(ref)
    for source in p['preload_sources']:
        for k in ['preload_report','preload_protocol','preload_saved_review']:checked(source[k])
    for k in ['source','launcher','protocol','stdout','stderr']:checked(t[k])
    checks={k:True for k in ['all_pins_sources_terminal_and_two_parents','all2x4x2_cases_and_actual_control_parent_bits','all16_initial_Fn_bits_actual3pad_loaded',
        'all_trace_finite_ticks_PRE_POST_prefix_binding','contact24_classification_new3pad_and_old2pad','actual_control_and_force_margin_arithmetic',
        'all_original_new_old_metrics_gates_null_tail','all_first_abort_times_and_incomplete_dt_pairs','counts9200steps7900forces16initial0prep','unchanged_model_and_noaid_SOURCE_scope']}
    checks['all_pins_sources_terminal_and_two_parents']=t['returncode']==0 and t['stderr']['bytes']==0 and t['protocol']==r['protocol'] and t['source']==p['source_pins']['runner'] and len(p['preload_sources'])==2
    variants={v['name']:v for v in r['variants']};checks['all2x4x2_cases_and_actual_control_parent_bits']=len(variants)==8 and len(r['cases'])==16
    source_by_id={v['source_id']:v for v in p['preload_sources']};case_reviews=[];ends={};total_steps=total_contacts=initial_contacts=0
    for c in r['cases']:
        v=variants[c['name']];probe=read(checked(v['source_native_probe']));checked(v['candidate']);checked(v['state_npz']);src=source_by_id[v['source_id']];pre=read(src['preload_report']['path']);parent=pre['candidates'][0]
        expected=parent['final_repeat_probes'][0] if v['route']=='GN_best_final_fresh' else next(x['probe'] for x in parent['routes'] if x['route']==v['route'])
        checks['all2x4x2_cases_and_actual_control_parent_bits'] &= v['source_native_probe']==expected and u.eq(np.asarray(v['ctrl20']),np.asarray(probe['raw']['ctrl20'])) and u.eq(np.asarray(v['qpos27']),np.asarray(probe['raw']['qpos27'])) and c['command']==v
        a=arrays(checked(c['trace']));dt=c['dt_s'];n=len(a['post_time']);total_steps+=n;total_contacts+=len(a['contact_rows']);q0=a['initial_qpos27'];initial=c['initial_native'];initial_contacts+=len(initial['contact_rows24']);ticks=np.arange(n)
        checks['all_trace_finite_ticks_PRE_POST_prefix_binding'] &= all(np.isfinite(x).all() for x in a.values()) and c['integration_steps']==c['finite_steps_saved']==n and u.eq(a['ctrl20'],np.asarray(probe['raw']['ctrl20'])) and u.eq(q0,np.asarray(probe['raw']['qpos27'])) and u.eq(a['pre_time_from_integer_tick'],ticks*dt) and u.close(a['post_time'],(ticks+1)*dt)
        sample=a['sample_pre_tick'];checks['all_trace_finite_ticks_PRE_POST_prefix_binding'] &= u.eq(a['sample_pre_qpos'],np.vstack([q0,a['post_qpos'][:-1]])[sample]) and u.eq(sample,np.arange(0,n,round(.01/dt)))
        pad,other,table,pen,sp,ncon=u.decode_contacts(a['contact_rows'],n);oldother=other+pad[:,2]
        checks['contact24_classification_new3pad_and_old2pad'] &= all(u.eq(x,a[k]) for x,k in [(pad,'pre_pad_normal_N'),(other,'pre_phone_nonpad_normal_N'),(oldother,'pre_old_two_pad_phone_nonpad_normal_N'),(table,'pre_table_normal_N'),(pen,'pre_penetration_m'),(sp,'pre_self_penetration_m'),(ncon,'pre_ncon')])
        ic=np.asarray(initial['contact_rows24']).reshape(-1,24).copy();ic[:,0]=0;ip,io,it,_,_,_=u.decode_contacts(ic,1)
        checks['all16_initial_Fn_bits_actual3pad_loaded'] &= u.eq(ip[0],np.asarray(initial['pad_normal_N3'])) and u.eq(ip[0],np.asarray(probe['pad_normal_N3'])) and bool(np.all(ip[0]>.1)) and initial['phase']=='RESET_FORWARD' and io[0]==initial['phone_nonpad_normal_N'] and it[0]==initial['table_normal_N']
        pd=probe['PD'];control=np.asarray(probe['raw']['ctrl20']);cr=np.asarray(pd['ctrlrange20x2']);ar=np.asarray(pd['actuator_forcerange20x2']);jr=np.asarray(pd['joint_actfrcrange20x2']);cm=np.c_[control-cr[:,0],cr[:,1]-control]
        am=np.stack([a['pre_actuator_force']-ar[:,0],ar[:,1]-a['pre_actuator_force']],axis=-1);jm=np.stack([a['pre_qfrc_actuator'][:,:20]-jr[:,0],jr[:,1]-a['pre_qfrc_actuator'][:,:20]],axis=-1)
        checks['actual_control_and_force_margin_arithmetic'] &= u.eq(a['ctrl_margin20x2'],cm) and u.eq(a['pre_actuator_force_margin20x2'],am) and u.eq(a['pre_joint_actuator_force_margin20x2'],jm) and bool(np.all(cm>=0) and np.all(am>=0) and np.all(jm>=0))
        drift=np.linalg.norm(a['post_COM']-a['initial_COM_world3'],axis=1)
        angle=np.asarray([2*np.arccos(np.clip(abs(q[23:27]@q0[23:27]),0,1)) for q in a['post_qpos']])
        # Replay exactly the original row-wise quaternion observer; no new
        # physical rotation truth/converter or batched near-one dot substitute.
        assert u.eq(angle,a['post_angle'])
        qvio=0.
        # Saved case metric0 is independently checked against all explicit
        # source hardware ranges captured with the compressed parent model.
        # The model's jnt_range is pinned in the original geometry archive.
        ranges_path=D.parent/'functional_tripod_support_v1/completed_v1/compiled_geometry_and_limits.npz'
        if not hasattr(main,'limits'):
            with np.load(ranges_path,allow_pickle=False) as z:main.limits=z['jnt_range'][:20].copy()
        ql=main.limits;qvio=float((np.maximum(ql[:,0]-a['post_qpos'][:,:20],0)+np.maximum(a['post_qpos'][:,:20]-ql[:,1],0)).max())
        assert n*dt<1.5
        metrics=dict(final_COM_drift_m=float(drift[-1]),final_angle_rad=float(angle[-1]),tail_linear_velocity_RMS_m_s=None,tail_angular_velocity_RMS_rad_s=None,tail_triple_pad_force_coverage=None,max_native_penetration_m=float(pen.max()),max_self_penetration_m=float(sp.max()),max_phone_nonpad_normal_force_N=float(other.max()),max_table_normal_force_N=float(table.max()),max_joint_limit_excursion_rad=qvio,warning_count=sum(c['warnings']))
        old={k:x for k,x in metrics.items() if k!='tail_triple_pad_force_coverage'};old.update(tail_dual_pad_force_coverage=None,max_phone_nonpad_normal_force_N=float(oldother.max()))
        for actual,saved,thresholds,key in [(metrics,c['metrics'],p['gates'],'tail_triple_pad_force_coverage'),(old,c['old_two_pad_metrics'],p['old_two_pad_gates'],'tail_dual_pad_force_coverage')]:
            gates=dict(duration_complete=False);gates.update({k:bool(x is not None and (x>=thresholds[k] if k==key else x<=thresholds[k])) for k,x in actual.items()})
            checks['all_original_new_old_metrics_gates_null_tail'] &= gates==(c['gates'] if key=='tail_triple_pad_force_coverage' else c['old_two_pad_gates']) and all(saved[k] is None if x is None else saved[k] is not None and u.close(saved[k],x) for k,x in actual.items())
        checks['all_original_new_old_metrics_gates_null_tail'] &= not c['nominal_holding_dynamics_pass'] and not c['old_two_pad_pass']
        abort=(drift>.05)|(angle>math.pi/4);checks['all_first_abort_times_and_incomplete_dt_pairs'] &= c['termination']=='ABORT_LOSS_OF_HOLD' and bool(abort[-1]) and not np.any(abort[:-1])
        groups={k:[] for k in ['pad15','pad28','pad41','other_hand','table']}
        for row in a['contact_rows']:
            g0,g1=map(int,row[1:3])
            if 71 not in (g0,g1):continue
            g=g0 if g1==71 else g1;sign=1 if g1==71 else -1;frame=row[9:18].reshape(3,3)
            force=np.asarray([sign*math.fsum(float(frame[i,j])*float(row[18+i]) for i in range(3)) for j in range(3)])
            groups['pad'+str(g) if g in [15,28,41] else 'table' if g==0 else 'other_hand'].append(force)
        impulse={k:[dt*math.fsum(float(v[j]) for v in values) for j in range(3)] for k,values in groups.items()}
        gravityJ=probe['mass_kg']*np.asarray(probe['gravity_world_m_s2'])*n*dt;padJ=np.sum([impulse[k] for k in ['pad15','pad28','pad41']],axis=0);otherJ=np.asarray(impulse['other_hand'])+impulse['table']
        case_reviews.append(dict(source_id=v['source_id'],route=v['route'],name=c['name'],dt_s=dt,steps=n,failed_prefix_duration_s=n*dt,initial_Fn3=ip[0].tolist(),initial_Fn_bits_match_native_parent=True,termination=c['termination'],trace=c['trace'],native_parent_probe=v['source_native_probe'],
            prefix_all3_Fn_gt_0p1_coverage=float(np.mean(np.all(pad>.1,axis=1))),first_any_pad_PRE_le_0p1N=u.first(np.any(pad<=.1,axis=1),dt,'PRE'),first_nonpad_PRE_gt_0p1N=u.first(other>.1,dt,'PRE'),first_COM_POST_gt_5mm=u.first(drift>.005,dt,'POST'),first_angle_POST_gt_5deg=u.first(angle>math.pi/36,dt,'POST'),
            impulse_world_Ns_by_group=impulse,designated_pad_impulse_world_Ns=padJ.tolist(),other_hand_table_impulse_world_Ns=otherJ.tolist(),gravity_impulse_world_Ns=gravityJ.tolist(),
            upward_designated_impulse_over_gravity=float(padJ[2]/(-gravityJ[2])),upward_other_impulse_over_gravity=float(otherJ[2]/(-gravityJ[2])),
            max_nonpad_positive_Fn_sum_N=float(other.max()),min_actual_force_margin_Nm=float(am.min()),min_actual_joint_force_margin_Nm=float(jm.min()),all_tail_metrics=None,holding_qualified=False))
        ends.setdefault(c['name'],[]).append((dt,a['post_COM'][-1],a['post_qpos'][-1,23:27]))
    for pair in r['dt_pair_consistency']:
        e=sorted(ends[pair['name']],key=lambda x:x[0],reverse=True);dist=float(np.linalg.norm(e[0][1]-e[1][1]));angle=u.quat_angle(e[0][2],e[1][2]);g=dict(both_duration_complete=False,final_COM_difference_m=dist<=.002,final_orientation_difference_rad=angle<=p['dt_pair_gates']['final_orientation_difference_rad'])
        checks['all_first_abort_times_and_incomplete_dt_pairs'] &= len(e)==2 and pair['gates']==g and u.close(dist,pair['final_COM_difference_m']) and u.close(angle,pair['final_orientation_difference_rad']) and pair['comparison_status']=='INCOMPLETE_DURATION_NOT_2S_CONSISTENCY' and pair['dt_pair_consistency_pass'] is False
    cc=r['counts'];checks['counts9200steps7900forces16initial0prep']=total_steps==cc['integration_steps']==9200 and total_contacts+initial_contacts==cc['contact_force_reads']==7900 and cc['initial_forward']==cc['MjData']==16 and cc['MJB_reads']==1 and cc['preload_preparation_forward']==cc['preload_point_jacobians']==cc['neural_training_GPU']==0
    checks['unchanged_model_and_noaid_SOURCE_scope']=r['model_audit']['before']==r['model_audit']['after'] and r['model_audit']['unchanged_except_dt'] is True and all(c['no_external_forces_equality_mocap_or_pose_reset'] is True for c in r['cases'])
    out=dict(schema='functional_tripod_loaded_dynamic_saved_independent:1',status='PASS_ALL16_SAVED_ABORTED_PREFIXES_NOT_HOLDING' if all(checks.values()) else 'FAILED_SAVED_REVIEW_PRESERVED',passed=sum(map(bool,checks.values())),total=len(checks),checks={k:bool(v) for k,v in checks.items()},producer_report=pin(RUN/'report.json'),protocol=r['protocol'],terminal=pin(D/'process_run1/terminal.json'),source_auditor=pin(__file__),saved_math_helper=pin(UTIL),hardware_ranges_archive=pin(D.parent/'functional_tripod_support_v1/completed_v1/compiled_geometry_and_limits.npz'),case_reviews=case_reviews,
        actual_counts=dict(steps=total_steps,per_step_contact_rows=total_contacts,initial_contact_rows=initial_contacts,force_reads=total_contacts+initial_contacts,initial_forward=16,preload_forward=0),reviewer_new_native_NN_calls=0,holding_qualified=False,
        conclusion='Initial three-pad loading is present and bit-matches the native parent under bothdt, so missing initial contact was removed. All16 still fail2s: sustained designated-pad wrench/force/trajectory is insufficient. Static GN loading did not provide successful holding. Failure-prefix duration must not be called stable duration.',
        scope=['Independent saved contact grouping/impulses/force margins/ticks/gates/counts and parent control/source joins; no native/NN/solver execution.',
            'Row-wise raw quaternion-dot observer is an exact same-arithmetic replay, not independent angle/geometry truth. COM uses saved native observer values.',
            'All tails null and eightdt pairs incomplete. Prefix coverage/impulses are descriptors, never tail estimates or full2s qualification.',
            'Signed force frame.T/sign is used for impulse without positiveFn filtering. Fn gate classification uses the original positiveFn sum. Other-hand/table impulse is kept separate from designated-pad support.',
            'Native noaid/model invariance use source/runtime and saved hash guards; unsaved appliedforce vectors are not independently reconstructed. Hardware-range archive is pin-captured from the same original model; no current model read.'])
    with (H/'report.json').open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(H/'report.json'),passed=out['passed'],total=out['total'],checks=out['checks'],counts=out['actual_counts'],cases=[{k:c[k] for k in ['source_id','route','dt_s','failed_prefix_duration_s','prefix_all3_Fn_gt_0p1_coverage','upward_designated_impulse_over_gravity','upward_other_impulse_over_gravity']} for c in case_reviews])))
    return 0 if all(checks.values()) else 1
if __name__=='__main__':sys.exit(main())
