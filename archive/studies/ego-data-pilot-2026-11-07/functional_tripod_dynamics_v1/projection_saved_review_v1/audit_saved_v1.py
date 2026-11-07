"""Saved-only tripod diagnostic arithmetic; no producer/native/NN import."""
from pathlib import Path
import hashlib,json,math,sys
import numpy as np
H=Path(__file__).resolve().parent;D=H.parent;RUN=D/'projection_completed_run1'
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(p):
    if pin(p['path'])!=p:raise ValueError('Changed pin '+p['path'])
    return Path(p['path'])
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def eq(a,b):
    a,b=np.asarray(a),np.asarray(b);return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def close(a,b):return bool(np.allclose(a,b,atol=2e-12,rtol=2e-12,equal_nan=False))
def first(mask,dt,phase):
    idx=np.nonzero(mask)[0]
    if not len(idx):return None
    i=int(idx[0]);tick=i+(phase=='POST')
    return dict(row=i,tick=tick,time_s=tick*dt,phase=phase)
def decode_contacts(rows,n):
    rows=np.asarray(rows,dtype=float).reshape(-1,24);pad=np.zeros((n,3));other=np.zeros(n);table=np.zeros(n);pen=np.zeros(n);sp=np.zeros(n);ncon=np.zeros(n,dtype=np.int64)
    for row in rows:
        t=int(row[0]);g1,g2=map(int,row[1:3]);assert row[0]==t and 0<=t<n
        ncon[t]+=1;pen[t]=max(pen[t],-float(row[5]));fn=max(0.,float(row[18]))
        if 71 in (g1,g2):
            g=g2 if g1==71 else g1
            if g in [15,28,41]:pad[t,[15,28,41].index(g)]+=fn
            elif g==0:table[t]+=fn
            else:other[t]+=fn
        elif 0 not in (g1,g2):sp[t]=max(sp[t],-float(row[5]))
    return pad,other,table,pen,sp,ncon
def quat_angle(a,b):
    # Pair observer explicitly normalizes endpoints; per-case source uses raw dot.
    a,b=np.asarray(a),np.asarray(b);a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    return float(2*np.arccos(np.clip(abs(a@b),0,1)))
def main():
    rp=RUN/'report.json';r=read(rp);p=read(checked(r['protocol']));t=read(D/'process_projection_run1/terminal.json')
    for x in [*p['inputs'].values(),*p['source_pins'].values()]:checked(x)
    for k in ['source','launcher','protocol','stdout','stderr']:checked(t[k])
    params=arrays(RUN/'actual_frozen_model_parameters.npz');prep=read(RUN/'control_preparation.json')
    binding=read(p['inputs']['model_binding']['path']);mass=float(binding['mass_kg']);gravity=params['option_gravity'];mg=mass*gravity
    gains=params['actuator_gainprm'][:,0];bp=params['actuator_biasprm'];cr=params['actuator_ctrlrange'];ar=params['actuator_forcerange'];jr=params['jnt_actfrcrange'][:20];ql=params['jnt_range'][:20]
    checks={k:True for k in ['source_protocol_terminal_current','all2_candidates_6routes_12cases_retained','inverse_PD_signed_reaction_and_real_margins',
        'all_raw_trace_shapes_finite_and_ticks','all_PRE_contact24_new3_old2_classification','initial_RESET_FORWARD_forces_reproduced',
        'actual_PD_forces_joint_torques_and_margin_arrays','all_POST_metrics_and_old_new_gates_reproduced','all_tail_absent_null_no_2s_imputation',
        'first_abort_boundary_no_missing_prefix','all6_dt_pairs_incomplete_and_endpoint_math','actual6845steps_6136force_reads_and_counts',
        'geometry_positive_gap_and_initial_unloaded_conditions','source_noaid_full_model_audit_scope']}
    checks['source_protocol_terminal_current']=t['returncode']==0 and t['stderr']['bytes']==0 and t['protocol']==r['protocol'] and t['source']==p['source_pins']['runner']
    checks['all2_candidates_6routes_12cases_retained']=len(prep)==2 and len(r['control_routes'])==6 and len(r['cases'])==12 and all(v['available'] for v in r['control_routes'])
    route_by_name={v['name']:v for v in r['control_routes']};candidates={};route_reviews=[]
    for item in prep:
        candidate=read(checked(item['candidate']));candidates[candidate['candidate_id']]=candidate;raw=item['raw'];J=np.asarray(raw['J_hand_anchor3x3x20']);normals=np.asarray(raw['normals_force_on_object_world']);gain=np.asarray(raw['gain20']);offset=np.asarray(raw['offset20'])
        base=np.asarray(raw['qfrc_bias26'])[:20]-np.asarray(raw['qfrc_passive26'])[:20];q=np.asarray(raw['qpos27'])[:20]
        checks['inverse_PD_signed_reaction_and_real_margins'] &= eq(gain,gains) and close(offset,bp[:,0]+bp[:,1]*np.asarray(raw['actuator_length20'])+bp[:,2]*np.asarray(raw['actuator_velocity20']))
        for v in item['routes']:
            checked(v['state_npz'])
            f=np.asarray(v['intended_force_on_object_world_N3x3']);load=v['requested_normal_N_each'];u=np.asarray(v['ctrl20']);desired=np.asarray(v['desired_generalized_actuator_tau20'])
            jt=np.asarray([math.fsum(float(J[i,j,k])*float(f[i,j]) for i in range(3) for j in range(3)) for k in range(20)])
            target=gain*q+offset if load==0 else base+jt
            cm=np.c_[u-cr[:,0],cr[:,1]-u];am=np.c_[desired-ar[:,0],ar[:,1]-desired];jm=np.c_[desired-jr[:,0],jr[:,1]-desired]
            checks['inverse_PD_signed_reaction_and_real_margins'] &= close(f,load*normals) and close(desired,target) and close(gain*u+offset,desired) and close(cm,v['ctrl_margin20x2']) and close(am,v['actuator_force_margin20x2']) and close(jm,v['joint_actuator_force_margin20x2']) and bool(np.all(cm>=0) and np.all(am>=0) and np.all(jm>=0))
            route_reviews.append(dict(name=v['name'],available=v['available'],minimum_ctrl_margin_rad=float(cm.min()),minimum_desired_actuator_force_margin_Nm=float(am.min()),minimum_desired_joint_actuator_force_margin_Nm=float(jm.min())))
    totalsteps=0;totalcontacts=0;initial_contacts=0;case_reviews=[];endpoints={};max_pd_error=0.
    for case in r['cases']:
        a=arrays(checked(case['trace']));dt=case['dt_s'];n=len(a['post_time']);totalsteps+=n;totalcontacts+=len(a['contact_rows']);v=route_by_name[case['name']];q0=np.asarray(v['qpos27']);u=np.asarray(v['ctrl20']);initial=case['initial_native'];initial_contacts+=len(initial['contact_rows24'])
        ticks=np.arange(n);shapes=all(len(a[k])==n for k in ['post_qpos','post_qvel','pre_pad_normal_N','pre_actuator_force','pre_qfrc_actuator','pre_ncon','pre_actuator_length20','pre_actuator_velocity20'])
        checks['all_raw_trace_shapes_finite_and_ticks'] &= shapes and all(np.isfinite(x).all() for x in a.values()) and eq(a['initial_qpos27'],q0) and eq(a['ctrl20'],u) and eq(a['pre_time_from_integer_tick'],ticks*dt) and close(a['post_time'],(ticks+1)*dt) and case['integration_steps']==case['finite_steps_saved']==n
        sample=a['sample_pre_tick'];expected=np.vstack([q0,a['post_qpos'][:-1]])
        checks['all_raw_trace_shapes_finite_and_ticks'] &= eq(sample,np.arange(0,n,round(.01/dt))) and eq(a['sample_pre_qpos'],expected[sample])
        pad,other,table,pen,sp,ncon=decode_contacts(a['contact_rows'],n);oldother=other+pad[:,2]
        checks['all_PRE_contact24_new3_old2_classification'] &= all(eq(x,a[k]) for x,k in [(pad,'pre_pad_normal_N'),(other,'pre_phone_nonpad_normal_N'),(oldother,'pre_old_two_pad_phone_nonpad_normal_N'),(table,'pre_table_normal_N'),(pen,'pre_penetration_m'),(sp,'pre_self_penetration_m'),(ncon,'pre_ncon')])
        ic=np.asarray(initial['contact_rows24'],dtype=float).reshape(-1,24).copy()
        if len(ic):ic[:,0]=0
        ip,io,it,_,_,_=decode_contacts(ic,1)
        checks['initial_RESET_FORWARD_forces_reproduced'] &= eq(ip[0],np.asarray(initial['pad_normal_N3'])) and initial['phone_nonpad_normal_N']==io[0] and initial['table_normal_N']==it[0] and initial['old_two_pad_phone_nonpad_normal_N']==io[0]+ip[0,2] and initial['phase']=='RESET_FORWARD'
        raw=gains*u+bp[:,0]+bp[:,1]*a['pre_actuator_length20']+bp[:,2]*a['pre_actuator_velocity20'];limited=np.clip(raw,ar[:,0],ar[:,1]);joint=np.clip(limited,jr[:,0],jr[:,1])
        max_pd_error=max(max_pd_error,float(np.max(abs(limited-a['pre_actuator_force']))),float(np.max(abs(joint-a['pre_qfrc_actuator'][:,:20]))))
        am=np.stack([a['pre_actuator_force']-ar[:,0],ar[:,1]-a['pre_actuator_force']],axis=-1);jm=np.stack([a['pre_qfrc_actuator'][:,:20]-jr[:,0],jr[:,1]-a['pre_qfrc_actuator'][:,:20]],axis=-1)
        checks['actual_PD_forces_joint_torques_and_margin_arrays'] &= close(limited,a['pre_actuator_force']) and close(joint,a['pre_qfrc_actuator'][:,:20]) and eq(am,a['pre_actuator_force_margin20x2']) and eq(jm,a['pre_joint_actuator_force_margin20x2']) and close(a['ctrl_margin20x2'],v['ctrl_margin20x2']) and np.all(a['pre_qfrc_actuator'][:,20:]==0)
        drift=np.linalg.norm(a['post_COM']-a['initial_COM_world3'],axis=1);angle=2*np.arccos(np.clip(np.abs(a['post_qpos'][:,23:27]@q0[23:27]),0,1));qvio=np.maximum(ql[:,0]-a['post_qpos'][:,:20],0)+np.maximum(a['post_qpos'][:,:20]-ql[:,1],0)
        checks['all_POST_metrics_and_old_new_gates_reproduced'] &= close(angle,a['post_angle'])
        assert n*dt<1.5  # All this run is genuinely before the fixed tail window.
        metrics=dict(final_COM_drift_m=float(drift[-1]),final_angle_rad=float(angle[-1]),tail_linear_velocity_RMS_m_s=None,tail_angular_velocity_RMS_rad_s=None,tail_triple_pad_force_coverage=None,max_native_penetration_m=float(pen.max()),max_self_penetration_m=float(sp.max()),max_phone_nonpad_normal_force_N=float(other.max()),max_table_normal_force_N=float(table.max()),max_joint_limit_excursion_rad=float(qvio.max()),warning_count=sum(case['warnings']))
        oldmetrics={k:x for k,x in metrics.items() if k!='tail_triple_pad_force_coverage'};oldmetrics.update(tail_dual_pad_force_coverage=None,max_phone_nonpad_normal_force_N=float(oldother.max()))
        for saved,actual in [(case['metrics'],metrics),(case['old_two_pad_metrics'],oldmetrics)]:
            checks['all_POST_metrics_and_old_new_gates_reproduced'] &= set(saved)==set(actual) and all(saved[k] is None if x is None else saved[k] is not None and close(saved[k],x) for k,x in actual.items())
        for saved,actual,thresholds,coverage in [(case['gates'],metrics,p['gates'],'tail_triple_pad_force_coverage'),(case['old_two_pad_gates'],oldmetrics,p['old_two_pad_gates'],'tail_dual_pad_force_coverage')]:
            gates=dict(duration_complete=False);gates.update({k:bool(x is not None and (x>=thresholds[k] if k==coverage else x<=thresholds[k])) for k,x in actual.items()})
            checks['all_POST_metrics_and_old_new_gates_reproduced'] &= gates==saved
        checks['all_tail_absent_null_no_2s_imputation'] &= case['nominal_holding_dynamics_pass'] is False and case['old_two_pad_pass'] is False and all(case['metrics'][k] is None for k in ['tail_linear_velocity_RMS_m_s','tail_angular_velocity_RMS_rad_s','tail_triple_pad_force_coverage']) and case['old_two_pad_metrics']['tail_dual_pad_force_coverage'] is None
        abort=(drift>.05)|(angle>math.pi/4)
        checks['first_abort_boundary_no_missing_prefix'] &= case['termination']=='ABORT_LOSS_OF_HOLD' and bool(abort[-1]) and not np.any(abort[:-1])
        events=dict(first_all3_PRE_above_0p1N=first(np.all(pad>.1,axis=1),dt,'PRE'),first_nonpad_PRE_above_0p1N=first(other>.1,dt,'PRE'),first_COM_POST_above_5mm=first(drift>.005,dt,'POST'),first_angle_POST_above_5deg=first(angle>math.pi/36,dt,'POST'))
        # Every saved solver-PRE signed force contributes once; RESET_FORWARD
        # observation is instantaneous and is not integrated a second time.
        forces={k:[] for k in ['pad15','pad28','pad41','other_hand','table']}
        for row in a['contact_rows']:
            g1,g2=map(int,row[1:3])
            if 71 not in (g1,g2):continue
            other_geom=g2 if g1==71 else g1;sign=1. if g2==71 else -1.
            world=sign*(row[9:18].reshape(3,3).T@row[18:21])
            group='pad'+str(other_geom) if other_geom in [15,28,41] else 'table' if other_geom==0 else 'other_hand'
            forces[group].append(world)
        impulses={k:[dt*math.fsum(float(v[j]) for v in values) for j in range(3)] for k,values in forces.items()}
        totalJ=[math.fsum(v[j] for v in impulses.values()) for j in range(3)];gravityJ=(mg*n*dt).tolist()
        case_reviews.append(dict(name=case['name'],candidate_id=case['candidate_id'],route=case['route'],dt_s=dt,steps=n,abort_prefix_s=n*dt,termination=case['termination'],initial_pad_normal_N3=ip[0].tolist(),initial_all3_above_0p1N=bool(np.all(ip[0]>.1)),events=events,
            prefix_all3_loaded_fraction_descriptor=float(np.mean(np.all(pad>.1,axis=1))),prefix_signed_contact_impulse_world_Ns_by_group=impulses,
            prefix_total_external_contact_impulse_world_Ns=totalJ,prefix_gravity_impulse_world_Ns=gravityJ,
            prefix_upward_contact_impulse_over_downward_gravity=float(totalJ[2]/(-gravityJ[2])),
            max_nonpad_positive_Fn_sum_N=float(other.max()),max_old2_nonpad_positive_Fn_sum_N=float(oldother.max()),minimum_actual_actuator_force_margin_Nm=float(am.min()),minimum_actual_joint_actuator_force_margin_Nm=float(jm.min()),new3pad_pass=False,old2pad_pass=False,tail_metrics_available=False,trace=case['trace']))
        endpoints.setdefault(case['name'],[]).append((dt,a['post_COM'][-1],a['post_qpos'][-1,23:27]))
    for pair in r['dt_pair_consistency']:
        e=sorted(endpoints[pair['name']],reverse=True,key=lambda x:x[0]);d=float(np.linalg.norm(e[0][1]-e[1][1]));angle=quat_angle(e[0][2],e[1][2]);expected=dict(both_duration_complete=False,final_COM_difference_m=d<=.002,final_orientation_difference_rad=angle<=p['dt_pair_gates']['final_orientation_difference_rad'])
        checks['all6_dt_pairs_incomplete_and_endpoint_math'] &= len(e)==2 and close(d,pair['final_COM_difference_m']) and close(angle,pair['final_orientation_difference_rad']) and pair['gates']==expected and pair['comparison_status']=='INCOMPLETE_DURATION_NOT_2S_CONSISTENCY' and pair['dt_pair_consistency_pass'] is False
    counts=r['counts'];checks['actual6845steps_6136force_reads_and_counts']=totalsteps==6845==counts['integration_steps'] and totalcontacts+initial_contacts==6136==counts['contact_force_reads'] and counts['MJB_reads']==1 and counts['MjData']==14 and counts['preload_preparation_forward']==2 and counts['initial_forward']==12 and counts['preload_point_jacobians']==6 and counts['neural_training_GPU']==0
    gaps=[]
    for cid,c in candidates.items():
        gg={x['hand_geom_id']:x['query_gap_m'] for x in c['pad_bindings']};gaps.append(dict(candidate_id=cid,independent_convex_query_gaps_m=gg,native_initial_three_pad_loaded=False))
        if cid.startswith('exact_source'):checks['geometry_positive_gap_and_initial_unloaded_conditions'] &= abs(gg[15]-.0001970405625371646)<1e-15
        else:checks['geometry_positive_gap_and_initial_unloaded_conditions'] &= all(x>0 for x in gg.values()) and abs(gg[28]-.00019804777470188586)<1e-15
    checks['geometry_positive_gap_and_initial_unloaded_conditions'] &= all(not x['initial_all3_above_0p1N'] for x in case_reviews)
    checks['source_noaid_full_model_audit_scope']=r['model_audit']['before']==r['model_audit']['after'] and r['model_audit']['all_exposed_arrays_and_numeric_options_unchanged_except_dt'] is True and all(x['no_external_forces_equality_mocap_or_pose_reset'] is True for x in r['cases'])
    output=dict(schema='functional_tripod_native_dynamic_saved_independent:1',status='PASS_SAVED_ALL12_FAILED_HOLD_TIMING_CONTROL_AND_FORCE_ARITHMETIC' if all(checks.values()) else 'FAILED_SAVED_REVIEW_PRESERVED',passed=sum(map(bool,checks.values())),total=len(checks),checks={k:bool(v) for k,v in checks.items()},producer_report=pin(rp),protocol=r['protocol'],terminal=pin(D/'process_projection_run1/terminal.json'),source_auditor=pin(__file__),parameter_arrays=pin(RUN/'actual_frozen_model_parameters.npz'),control_preparation=pin(RUN/'control_preparation.json'),route_reviews=route_reviews,candidate_initial_gaps=gaps,case_reviews=case_reviews,
        saved_counts=dict(integration_steps=totalsteps,per_step_contact_rows=totalcontacts,initial_RESET_FORWARD_contact_rows=initial_contacts,total_contact_force_reads=totalcontacts+initial_contacts),max_PD_reconstruction_error_Nm=max_pd_error,reviewer_new_model_native_forward_steps_NN=0,holding_qualified=False,
        next_native_entry_proposal='New frozen protocol: a functional proposal such as -75um gap on all3 pads, within unchanged hard[-100,+200]um, with all nonpad/self>=0 and real joint/ctrl bounds. Fresh native RESET_FORWARD must verify actual3pad Fn>.1N plus no unwanted/table support before admission. Initial load alone cannot solve observed later intermittent/insufficient support: use actual native wrench/COM feedback in a separately bounded staged closure/control design, and recheck actual friction directions, gravity/moment support and real torque margins. Do not blindly increaseN, require allqacc=0, change materials, add helper forces or call initial loading holding success. Any2s diagnostic must retain its original gates and all failures.',
        scope=['All12 are aborted failure prefixes, not stable hold durations. No2s complete/tail or dt-consistency qualification; all tails remain null.',
            'Old2pad nonpad includes geom41; new3pad excludes41. Both use per-tick summed positive Fn, not individual-contact max.',
            'Nominal normal1N/3N inverse-PD commands are feasible commands, not actual initial forces. Positive geometric gap can yield no native contact; no force inferred from category labels.',
            'Impulse descriptor sums signed frame.T forces on phone at every solverPRE tick times actualdt; gravity comparison is mass from pinned binding times saved actual gravity. It is a prefix descriptor, not proof of momentum closure or sole-cause attribution. Initial-gap absence is not claimed the unique reason for later loss.',
            'Initial/step contacts, margins and stored controls were recomputed independently from saved operands. Raw NN/native data are not re-executed.',
            'POST COM is saved observer evidence; no new FK/rotation converter. Source/runtime no-aid assertions and full-model hash are delegated evidence; unsaved applied-force arrays were not independently reconstructed.'])
    with (H/'report.json').open('x',encoding='utf8') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=pin(H/'report.json'),passed=output['passed'],total=output['total'],checks=output['checks'],saved_counts=output['saved_counts'],max_PD_error=max_pd_error,case_summaries=[dict(route=x['route'],candidate=x['candidate_id'].split('__')[0],dt=x['dt_s'],prefix_s=x['abort_prefix_s'],events=x['events']) for x in case_reviews])))
    return 0 if all(checks.values()) else 1
if __name__=='__main__':sys.exit(main())
