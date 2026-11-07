"""Saved four-case timing/contact/gate review; no backend/model import."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

OUT=Path(__file__).resolve().parent
R=OUT.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def current(p):return pin(p['path'])==p
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def first_event(mask,dt,phase,values):
    rows=np.flatnonzero(mask)
    if not len(rows):return None
    i=int(rows[0]);tick=i if phase=='PRE' else i+1
    return {'phase':phase,'record_row_index':i,'tick':tick,'nominal_time_s':tick*dt,'value':float(values[i])}

def main():
    report_path=R/'cpu_nominal_v1/report.json';report=read(report_path);protocol=read(report['protocol']['path']);terminal=read(R/'process_run1/terminal.json')
    gates=protocol['gates'];inputs={'report':pin(report_path),'protocol':report['protocol'],'terminal':pin(R/'process_run1/terminal.json')}
    checks={k:True for k in ['source_protocol_terminal_and_logs_pinned','all_input_source_pins_current','four_case_steps_finite_arrays_and_integer_ticks','source_pre_geometry_qpos_equals_previous_post','all_contact_rows_and_PRE_aggregates_reconstructed','initial_qpos_and_command_match_selected_native_record','reported_metrics_and_all_original_gates_reproduced','all4_loss_abort_no_missing_or_extra_ticks','noaid_report_and_source_guards_preserved','both_dt_pairs_incomplete_not_2s_qualified','actual2959steps_and2581contact_reads_match']}
    checks['source_protocol_terminal_and_logs_pinned']=(terminal['returncode']==0 and all(current(terminal[k]) for k in ['source','runner','protocol','stdout','stderr']) and terminal['stderr']['bytes']==0 and terminal['protocol']==report['protocol'] and read(R/'cpu_nominal_v1/protocol.json')==protocol)
    checks['all_input_source_pins_current']=all(current(p) for p in [*protocol['inputs'].values(),*protocol['source_pins'].values()])
    cases=[];events_csv=[];byname={};contact_total=0;steps_total=0;max_scalar_metric_error=0.
    for c in report['cases']:
        assert current(c['trace']) and current(c['source_candidate_state'])
        with np.load(c['trace']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        n=len(a['post_time']);dt=c['dt_s'];ticks=np.arange(n,dtype=np.int64)
        branch=c['name'].removeprefix('source_mapping_');screen=next(x for x in report['screening'] if x['branch']==branch);selected=read(screen['selected_record']['path'])
        assert current(screen['selected_record'])
        native=selected['native_probe'];names={int(x['geom_id']):x['name'] for x in selected['final_geometry']['state']['phone_geometry_rows']}
        names.update({71:'phone24_collision_convex',15:'right_finger1_tip_sensor_col',28:'right_finger2_tip_sensor_col'})
        inputs[branch+'_selected_native_record']=screen['selected_record']
        shapes={'post_COM':(n,3),'post_qpos':(n,27),'post_qvel':(n,26),'pre_pad_normal_N':(n,2),'pre_qfrc_actuator':(n,26),'pre_actuator_force':(n,20)}
        good=all(a[k].shape==v for k,v in shapes.items()) and all(np.isfinite(v).all() for v in a.values())
        checks['four_case_steps_finite_arrays_and_integer_ticks'] &= good and np.array_equal(a['pre_time_from_integer_tick'],ticks*dt) and np.max(abs(a['post_time']-(ticks+1)*dt))<1e-10 and c['finite_steps_saved']==c['integration_steps']==n
        sample=a['sample_pre_tick'].astype(int);expected_pre=np.array([a['initial_qpos27'] if t==0 else a['post_qpos'][t-1] for t in sample])
        checks['source_pre_geometry_qpos_equals_previous_post'] &= np.array_equal(expected_pre,a['sample_pre_qpos']) and np.array_equal(sample,np.arange(0,n,round(.01/dt)))
        checks['initial_qpos_and_command_match_selected_native_record'] &= np.array_equal(a['initial_qpos27'],native['raw']['qpos27']) and np.array_equal(a['ctrl20'],native['raw']['ctrl20']) and np.array_equal(a['initial_COM_world3'],c['initial_COM_world_m'])
        contacts=a['contact_rows'];contact_total+=len(contacts);steps_total+=n
        pad=np.zeros((n,2));other=np.zeros(n);table=np.zeros(n);pen=np.zeros(n);selfpen=np.zeros(n);ncon=np.zeros(n,dtype=np.int64);nonpad_rows=[]
        for row in contacts:
            t,g1,g2=map(int,row[:3]);assert 0<=t<n and row[0]==t and row[1]==g1 and row[2]==g2
            ncon[t]+=1;fn=max(0.,float(row[18]));pen[t]=max(pen[t],-float(row[5]))
            if 71 in [g1,g2]:
                g=g2 if g1==71 else g1
                if g in [15,28]:pad[t,[15,28].index(g)]+=fn
                elif g==0:table[t]+=fn
                else:
                    other[t]+=fn
                    nonpad_rows.append({'pre_tick':t,'pre_nominal_time_s':t*dt,'hand_geom_id':g,'name':names.get(g,'UNKNOWN'),
                                        'positive_normal_N':fn,'raw_signed_normal_N':float(row[18]),'native_distance_m':float(row[5])})
            elif 0 not in [g1,g2]:selfpen[t]=max(selfpen[t],-float(row[5]))
        checks['all_contact_rows_and_PRE_aggregates_reconstructed'] &= (contacts.shape[1]==24 and np.array_equal(ncon,a['pre_ncon']) and np.array_equal(pad,a['pre_pad_normal_N']) and np.array_equal(other,a['pre_phone_nonpad_normal_N']) and np.array_equal(table,a['pre_table_normal_N']) and np.array_equal(pen,a['pre_penetration_m']) and np.array_equal(selfpen,a['pre_self_penetration_m']))
        qlimits=np.array(native['contact_design']['PD_raw']['joint_range20x2']);qvio=np.maximum(qlimits[:,0]-a['post_qpos'][:,:20],0)+np.maximum(a['post_qpos'][:,:20]-qlimits[:,1],0)
        drift=np.linalg.norm(a['post_COM']-a['initial_COM_world3'],axis=1)
        metrics={'final_COM_drift_m':float(drift[-1]),'final_angle_rad':float(a['post_angle'][-1]),
            'tail_linear_velocity_RMS_m_s':None,'tail_angular_velocity_RMS_rad_s':None,'tail_dual_pad_force_coverage':None,
            'max_native_penetration_m':float(pen.max()),'max_self_penetration_m':float(selfpen.max()),
            'max_phone_nonpad_normal_force_N':float(other.max()),'max_table_normal_force_N':float(table.max()),
            'max_joint_limit_excursion_rad':float(qvio.max()),'warning_count':sum(c['warnings'])}
        assert n*dt<gates['duration_s']-gates['tail_duration_s']
        expected_gates={'duration_complete':False}
        for k,v in metrics.items():expected_gates[k]=v is not None and (v>=gates[k] if k=='tail_dual_pad_force_coverage' else v<=gates[k])
        scalar_errors=[abs(metrics[k]-c['metrics'][k]) for k in metrics if metrics[k] is not None]
        max_scalar_metric_error=max(max_scalar_metric_error,max(scalar_errors))
        checks['reported_metrics_and_all_original_gates_reproduced'] &= all((metrics[k] is None and c['metrics'][k] is None) or (metrics[k] is not None and metrics[k]==c['metrics'][k]) for k in metrics) and expected_gates==c['gates'] and c['nominal_holding_dynamics_pass'] is False
        abort=(drift>gates['abort_COM_drift_m']) | (a['post_angle']>gates['abort_angle_rad'])
        checks['all4_loss_abort_no_missing_or_extra_ticks'] &= c['termination']=='ABORT_LOSS_OF_HOLD' and np.flatnonzero(abort).tolist()==[n-1] and c['actual_final_time_s']==float(a['post_time'][-1])
        checks['noaid_report_and_source_guards_preserved'] &= c['no_external_forces_equality_mocap_or_pose_reset'] is True and c['training_GPU'] is False and native['model_configuration']['neq']==native['model_configuration']['nmocap']==0 and not any(native['raw']['qfrc_applied26']) and not any(x for row in native['raw']['xfrc_applied'] for x in row)
        events={'thumb_first_PRE_Fn_le_0p1N':first_event(pad[:,0]<=.1,dt,'PRE',pad[:,0]),
                'index_first_PRE_Fn_le_0p1N':first_event(pad[:,1]<=.1,dt,'PRE',pad[:,1]),
                'first_PRE_nonpad_Fn_gt_0p1N':first_event(other>.1,dt,'PRE',other),
                'first_POST_COM_drift_gt_5mm':first_event(drift>.005,dt,'POST',drift),
                'first_POST_angle_gt_5deg':first_event(a['post_angle']>math.radians(5),dt,'POST',a['post_angle'])}
        for k,v in events.items():
            if v is not None:
                if v['phase']=='POST':v['saved_native_post_time_s']=float(a['post_time'][v['record_row_index']])
                if k=='first_PRE_nonpad_Fn_gt_0p1N':v['contributing_nonpad_contacts']=[q for q in nonpad_rows if q['pre_tick']==v['tick'] and q['positive_normal_N']>0]
                events_csv.append({'case':c['name'],'dt_s':dt,'event':k,**{q:v[q] for q in ['phase','tick','nominal_time_s','value']}})
        peak_tick=int(np.argmax(other));peak_contacts=[q for q in nonpad_rows if q['pre_tick']==peak_tick and q['positive_normal_N']>0]
        first_nonpad=min(nonpad_rows,key=lambda q:q['pre_tick']) if nonpad_rows else None
        byname.setdefault(c['name'],[]).append((c,a))
        cases.append({'name':c['name'],'dt_s':dt,'trace':c['trace'],'selected_native_record':screen['selected_record'],
            'steps':n,'termination':c['termination'],'nominal_abort_time_s':n*dt,'saved_native_final_time_s':c['actual_final_time_s'],
            'events':events,'first_generated_nonpad_contact':first_nonpad,
            'peak_aggregate_nonpad':{'pre_tick':peak_tick,'nominal_time_s':peak_tick*dt,'aggregate_positive_Fn_N':float(other[peak_tick]),'contacts':peak_contacts},
            'largest_individual_nonpad_contact':max(nonpad_rows,key=lambda q:q['positive_normal_N']) if nonpad_rows else None,
            'recomputed_metrics':metrics,'recomputed_gates':expected_gates,'original_holding_pass':False,
            'phase':'For record row i, all contact/energy values are solver PRE tick i; qpos/qvel/COM/angle are POST tick i+1.',
            'noaid_evidence_scope':'Saved initial raw zero applied forces + exact prepared q/control pins, producer per-step applied-force/constant-control guards and no-equality/mocap model checks. Full per-tick applied-force/ctrl vectors are not saved; do not claim an independent raw-byte audit of those unsaved arrays.'})
    pairs=[]
    for pair in report['dt_pair_consistency']:
        cases2=byname[pair['name']];(_,a),(_,b)=cases2
        com=float(np.linalg.norm(a['post_COM'][-1]-b['post_COM'][-1]));qa=a['post_qpos'][-1,23:27];qb=b['post_qpos'][-1,23:27]
        angle=float(2*np.arccos(np.clip(abs((qa/np.linalg.norm(qa))@(qb/np.linalg.norm(qb))),0,1)))
        checks['both_dt_pairs_incomplete_not_2s_qualified'] &= pair['comparison_status']=='INCOMPLETE_DURATION_NOT_2S_CONSISTENCY' and pair['gates']['both_duration_complete'] is False and pair['dt_pair_consistency_pass'] is False and com==pair['final_COM_difference_m'] and angle==pair['final_orientation_difference_rad']
        pairs.append({'name':pair['name'],'prefix_endpoint_COM_difference_m':com,'prefix_endpoint_angle_rad':angle,'both_duration_complete':False,'two_second_consistency_qualified':False,'scope':'Different early-abort endpoint times; proximity is descriptive, not complete2s or same-time convergence.'})
    checks['actual2959steps_and2581contact_reads_match'] = steps_total==2959==report['counts']['integration_steps'] and contact_total==2581==report['counts']['contact_force_reads'] and len(cases)==4
    with (OUT/'events.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case','dt_s','event','phase','tick','nominal_time_s','value']);w.writeheader();w.writerows(events_csv)
    md='# Source-mapping dynamic saved review\n\nAll four cases ABORT_LOSS_OF_HOLD. These are loss-of-hold prefixes, not stable holding durations. Neither dt pair completed2s; all original tail metrics remain null.\n\n|Mapping/dt ms|Steps / abort s|Thumb <=.1N PRE s|Index <=.1N PRE s|Nonpad >.1N PRE s|COM >5mm POST s|Angle >5deg POST s|\n|---|---|---|---|---|---|---|\n'
    def timeof(v):return 'not observed' if v is None else f"{v['nominal_time_s']:.7g} (tick{v['tick']})"
    for c in cases:
        e=c['events'];md+=f"|{c['name'].removeprefix('source_mapping_')}/{c['dt_s']*1000:g}|{c['steps']} / {c['nominal_abort_time_s']:.7g}|{timeof(e['thumb_first_PRE_Fn_le_0p1N'])}|{timeof(e['index_first_PRE_Fn_le_0p1N'])}|{timeof(e['first_PRE_nonpad_Fn_gt_0p1N'])}|{timeof(e['first_POST_COM_drift_gt_5mm'])}|{timeof(e['first_POST_angle_gt_5deg'])}|\n"
    md+='\nEach row i stores solverPRE force/energy at tick i and POST state at tick i+1; integer nominal times are distinct from accumulated native clocks. JSON identifies first/peak nonpad contacts via saved22-row geometry ID/name mappings. No model was loaded to infer names. Terminal/pins,2959steps,2581contact rows, PRE aggregation and all original case gates were checked. No missing tail/root/COM RMS was filled. Noaid relies on raw initial evidence and the preserved runtime/source guard; unsaved per-tick applied-force vectors are not independently reconstructed.\n'
    with (OUT/'report.md').open('x',encoding='utf8') as f:f.write(md)
    result={'schema':'hand2_source_mapping_dynamic_saved_review:1','status':'PASS_SAVED_TRACE_TIMING_AND_FAILED_HOLD_GATES' if all(checks.values()) else 'FAILED_SAVED_REVIEW_PRESERVED',
        'source':pin(__file__),'inputs':inputs,'checks':checks,'passed':sum(checks.values()),'total':len(checks),
        'case_reviews':cases,'dt_pair_reviews':pairs,'counts':{'unique_dynamic_traces':4,'audit_NPZ_reads':4,'prior_ABI_shape_only_NPZ_reads':4,'actual_original_steps':steps_total,'saved_contact_rows':contact_total,'reviewer_new_model_FK_LP_physics_steps_NN_training_GPU':0},
        'metric_scalar_max_abs_difference':max_scalar_metric_error,'event_csv':pin(OUT/'events.csv'),'markdown':pin(OUT/'report.md'),
        'holding_or_two_second_dt_consistency_qualified':False,'scope':'Saved native simulator diagnostic only, not stable holding duration, calibrated real contact/ego success or learned benefit. No source/model/result mutations and no physics or solver re-execution.'}
    save(OUT/'report.json',result)
    print(json.dumps({'report':pin(OUT/'report.json'),'checks':checks,'events':[{'case':c['name'],'dt':c['dt_s'],'events':c['events'],'peak_nonpad':c['peak_aggregate_nonpad']} for c in cases]}))

if __name__=='__main__':main()
