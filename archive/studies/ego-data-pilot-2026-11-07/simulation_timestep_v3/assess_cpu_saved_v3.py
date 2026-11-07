"""Independent saved-only v3 adjacent-timestep screen/confirmation.

Reuse frozen independent saved arithmetic, never import a physics runner. Full
case and pair coverage precedes the fixed largest-coarse selection rule.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import itertools
import json
import pathlib

import numpy as np

BASE=pathlib.Path(__file__).resolve().parents[1]
PURE=BASE/'simulation_convex_v2/assess_cpu_saved_v2.py'
assert hashlib.sha256(PURE.read_bytes()).hexdigest()=='0091b055376d650139c3cca22d5a798443dd99ba96b355220d9b35dcd14b7dfb'
spec=importlib.util.spec_from_file_location('independent_v2_math',PURE)
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
A=V.A
PROTO=BASE/'protocols/phone_convex_timestep_v3.json'
PROTO_SHA='177cc53fb989bee15f923920487856ef07c7778da2c1ac105d6898c7357edac3'
RUNNER_SHA='30d56ba08b853a04bb4b7c72214347d6246664e6177a6064c8dc9e3766cb1436'
LADDER=[.000625,.0003125,.00015625]
PAIRS=[[.000625,.0003125],[.0003125,.00015625]]
TESTS=['drop','table_settling','force_push_release']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    p=A.read(PROTO);assert A.pin(PROTO)['sha256']==PROTO_SHA
    for value in p['source_pins'].values():A.check_pin(value)
    p2=A.read(p['source_pins']['v2_protocol']['path']);A.check_pin(p['source_pins']['v2_protocol'])
    assert p['screening']['timesteps_s']==LADDER and p['screening']['adjacent_pairs_s']==PAIRS
    fixture=A.focused_closed_form()
    rp=args.run_dir/'result.json';r=A.read(rp);mode=r['mode'];assert mode in ['screen','confirm']
    assert r['schema']==('phone_convex_timestep_CPU_screening:3' if mode=='screen' else 'phone_convex_timestep_CPU_confirmation:3')
    assert A.check_pin(r['protocol'])['sha256']==PROTO_SHA
    bp=A.check_pin(r['binding']);b=A.read(bp['path']);assert pathlib.Path(bp['path']).samefile(args.run_dir/'execution_binding.json')
    for k in ['protocol','runner','common','capture_source','legacy_helpers','asset','geometry_report','collision_OBJ','visual_OBJ']:A.check_pin(b[k])
    assert b['runner']['sha256']==RUNNER_SHA and b['common']['sha256']==p['source_pins']['common']['sha256']
    assert b['capture_source']['sha256']==p['source_pins']['CPU_run_case_source']['sha256']
    assert b['asset']['sha256']==p['source_pins']['convex_asset']['sha256']
    assert b['geometry_report']['sha256']==p['source_pins']['convex_geometry_report']['sha256']
    assert b['timeconst_s']==.0025 and b['integration_steps_before_binding']==0
    A.check_pin(p['source_pins']['models_info'])
    selection_pin=None
    if mode=='screen':
        params=[(.2,.6,dt,t) for dt in LADDER for t in TESTS];stage_pairs=PAIRS
        expected_steps=100800;chosen=None
    else:
        selection_pin=A.check_pin(b['selection']);s=A.read(selection_pin['path']);chosen=s['selected_timesteps_s']
        assert s['schema']=='phone_convex_timestep_selection:3' and s['screening_complete'] and s['selection_complete'] and s['audit_integrity_qualified']
        assert chosen in PAIRS and chosen==s['eligible_timesteps_s'][0] and s['selected_timeconst_s']==.0025
        assert s['protocol_pin']['sha256']==PROTO_SHA
        mm=[(.2,.6)]+[x for x in itertools.product([.1,.2,.3],[.3,.6,.9]) if x!=(.2,.6)]
        params=[(m,mu,dt,t) for m,mu in mm for dt in chosen for t in TESTS];stage_pairs=[chosen]
        expected_steps=9*3*sum(round(3/dt) for dt in chosen)
    assert b['parameters']==[list(x) for x in params] and len(r['cases'])==r['case_count']==len(params)
    rows=[];traces={}
    for c,key in zip(r['cases'],params):
        assert (c['mass_kg'],c['sliding_friction'],c['dt_s'],c['test'])==key and c['timeconst_s']==.0025
        assert c['case']==f'{key[3]}_mass{key[0]:g}_mu{key[1]:g}_tau0.0025_dt{key[2]:g}'
        row,d=V.inspect_case(c,args.run_dir);rows.append(row);traces[key]=d
    assert sum(x['steps'] for x in traces.values())==r['integration_steps']==expected_steps
    geom=A.read(b['geometry_report']['path']);geomchecks={
       'outside':geom['maximum_original_vertex_outside_hull_m']<=.0005,
       'p95':geom['full_sampled_hull_to_scan']['p95_absolute_m']<=.002,
       'max':geom['full_sampled_hull_to_scan']['maximum_m']<=.004}
    geometry_ok=all(geomchecks.values());limits=p2['engineering_gates']['G7_timestep_pair']
    pair_results=[];candidate_results=[]
    for coarse,fine in stage_pairs:
        relevant=[x for x in rows if x['dt_s'] in [coarse,fine]];these=[]
        for m,mu in dict.fromkeys((x[0],x[1]) for x in params):
            for test in TESTS:
                a,z=traces[(m,mu,coarse,test)],traces[(m,mu,fine,test)]
                assert len(z['com'])==2*len(a['com'])-1
                metrics={'final_COM_position_difference_m':float(np.linalg.norm(a['com'][-1]-z['com'][-1])),
                   'final_orientation_difference_deg':float(np.degrees(A.quat_angle(a['q'][-1,3:],z['q'][-1,3:]))),
                   'final_COM_velocity_vector_difference_m_s':float(np.linalg.norm(a['v'][-1]-z['v'][-1])),
                   'final_angular_velocity_vector_difference_rad_s':float(np.linalg.norm(a['w'][-1]-z['w'][-1])),
                   'shared_tick_COM_trajectory_RMS_difference_m':A.rms_vec(a['com']-z['com'][::2]),
                   'first_contact_time_difference_s':abs(a['first']-z['first']) if a['first'] is not None and z['first'] is not None else None}
                checks={k:(v is not None and v<=limits[k]) for k,v in metrics.items()}
                pid=f'{test}_mass{m:g}_mu{mu:g}_dt{coarse:g}_vs{fine:g}'
                pr={'pair_id':pid,'mass_kg':m,'sliding_friction':mu,'test':test,'timesteps_s':[coarse,fine],
                    'metrics':metrics,'checks':checks,'status':A.status(all(checks.values()))}
                pair_results.append(pr);these.append(pr)
        case_pass=all(all(v=='PASS' for k,v in row['gates'].items() if not k.startswith('G8')) for row in relevant)
        eligible=geometry_ok and case_pass and all(x['status']=='PASS' for x in these)
        candidate_results.append({'timesteps_s':[coarse,fine],'case_ids':[x['case'] for x in relevant],
          'G2_to_G6_case_gate_results':{x['case']:{k:v for k,v in x['gates'].items() if k not in ['G1_identity_and_coverage','G8_CPU_Warp_parity']} for x in relevant},
          'timestep_comparisons':these,'geometry_qualified':geometry_ok,'eligible':eligible})
    pins={x['case']:x['trajectory_pin'] for x in rows};assert len(pins)==len(params)
    case_counts={k:dict(collections.Counter(x['gates'][k] for x in rows)) for k in rows[0]['gates']}
    complete_cpu=geometry_ok and all(x['eligible'] for x in candidate_results)
    report={'schema':('phone_convex_timestep_CPU_screening_assessment:3' if mode=='screen' else 'phone_convex_timestep_CPU_confirmation_assessment:3'),
       'status':'COMPLETED_SAVED_TIMESTEP_ASSESSMENT_NOT_REAL_PARAMETER_CALIBRATION','created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'auditor_pin':A.pin(__file__),'independent_v2_math_pin':A.pin(PURE),'focused_fixture':fixture,
       'assessment_complete':True,'audit_integrity_qualified':True,'confirmation_complete':mode=='confirm',
       'producer_result_pin':A.pin(rp),'protocol_pin':A.pin(PROTO),'execution_binding_pin':bp,'selection_receipt_pin':selection_pin,
       'CPU_case_count':len(rows),'integration_steps':expected_steps,'selected_timeconst_s':.0025,'selected_timesteps_s':chosen,
       'trajectory_pins':pins,'case_results':rows,'case_gate_counts':case_counts,'timestep_comparisons':pair_results,
       'per_adjacent_pair_gate_results':candidate_results,'CPU_bench_qualified':complete_cpu,
       'geometry_qualified':geometry_ok,'geometry_checks':geomchecks,'G8_Warp_status':'NOT_RUN','stage3_qualified':False,
       'real_parameter_calibrated':False,'hand_task_qualified':False,'ego_retargeting_qualified':False,
       'scope':'Same pinned independent v2 saved arithmetic, unchanged physical bounds. qacc/native warnings/compiled options are saved runtime captures, not a new API interceptor. No physics/model calls. Shared middle-dt cases reused as dependent comparison inputs, not independent samples.'}
    A.write(args.output/'all_case_results.json',rows);A.write(args.output/'all_timestep_pairs.json',pair_results)
    if mode=='screen':
        eligible=[x['timesteps_s'] for x in candidate_results if x['eligible']]
        selected=eligible[0] if eligible else None
        sel={'schema':'phone_convex_timestep_selection:3','selected_timeconst_s':.0025,'selected_timesteps_s':selected,
          'eligible_timesteps_s':eligible,'screening_complete':True,'selection_complete':True,'audit_integrity_qualified':True,
          'protocol_pin':A.pin(PROTO),'screening_result_pin':A.pin(rp),'trajectory_pins':pins,
          'per_adjacent_pair_gate_results':candidate_results,'auditor_pin':A.pin(__file__),
          'rule':'First eligible adjacent pair in descending coarse-dt order; all9complete; no fallback after downstream failure.',
          'real_parameter_calibrated':False}
        A.write(args.output/'selection.json',sel);report['selection_output_pin']=A.pin(args.output/'selection.json');report['selected_timesteps_s']=selected
    A.write(args.output/'report.json',report)
    print(json.dumps({'report':A.pin(args.output/'report.json'),'selection':report.get('selection_output_pin'),
       'selected_timesteps_s':report['selected_timesteps_s'],'case_gate_counts':case_counts,
       'candidates':[{'pair':x['timesteps_s'],'eligible':x['eligible'],'failed_pairs':[z for z in x['timestep_comparisons'] if z['status']=='FAIL']} for x in candidate_results]},indent=2,allow_nan=False))


if __name__=='__main__':main()
