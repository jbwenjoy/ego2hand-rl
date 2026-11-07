"""Fixed-grid v4 saved-only assessment; no physics/model imports or calls.

Reuse frozen independent case arithmetic0091 and original unchanged G7 bounds.
Every original case and coarse/fine comparison is retained, including failures.
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
spec=importlib.util.spec_from_file_location('independent_saved_math_v2',PURE)
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
A=V.A
PROTO=BASE/'protocols/phone_convex_fixedgrid_v4.json'
PROTO_SHA='7a84cdda18952e24e45cfd96ba0bdcb846ff656794512112911e64602b1ed14f'
RUNNER_SHA='0e78a7c0b149ff121ed18368bdd1c49f7c407522263581a047241bd32db9a9b0'
TESTS=['drop','table_settling','force_push_release']
DTS=[.0003125,.00015625]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    p=A.read(PROTO);assert A.pin(PROTO)['sha256']==PROTO_SHA
    for value in p['source_pins'].values():A.check_pin(value)
    p2=A.read(p['source_pins']['v2_protocol']['path']);grid=p['fixed_grid']
    assert grid['timesteps_s']==DTS and grid['timeconst_s']==.0025 and grid['test_names']==TESTS
    fixture=A.focused_closed_form()
    rp=args.run_dir/'result.json';r=A.read(rp)
    assert r['schema']=='phone_convex_fixedgrid_CPU:4' and r['mode']=='fixedgrid'
    assert r['timeconst_s']==.0025 and r['timesteps_s']==DTS and A.check_pin(r['protocol'])['sha256']==PROTO_SHA
    bp=A.check_pin(r['binding']);b=A.read(bp['path']);assert pathlib.Path(bp['path']).samefile(args.run_dir/'execution_binding.json')
    for key in ['protocol','runner','common','capture_source','legacy_helpers','asset','geometry_report','collision_OBJ','visual_OBJ']:A.check_pin(b[key])
    assert b['runner']['sha256']==RUNNER_SHA
    assert b['common']['sha256']==p['source_pins']['common']['sha256'] and b['capture_source']['sha256']==p['source_pins']['CPU_run_case_source']['sha256']
    assert b['asset']['sha256']==p['source_pins']['convex_asset']['sha256'] and b['geometry_report']['sha256']==p['source_pins']['finite_sample_geometry_report']['sha256']
    assert b['timeconst_s']==.0025 and b['integration_steps_before_binding']==0
    masses,mus=grid['cartesian_grid']['mass_kg'],grid['cartesian_grid']['sliding_friction']
    assert masses==[.1,.2,.3] and mus==[.3,.6,.9]
    mm=[(.2,.6)]+[x for x in itertools.product(masses,mus) if x!=(.2,.6)]
    params=[(mass,mu,dt,test) for mass,mu in mm for dt in DTS for test in TESTS]
    assert b['parameters']==[list(x) for x in params] and len(r['cases'])==r['case_count']==grid['case_count']==54
    rows=[];data={}
    for c,key in zip(r['cases'],params):
        assert (c['mass_kg'],c['sliding_friction'],c['dt_s'],c['test'])==key and c['timeconst_s']==.0025
        assert c['case']==f'{key[3]}_mass{key[0]:g}_mu{key[1]:g}_tau0.0025_dt{key[2]:g}'
        row,d=V.inspect_case(c,args.run_dir);rows.append(row);data[key]=d
    assert sum(x['steps'] for x in data.values())==r['integration_steps']==grid['integration_steps']==777600
    limits=p2['engineering_gates']['G7_timestep_pair'];pair_results=[]
    for mass,mu in mm:
        for test in TESTS:
            a,z=data[(mass,mu,DTS[0],test)],data[(mass,mu,DTS[1],test)]
            assert len(z['com'])==2*len(a['com'])-1
            metrics={'final_COM_position_difference_m':float(np.linalg.norm(a['com'][-1]-z['com'][-1])),
                'final_orientation_difference_deg':float(np.degrees(A.quat_angle(a['q'][-1,3:],z['q'][-1,3:]))),
                'final_COM_velocity_vector_difference_m_s':float(np.linalg.norm(a['v'][-1]-z['v'][-1])),
                'final_angular_velocity_vector_difference_rad_s':float(np.linalg.norm(a['w'][-1]-z['w'][-1])),
                'shared_tick_COM_trajectory_RMS_difference_m':A.rms_vec(a['com']-z['com'][::2]),
                'first_contact_time_difference_s':abs(a['first']-z['first']) if a['first'] is not None and z['first'] is not None else None}
            checks={k:(v is not None and v<=limits[k]) for k,v in metrics.items()}
            state=A.status(all(checks.values()))
            pr={'mass_kg':mass,'sliding_friction':mu,'test':test,'timesteps_s':DTS,'metrics':metrics,'checks':checks,'status':state}
            pair_results.append(pr)
            for row in rows:
                if (row['mass_kg'],row['sliding_friction'],row['test'])==(mass,mu,test):row['gates']['G7_timestep_pair']=state
    geom=A.read(b['geometry_report']['path']);geo_checks={
       'outside':geom['maximum_original_vertex_outside_hull_m']<=.0005,
       'p95':geom['full_sampled_hull_to_scan']['p95_absolute_m']<=.002,
       'max':geom['full_sampled_hull_to_scan']['maximum_m']<=.004}
    pins={x['case']:x['trajectory_pin'] for x in rows};assert len(pins)==54
    CPU_ok=all(geo_checks.values()) and all(all(v=='PASS' for k,v in row['gates'].items() if not k.startswith('G8')) for row in rows)
    gate_counts={k:dict(collections.Counter(x['gates'][k] for x in rows)) for k in rows[0]['gates']}
    report={'schema':'phone_convex_fixedgrid_CPU_assessment:4','status':'COMPLETED_SAVED_FIXED_GRID_ASSESSMENT_NOT_REAL_PARAMETER_CALIBRATION',
      'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'auditor_pin':A.pin(__file__),
      'independent_case_math_pin':A.pin(PURE),'focused_fixture':fixture,
      'assessment_complete':True,'audit_integrity_qualified':True,'CPU_case_count':54,
      'CPU_bench_qualified':CPU_ok,'producer_result_pin':A.pin(rp),'protocol_pin':A.pin(PROTO),
      'execution_binding_pin':bp,'timeconst_s':.0025,'timesteps_s':DTS,'trajectory_pins':pins,
      'integration_steps':777600,'saved_state_rows':sum(x['steps']+1 for x in data.values()),
      'case_results':rows,'timestep_comparisons':pair_results,'gate_case_counts':gate_counts,
      'geometry_qualified':all(geo_checks.values()),'geometry_checks':geo_checks,'G8_Warp_status':'NOT_RUN','stage4_qualified':False,
      'real_parameter_calibrated':False,'hand_task_qualified':False,'ego_retargeting_qualified':False,
      'scope':'All54 raw cases/all27 timestep pairs; no case/threshold selection. Independent COM/energy/contact/force arithmetic, source-backed compiled options/native warnings, no model or physics reload. GPU confirmation is permitted by protocol only if CPU_bench_qualified=True; this report never establishes GPU parity itself.'}
    A.write(args.output/'all54_case_results.json',rows);A.write(args.output/'all27_timestep_pairs.json',pair_results)
    A.write(args.output/'report.json',report)
    print(json.dumps({'report':A.pin(args.output/'report.json'),'CPU_bench_qualified':CPU_ok,
       'gate_case_counts':gate_counts,'failed_pairs':[x for x in pair_results if x['status']=='FAIL'],
       'max_penetration_m':max(x['metrics']['max_penetration_m'] for x in rows)},indent=2,allow_nan=False))


if __name__=='__main__':main()
