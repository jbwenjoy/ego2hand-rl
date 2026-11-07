"""Seven semantic boundary fixtures; no actual contact-regime classification."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MATH=ROOT/'src/full151_gradient/contact_regime_saved_independent_math_v1.py'
OUT=ROOT/'outputs/full151105_contact_regime_independent_math_preparation_v1'


def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False);sources={'math':pin(MATH),'checker':pin(Path(__file__))}
    for name,path in [('math',MATH),('checker',Path(__file__))]:
        ast.parse(path.read_bytes());(OUT/(name+'_source_before_execution.py')).write_bytes(path.read_bytes())
    (OUT/'predata_freeze.json').write_text(json.dumps({'source_pins':sources,'actual_full608_or_contact_longtable_read':False},indent=2))
    spec=importlib.util.spec_from_file_location('independent_saved_contact_regime',MATH);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    catalog=[]
    for i,(name,body,ct,ca) in enumerate([('terrain',1,1,1),('object/cube',2,1,2),('robot/right_palm_collision',9,1,1),
        ('robot/right_finger1_link1_col',10,0,0),('robot/other',11,1,1),('object/A_top',2,1,2),('robot/right_finger2_tip_sensor_col',12,0,1)]):
        catalog.append({'id':i,'name':name,'native_body_id':body,'contype':ct,'conaffinity':ca})
    sets=m.catalog_sets(7,2,catalog)
    ho=m.pair_classification([5,6],sets);hh=m.pair_classification([2,6],sets);hn=m.pair_classification([2,4],sets);on=m.pair_classification([1,0],sets)
    checks={'original_enabled_regex_and_nativeID_classification':sets['hand']=={2,6} and sets['object']=={1,5}
        and ho['category']=='hand_object' and hh['category']=='hand_self' and hn['category']=='other' and hn['hand_other'] and on['object_nonhand']}
    c={'geom':[1,2],'efc_address':7,'dist':-.004,'pos':[0.,0.,0.],'frame':np.eye(3),'force6':[2.,.1,-.2,.003,0.,0.]}
    raw={'geom':[1,2],'type':3,'worldid':0,'efc_address':[0,1,2,3,-1,-1]}
    force=m.inspect_contact(0,c,raw,4,sets)
    threshold=m.inspect_contact(1,{**c,'force6':[1e-6,0,0,0,0,0]},raw,4,sets)
    checks['constraint_bitmask3_and_padding_legal_strict_original_Fn_threshold']=force['force_evidence_eligible'] and force['secondary_positive_force'] and force['sensor_type_bit'] and not threshold['secondary_positive_force']
    sensor=m.inspect_contact(2,c,{**raw,'type':2},4,sets)
    noslots=m.inspect_contact(3,c,{**raw,'efc_address':[-1]*6},4,sets)
    badslot=m.inspect_contact(4,c,{**raw,'efc_address':[0,4,-1,-1,-1,-1]},4,sets)
    checks['sensoronly_or_rawinvalid_not_rescued_by_positive_host_address_and_Fn']=all(x['generated_candidate'] and not x['force_evidence_eligible'] and not x['secondary_positive_force'] for x in (sensor,noslots,badslot))
    deepzero=m.inspect_contact(5,{**c,'force6':[0]*6},raw,4,sets);summary=m.generated_summary([deepzero]);empty=m.generated_summary([])
    checks['generated_depth_not_filtered_by_zero_force_empty_not_clearance']=summary['hand_object']['generated_candidate_count']==1 and summary['hand_object']['observed_candidate_depth_m']==.004 and summary['hand_object']['secondary_positive_force_count']==0 and empty['hand_object']=={
       'generated_candidate_count':0,'minimum_generated_distance_m':None,'observed_candidate_depth_m':0.,'generated_candidate_absent':True,'secondary_positive_force_count':0,'global_clearance':'UNKNOWN'}
    positivegap=m.generated_summary([m.inspect_contact(6,{**c,'dist':.0001},raw,4,sets)])
    checks['positive_saved_distance_is_generated_not_absent_certificate']=positivegap['hand_object']['generated_candidate_count']==1 and positivegap['hand_object']['observed_candidate_depth_m']==0 and positivegap['hand_object']['global_clearance']=='UNKNOWN'
    um=np.zeros(20,dtype='f4');up=um.copy();up[0]=np.float32(-0.)
    tm=np.ones(58,dtype='f4');tp=tm.copy();tp[0]=np.nextafter(np.float32(1),np.float32(2));pm=np.zeros(58,dtype='f4');pp=pm.copy();pp[0]=np.float32(-0.)
    z=m.zero_descriptors(um,up,tm,tp,pm,pp,0)
    checks['zero_executed_delta_and_encodedflat_numeric_vs_bit_distinct']=z['zero_actual_coordinate_delta'] and z['zero_actual_all20_delta'] and not z['control_endpoints_all_bits_equal'] and not z['physical_encoded_numeric_zero58'][0] and z['model_encoded_numeric_zero58'][0] and not z['model_endpoint_bits_equal58'][0]
    rejected=0
    for call in [lambda:m.pair_classification([1,99],sets),lambda:m.inspect_contact(0,c,{**raw,'geom':[2,1]},4,sets),lambda:m.catalog_sets(True,2,catalog)]:
        try:call()
        except ValueError:rejected+=1
    checks['unknown_ID_order_drift_bool_nativecount_rejected_no_backend']=rejected==3 and not any(k in sys.modules for k in ('torch','mujoco','warp','scipy','dex_supermimic','mjlab'))
    checks={k:bool(v) for k,v in checks.items()}
    report={'schema':'full151105_contact_regime_independent_math_CPU_proof:1','status':'PASS_SYNTHETIC_SEMANTICS_NOT_ACTUAL_CONTACT_REGIME' if all(checks.values()) else 'FAILED_SYNTHETIC_SEMANTICS',
       'source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,'actual_full608_or_contact_longtable_reads':0,
       'new_world_force_COM_moment_calculations':0,'new_model_MJB_FK_WM_Env_physics_GPU':0,'synthetic_only':True}
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    ready={'source_pins':sources,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready,indent=2));print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2))
    return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
