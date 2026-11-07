"""Three ABI fixtures plus metadata; no actual608 classification or new models."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ADAPTER=ROOT/'scripts/audit_contact_regime608_saved_cpu_v2.py'
OUT=ROOT/'outputs/full151105_contact_regime608_adapter_preparation_v2'


def main():
    OUT.mkdir(exist_ok=False);spec=importlib.util.spec_from_file_location('contact_adapter',ADAPTER);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    pins={'auditor':a.pin(ADAPTER),'math':a.pin(a.MATH),'checker':a.pin(__file__)}
    for k,p in [('auditor',ADAPTER),('math',a.MATH),('checker',Path(__file__))]:ast.parse(p.read_bytes());(OUT/(k+'_source_before_execution.py')).write_bytes(p.read_bytes())
    a.dump(OUT/'predata_freeze.json',{'source_pins':pins,'actual_regime_output_read':False,'actual608_classifications':0})
    m=a.independent_math()
    row={'category':'hand_object','object_nonhand':False,'hand_other':False,'dist_m':-.004,'secondary_positive_force':False}
    s=a.summaries((x for x in [row]),m)
    checks={'all_six_categories_survive_generator_input_once':s['all']['generated_candidate_count']==s['hand_object']['generated_candidate_count']==1 and s['hand_object']['observed_candidate_depth_m']==.004 and s['other']['minimum_generated_distance_m'] is None}
    samples=[{'substep':i,'categories':s} for i in range(1,5)];p=a.profiles(samples)
    checks['all_four_PRE_profiles_are_repeated_sample_counts_not_unique_pairs']=p['hand_object']['generated_count_by_substep']==[1]*4 and p['hand_object']['observed_candidate_depth_by_substep_m']==[.004]*4 and p['other']['generated_count_by_substep']==[0]*4
    values=[{'identity':0,'nested':{'a':True,'signedzero':-0.},'minimum':None}]
    with (OUT/'tiny_table.csv').open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['nested','minimum','identity']);w.writeheader();w.writerow({'nested':json.dumps(values[0]['nested']),'minimum':'','identity':0})
    z=np.zeros(58,dtype='f4');zz=z.copy();zz[43]=np.float32(-0.);e=a.encoded(zz,z)
    checks['CSV_column_order_semantics_and_signedzero_raw_encoding_distinct']=a.CSV_matches(OUT/'tiny_table.csv',values) and e['R_all_numeric_zero'] and not e['R_all_bit_equal'] and not a.strict_json(-0.,0.)
    hc={'category':'hand_object','source_helper_active_by_original_formula':True,'force_evidence_eligible':False}
    oc={'category':'other','source_helper_active_by_original_formula':False}
    joined=a.join_original_helper_activity([hc,oc],[True])
    checks['finalABI_original_HO_activity_separate_from_forcevalid_and_snapshot_summary']=joined[0]['source_measured_hand_object_active'] is True and not joined[0]['force_evidence_eligible'] and 'source_measured_hand_object_active' not in joined[1] and a.summaries((x for x in [row]*19),m)['hand_object']['generated_candidate_count']==19
    proto=ROOT/'protocols/full151105_saved_contact_regime608_CPU_prereg_v1.json';meta=a.metadata(proto)
    checks['real_protocol_source_hash_JSON_default_only_parents_join']=len(meta['inputs'])==22 and len(meta['source_pins'])==12
    checks={k:bool(v) for k,v in checks.items()};report={'schema':'full151105_contact_regime608_saved_adapter_CPU_proof:1','source_pins':pins,'protocol':a.pin(proto),
       'passed':sum(checks.values()),'total':len(checks),'checks':checks,'actual608_contact_classifications':0,'actual_new_regime_output_reads':0,'new_models_FW_AD_MJB_FK_Env_physics_GPU':0}
    a.dump(OUT/'report.json',report);ready={'source_pins':pins,'cpu_report':a.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    a.dump(OUT/'READY.json',ready);print(json.dumps({'READY':a.pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
