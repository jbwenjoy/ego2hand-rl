"""Single old152 table/index/CSV ABI fixture, no actual label arrays."""
import ast,csv,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'scripts/audit_boundary_impulse_dataset_saved_cpu_v1.py';OUT=ROOT/'outputs/full151105_boundary_impulse_saved_adapter_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);ast.parse(SRC.read_bytes());s=importlib.util.spec_from_file_location('boundary_saved_adapter',SRC);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    sources={'auditor':a.pin(SRC),'checker':a.pin(__file__),'independent_math':a.pin(a.MATH),'generic_IO':a.pin(a.UTIL)}
    for k,p in [('auditor',SRC),('checker',Path(__file__))]:(OUT/(k+'_before_execution.py')).write_bytes(p.read_bytes())
    a.dump(OUT/'source_freeze.json',{'source_pins':sources,'actual_label_values_read':False,'F64_implementation_comparison':{'rtol':a.U.F64_RTOL,'atol':a.U.F64_ATOL,'not_a_defect_zero_threshold':True}})
    m=a.module(a.MATH,a.MATH_SHA);rows=[{**a.plan_record(r),'boundary_HO_constraint_witness':False,'r_w_saved_rad_s':[1.,2.,3.]} for r in m.expected_restored_schedule()]
    p=OUT/'synthetic152.csv'
    with p.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    wrong=[dict(r) for r in rows];wrong[-1]['group_id']=450
    checks={'all152_table_indices_lists_bools_nonzero_defects_and_wrongjoin_reject':len(rows)==152 and rows[0]['physical_rows']==[600,601,602,603] and rows[-1]['physical_rows']==[1804,1805,1806,1807] and rows[-1]['terminal_observation']==150 and a.csv_matches(p,rows) and not a.csv_matches(p,wrong)}
    if any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules):raise ValueError('Unexpected backend')
    report={'schema':'full151105_boundary_impulse_saved_adapter_CPU_proof:1','source_pins':sources,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'actual_target_array_reads':0,'model_FW_AD_MJB_FK_Env_physics_GPU':0}
    a.dump(OUT/'report.json',report);ready={'source_pins':sources,'cpu_report':a.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},'api':{'entry':str(SRC)},'actual_saved_review_completed':False};a.dump(OUT/'READY.json',ready);print(json.dumps({'READY':a.pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
