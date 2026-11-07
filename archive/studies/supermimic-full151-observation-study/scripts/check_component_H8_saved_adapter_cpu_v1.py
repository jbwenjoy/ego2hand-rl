"""One synthetic saved-table ABI/order case; no actual H8 or model values."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'scripts/audit_component_H8_donor_saved_cpu_v1.py'
OUT=ROOT/'outputs/full151105_component_H8_saved_adapter_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);ast.parse(SRC.read_bytes())
    s=importlib.util.spec_from_file_location('H8_saved_adapter',SRC);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    sources={'auditor':m.pin(SRC),'checker':m.pin(__file__),'math':m.pin(m.MATH_PATH),'generic_saved_helper':m.pin(m.UTIL_PATH)}
    for k,p in [('auditor',SRC),('checker',Path(__file__))]:(OUT/(k+'_before_execution.py')).write_bytes(p.read_bytes())
    m.dump(OUT/'source_freeze.json',{'source_pins':sources,'actual_H8_arrays_model_or_statistics_read':False})
    math=m.module(m.MATH_PATH,m.MATH_SHA);p=np.zeros((286,8,58),dtype='f4');t=p.copy();p[...,40:43]=[3,4,0]
    stats=math.raw58_error_statistics(p,t,np.repeat(np.arange(2,dtype='i8'),143))['statistics']
    rows=[]
    for role in m.ROLES:rows.extend(m.producer_style_statistics(stats,role))
    table=OUT/'synthetic_horizons_units.csv'
    with table.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    wrong=rows.copy();wrong[0],wrong[144]=wrong[144],wrong[0]
    checks={'all3x8x3x6_tabletypes_roles_horizons_units_emptyfields_and_wrongorder_guard':len(rows)==432 and rows[0]['model_role']==m.ROLES[0] and rows[-1]['model_role']==m.ROLES[2] and rows[-1]['horizon']==8 and rows[-1]['path']=='A2' and rows[-1]['window_count']==143 and rows[-1]['block']=='omega' and all(r['vector_or_Frobenius_RMS']==5. for r in rows if r['block']=='p') and m.UTIL.csv_matches(table,rows) and not m.UTIL.csv_matches(table,wrong)}
    checks={k:bool(v) for k,v in checks.items()}
    if any(k.split('.')[0] in ('torch','warp','mujoco','mjlab','dex_supermimic') for k in sys.modules):raise ValueError('Unexpected model/backend imported')
    report={'schema':'full151105_component_H8_saved_adapter_CPU_proof:1','source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,
      'actual_H8_values_or_model_reads':0,'model_FW_AD_Env_physics_GPU':0,'independent_math_fixture_repeated':False}
    m.dump(OUT/'report.json',report);ready={'source_pins':sources,'cpu_report':m.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},'api':{'entry':str(SRC)},'actual_saved_audit_completed':False}
    m.dump(OUT/'READY.json',ready);print(json.dumps({'READY':m.pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
