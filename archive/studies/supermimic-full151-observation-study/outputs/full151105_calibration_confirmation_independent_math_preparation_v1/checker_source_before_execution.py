"""Two closed-form confirmation/units fixtures; no sealed values or models."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/full151_gradient/calibration_confirmation_independent_math_v1.py'
OUT=ROOT/'outputs/full151105_calibration_confirmation_independent_math_preparation_v1'


def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False);sources={'math':pin(SRC),'checker':pin(Path(__file__))}
    for k,p in [('math',SRC),('checker',Path(__file__))]:ast.parse(p.read_bytes());(OUT/(k+'_source_before_execution.py')).write_bytes(p.read_bytes())
    (OUT/'predata_freeze.json').write_text(json.dumps({'source_pins':sources,'sealed_test_values_read':False},indent=2))
    spec=importlib.util.spec_from_file_location('confirmation_math',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    point=np.zeros((152,52),dtype='f4');model=point.copy();target=point.copy();s=np.full(52,2,dtype='f4');mi=np.arange(0,128,2,dtype='i8');pl=mi+1
    model[pl]=4.;point[:,43:46]=3.;p=np.zeros((152,58),dtype='f4');t=p.copy();p[:,43:52]=3.
    result=m.all_confirmation_arithmetic(point,model,target,s,mi,pl,np.ones(64,dtype='f8'),p,t)
    checks={'all64_closedform_MSE4_and_Log3rad_vs_R9dimensionless_separate':result['all64_normalized_pair_MSE_F64']==4 and result['arrays']['pair_normalized_MSE64_F64'].shape==(64,) and result['native52_point_coordinates']['blocks']['R_commonpre_Log3']['units']=='rad' and result['raw58_state_error_original_units']['blocks']['R_raw9']['units']=='dimensionless' and result['raw58_state_error_original_units']['blocks']['R_raw9']['Frobenius_RMS']==9.}
    zero=np.ones(64,dtype='f8');zero[42]=0
    try:m.all_confirmation_arithmetic(point,model,target,s,mi,pl,zero,p,t);accepted=True
    except ValueError:accepted=False
    checks['strict_confirmation_no_equal_pair_tradeoff_or_zero_denominator_filter']=not m.confirmation_rule(1.,1.,2.,2.)['local_component_confirmation'] and not m.confirmation_rule(1.,1.1,2.,1.)['local_component_confirmation'] and m.confirmation_rule(1.,1.,2.,1.)['local_component_confirmation'] and not accepted and not any(k in sys.modules for k in ('torch','mujoco','warp','dex_supermimic','mjlab'))
    checks={k:bool(v) for k,v in checks.items()};report={'schema':'full151105_calibration_confirmation_independent_math_CPU_proof:1','source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,'actual_sealed_test_values_or_model_reads':0,'new_FW_AD_physics_GPU':0}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));ready={'source_pins':sources,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready,indent=2));print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
