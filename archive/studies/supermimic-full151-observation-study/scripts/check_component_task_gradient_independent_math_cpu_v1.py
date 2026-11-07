"""Three closed-form scalar/projection boundaries, no actual score values."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/full151_gradient/component_task_gradient_independent_math_v1.py'
OUT=ROOT/'outputs/full151105_component_task_gradient_independent_math_preparation_v1'


def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False);sources={'math':pin(SRC),'checker':pin(Path(__file__))}
    for k,p in [('math',SRC),('checker',Path(__file__))]:ast.parse(p.read_bytes());(OUT/(k+'_source_before_execution.py')).write_bytes(p.read_bytes())
    (OUT/'source_freeze.json').write_text(json.dumps({'source_pins':sources,'actual_physical_taskscore_or_model_values_read':False},indent=2))
    spec=importlib.util.spec_from_file_location('task_gradient_numpy',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    center=np.zeros(20,dtype='f4');minus=np.zeros((2,20),dtype='f4');plus=minus.copy();minus[:,0]=[-2**-9,-2**-10];plus[:,0]=[2**-10,2**-11]
    g=np.zeros(20,dtype='f4');g[0]=4;g[19]=7
    # Binary-exact bounded linear toy score, with asymmetric clipped intervals.
    rm=np.float32(2**-5)+np.float32(4)*minus[:,0];rp=np.float32(2**-5)+np.float32(4)*plus[:,0]
    v=m.projection_conventions(g,center,minus,plus,rm,rp,0)
    checks={'clipped_asymmetric_linear_toy_actual4_legacy_effective_not_nominal4':np.array_equal(v['physical_actual_coordinate_secant'],[4.,4.]) and np.array_equal(v['model_actual_coordinate_projection'],[4.,4.]) and np.array_equal(v['physical_legacy_nominal_slope'],v['model_legacy_effective_projection']) and not np.array_equal(v['model_legacy_effective_projection'],[4.,4.]) and (v['center_offset_rad'][:,0]!=0).all()}
    scores0=np.zeros(2,dtype='f4');scores1=np.full(2,.004,dtype='f4');b=np.zeros(3,dtype='f4');d=np.float32(0)
    stable=m.physical_classification(scores0,scores1,b,d);unstable_scores=scores1.copy();unstable_scores[1]=np.nextafter(unstable_scores[1],np.float32(1))
    unstable=m.physical_classification(scores0,unstable_scores,b,d)
    atfloor=m.physical_classification(scores0,np.full(2,1.25,dtype='f4'),np.array([0,.125,.125],dtype='f4'),np.float32(0))
    checks['shared_old_physical_floor_strict_signal_relative_half_inclusive']=stable['stable_nonflat'] and stable['relative_epsilon_disagreement']==.5 and not unstable['stable_nonflat'] and atfloor['signal_floor']==1.25 and not atfloor['signal_above_floor'].any() and atfloor['label']=='FLAT_OR_BELOW_NOISE'
    delta=m.projection_conventions(g,center,minus,plus,np.full(2,1e-8,dtype='f4'),np.ones(2,dtype='f4'),0)['physical_delta_score_F64']
    zm=minus.copy();zp=plus.copy();zp[1]=zm[1]
    try:m.projection_conventions(g,center,zm,zp,rm,rp,0);zeroaccepted=True
    except ValueError:zeroaccepted=False
    checks['physical_scores_castF64_before_subtract_zero_actualden_whole_stop']=delta[0]!=float(np.float32(np.float32(1)-np.float32(1e-8))) and not zeroaccepted and not any(k in sys.modules for k in ('torch','mujoco','warp','dex_supermimic','mjlab'))
    checks={k:bool(v) for k,v in checks.items()};report={'schema':'full151105_component_task_gradient_independent_math_CPU_proof:1','source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,
      'actual_taskscore_or_model_data_reads':0,'new_model_FW_AD_Env_physics_GPU':0,'scope':'Closed-form arithmetic only; not an actual physical/model derivative qualification.'}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));ready={'source_pins':sources,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready,indent=2));print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
