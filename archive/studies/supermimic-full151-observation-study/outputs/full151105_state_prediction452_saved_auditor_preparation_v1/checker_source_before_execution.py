"""Small synthetic-only saved-adapter metadata and raw-byte checks."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
AUDITOR=ROOT/'scripts/audit_state_prediction452_saved_cpu_v1.py'
OUT=ROOT/'outputs/full151105_state_prediction452_saved_auditor_preparation_v1'


def pin(p):return {'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False);pins={'auditor':pin(AUDITOR),'checker':pin(Path(__file__))}
    for k,p in [('auditor',AUDITOR),('checker',Path(__file__))]:
        ast.parse(p.read_bytes());(OUT/(k+'_source_before_execution.py')).write_bytes(p.read_bytes())
    (OUT/'predata_freeze.json').write_text(json.dumps({'source_pins':pins,'actual_predictions_read':False},indent=2))
    spec=importlib.util.spec_from_file_location('independent452saved',AUDITOR);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    ready={'schema':'full151105_state_prediction452_CPU_READY:1'}
    proof={'schema':'full151105_state_prediction452_CPU_contract:1','status':'PASS_SYNTHETIC_GROUP_ERROR_AND_SINGLE_FORWARD_CONTRACT','passed':38,'total':38,'checks':{str(i):True for i in range(38)}}
    checks={'CPU38_metadata_positive_fixture':a.CPU38_valid(ready,proof)}
    bad=copy.deepcopy(proof);bad['passed']=True;checks['reject_bool_as_native_count']=not a.CPU38_valid(ready,bad)
    bad=copy.deepcopy(proof);bad['checks']['0']=False;checks['reject_one_false_sourceproof']=not a.CPU38_valid(ready,bad)
    rng={k:np.array([1,2,3],dtype='u1') for k in ('Python_pickle','NumPy_pickle','Torch_CPU')}
    checks['raw_serialized_RNG_equal']=a.raw_rng_equal(rng,rng)
    mutated={k:v.copy() for k,v in rng.items()};mutated['Torch_CPU'][0]=4;checks['raw_RNG_mutation_rejected']=not a.raw_rng_equal(rng,mutated)
    m=a.module(a.MATH,a.MATH_SHA);r=a.module(a.RULE,a.RULE_SHA)
    checks['frozen452_order_and_units_math']=m.validate_identity_order(m.expected_identities()) and set(m.BLOCKS)=={'q','qd','p','R','v','omega'}
    x=np.zeros((2,58),dtype='f4');y=x.copy();y[0,0]=np.nextafter(np.float32(0),np.float32(1))
    checks['original_full58_q_exact_rule_retained']=not r.old_rule(x,y,True)['all_passed']
    checks['no_native_backend_or_model_import']=not any(k in sys.modules for k in ('torch','warp','mujoco','mjlab','dex_supermimic','scipy'))
    checks={k:bool(v) for k,v in checks.items()};pins.update(math=pin(a.MATH),old_rule=pin(a.RULE))
    report={'schema':'full151105_state_prediction452_saved_auditor_preparation:1','source_pins':pins,'passed':sum(checks.values()),'total':len(checks),'checks':checks,
        'synthetic_only':True,'actual_prediction_reads':0,'new_Torch_loader_model_Reader_FK_MJB_GPU_calls':0}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));ready={'source_pins':pins,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready,indent=2));print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2))
    return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
