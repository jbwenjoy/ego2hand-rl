"""Three source/schema/order checks; no real fresh predictions or model read."""
import ast
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'scripts/audit_WM_component_confirmation_saved_cpu_v1.py'
OUT=ROOT/'outputs/full151105_component_confirmation_saved_adapter_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);s=importlib.util.spec_from_file_location('confirmation_saved',SRC);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    pins={'auditor':a.pin(SRC),'math':a.pin(a.MATH),'native52_arithmetic_helper':a.pin(a.LOSS),'checker':a.pin(__file__)}
    for k,p in [('auditor',SRC),('checker',Path(__file__))]:ast.parse(p.read_bytes());(OUT/(k+'_source_before_execution.py')).write_bytes(p.read_bytes())
    a.dump(OUT/'predata_freeze.json',{'source_pins':pins,'actual_fresh_test_or_eval_arrays_read':False})
    rows,pairs=a.symbolic_order()
    checks={'all152_roles_and64_s_j_epsilon_pairs_explicit':len(rows)==152 and len(pairs)==64 and sum(r['type']=='baseline' for r in rows)==24
       and pairs[1]['minus_branch_id']==11 and pairs[2]['minus_branch_id']==5 and rows[-1]['path']=='A2' and rows[-1]['k']==148 and rows[-1]['sign']==1}
    fake=OUT/'synthetic_RNG';fake.mkdir();host={'python':[3,[123,456],None],'numpy_header':['MT19937',623,0,0.0],'CUDA_initialized':False}
    for label in ('before','after'):
        np.savez(fake/(label+'_RNG_arrays.npz'),numpy_MT=np.arange(624,dtype='u4'),torch_CPU=np.arange(32,dtype='u1'))
        a.dump(fake/(label+'_RNG_python_numpy.json'),host)
    helper=a.module(a.LOSS,a.LOSS_SHA);okay=a.raw_RNG_before_after(fake,helper)
    changed={**host,'numpy_header':['MT19937',624,0,0.0]};(fake/'after_RNG_python_numpy.json').unlink();a.dump(fake/'after_RNG_python_numpy.json',changed)
    checks['all3_RNG_groups_header_mutation_not_hidden_by_equal_vectors']=all(okay.values()) and not all(a.raw_RNG_before_after(fake,helper).values())
    # Only JSON/hash metadata, never sealed branch/snapshot values or NPZ.
    proto=ROOT/'protocols/full151105_WM_component_confirmation_CPU_run_v1.json';metadata=a.metadata(proto)
    checks['actual_released_protocol_and_final29_hash_JSON_only']=len(metadata['inputs'])==16 and a.CORE_SHA=='f75302fa70a894300ceb3a71492b9cffd973244990f13be26cee5b7e5e112659'
    checks={k:bool(v) for k,v in checks.items()};report={'schema':'full151105_component_confirmation_saved_adapter_CPU_proof:1','source_pins':pins,
       'passed':sum(checks.values()),'total':len(checks),'checks':checks,'actual_fresh_or_predicted_arrays_read':0,'model_checkpoint_loads_Log_FW_AD_physics_GPU':0}
    a.dump(OUT/'report.json',report);ready={'source_pins':pins,'cpu_report':a.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},'actual_saved_audit_authorized':False}
    a.dump(OUT/'READY.json',ready);print(json.dumps({'READY':a.pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
