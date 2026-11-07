"""One scale-ABI correction fixture: source-bound capture, no kernel surrogate."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'scripts/audit_boundary_initial_parity_saved_cpu_v2.py'
s=importlib.util.spec_from_file_location('saved_parity_v2',P);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)

def main():
    out=ROOT/'outputs/full151105_boundary_initial_parity_saved_adapter_preparation_v2';out.mkdir(exist_ok=False)
    sources={'auditor_v2':a.pin(P),'checker':a.pin(__file__),'original_preserved_v1':a.pin(ROOT/'scripts/audit_boundary_initial_parity_saved_cpu_v1.py'),'generic_saved_helper':a.pin(a.UTILITY),'independent_math':a.pin(a.MATH)}
    for k,v in sources.items():(out/(k+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    a.dump(out/'source_before_fixture.json',{'source_pins':sources,'actual_v2_arrays_read':False})
    native=np.ones(52,dtype='f4');native[46]=np.nextafter(np.float32(1),np.float32(2));numpy_surrogate=np.ones(52,dtype='f4')
    result=a.captured_effective52(native,native.copy(),native.copy());rejected=False
    try:a.captured_effective52(native,numpy_surrogate,native.copy())
    except ValueError:rejected=True
    one=a.bits(result,native) and not a.bits(result,numpy_surrogate) and rejected
    one &= not any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules)
    report={'schema':'full151105_initial_parity_saved_adapter_CPU_contract:2','checks':{'captured_scale_exact_to_both_native_evidence_arrays_not_NumPy_sqrt':bool(one)},'passed':int(one),'total':1,'source_pins':sources,'actual_data_Torch_models_FW_AD_MJB_physics_GPU':0,'v1_proof_1_unchanged':True}
    a.dump(out/'report.json',report);a.dump(out/'READY.json',{'schema':'full151105_initial_parity_saved_adapter_READY:2','source_pins':sources,'cpu_report':a.pin(out/'report.json'),'cpu_checks':{'passed':int(one),'total':1},'actual_v2_arrays_read':False,'api':{'entry':str(P),'mode':'NumPy saved files only'}})
    print(json.dumps({'ready':a.pin(out/'READY.json'),'report':a.pin(out/'report.json'),'auditor':a.pin(P),'passed':int(one),'total':1},indent=2));return 0 if one else 2
if __name__=='__main__':raise SystemExit(main())
