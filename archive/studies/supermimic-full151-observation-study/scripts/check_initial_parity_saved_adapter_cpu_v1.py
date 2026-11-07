"""One closed saved-ABI case; no actual data, Torch, model or backend."""
import importlib.util,json,math,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'scripts/audit_boundary_initial_parity_saved_cpu_v1.py'
s=importlib.util.spec_from_file_location('initial_parity_saved_adapter',P);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)

def main():
    out=ROOT/'outputs/full151105_boundary_initial_parity_saved_adapter_preparation_v1';out.mkdir(exist_ok=False)
    sources={'auditor':a.pin(P),'checker':a.pin(__file__),'generic_saved_helper':a.pin(a.UTILITY),'independent_math':a.pin(a.MATH)}
    for k,v in sources.items():(out/(k+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    a.dump(out/'source_before_fixture.json',{'source_pins':sources,'actual_parity_arrays_read':False})
    z=np.zeros((152,2,6),dtype='f4');negative=z.copy();negative[17,1,4]=np.float32(-0.)
    x=np.ones((152,58),dtype='f4');shift=x.copy();shift[151,57]=np.nextafter(np.float32(1),np.float32(2))
    public,total=a.public_order(np.float32(1),np.float32(2**-24),np.float64(1),np.float64(.5),1.)
    reassociated=float(np.float64(1)+np.float64(2**-24)+np.float64(1))
    one=a.positive_zero(z) and not a.positive_zero(negative) and np.array_equal(z,negative) and not a.bits(z,negative)
    one &= not a.bits(x,shift) and public==2. and total==2.5 and public!=reassociated
    values=np.full((152,2),np.float32(math.log(2)),dtype='f4');captured=np.nextafter(np.float32(math.log(2)),np.float32(np.inf))
    reduction=a.fp32_mean_evidence(captured,values);one &= reduction['within_roundoff'] and not reduction['bitexact_kernel_replay']
    one &= not any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules)
    report={'schema':'full151105_initial_parity_saved_adapter_CPU_contract:1','checks':{'positive_zero_bitmask_ulp_mutation_F32_subtotal_and_reduction_roundoff':bool(one)},'passed':int(one),'total':1,'source_pins':sources,'actual_data_Torch_models_FW_AD_MJB_physics_GPU':0,'fixture':{'public_F32_then_F64':public,'all_F64_reassociated':reassociated,'reduction':reduction}}
    a.dump(out/'report.json',report)
    ready={'schema':'full151105_initial_parity_saved_adapter_READY:1','source_pins':sources,'cpu_report':a.pin(out/'report.json'),'cpu_checks':{'passed':int(one),'total':1},'actual_parity_arrays_read':False,'api':{'entry':str(P),'mode':'NumPy saved files only'}}
    a.dump(out/'READY.json',ready);print(json.dumps({'ready':a.pin(out/'READY.json'),'report':a.pin(out/'report.json'),'passed':int(one),'total':1},indent=2))
    return 0 if one else 2

if __name__=='__main__':raise SystemExit(main())
