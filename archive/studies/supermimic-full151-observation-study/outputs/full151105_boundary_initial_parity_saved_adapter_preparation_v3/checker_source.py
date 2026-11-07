"""One signed native difference fixture evaluating the actual saved guard AST."""
import ast,importlib.util,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'scripts/audit_boundary_initial_parity_saved_cpu_v3.py'
s=importlib.util.spec_from_file_location('parity_saved_v3',P);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)

def main():
    out=ROOT/'outputs/full151105_boundary_initial_parity_saved_adapter_preparation_v3';out.mkdir(exist_ok=False)
    sources={'auditor_v3':a.pin(P),'checker':a.pin(__file__),'preserved_v2':a.pin(ROOT/'scripts/audit_boundary_initial_parity_saved_cpu_v2.py'),'native_state':a.pin(Path(r'D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py'))}
    for k,v in sources.items():(out/(k+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    a.dump(out/'source_before_fixture.json',{'source_pins':sources,'actual_v3_arrays_read':False})
    calls=[v for v in ast.walk(ast.parse(P.read_text())) if isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='bits' and len(v.args)==2 and isinstance(v.args[1],ast.BinOp) and isinstance(v.args[1].op,ast.Sub)]
    a.require(len(calls)==1,'One exact production nonrotation bit guard')
    expression=compile(ast.fix_missing_locations(ast.Expression(calls[0])),'native_nonrotation_guard','eval')
    pred=np.asarray([[3.,-4.,2.]],dtype='f4');target=np.asarray([[8.,-7.,2.]],dtype='f4');D=np.asarray([[5.,-3.,0.]],dtype='f4')
    env={'bits':a.bits,'D':D,'pred':pred,'target':target,'lo':0,'hi':3,'slo':0,'shi':3}
    positive=bool(eval(expression,env));env['D']=-D;negative=not bool(eval(expression,env));one=positive and negative
    one &= not any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules)
    report={'schema':'full151105_initial_parity_saved_adapter_CPU_contract:3','checks':{'native_target_minus_pred_positive_and_negative_signed_capture':bool(one)},'passed':int(one),'total':1,'source_pins':sources,'actual_data_Torch_models_FW_AD_MJB_physics_GPU':0}
    a.dump(out/'report.json',report);a.dump(out/'READY.json',{'schema':'full151105_initial_parity_saved_adapter_READY:3','source_pins':sources,'cpu_report':a.pin(out/'report.json'),'cpu_checks':{'passed':int(one),'total':1},'actual_v3_arrays_read':False,'api':{'entry':str(P),'mode':'NumPy saved files only'}})
    print(json.dumps({'ready':a.pin(out/'READY.json'),'report':a.pin(out/'report.json'),'auditor':a.pin(P),'passed':int(one),'total':1},indent=2));return 0 if one else 2
if __name__=='__main__':raise SystemExit(main())
