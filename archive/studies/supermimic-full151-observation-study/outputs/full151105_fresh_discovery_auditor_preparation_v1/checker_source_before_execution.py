"""Three small saved-byte/identity boundaries, no actual discovery input read."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'scripts/audit_fresh_calibration_discovery_saved_cpu_v1.py'
OUT=ROOT/'outputs/full151105_fresh_discovery_auditor_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);s=importlib.util.spec_from_file_location('fresh_saved_auditor',SRC);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    pins={'auditor':a.pin(SRC),'helper':a.pin(a.HELPER),'checker':a.pin(__file__)}
    for key,path in [('auditor',SRC),('checker',Path(__file__))]:ast.parse(path.read_bytes());(OUT/(key+'_source_before_execution.py')).write_bytes(path.read_bytes())
    a.dump(OUT/'predata_freeze.json',{'source_pins':pins,'actual_fresh_discovery_read':False,'new_native_execution':False})
    h=a.helper();values={'np':np.asarray([1.,2.],dtype='f4').view('u1').copy(),'empty':np.array([],dtype='u1')}
    meta=[['numpy','<f4',[2],[4],123,True],['warp',"<class 'warp._src.types.float32'>",[0],'cuda:0',None]]
    top={'array_values':[{'key':k,'target_meta':m,'value_sha256':hashlib.sha256(values[k].tobytes()).hexdigest()} for k,m in zip(values,meta)]}
    desc={'native_Warp_arrays':1,'native_Torch_tensors':0,'explicit_arrays':1}
    valid=a.byte_inventory(top,values,desc,h)
    checks={'saved_numpy_bytes_and_native_emptyNull_metadata':valid['all_valid'] and valid['empty_Warp_ptr_None']==1 and valid['array_bytes_total']==8}
    wrong={**top,'array_values':[dict(x) for x in top['array_values']]};wrong['array_values'][1]['target_meta']=meta[1].copy();wrong['array_values'][1]['target_meta'][2]=[1]
    checks['nonempty_Warp_nullptr_or_SHA_drift_rejected']=not a.byte_inventory(wrong,values,desc,h)['all_valid']
    labels=[name+'/'+str(k) for name in ('A0','A2') for k in a.PHASES]
    checks['new8labels_and_oldshape_signedzero_not_silently_equal']=labels==['A0/24','A0/74','A0/124','A0/148','A2/24','A2/74','A2/124','A2/148'] and not a.f32same(np.zeros((1,20),dtype='f4'),np.zeros(20,dtype='f4')) and not a.f32same([0.],[-0.]) and not any(x in sys.modules for x in ('torch','mujoco','warp','scipy','dex_supermimic'))
    checks={k:bool(v) for k,v in checks.items()};report={'schema':'full151105_fresh_discovery_saved_auditor_CPU_proof:1','source_pins':pins,'passed':sum(checks.values()),'total':len(checks),'checks':checks,'actual_fresh_discovery_reads':0,'new_models_native_Env_physics_GPU':0}
    a.dump(OUT/'report.json',report);ready={'source_pins':pins,'cpu_report':a.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    a.dump(OUT/'READY.json',ready);print(json.dumps({'READY':a.pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
