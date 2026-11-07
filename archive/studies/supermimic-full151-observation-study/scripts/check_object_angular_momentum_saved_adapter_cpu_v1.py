"""One contact-evidence ABI fixture; no actual angular output reads."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'scripts/audit_object_angular_momentum_saved_cpu_v1.py'
OUT=ROOT/'outputs/full151105_object_angular_saved_adapter_preparation_v1'


def main():
    OUT.mkdir(exist_ok=False);ast.parse(SRC.read_bytes())
    s=importlib.util.spec_from_file_location('angular_saved_adapter',SRC);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    sources={'auditor':a.pin(SRC),'checker':a.pin(__file__),'independent_math':a.pin(a.MATH)}
    for key,p in [('auditor',SRC),('checker',Path(__file__))]:(OUT/(key+'_before_execution.py')).write_bytes(p.read_bytes())
    a.dump(OUT/'source_freeze.json',{'source_pins':sources,'actual_angular_output_arrays_read':False,'F64_compare_only_not_residual_accuracy':{'relative':a.F64_RTOL,'absolute':a.F64_ATOL}})
    m=a.math_module();contact={'geom':[7,11],'efc_address':0,'force6':[2,3,5,7,11,13],'frame':np.eye(3).tolist(),'pos':[1,2,3]}
    raw={'geom':[7,11],'type':3,'worldid':0,'efc_address':[0,-1]}
    v=a.contact_evidence(contact,raw,{7},{11},1,np.zeros(3),m)
    unknown=a.contact_evidence(contact,{**raw,'efc_address':[-1,-1]},{7},{11},1,np.zeros(3),m)
    tiny={**contact,'force6':[1e-7,0,0,0,0,0]};small=a.contact_evidence(tiny,raw,{7},{11},1,np.zeros(3),m)
    checks={'type3_valid_original_order_direct_sign_once_noFncut_and_unknown_notzero':v['constraint_force_evidence_valid'] and v['external'] and v['category']=='HO' and np.array_equal(v['total_torque_world_Nm'],[-8,-12,-12]) and not unknown['constraint_force_evidence_valid'] and np.isnan(unknown['total_torque_world_Nm']).all() and small['constraint_force_evidence_valid'] and np.any(small['total_torque_world_Nm']!=0)}
    checks={k:bool(v) for k,v in checks.items()}
    if any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules):raise ValueError('Unexpected backend/model import')
    report={'schema':'full151105_object_angular_saved_adapter_CPU_proof:1','source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,'actual_angular_output_arrays_read':0,'MJB_backend_models_FW_AD_Env_physics_GPU':0}
    a.dump(OUT/'report.json',report);ready={'source_pins':sources,'cpu_report':a.pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']},'api':{'entry':str(SRC)},'actual_saved_audit_completed':False}
    a.dump(OUT/'READY.json',ready);print(json.dumps({'READY':a.pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
