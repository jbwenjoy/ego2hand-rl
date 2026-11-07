"""One closed-form nonprincipal-inertia/endpoint/sign fixture; no real arrays."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/full151_gradient/object_angular_momentum_independent_math_v1.py'
OUT=ROOT/'outputs/full151105_object_angular_independent_math_preparation_v1'


def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def dump(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)


def main():
    OUT.mkdir(exist_ok=False);sources={'math':pin(SRC),'checker':pin(__file__)}
    for k,p in [('math',SRC),('checker',Path(__file__))]:ast.parse(p.read_bytes());(OUT/(k+'_before_execution.py')).write_bytes(p.read_bytes())
    dump(OUT/'source_freeze.json',{'source_pins':sources,'actual_endpoint_force_or_inertia_values_read':False})
    s=importlib.util.spec_from_file_location('independent_angular_vectors',SRC);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    inertia=np.array([2,3,5],dtype='f4');qi=np.array([.6,0,0,.8],dtype='f4');omega=np.array([1,0,0],dtype='f4')
    pre=m.endpoint_world_L(np.array([1,0,0,0],dtype='f4'),omega,inertia,qi);post=m.endpoint_world_L(np.array([0,0,0,1],dtype='f4'),omega,inertia,qi)
    w,z=map(float,qi[[0,3]]);n=w*w+z*z;c=(w*w-z*z)/n;t=2*w*z/n
    # Closed planar Ibody entries, not the producer's matrix transform.
    expected=np.array([2*c*c+3*t*t,(2-3)*c*t,0.],dtype='f8')
    minus=m.signed_contact_torque([2,3,5,7,11,13],np.eye(3),[1,2,3],[0,0,0],-1)
    plus=m.signed_contact_torque([2,3,5,7,11,13],np.eye(3),[1,2,3],[0,0,0],1)
    checks={'nonprincipal_inertia_both_endpoint_rotations_and_direct_torque_signed_once':bool(np.allclose(pre['world_L_F64'],expected,atol=2e-15,rtol=2e-15) and np.allclose(post['world_L_F64'],-expected,atol=2e-15,rtol=2e-15)
      and abs(expected[1])>.01 and np.array_equal(minus['lever_torque_COM_Nm_F64'],[-1,-1,1]) and np.array_equal(minus['direct_torque_Nm_F64'],[-7,-11,-13]) and np.array_equal(minus['total_torque_COM_Nm_F64'],[-8,-12,-12]) and np.array_equal(m.sum_vectors([minus['total_torque_COM_Nm_F64'],plus['total_torque_COM_Nm_F64']]),[0,0,0])
      and not m.endpoint_world_L(np.zeros(4,dtype='f4'),omega,inertia,qi)['endpoint_valid'])}
    if any(k.split('.')[0] in ('torch','mujoco','warp','mjlab','dex_supermimic') for k in sys.modules):raise ValueError('Unexpected native/model import')
    report={'schema':'full151105_object_angular_independent_math_CPU_proof:1','source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,
      'actual_endpoint_force_or_inertia_array_reads':0,'MJB_model_FW_AD_Env_physics_GPU':0,'synthetic_only':True}
    dump(OUT/'report.json',report);ready={'source_pins':sources,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    dump(OUT/'READY.json',ready);print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
