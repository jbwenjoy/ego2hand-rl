"""One boundary/mapping/nonorthogonal-observer closed fixture, no real arrays."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/full151_gradient/boundary_impulse_independent_math_v1.py'
OUT=ROOT/'outputs/full151105_boundary_impulse_independent_math_preparation_v1'


def pin(p):return {'path':str(Path(p).resolve()),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def dump(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)


def main():
    OUT.mkdir(exist_ok=False);sources={'math':pin(SRC),'checker':pin(__file__)}
    for k,p in [('math',SRC),('checker',Path(__file__))]:ast.parse(p.read_bytes());(OUT/(k+'_before_execution.py')).write_bytes(p.read_bytes())
    dump(OUT/'source_freeze.json',{'source_pins':sources,'actual_boundary_label_or_input_arrays_read':False})
    s=importlib.util.spec_from_file_location('boundary_independent_math',SRC);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    pre=np.zeros(58,dtype='f4');post=pre.copy();pre[43:52]=np.eye(3,dtype='f4').reshape(9);post[43:52]=np.diag([2,1,1]).astype('f4').reshape(9)
    pre[55:58]=[1,2,3];post[52:55]=[1,2,3];post[55:58]=[4,5,6]
    P=np.array([[1,.25,0],[0,1,0],[0,0,1]],dtype='f4');dt=np.full(4,.005,dtype='f4');h=float(dt[0]);F=np.zeros((4,2,3),dtype='f8');F[:,0,0]=2;T=F.copy()
    v=m.four_substep_labels(pre,post,P,dt,F,T,2.,np.array([0,0,-10.],dtype='f8'),np.diag([2.,3.,4.]))
    witness=np.zeros((4,2),bool);witness[2,0]=True;unknown=np.ones((152,4,2),bool);unknown[151,3,1]=False
    try:m.require_complete_dataset(unknown);subsetaccepted=True
    except ValueError:subsetaccepted=False
    schedule=m.expected_restored_schedule();last=schedule[-1]
    checks={'old152_lastgroup_native58_raw_transpose_nativeR9_solve_defect_and_global_unknown':len(schedule)==152 and schedule[0]['group_id']==150 and schedule[0]['substep_source_rows'][0]==600 and last['group_id']==451 and last['branch_id']==151 and last['substep_source_rows']==[1804,1805,1806,1807] and last['terminal_observation']==150
      and np.array_equal(v['J_palm_Ns_F64'][0],[8*h,2*h,0]) and v['DeltaT_s_F64']==4*h and v['DeltaT_s_F64']!=.02
      and np.allclose(v['r_v_saved_m_s_F64'],[1-4*h,2-h,3+40*h],rtol=0,atol=2e-15) and np.allclose(v['r_w_saved_rad_s_F64'],[3.75-h,3-2*h/3,3],rtol=0,atol=2e-15)
      and v['I_p1_F64'][0,0]==8 and v['raw_R1_det_F64']==2 and np.linalg.norm(v['r_w_saved_rad_s_F64'])>0 and m.source_witness(witness).tolist()==[True,False] and not subsetaccepted}
    checks={k:bool(v) for k,v in checks.items()}
    if any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules):raise ValueError('Unexpected model/backend')
    report={'schema':'full151105_boundary_impulse_independent_math_CPU_proof:1','source_pins':sources,'passed':sum(checks.values()),'total':len(checks),'checks':checks,'actual_boundary_target_arrays_read':0,'NN_FW_AD_MJB_FK_Env_physics_GPU':0}
    dump(OUT/'report.json',report);ready={'source_pins':sources,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':report['passed'],'total':report['total']}}
    dump(OUT/'READY.json',ready);print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2));return 0 if all(checks.values()) else 2


if __name__=='__main__':raise SystemExit(main())
