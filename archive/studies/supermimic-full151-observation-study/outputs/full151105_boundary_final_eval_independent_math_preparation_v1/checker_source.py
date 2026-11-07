"""One nonzero head / target cancellation / saved nonzero defect closed case."""
import hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'src/full151_gradient/boundary_final_eval_independent_math_v1.py'
def pin(p):return {'path':str(Path(p).resolve(strict=True)),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def dump(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
    out=ROOT/'outputs/full151105_boundary_final_eval_independent_math_preparation_v1';out.mkdir(exist_ok=False)
    sources={'math':pin(P),'checker':pin(__file__),'scalar_pivot':pin(P.with_name('boundary_impulse_independent_math_v1.py'))}
    for k,v in sources.items():(out/(k+'_source.py')).write_bytes(Path(v['path']).read_bytes())
    dump(out/'source_before_fixture.json',{'source_pins':sources,'actual_arrays_models_read':False})
    s=importlib.util.spec_from_file_location('independent_final_aux',P);a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
    x=np.zeros((1,58),dtype='f4');x[:,43:52]=np.eye(3,dtype='f4').reshape(9);x[:,55:58]=[1,2,3];pred=x.copy();pred[:,52:55]=[1,2,3];pred[:,55:58]=[2,3,4]
    z=np.asarray([[[1,2,3,1,1,1],[-1,-2,-3,-1,-1,-1]]],dtype='f4');J=np.asarray([[[4,6,8],[-4,-6,-8]]],dtype='f8');K=np.asarray([[[2,6,12],[-2,-6,-12]]],dtype='f8')
    r=a.auxiliary(x,pred,z,np.ones(52,dtype='f4'),2.,np.diag([2.,3.,4.]),np.zeros(3),.5,J,K,np.asarray([[.5,1,1.5]]),np.asarray([[.25,.5,.75]]))
    one=np.array_equal(r['J_hat_palm_F64'][0],[[2,4,6],[-2,-4,-6]]) and np.array_equal(r['K_hat_palm_F64'][0],[[2,3,4],[-2,-3,-4]])
    one &= np.array_equal(r['r_v_hat_F64'],[[1,2,3]]) and np.array_equal(r['r_w_hat_F64'],[[1,1,1]]) and abs(float(r['L_imp_F64'])-4/3)<1e-14 and abs(float(r['L_cons_F64'])-4.375/6)<1e-14
    one &= np.array_equal(r['z_target_F64'][0],[[2,3,4,1,2,3],[-2,-3,-4,-1,-2,-3]]) and np.max(abs(r['solve_residual_F64']))==0
    b=a.bce_elements(np.asarray([[2.,-2.]]),np.asarray([[True,False]]));one &= abs(a.fmean(b)-math.log1p(math.exp(-2)))<1e-15
    one &= not any(k.split('.')[0] in ('torch','mujoco','warp','dex_supermimic','mjlab') for k in sys.modules)
    report={'schema':'full151105_boundary_final_eval_independent_math_contract:1','source_pins':sources,'checks':{'nonzero_z_two_source_cancellation_actual_units_nonzero_saved_defect_scalar_solve':bool(one)},'passed':int(one),'total':1,'actual_arrays_models_FW_AD_MJB_physics_GPU':0}
    dump(out/'report.json',report);dump(out/'READY.json',{'schema':'full151105_boundary_final_eval_independent_math_READY:1','source_pins':sources,'cpu_report':pin(out/'report.json'),'cpu_checks':{'passed':int(one),'total':1}});print(json.dumps({'ready':pin(out/'READY.json'),'report':pin(out/'report.json'),'passed':int(one),'total':1},indent=2));return 0 if one else 2
if __name__=='__main__':raise SystemExit(main())
