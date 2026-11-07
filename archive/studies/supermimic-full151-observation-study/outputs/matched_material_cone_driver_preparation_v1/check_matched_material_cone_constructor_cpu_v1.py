"""Pure synthetic array checks; no material inference or LP/physics calls."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'scripts/screen_full151_matched_material_cones_cpu_v1.py';(out/'worker_source.py').write_bytes(source.read_bytes());(out/'checker_source.py').write_bytes(Path(__file__).read_bytes())
    sp=importlib.util.spec_from_file_location('_matched_cone_synthetic_constructor',source);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);checks={}
    def check(n,v):checks[n]=bool(v)
    local=np.zeros((6,6));local[0]=1.;local[1,:2]=[.7,-.7];local[2,2:4]=[.7,-.7];local[3,4:]=[.005,-.005]
    point=np.array([[1.,0.,0.]]);outward=np.array([[0.,0.,-1.]])
    A=m.wrench_matrix(point,outward,np.zeros(3),2.,local)
    check('six_columns_not_eightedge_product',A.shape==(6,6))
    check('normalforce1_not_unitvector_norm',np.array_equal(A[2],np.ones(6)) and np.linalg.norm(A[:3,0])>1)
    check('torsion_pairs_keep_normalforce_and_lever',np.array_equal(A[:3,4:],np.array([[0.,0.],[0.,0.],[1.,1.]])) and np.array_equal(A[3:5,4:],np.array([[0.,0.],[-.5,-.5]])) and np.array_equal(A[5,4:],np.array([.0025,-.0025])))
    check('moment_radius_divided_once',A[5,2]==.35 and A[5,3]==-.35)
    P=np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[.2,.3,.4],[-.1,.2,-.3]])
    N=np.tile(outward,(5,1));a3=m.wrench_matrix(P[:3],N[:3],np.zeros(3),2.,local);a5=m.wrench_matrix(P,N,np.zeros(3),2.,local)
    check('H123_exact_nested_columns',a3.shape==(6,18) and a5.shape==(6,30) and np.array_equal(a3,a5[:,:18]))
    check('normalforce_objective_coefficients_all1',np.array_equal(a5[2],np.ones(30)))
    check('jsonpointer_unambiguous_escaped_contract_fields',m.pointer({'fields':{'~x/y':4}},'/fields/~0x~1y')==4)
    for label,p,n,t in [('unknown_normal',point,np.zeros_like(outward),local),('nonfinite_point',point*np.nan,outward,local),('independent_torsion_no_normalF',point,outward,np.zeros((6,6))),('nonunit_normal',point,outward*2,local)]:
        try:m.wrench_matrix(p,n,np.zeros(3),2.,t)
        except ValueError:check('reject_'+label,True)
        else:check('reject_'+label,False)
    check('no_Torch_SciPy_Mujoco_Warp_imports',all(k not in sys.modules for k in ('torch','scipy','mujoco','warp','mujoco_warp')))
    report={'schema':'matched_material_cone_constructor_synthetic_CPU:1','status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},'LP_calls':0,'native_geometry_queries':0,'Env_GPU_physics':False,'scope':'Synthetic contact6 template only, not actual material verification or a physical/LP witness. Actual template and cue producers require independent contracts.'}
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(report,indent=2));raise SystemExit(0 if all(checks.values()) else 1)
if __name__=='__main__':main()
