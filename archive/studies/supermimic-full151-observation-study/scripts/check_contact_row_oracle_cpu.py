"""Synthetic row/Jacobian algebra only, no native import, query or solve."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151.contact_row_oracle import geometry_rows,distance_jacobian_from_point_jacobians

def run(out):
    if out.exists():raise FileExistsError(out)
    checks={}
    def C(k,v):checks[k]=bool(v)
    def reject(k,fn):
        try:fn()
        except ValueError:checks[k]=True
        else:checks[k]=False
    B=np.zeros((601,151));B[1,[0,1,2,3]]=[.1,.2,.3,.4];B[2,[73,74,75,76]]=[.4,.3,.2,.1];B[600,150]=1
    si=np.array([1,2,600,0],np.int64);d=np.array([-.004,-.001,0.,-.004]);J=np.zeros((4,20));J[0,[0,19]]=[.02,-.03];J[1,[1,7]]=[-.01,.04];J[2,19]=.05;J[3,0]=.1
    ids=[{'pair':1,'native':i} for i in range(4)];r=geometry_rows(B,si,d,J,witness_ids=ids)
    dense=np.zeros((4,3020))
    for k in range(4):
        for j in range(20):
            for c in range(151):dense[k,j*151+c]=-J[k,j]*B[si[k],c]
    C('independent_explicit_joint_major_dense_chain_exact',np.array_equal(r['A_step'].toarray(),dense))
    C('joint19_final_coefficient_index3019',r['A_step'][2,3019]==-.05)
    C('rhs_metres_sign_depth_exact',np.array_equal(r['rhs_m'],d+.002999))
    C('current_penetration_violation_preserved',r['current_violation_m'][0]>.001 and r['current_violation_m'][3]>.001)
    C('fixed_P0_P1_position_support_no_free_columns',r['A_step'][3].nnz==0 and r['fixed_sample_no_free_position_support'][3] and r['rhs_m'][3]<0)
    C('native_or_smoothness_not_claimed',r['metadata']['native_or_physical_derivative_qualified'] is False and r['metadata']['source_completeness_verified_by_helper'] is False)
    rng=np.random.default_rng(1984);delta=rng.normal(size=(20,151))*.001
    dq=np.stack([B[s]@delta.T for s in si]);newdist=d+np.einsum('ij,ij->i',J,dq)
    C('affine_distance_inequality_equivalent',np.allclose(r['rhs_m']-r['A_step']@delta.ravel(),newdist+.002999,rtol=0,atol=1e-17))
    eps=1e-6;direction=rng.normal(size=(20,151))
    def gd(alpha):return -(d+np.einsum('ij,ij->i',J,np.stack([B[s]@(alpha*direction).T for s in si])))-.002999
    fd=(gd(eps)-gd(-eps))/(2*eps)
    C('synthetic_central_FD_chain',np.max(np.abs(fd-r['A_step']@direction.ravel()))<1e-11)
    ties=geometry_rows(B,np.array([1,1]),np.array([-.004,-.004]),np.stack([J[0],-J[0]]),witness_ids=[{'pair':1,'native':0},{'pair':1,'native':1}])
    C('tied_witnesses_retained_no_averaging',ties['A_step'].shape[0]==2 and np.array_equal(ties['A_step'][0].toarray(),-ties['A_step'][1].toarray()) and len(ties['witness_ids'])==2)
    z=geometry_rows(B,np.empty(0,np.int64),np.empty(0),np.empty((0,20)))
    C('empty_rows_unknown_not_clearance',z['A_step'].shape==(0,3020) and z['metadata']['native_or_physical_derivative_qualified'] is False)
    Ja=rng.normal(size=(3,26));Jb=rng.normal(size=(3,26));n=np.array([0.,0.,1.]);cols=np.arange(6,26)
    qj=distance_jacobian_from_point_jacobians(n,Ja,Jb,cols)
    C('native_pointJac_formula_direct',np.array_equal(qj,n@(Jb[:,cols]-Ja[:,cols])))
    C('reverse_geomorder_and_normal_same_derivative',np.array_equal(qj,distance_jacobian_from_point_jacobians(-n,Jb,Ja,cols)))
    obj=np.zeros((3,26));obj[:,:6]=rng.normal(size=(3,6))
    C('fixed_object_freeDOFs_excluded_by_explicit_robotcols',np.array_equal(distance_jacobian_from_point_jacobians(n,obj,Jb,cols),n@Jb[:,cols]))
    reject('reject_missing_J_no_saved_normal_fabrication',lambda:geometry_rows(B,si,d,None))
    bad=J.copy();bad[0,0]=np.nan;reject('reject_NaN_J_not_zero_fill',lambda:geometry_rows(B,si,d,bad))
    reject('reject_bool_sample_index',lambda:geometry_rows(B,np.array([True]),d[:1],J[:1]))
    reject('reject_outside_sample_index',lambda:geometry_rows(B,np.array([601]),d[:1],J[:1]))
    reject('reject_wrong151_free_dimension',lambda:geometry_rows(B[:,:150],si,d,J))
    reject('reject_changed_planning_depth',lambda:geometry_rows(B,si,d,J,.003))
    reject('reject_missing_witness_ID',lambda:geometry_rows(B,si,d,J,witness_ids=[0]))
    reject('reject_normal_not_unit',lambda:distance_jacobian_from_point_jacobians(n*2,Ja,Jb,cols))
    reject('reject_missing_robot_columns',lambda:distance_jacobian_from_point_jacobians(n,Ja,Jb,cols[:19]))
    C('no_native_Torch_Warp_WM_import',all(k not in sys.modules for k in ['mujoco','torch','warp']))
    out.mkdir(parents=True,exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    result={'schema':'full151_contact_row_oracle_synthetic_CPU_checks:1','status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,
            'source_pins':{'module':{'path':str(ROOT/'src/full151/contact_row_oracle.py'),'sha256':sha(ROOT/'src/full151/contact_row_oracle.py')},'checker':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}},
            'scope':'Synthetic affine distance, explicit dense chain and algebra sign/order tests only. No true geometry Jacobian certificate.','real_FK_collision_solver_Env_WM_physics_GPU_calls':0}
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=run(a.output);print(json.dumps(x,indent=2));sys.exit(0 if x['status']=='PASS' else 1)
