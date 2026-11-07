"""Saved-array/plan/IO checks only. No native queries or optimization calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import argparse,ast,copy,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import scipy.optimize
from scipy import sparse
from runtime_io import write_json
from restoration_prefix import math as m
from restoration_prefix.inputs import checked_plan,required_sources,PROTOCOL,PROTOCOL_SHA,TARGET_PROTOCOL,TARGET_SHA,PARENT_PROTOCOL,PARENT_SHA,RETARGET_VARIANT_ID
from restoration_prefix.io import sha,read,atomic_npz,atomic_bytes
from restoration_prefix.problem import load_linear_contract


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    checks={};calls={'linprog':0,'minimize':0};original={k:getattr(scipy.optimize,k) for k in calls}
    def deny(k):
        def fail(*a,**kw):calls[k]+=1;raise AssertionError('Selfcheck forbids solver')
        return fail
    for k in calls:setattr(scipy.optimize,k,deny(k))
    try:
        proto=read(PROTOCOL);parent=read(PARENT_PROTOCOL)
        with np.load(proto['seed_candidate']['path'],allow_pickle=False) as src:z={k:src[k].copy() for k in src.files}
        con=load_linear_contract(proto['linear_contract']['path']);x=z['optimizer_x'];d=z['constraint_scaled_mm'];initial=m.restoration_z0(x,d);lo,hi=m.restoration_box(x,con['lower'],con['upper'])
        checks['seed64aa_hash_bound']=sha(proto['seed_candidate']['path'])=='64aa2cfe44f20fee4546b07c4546916c95d57771404cb5fb4e39156206ad9b43'
        checks['original5960x340_matrix_shape']=con['A_ub'].shape==(5960,340)
        checks['R341_z0_preserves_exact_seed']=initial.shape==(341,) and np.array_equal(initial[:340],x)
        checks['R_z0_declared_buffer']=abs(initial[-1]-(max(0.,d.max()-2.999)+1e-6))<1e-15
        checks['R_z0_soft_geometry_slack_positive']=m.restoration_geometry_slack(initial,d).min()>0
        raw=m.primal_audit(x,d,con,trust_lower=lo,trust_upper=hi)
        checks['slack_feasible_does_not_make_original_seed_feasible']=not raw['passed'] and raw['declared_arithmetic']['linear'] and raw['declared_arithmetic']['box'] and raw['declared_arithmetic']['restoration_trustbox']
        changed=initial.copy();changed[0]+=.001
        checks['Robjective_slack_only']=m.restoration_objective(initial)==m.restoration_objective(changed)==initial[-1] and np.array_equal(m.restoration_gradient(initial),np.r_[np.zeros(340),1.])
        checks['Rtrust_intersects_original_without_clipping_seed']=np.all(lo<=x) and np.all(x<=hi) and np.max(x-lo)<=.05+1e-14 and np.max(hi-x)<=.05+1e-14
        C=sparse.csr_matrix((z['constraint_J_CSR_data'],z['constraint_J_CSR_indices'],z['constraint_J_CSR_indptr']),shape=z['constraint_J_shape'])
        RJ=m.restoration_geometry_jacobian(C)
        checks['RslackJacobian122x341']=RJ.shape==(122,341) and np.array_equal(RJ[:,-1],np.ones(122)) and np.array_equal(RJ[:,:340],-C.toarray())
        J=sparse.csr_matrix((z['jacobian_CSR_data'],z['jacobian_CSR_indices'],z['jacobian_CSR_indptr']),shape=z['jacobian_shape']);g,H=m.gauss_newton_model(z['residual'],J)
        checks['GN_unchanged_fit_g_and_regularizer']=np.array_equal(g,np.asarray(J.T@z['residual']).ravel()) and np.array_equal(H,(J.T@J).toarray()+1e-4*np.eye(340))
        qp=m.qp_contract(x,d,C,con)
        checks['QP_original_linear_rhs_and_depth_rhs']=np.array_equal(qp['b_original'],con['b_centered']-con['A_ub']@(x-con['x0'])) and np.array_equal(qp['b_geometry'],2.999-d)
        checks['QPstep_bounds_no_Rtrust_inheritance']=np.array_equal(qp['lower'],np.maximum(con['lower']-x,-.02)) and np.array_equal(qp['upper'],np.minimum(con['upper']-x,.02))
        checks['zero_QP_direction_never_eligible']=not m.qp_direction_audit(np.zeros(340),g,qp)['eligible']
        fake={'passed':True}
        checks['unchangedx_never_Armijo_accepted']=not m.armijo_decision(10.,9.,1.,-1.,fake,trial_x_changed=False)['accepted']
        checks['samecost_never_accepted_despite_RHS_rounding']=not m.armijo_decision(10.,10.,.03125,-1e-20,fake,trial_x_changed=True)['accepted']
        checks['nonfinite_trialcost_rejected']=not m.armijo_decision(10.,float('nan'),1.,-1.,fake,trial_x_changed=True)['accepted']
        files=required_sources(parent);pins={str(i):{'path':str(f),'sha256':sha(f)} for i,f in enumerate(files)}
        p={'schema':'restoration_prefix_cpu_plan:1','optimization_authorized':False,'retarget_variant_id':RETARGET_VARIANT_ID,'protocol':{'path':str(PROTOCOL),'sha256':PROTOCOL_SHA},'target_protocol':{'path':str(TARGET_PROTOCOL),'sha256':TARGET_SHA},'parent_protocol':{'path':str(PARENT_PROTOCOL),'sha256':PARENT_SHA},'inputs':proto['inputs'],'seed_candidate':proto['seed_candidate'],'linear_contract':proto['linear_contract'],'linear_contract_metadata':proto['linear_metadata'],'source_pins':pins,'R_maxiter':60,'R_max_objective_calls':200,'R_ftol':1e-9,'F_max_major':8,'QP_maxiter':25,'QP_max_objective_calls':100,'QP_ftol':1e-9,'soft_wall_seconds':340,'owned_process_seconds':360,'automatic_retry':False}
        write_json(out/'valid_check_plan.json',p);checked_plan(out/'valid_check_plan.json');checks['full_source_plan_valid_no_solver']=True
        def reject(label,mutate,run=False):
            q=copy.deepcopy(p);mutate(q);path=out/(label+'.json');write_json(path,q)
            try:checked_plan(path,run=run);checks[label]=False
            except ValueError:checks[label]=True
        reject('unauthorized_run_rejected',lambda q:None,True)
        reject('wrong_protocol_rejected',lambda q:q['protocol'].__setitem__('sha256','0'*64))
        reject('wrong_seed_rejected',lambda q:q['seed_candidate'].__setitem__('sha256','0'*64))
        reject('wrong_linear_artifact_rejected',lambda q:q['linear_contract'].__setitem__('sha256','0'*64))
        reject('missing_entry_pin_rejected',lambda q:q['source_pins'].pop('0'))
        reject('wrong_math_source_hash_rejected',lambda q:q['source_pins'][next(k for k,v in q['source_pins'].items() if Path(v['path']).name=='math.py')].__setitem__('sha256','0'*64))
        reject('wrong_variant_rejected',lambda q:q.__setitem__('retarget_variant_id','parent_is_not_new_variant'))
        reject('QPbudget101_rejected',lambda q:q.__setitem__('QP_max_objective_calls',101))
        reject('retry_rejected',lambda q:q.__setitem__('automatic_retry',True))
        tree=ast.parse((ROOT/'scripts/build_restoration_prefix_cpu.py').read_text());allcalls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        name=lambda n:n.func.id if isinstance(n.func,ast.Name) else n.func.attr if isinstance(n.func,ast.Attribute) else None
        checks['exactly_two_minimize_call_sites_R_and_QP']=sum(name(n)=='minimize' for n in allcalls)==2
        checks['no_linprog_call_site']=not any(name(n)=='linprog' for n in allcalls)
        checks['only_lastaccepted_normal_canonical_call']=all(len(n.args)==1 and isinstance(n.args[0],ast.Name) and n.args[0].id=='last_accepted' for n in allcalls if name(n)=='final_candidate')
        atomic_npz(out/'io_test.npz',unknown_gap=np.array([np.nan]),seed_x=x)
        payload=(out/'io_test.npz').read_bytes();atomic_bytes(out/'io_copy.npz',payload)
        checks['samepayload_bounded_copy_hash']=sha(out/'io_test.npz')==sha(out/'io_copy.npz')
        with np.load(out/'io_copy.npz',allow_pickle=False) as copied:checks['NaN_unknown_survives_NPZ']=bool(np.isnan(copied['unknown_gap'][0]) and np.array_equal(copied['seed_x'],x))
        try:write_json(out/'invalid_nan.json',{'invalid':float('nan')});checks['nonfinite_JSON_rejected']=False
        except ValueError:checks['nonfinite_JSON_rejected']=True
        checks['no_solver_calls']=all(v==0 for v in calls.values())
        checks['no_MuJoCo_Torch_Warp_imports']=all(k not in sys.modules for k in ('mujoco','torch','warp','mujoco_warp'))
        result={'status':'PASS_SAVED_ARRAY_PLAN_IO_SELF_CHECKS_NO_QUERY_SOLVER' if all(checks.values()) else 'FAIL_SELF_CHECKS','checks':{k:bool(v) for k,v in checks.items()},'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'source_pins':{str(f):sha(f) for f in [Path(__file__),ROOT/'scripts/build_restoration_prefix_cpu.py',ROOT/'src/restoration_prefix/inputs.py',ROOT/'src/restoration_prefix/math.py',ROOT/'src/restoration_prefix/problem.py']},'R_seed_original_primal':raw,'R_z0_slack_mm':float(initial[-1]),'geometry_queries':0,'physics_steps':0,'solver_calls':calls,'GPU_used':False,'scope':'Saved64aa matrices/vectors and plan/AST/IO checks; no native geometry construction, defaultCLI check, Rphase or QP solver run.'}
        write_json(out/'report.json',result);print(json.dumps(result,indent=2));return 0 if all(checks.values()) else 1
    finally:
        for k,v in original.items():setattr(scipy.optimize,k,v)
if __name__=='__main__':raise SystemExit(main())
