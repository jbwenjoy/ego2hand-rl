"""Synthetic sparse/roundoff proof only; no LP, real geometry or path fit."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys,traceback
import numpy as np
from scipy import sparse

ROOT=Path(__file__).resolve().parents[1]
SOURCES={'global':ROOT/'src/full151/sparse_global_scp.py',
         'old_control':ROOT/'src/full151/continuous_control.py',
         'row_oracle_v2':ROOT/'src/full151/contact_row_oracle_v2.py'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}


def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def main(output):
    if output.exists():raise FileExistsError('Fresh proof output required')
    output.mkdir(parents=True)
    for name,p in {**SOURCES,'checker':Path(__file__)}.items():(output/(name+'_source_before.py')).write_bytes(p.read_bytes())
    pins={name:pin(p) for name,p in {**SOURCES,'checker':Path(__file__)}.items()}
    checks={};details={}
    def ok(name,v):
        checks[name]=bool(v)
        if not checks[name]:raise AssertionError(name)
    def reject(name,fn):
        try:fn()
        except (ValueError,TypeError):ok(name,True)
        else:ok(name,False)
    try:
        ok('old3c32_source_unchanged',pins['old_control']['sha256']=='3c32f3022d5bade3b6acfd0423c94b4f0dcee3b8f4832a6f0cbd45c9c2da4d17')
        ok('source_oracle143e_units_v2_bound',pins['row_oracle_v2']['sha256']=='143eaa3dd68dc6cbb579a84126767bf6d95f6225200ffa2e40c3e318a5f5583a')
        scp=module(SOURCES['global'],'synthetic_full151_global_scp')
        old=module(SOURCES['old_control'],'synthetic_full151_old_cubic')
        oracle=module(SOURCES['row_oracle_v2'],'synthetic_full151_all_witness_v2')
        P0=(np.arange(20)-10)/128;P1=P0.copy()
        center=np.repeat(P0[None,:],153,axis=0)
        center[2:]+=.002*np.sin(np.arange(151)[:,None]/150*np.pi)*(1+np.arange(20)[None,:]/40)
        lo=-np.ones(20);hi=np.ones(20);kp=np.ones(20);bias=np.c_[np.zeros(20),-kp,-.02*kp]
        caps=np.tile([-4.,4.],(20,1))
        params={'kp':kp,'bias':bias,'target_lower':lo,'target_upper':hi,'actuator_force_bounds':caps,'joint_force_bounds':caps}
        target=old.eval_controls(center,np.arange(151)*.02)
        u0=P0.astype(np.float32);tau=np.zeros(20,np.float32)
        problems=[old.build_joint_lp(j,target,P0,P1,u0,tau,params,lo,hi) for j in range(20)]
        B=old.basis().evaluate(np.arange(601)*.005)[:,2:]
        J=np.zeros((3,20));J[0,0]=.1;J[0,1]=-.08;J[1,12]=.07;J[2]=-J[0]
        d=np.array([-.0038,-.0025,-.0038]);sample=np.array([5,17,5])
        geom=oracle.geometry_rows(B,sample,d,J,witness_ids=['fake_tieA','fake_other','fake_tieB'])
        built=scp.build_scp_lp(problems,center,geom)
        kw,c=built['linprog_kwargs'],built['contract']
        M=geom['A_step'].shape[0]
        ok('9061_variables_and48040_plus3_rows',kw['c'].shape==(9061,) and kw['A_ub'].shape==(48043,9061))
        S=scp.geometry_to_global_map()
        free=np.concatenate([center[2:,j]-P0[j] for j in range(20)])
        ok('jointmajor151_to453_mapping_exact',np.array_equal(S@c['center_vector'][:9060],free))
        expectedcols=np.repeat(np.arange(20)*453,151)+np.tile(np.arange(151),20)
        ok('all3020_column_indices_exact',np.array_equal(S.indices,expectedcols) and np.array_equal(S.data,np.ones(3020)))
        raw=c['A_raw'];rhs=c['b_raw'];start=c['geometry_row_start']
        ok('eta_absent_from_all_original_control_rows',raw[:48040,-1].nnz==0)
        ok('eta_minus1_only_each_geometry_row',np.array_equal(raw[48040:,-1].toarray().ravel(),-np.ones(M)))
        ok('geometry_all_witness_and_opposing_ties_retained',c['geometry_witness_ids']==geom['witness_ids'] and len(c['geometry_witness_ids'])==3 and np.any(J[0]!=J[2]))
        mapped=geom['A_step']@S
        ok('center_rhs_shift_exact',np.array_equal(rhs[start:],geom['rhs_m']+mapped@c['center_vector'][:9060]))
        step=np.zeros(3020);step[2]=.003;step[151+2]=-.004
        z=c['center_vector'].copy()
        for j in range(20):z[j*453:j*453+151]+=step[j*151:(j+1)*151]
        z[-1]=.0007
        expected=geom['A_step']@step-geom['rhs_m']-z[-1]
        actual=raw[start:]@z-rhs[start:]
        ok('geometry_step_sign_center_and_eta_units_m',np.max(np.abs(actual-expected))<1e-15)
        ok('all_rows_scaled_positive1e4_exact',(kw['A_ub']-raw*1e4).nnz==0 and np.array_equal(kw['b_ub'],rhs*1e4))
        ok('old_sum20_cost_unchanged_eta1e4',np.array_equal(kw['c'][:-1],np.concatenate([p['linprog_kwargs']['c'] for p in problems])) and kw['c'][-1]==1e4)
        ok('one_major_HiGHS_budget10s_no_solve',kw['method']=='highs' and kw['options']=={'time_limit':10.,'maxiter':100000,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
        bounds=kw['bounds'];trust=True;epis=True
        for j in range(20):
            for k in range(151):
                i=j*453+k;low,high=bounds[i]
                trust &= abs(low-max(-1-P0[j],c['center_vector'][i]-.15))<1e-15 and abs(high-min(1-P0[j],c['center_vector'][i]+.15))<1e-15
            epis &= all(bounds[j*453+k]==(0.,None) for k in range(151,453))
        ok('trust_intersects_hard_freeCP_bounds_only',trust)
        ok('epigraph_bounds_unchanged_eta_nonnegative',epis and bounds[-1]==(0,None))
        P,eta=scp.unpack_trial(z,c)
        ok('unpack_P0_P1_exact_and_eta_separate',np.array_equal(P[:2],center[:2]) and eta==z[-1])
        ok('unpack_joints_freeCP_order',all(np.array_equal(P[2:,j],z[j*453:j*453+151]+P0[j]) for j in range(20)))
        a=scp.audit_subproblem_solution(c['center_vector'],c,bounds)
        higher=c['center_vector'].copy();higher[-1]+=1.
        b=scp.audit_subproblem_solution(higher,c,bounds)
        ok('eta_cannot_relax_original_control_rows',np.array_equal(a['raw_residuals'][:48040],b['raw_residuals'][:48040]) and np.array_equal(a['original_absolute_raw_residuals'],b['original_absolute_raw_residuals']))
        ok('linear_audit_never_claims_actual_geometry',a['actual_geometry_qualified'] is False)
        ok('center_eta_is_max_current_observed_linear_violation',c['center_vector'][-1]==max(0.,float(np.max(-geom['rhs_m']))))
        # Concrete IEEE64 example: algebraic variable translation can still
        # make a centered row pass while direct original absolute arithmetic
        # exceeds1e-12. The extra original audit must catch it.
        offset=1.1;coef=10000.;original_b=11000.00001
        centered_b=original_b-coef*offset;candidate=centered_b/coef
        synthetic=dict(c)
        rr=sparse.csr_matrix(([coef],([0],[0])),shape=(48043,9061))
        aa=sparse.csr_matrix(([coef],([0],[0])),shape=(48040,9060))
        rb=np.zeros(48043);rb[0]=centered_b
        ab=np.zeros(48040);ab[0]=original_b
        shift=np.zeros(9060);shift[0]=offset
        synthetic.update(A_raw=rr,b_raw=rb,A_original_absolute=aa,b_original_absolute=ab,absolute_shift=shift)
        zz=np.zeros(9061);zz[0]=candidate
        roundoff=scp.audit_subproblem_solution(zz,synthetic,[(None,None)]*9060+[(0,None)])
        ok('centered_pass_original_absolute_fail_caught',roundoff['max_raw_Ax_minus_b']<=1e-12 and roundoff['max_original_absolute_raw_Ax_minus_b']>1e-12 and not roundoff['raw_primal_pass'])
        details['roundoff_counterexample']={'offset':offset,'coefficient':coef,'b_original':original_b,'b_centered':centered_b,'candidate_centered':candidate,'centered_residual':roundoff['max_raw_Ax_minus_b'],'original_absolute_residual':roundoff['max_original_absolute_raw_Ax_minus_b']}
        negative=c['center_vector'].copy();negative[-1]=-1e-6
        an=scp.audit_subproblem_solution(negative,c,bounds)
        ok('negative_eta_rejected_by_bounds_and_qual',not an['raw_primal_pass'] and not an['eta_gate_pass'])
        empty=oracle.geometry_rows(B,np.array([],dtype=int),np.empty(0),np.empty((0,20)),witness_ids=[])
        ec=scp.build_scp_lp(problems,center,empty)['contract']
        ok('no_generated_rows_not_clearance',ec['generated_witness_rows']==0 and 'UNKNOWN' in ec['missing_pair_gap_semantics'])
        bad=dict(geom);bad['units']='mm'
        reject('reject_implicit_mm_units',lambda:scp.build_scp_lp(problems,center,bad))
        bad=dict(geom);bad['column_order']='control-major'
        reject('reject_wrong_geometry_variable_order',lambda:scp.build_scp_lp(problems,center,bad))
        bad=dict(geom);bad['target_depth_m']=.003
        reject('reject_relaxed_planning_depth',lambda:scp.build_scp_lp(problems,center,bad))
        bad=dict(geom);bad['rhs_m']=np.array([np.nan,0,0])
        reject('reject_missing_geometry_value',lambda:scp.build_scp_lp(problems,center,bad))
        badP=center.copy();badP[1,0]+=.001
        reject('reject_changed_fixedP1',lambda:scp.build_scp_lp(problems,badP,geom))
        reject('reject_trust_expansion',lambda:scp.build_scp_lp(problems,center,geom,trust_radius_rad=.16))
        reject('reject_zero_trust',lambda:scp.build_scp_lp(problems,center,geom,trust_radius_rad=0))
        reject('reject_changed_row_scale',lambda:scp.build_scp_lp(problems,center,geom,row_scale=1))
        reject('reject_changed_eta_penalty',lambda:scp.build_scp_lp(problems,center,geom,eta_penalty=1))
        tree=ast.parse(SOURCES['global'].read_text('utf-8'))
        calls={getattr(n.func,'attr',getattr(n.func,'id','')) for n in ast.walk(tree) if isinstance(n,ast.Call)}
        ok('source_no_solver_native_or_dynamic_call',not calls.intersection({'linprog','minimize','mj_jac','mj_collision','forward','step'}))
        ok('no_Torch_Warp_MuJoCo_imported',not any(k in sys.modules for k in ('torch','warp','mujoco')))
        ok('all_sources_unchanged_after_execution',all(sha(p['path'])==p['sha256'] for p in pins.values()))
        details.update({'geometry_rows':3,'global_variables':9061,'global_rows':48043,'LP_calls':0,'real_path_reads':0,'FK':0,'collision':0,'Env':0,'physics':0,'WM':0,'GPU':0,'scope':'Synthetic sparse and floating-point algebra only; no actual contact Jacobian or physical geometry is certified.'})
        status='PASS'
    except Exception:
        status='FAILED_HARNESS_OR_CHECK';details['traceback']=traceback.format_exc()
    result={'schema':'full151_global_SCP_synthetic_CPU_proof:1','status':status,'source_pins':pins,'checks':checks,'passed':sum(checks.values()),'total':len(checks),'details':details}
    (output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n','utf-8')
    print(json.dumps({'status':status,'passed':result['passed'],'total':result['total'],'report':pin(output/'report.json')}))
    return 0 if status=='PASS' else 1


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args();raise SystemExit(main(Path(args.output).resolve()))
