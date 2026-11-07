"""Small integration checks only: one16+61 geometry evaluation, no LP/solve."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import argparse,hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
import mujoco
import scipy.optimize
from runtime_io import write_json
from constrained_prefix.problem import ConstrainedPrefixGeometryProblem,PLANNING_DEPTH_BOUND_M,CATEGORY_ORDER
from constrained_prefix.guard import geometry_only_guard,FORBIDDEN
from constrained_prefix.io import atomic_npz,atomic_jsonl,sha
import bspline_control as b


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();checks={};calls={'linprog':0,'minimize':0}
    originals={name:getattr(scipy.optimize,name) for name in calls}
    def block(name):
        def fail(*a,**kw):calls[name]+=1;raise AssertionError('CPU implementation check forbids '+name)
        return fail
    for name in calls:setattr(scipy.optimize,name,block(name))
    try:
        original_native={name:getattr(mujoco,name) for name in FORBIDDEN}
        with geometry_only_guard(mujoco) as forbidden:
            p=ConstrainedPrefixGeometryProblem(b.make_basis())
            with p.stage('implementation_checks_single_repeated_q0'):
                r,J,local=p.fit(p.x0);depth,C,samples=p.nonlinear(p.x0)
            after=p.parent.g.collision_query_count
            checks['shape_q16_20']=p.q_from_x(p.x0).shape==(16,20)
            checks['shape_q61_20']=p.sample_q(p.x0,p.B61).shape==(61,20)
            checks['hardP0_outside340_and_exact']=p.controls_from_x(p.x0).shape==(18,20) and np.array_equal(p.controls_from_x(p.x0)[0],p.q0)
            checks['repeat_q0_fit16_bitexact']=np.array_equal(p.q_from_x(p.x0),np.tile(p.q0,(16,1)))
            checks['repeat_q0_sample61_bitexact']=np.array_equal(p.sample_q(p.x0,p.B61),np.tile(p.q0,(61,1)))
            checks['fit6596_J6596x340']=r.shape==(6596,) and J.shape==(6596,340) and J.format=='csr'
            checks['constraint122_J122x340']=depth.shape==(122,) and C.shape==(122,340) and C.format=='csr'
            parent_r,parent_J,_=p.parent.evaluate(p.q_from_x(p.x0)[1:].ravel())
            checks['fit_residual_unchanged_parent_exact']=np.array_equal(r,parent_r)
            expected=parent_J@p.fit_chain
            checks['fit_chain_sparse_exact']=(J-expected).nnz==0
            checks['basis_chain_partition_and_hardq0']=p.fit_chain.shape==(300,340) and np.array_equal(p.B16[0],np.r_[1.,np.zeros(17)]) and np.array_equal(p.B61[0],np.r_[1.,np.zeros(17)])
            checks['fit_and_constraint_J_finite']=np.isfinite(r).all() and np.isfinite(depth).all() and np.isfinite(J.data).all() and np.isfinite(C.data).all()
            checks['planned_margin_fixed_1um']=PLANNING_DEPTH_BOUND_M==.002999
            checks['absent_pair_gap_and_J_remain_NaN']=all(np.isnan(a['evaluation'].signed_distance_m[~a['evaluation'].generated_mask]).all() and np.isnan(a['evaluation'].selected_distance_jacobian[~a['evaluation'].generated_mask]).all() for a in samples)
            checks['absent_category_operational_zero_not_gap']=all(np.all(a['operational_depth_m'][~a['category_generated']]==0) and np.all(a['depth_jac_q'][~a['category_generated']]==0) and np.all(a['selected_pair_row'][~a['category_generated']]==-1) for a in samples)
            expected_depth=[];selection_ok=True;ties_ok=True
            for a in samples:
                e=a['evaluation'];current=[]
                for k,kind in enumerate(CATEGORY_ORDER):
                    rows=np.flatnonzero(e.generated_mask&(p.parent.g.pair_kind==kind))
                    current.append(max(0.,-float(e.signed_distance_m[rows].min())) if len(rows) else 0.)
                    if len(rows):
                        selected=int(rows[np.argmin(e.signed_distance_m[rows])]);selection_ok &= selected==a['selected_pair_row'][k]
                        raw=[x for x in e.native_contact_records if p.parent.g.pair_kind[x[0]]==kind and abs(x[4]-e.signed_distance_m[selected])<=1e-9]
                        ties_ok &= [(int(x[0]),int(x[1])) for x in raw]==[(x['pair_row'],x['native_contact_index']) for x in a['category_global_tied_contacts'][k]]
                expected_depth.extend(current)
            checks['constraint_matches_generated_raw_min']=np.array_equal(np.asarray(expected_depth)/.001,depth)
            checks['deterministic_category_selection']=bool(selection_ok)
            checks['global_ties_recorded_across_pair_and_contact']=bool(ties_ok)
            checks['fixed_q0_constraint_rows_zero_J']=C[:2].nnz==0
            p.fit(p.x0);p.nonlinear(p.x0);arrays=p.diagnostic_arrays(p.x0);summary=p.diagnostic_summary(p.x0)
            checks['repeated_calls_full_diagnostics_no_new_query']=p.parent.g.collision_query_count==after
            checks['geometry_event_count_equals_actual_queries']=len(p.events)==after and sum(p.stage_counts.values())==after
            checks['same_original_CCD_options_and_clock0']=p.parent.g.data.time==0 and not np.any(p.parent.g.data.qvel) and p.parent.g.query_options==p.parent.g.metadata['query_options']
            checks['object_quintic_unchanged_knot_roundoff_gate']=all(v<1e-10 for v in p.object_knot_roundoff.values())
            checks['source_reconstruction_and_no_oldq_read']=all(p.parent.data['checks'].values())
            checks['actual_geometry_no_forbidden_calls']=all(v==0 for v in forbidden.values())
            bad=[]
            for value in (np.zeros(339),np.r_[np.nan,np.zeros(339)]):
                try:p.q_from_x(value);bad.append(False)
                except ValueError:bad.append(True)
            checks['wrongshape_and_nonfinite_freevars_rejected']=all(bad)
            atomic_npz(out/'diagnostic_arrays.npz',**arrays);write_json(out/'initial_geometry_summary.json',summary);write_json(out/'native_contacts.json',p.native_records(p.x0))
            counts=p.counters();actual_forbidden=dict(forbidden)
        checks['geometry_guard_restored_original_functions']=all(getattr(mujoco,k) is v for k,v in original_native.items())
        with geometry_only_guard(mujoco) as denied:
            for name in FORBIDDEN:
                try:getattr(mujoco,name)();raise RuntimeError('Guard not enforced')
                except AssertionError:pass
            checks['all7_forbidden_guard_negative_cases']=all(v==1 for v in denied.values())
        atomic_npz(out/'io_nan_roundtrip.npz',nan_gap=np.array([np.nan]),x=np.arange(3.))
        with np.load(out/'io_nan_roundtrip.npz',allow_pickle=False) as z:checks['samepayload_NPZ_preserves_unknown_NaN']=bool(np.isnan(z['nan_gap'][0]) and np.array_equal(z['x'],np.arange(3.)))
        payload={'same_payload':True,'source':'check'};write_json(out/'io_commit.json',payload);before=sha(out/'io_commit.json');write_json(out/'io_commit.json',payload);checks['samepayload_JSON_deterministic']=before==sha(out/'io_commit.json')
        try:write_json(out/'invalid_nonfinite.json',{'not_finite':float('nan')});checks['nonfinite_JSON_rejected']=False
        except ValueError:checks['nonfinite_JSON_rejected']=True
        atomic_jsonl(out/'events.jsonl',p.events)
        checks['no_LP_or_optimizer_called']=all(v==0 for v in calls.values())
        checks['no_Torch_or_Warp_imported']='torch' not in sys.modules and 'warp' not in sys.modules and 'mujoco_warp' not in sys.modules
        report={'status':'PASSED_SMALL_IMPLEMENTATION_CHECKS_NO_OPTIMIZATION' if all(checks.values()) else 'FAILED_IMPLEMENTATION_CHECKS','checks':{k:bool(v) for k,v in checks.items()},'passed':sum(bool(v) for v in checks.values()),'total':len(checks),'counts':counts,'actual_geometry_forbidden_calls':actual_forbidden,'negative_guard_expected_attempts':dict(denied),'solver_calls':calls,'physics_steps':0,'GPU_used':False,'source_pins':{str(f):sha(f) for f in [Path(__file__),ROOT/'src/constrained_prefix/problem.py',ROOT/'src/constrained_prefix/guard.py',ROOT/'src/bspline_control.py']},'scope':'One repeated-q0 16fit/61sample integration evaluation and algebra/cache/IO/guard negatives. No FD global contact-Jac certificate, LP, formal solve or physical qualification.','wall_s':time.perf_counter()-started}
        write_json(out/'report.json',report);print(json.dumps(report,indent=2));return 0 if all(checks.values()) else 1
    finally:
        for name,value in originals.items():setattr(scipy.optimize,name,value)
if __name__=='__main__':raise SystemExit(main())
