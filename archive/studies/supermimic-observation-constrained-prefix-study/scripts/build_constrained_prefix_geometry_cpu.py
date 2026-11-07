"""Constrained prefix CPU producer. Default check never calls an LP/optimizer."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
import argparse,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json
STARTED=time.perf_counter()
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path)
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--run',action='store_true');modes.add_argument('--check',action='store_true');args=ap.parse_args()
    planpath=args.plan.resolve(strict=True);plan_sha=sha(planpath);rawplan=json.loads(planpath.read_text(encoding='utf-8-sig'))
    if args.run and rawplan.get('optimization_authorized') is not True:raise ValueError('Root authorization absent: no LP/output/optimization')
    if args.run and args.output is None:raise ValueError('--output required for authorized run')
    out=args.output.resolve() if args.output else None
    if out is not None:out.mkdir(parents=True,exist_ok=False)
    runtime={'schema':'constrained_prefix_geometry_worker_runtime:1','status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'started_perf_counter':STARTED,'plan_sha256':plan_sha,'entry_sha256':sha(__file__),'LP_calls':0,'SLSQP_calls':0,'objective_fun_calls':0,'objective_jac_calls':0,'constraint_fun_calls':0,'constraint_jac_calls':0,'callback_calls':0,'unique_trial_x':0,'physics_steps':0,'GPU_used':False,**{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    if out is not None:write_json(out/'runtime.json',runtime)
    # Actual PID is published before slow scientific imports/constructors.
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    import mujoco
    from scipy import sparse
    from constrained_prefix.inputs import checked_plan,RETARGET_VARIANT_ID
    from constrained_prefix.problem import ConstrainedPrefixGeometryProblem,PLANNING_DEPTH_BOUND_M,PARENT_ROOT
    from constrained_prefix.guard import geometry_only_guard
    from constrained_prefix.io import atomic_npz,atomic_jsonl
    import bspline_control as linear
    sys.path.insert(0,str(PARENT_ROOT/'src'))
    import prefix_control as control
    problem=None;history=[];trial_hashes=set();last_x=None;lp_result=None;result=None;first_formal_x=None;formal_x0=None
    def deadline():
        if time.perf_counter()-STARTED>220:raise TimeoutError('220s whole-worker soft limit; preserve partial, no retry')
    def event_sink(event):
        if out is not None:write_json(out/'collision_events'/f"{event['query_index']:07d}.json",event)
    def mark(x):
        nonlocal last_x
        x=np.asarray(x,dtype=float);last_x=x.copy();key=hashlib.sha256(x.tobytes()).hexdigest()
        if key not in trial_hashes:
            trial_hashes.add(key)
            if out is not None:atomic_npz(out/'trial_vectors'/(key+'.npz'),x=x,plan_sha256=np.array(plan_sha),pid=np.array(os.getpid()),wall_seconds=np.array(time.perf_counter()-STARTED))
        runtime['unique_trial_x']=len(trial_hashes)
    def linear_audit(x):
        delta=np.asarray(x)-contract['x0'];v=np.asarray(contract['A_ub']@delta-contract['b_centered']).ravel()
        bounds=np.maximum(np.maximum(contract['lower']-x,x-contract['upper']),0)
        return {'max_Ax_minus_b_centered':float(v.max()),'max_bound_excess':float(bounds.max()),'all_inequalities_nonpositive':bool(np.all(v<=0)),'all_bounds_exact':bool(np.all(bounds==0))}
    def checkpoint(x,source):
        deadline()
        if out is None:return
        with problem.stage('checkpoint_'+source):arrays=problem.diagnostic_arrays(x);summary=problem.diagnostic_summary(x);native=problem.native_records(x)
        if first_formal_x is not None:arrays['solver_first_fun_x']=first_formal_x
        if formal_x0 is not None:arrays['phase0_SLSQP_requested_x0']=formal_x0
        arrays.update(retarget_variant_id=np.array(RETARGET_VARIANT_ID),checkpoint_plan_sha256=np.array(plan_sha),checkpoint_protocol_sha256=np.array(p['protocol']['sha256']),checkpoint_pid=np.array(os.getpid()),checkpoint_parent_pid=np.array(os.getppid()),checkpoint_wall_seconds=np.array(time.perf_counter()-STARTED),checkpoint_source=np.array(source),checkpoint_is_feasible_accepted=np.array(False))
        atomic_npz(out/'latest_iterate.npz',**arrays)
        write_json(out/'latest_iterate_native_contacts.json',native)
        audit=linear_audit(x)
        header={'source':source,'source_is_feasible_accepted':False,'pid':os.getpid(),'parent_pid':os.getppid(),'plan_sha256':plan_sha,'protocol_sha256':p['protocol']['sha256'],'wall_seconds':time.perf_counter()-STARTED,'npz_sha256':sha(out/'latest_iterate.npz'),'native_contacts_sha256':sha(out/'latest_iterate_native_contacts.json'),'cost':summary['cost'],'sampled_generated_overlap_constraints_pass':summary['sampled_generated_overlap_constraints_pass'],'maximum_generated_overlap_m':summary['maximum_generated_overlap_m'],'linear_float64_audit':audit,'counts':problem.counters(),'runtime_counts':{k:runtime[k] for k in ('LP_calls','SLSQP_calls','objective_fun_calls','objective_jac_calls','constraint_fun_calls','constraint_jac_calls','callback_calls','unique_trial_x')},'scope':'jac/trial/major-iterate checkpoint, never assumed feasible or physically qualified'}
        write_json(out/'latest_iterate.json',header);runtime.update(latest_checkpoint=header,wall_seconds=time.perf_counter()-STARTED)
        write_json(out/'runtime.json',runtime);atomic_jsonl(out/'optimizer_iterations.jsonl',history)
    def finish_candidate(x,*,status):
        with problem.stage('final_diagnostics'):
            arrays=problem.diagnostic_arrays(x);report=problem.diagnostic_summary(x);native=problem.native_records(x);checkpoint(x,'final')
        arrays['retarget_variant_id']=np.array(RETARGET_VARIANT_ID)
        atomic_npz(out/'candidate.npz',**arrays);write_json(out/'native_contacts.json',native)
        report.update(status=status,retarget_variant_id=RETARGET_VARIANT_ID,plan_sha256=plan_sha,protocol_sha256=p['protocol']['sha256'],entry_sha256=sha(__file__),candidate_sha256=sha(out/'candidate.npz'),linear_float64_audit=linear_audit(x),LP_calls=runtime['LP_calls'],SLSQP_calls=runtime['SLSQP_calls'],runtime_counts={k:runtime[k] for k in ('objective_fun_calls','objective_jac_calls','constraint_fun_calls','constraint_jac_calls','callback_calls','unique_trial_x')},continuous_clearance_qualified=False,controller_FP32_qualification='PENDING_INDEPENDENT_EXPORT',physical_reference_qualified=False)
        if result is not None:
            report.update(optimizer_success=bool(result.success),optimizer_status=int(result.status),optimizer_message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),njev=int(result.njev),solver_first_fun_delta_from_LP_x0=float(np.max(np.abs(first_formal_x-formal_x0))))
            atomic_npz(out/'optimizer_return.npz',**{k:np.asarray(result[k]) for k in ('x','fun','jac','multipliers') if k in result})
            primal=report['sampled_generated_overlap_constraints_pass'] and report['linear_float64_audit']['all_inequalities_nonpositive'] and report['linear_float64_audit']['all_bounds_exact']
            report.update(canonical_selection='Only normally returned terminal SLSQP result.x, never a best historical iterate',primal_operational_plan_exact_float64_feasible=primal,
                          terminal_classification='FEASIBLE_OPTIMIZER_SUCCESS' if primal and result.success else 'FEASIBLE_NOT_OPTIMAL' if primal else 'PRIMAL_CONSTRAINTS_NOT_ESTABLISHED_REGARDLESS_OF_OPTIMIZER_STATUS')
        write_json(out/'report.json',report)
        write_json(out/'manifest.json',{'schema':'constrained_prefix_geometry_candidate:1','status':status,'retarget_variant_id':RETARGET_VARIANT_ID,'protocol':p['protocol'],'parent_protocol':p['parent_protocol'],'source_identity':protocol.get('source_identity',parent_protocol['source_identity']),'candidate':{'path':str(out/'candidate.npz'),'sha256':sha(out/'candidate.npz')},'report':{'path':str(out/'report.json'),'sha256':sha(out/'report.json')},'q0_source':p['inputs']['canonical_H123_q0'],'mapped_pose_parent':p['inputs']['frozen_full151_geometry'],'rawFD':p['inputs']['rawFD_design_arrays'],'material':p['inputs']['frozen_material_anchors'],'physical_reference_qualified':False,'known_contacts':False,'qd_or_u_exported':False,'source_pins':p['source_pins']})
        runtime['status']='completed_CPU_constrained_geometry_only'
        return report
    with geometry_only_guard(mujoco) as forbidden:
        runtime['forbidden_calls']=forbidden
        try:
            p,parent_protocol,protocol=checked_plan(planpath,run=args.run)
            if p.get('output') is not None and out is not None and Path(p['output']).resolve()!=out:raise ValueError('Output differs from plan')
            if out is not None:write_json(out/'plan_snapshot.json',p);write_json(out/'protocol_snapshot.json',protocol)
            deadline();basis=linear.make_basis();problem=ConstrainedPrefixGeometryProblem(basis,parent_protocol=parent_protocol,event_sink=event_sink,deadline=deadline)
            inputs=p['inputs'];params,pdproof=control.load_actual_pd(inputs['model']['path'],inputs['actual_context']['path'],model_sha256=inputs['model']['sha256'],context_sha256=inputs['actual_context']['sha256'])
            design=json.loads(Path(inputs['static_preload_design']['path']).read_text(encoding='utf-8-sig'))
            if design.get('schema')!='seeded_static_PD_preload:1' or design.get('feasible') is not True or not np.array_equal(np.asarray(design['q0_rad']),problem.q0):raise ValueError('Static prior q0 identity')
            tau=control.static_prior_numpy(problem.q0,np.asarray(design['u0_rad'],np.float32),params)
            contract=linear.build_linear_contract(problem.q0,params,tau,problem.parent.robot.lower,problem.parent.robot.upper)
            if not np.array_equal(contract['x0'],problem.x0):raise ValueError('Only repeated canonical q0 initial seed')
            if out is not None:
                atomic_npz(out/'linear_contract.npz',x0=contract['x0'],lower=contract['lower'],upper=contract['upper'],b_ub=contract['b_ub'],b_centered=contract['b_centered'],A_CSR_data=contract['A_ub'].data,A_CSR_indices=contract['A_ub'].indices,A_CSR_indptr=contract['A_ub'].indptr,A_shape=np.array(contract['A_ub'].shape),tau_ff_static0_FP32_Nm=tau)
                write_json(out/'linear_contract_metadata.json',{'row_groups':contract['row_groups'],'margins':contract['margins'],'actual_PD_model':pdproof})
            with problem.stage('check_repeated_q0_fit16_and_geometry61'):
                r,J,_=problem.fit(problem.x0);depth,C,_=problem.nonlinear(problem.x0);initial=problem.diagnostic_summary(problem.x0)
                if out is not None:checkpoint(problem.x0,'check_repeated_q0')
            if out is not None:write_json(out/'initial_checks.json',initial)
            if not args.run:
                runtime.update(status='checked_no_LP_no_optimization',wall_seconds=time.perf_counter()-STARTED,counts=problem.counters())
                print(json.dumps({'status':'CPU_CHECK_ONLY_NO_LP_NO_OPTIMIZATION','fit_shape':list(r.shape),'fit_J_shape':list(J.shape),'constraint_shape':list(depth.shape),'constraint_J_shape':list(C.shape),'hardq0_bitexact':initial['hardq0_bitexact'],'initial_generated_overlap_constraints_pass':initial['sampled_generated_overlap_constraints_pass'],'linear_initial_audit':linear_audit(problem.x0),'input_checks':problem.parent.data['checks'],'counts':problem.counters(),'LP_calls':0,'SLSQP_calls':0,'physics_steps':0,'GPU_used':False},indent=2));return
            # One aggregate L1 phase0 LP. No fallback seed, tie-break or retry.
            from scipy.optimize import linprog,minimize,Bounds
            lp=linear.phase0_l1_problem(contract);runtime.update(status='phase0_L1_LP',LP_calls=1);write_json(out/'runtime.json',runtime)
            deadline();lp_result=linprog(lp['c'],A_ub=lp['A_ub'],b_ub=lp['b_ub'],bounds=lp['bounds'],method='highs',options={'time_limit':max(.001,220-(time.perf_counter()-STARTED)),'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            lpmeta={'success':bool(lp_result.success),'status':int(lp_result.status),'message':str(lp_result.message),'fun':float(lp_result.fun) if lp_result.fun is not None else None,'nit':int(lp_result.nit),'method':'HiGHS one aggregate L1 nearest-repeat-q0 LP; no tie-break','LP_calls':1,'scope':'A linear-feasible initialization only; not nonlinear geometry feasibility or global infeasibility proof'}
            if lp_result.x is not None:
                atomic_npz(out/'phase0_LP_solution.npz',solution=lp_result.x,requested_repeat_q0=problem.x0,inequality_residual=lp_result.ineqlin.residual,lower_residual=lp_result.lower.residual,upper_residual=lp_result.upper.residual)
                lpmeta['linear_audit']=linear_audit(lp_result.x[:340])
            write_json(out/'phase0_LP.json',lpmeta);deadline()
            if not lp_result.success:
                write_json(out/'report.json',{'status':'PHASE0_LINEAR_INITIALIZATION_NOT_ESTABLISHED_NO_SLSQP','LP':lpmeta,'canonical_candidate':None,'partial_latest_checkpoint':runtime.get('latest_checkpoint'),'counts':problem.counters(),'plan_sha256':plan_sha,'protocol_sha256':p['protocol']['sha256'],'physical_reference_qualified':False})
                runtime['status']='completed_phase0_no_SLSQP_no_canonical_candidate';return
            formal_x0=np.asarray(lp_result.x[:340]).copy();checkpoint(formal_x0,'phase0_LP_initialization_not_nonlinear_feasible')
            runtime.update(status='formal_SLSQP',SLSQP_calls=1);write_json(out/'runtime.json',runtime)
            class ObjectiveBudgetReached(RuntimeError):pass
            def fun(x):
                nonlocal first_formal_x
                deadline()
                if runtime['objective_fun_calls']>=250:raise ObjectiveBudgetReached('250 formal objective function calls reached')
                runtime['objective_fun_calls']+=1;mark(x)
                if first_formal_x is None:
                    first_formal_x=np.asarray(x).copy();atomic_npz(out/'solver_initialization.npz',phase0_LP_x0=formal_x0,first_SLSQP_fun_x=first_formal_x,free_variable_bound_adjustment=first_formal_x-formal_x0,q0_source64=problem.q0)
                with problem.stage('formal_objective_fun'):r=problem.fit(x)[0]
                cost=float(.5*r@r);history.append({'kind':'objective_fun','call':runtime['objective_fun_calls'],'cost':cost,'wall_s':time.perf_counter()-STARTED});return cost
            def jac(x):
                deadline();runtime['objective_jac_calls']+=1;mark(x)
                with problem.stage('formal_objective_jac'):r,J,_=problem.fit(x);gradient=np.asarray(J.T@r).ravel();checkpoint(x,'objective_jac_trial')
                return gradient
            def constraint_fun(x):
                deadline();runtime['constraint_fun_calls']+=1;mark(x)
                with problem.stage('formal_constraint_fun'):depth=problem.nonlinear(x)[0]
                return PLANNING_DEPTH_BOUND_M/.001-depth
            def constraint_jac(x):
                deadline();runtime['constraint_jac_calls']+=1;mark(x)
                with problem.stage('formal_constraint_jac'):
                    J=problem.nonlinear(x)[1];checkpoint(x,'constraint_jac_trial')
                return -J.toarray()
            def callback(x):
                deadline();runtime['callback_calls']+=1;mark(x);checkpoint(x,'SLSQP_major_iterate_not_feasible_accepted')
            A=contract['A_ub'];bcenter=contract['b_centered'];A_dense=A.toarray()
            constraints=[{'type':'ineq','fun':lambda x:bcenter-A@(x-contract['x0']),'jac':lambda x:-A_dense},{'type':'ineq','fun':constraint_fun,'jac':constraint_jac}]
            try:
                result=minimize(fun,formal_x0.copy(),jac=jac,bounds=Bounds(contract['lower'],contract['upper']),constraints=constraints,method='SLSQP',callback=callback,options={'maxiter':80,'ftol':1e-8,'disp':False})
                report=finish_candidate(result.x,status='FORMAL_CPU_CONSTRAINED_GEOMETRY_ATTEMPT_COMPLETED_ONLY')
            except ObjectiveBudgetReached as exc:
                runtime['budget_termination']=str(exc)
                checkpoint(last_x,'objective_budget_partial_not_canonical')
                write_json(out/'report.json',{'status':'FORMAL_OBJECTIVE_BUDGET_REACHED_PARTIAL_UNQUALIFIED','canonical_candidate':None,'partial_latest_checkpoint':runtime.get('latest_checkpoint'),'counts':problem.counters(),'plan_sha256':plan_sha,'protocol_sha256':p['protocol']['sha256'],'physical_reference_qualified':False})
                runtime['status']='completed_objective_budget_partial_no_canonical_candidate';return
            print(json.dumps({k:report[k] for k in ('status','cost','maximum_generated_overlap_m','hardq0_bitexact','sampled_generated_overlap_constraints_pass')},indent=2))
        except BaseException as exc:
            runtime.update(status='failed_engineering_partial_UNKNOWN',error_type=type(exc).__name__,error=str(exc))
            if out is not None:write_json(out/'failure.json',{**runtime,'latest_iterate_may_exist':(out/'latest_iterate.npz').exists(),'no_retry':True,'physics_inference_qualified':False})
            raise
        finally:
            runtime.update(forbidden_calls=forbidden,Torch_imported='torch'in sys.modules,Warp_imported='warp'in sys.modules or 'mujoco_warp'in sys.modules)
            if out is not None:
                if problem is not None:runtime['counts']=problem.counters();atomic_jsonl(out/'collision_events.jsonl',problem.events)
                atomic_jsonl(out/'optimizer_iterations.jsonl',history)
                runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write_json(out/'runtime.json',runtime)
if __name__=='__main__':main()
