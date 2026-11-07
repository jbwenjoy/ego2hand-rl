"""Bounded R -> feasible F producer; default check never runs any optimizer."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import argparse,hashlib,json,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json
STARTED=time.perf_counter()
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path)
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--check',action='store_true');modes.add_argument('--run',action='store_true');args=ap.parse_args()
    planpath=args.plan.resolve(strict=True);plan_sha=sha(planpath);rawplan=json.loads(planpath.read_text(encoding='utf-8-sig'))
    if args.run and rawplan.get('optimization_authorized') is not True:raise ValueError('Root authorization absent; no solve or output')
    if args.run and args.output is None:raise ValueError('--output required for authorized run')
    out=args.output.resolve() if args.output else None
    if out is not None:out.mkdir(parents=True,exist_ok=False)
    runtime={'schema':'restoration_prefix_cpu_worker_runtime:1','status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'started_perf_counter':STARTED,'plan_sha256':plan_sha,'entry_sha256':sha(__file__),'LP_calls':0,'R_SLSQP_calls':0,'R_objective_fun_calls':0,'R_objective_jac_calls':0,'R_constraint_fun_calls':0,'R_constraint_jac_calls':0,'R_linear_constraint_fun_calls':0,'R_linear_constraint_jac_calls':0,'R_callbacks':0,'F_QP_calls':0,'QP_objective_fun_calls':0,'QP_objective_jac_calls':0,'QP_callback_calls':0,'F_major_count':0,'F_alpha_trials':0,'F_actual_trials':0,'F_accepted_steps':0,'unique_x_vectors':0,'unique_z_vectors':0,'true_state_archives':0,'physics_steps':0,'GPU_used':False,**{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}}
    if out is not None:write_json(out/'runtime.json',runtime)
    # Publish actual PID before NumPy/SciPy/native constructors for the owner.
    sys.path.insert(0,str(ROOT/'src'))
    import numpy as np
    from scipy.optimize import minimize,Bounds
    import mujoco
    from restoration_prefix.inputs import checked_plan,RETARGET_VARIANT_ID
    from restoration_prefix.problem import RestorationPrefixProblem
    from restoration_prefix import math as m
    from restoration_prefix.io import atomic_npz,atomic_bytes,atomic_jsonl
    from constrained_prefix.guard import geometry_only_guard
    problem=None;events=[];qp_events=[];archives={};x_seen=set();z_seen=set();last_accepted=None;accepted_seq=0;R_result=None;R_terminal_record=None;F_stop=None
    class RObjectiveBudgetReached(RuntimeError):pass
    class QPObjectiveBudgetReached(RuntimeError):pass
    def deadline():
        if time.perf_counter()-STARTED>340:raise TimeoutError('340s whole-worker soft limit; partial only, no canonical/retry')
    def geometry_event(event):
        if out is not None:write_json(out/'collision_events'/f"{event['query_index']:07d}.json",event)
    def register_x(x):
        x_seen.add(hashlib.sha256(np.asarray(x,dtype=float).tobytes()).hexdigest());runtime['unique_x_vectors']=len(x_seen)
    def state_at(x,stage,*,fresh=False):
        deadline();register_x(x)
        with problem.stage(stage):return problem.evaluate(x,force_constraint_refresh=fresh)
    def archive(state):
        generation=state['generation']
        if generation in archives:return archives[generation]
        deadline()
        folder=out/'states'/f"{generation:05d}_{hashlib.sha256(state['x'].tobytes()).hexdigest()[:16]}"
        folder.mkdir(parents=True,exist_ok=False)
        # Called first at the freshly evaluated state; the underlying numerical
        # fit/61 cache is reused, not an uncounted physical or geometric step.
        with problem.stage('archive_cached_state'):
            arrays=problem.diagnostic_arrays(state);native=problem.geometry.native_records(state['x'])
        arrays.update(retarget_variant_id=np.array(RETARGET_VARIANT_ID),archive_generation=np.array(generation),archive_plan_sha256=np.array(plan_sha),archive_protocol_sha256=np.array(p['protocol']['sha256']),archive_pid=np.array(os.getpid()),archive_parent_pid=np.array(os.getppid()),archive_wall_seconds=np.array(time.perf_counter()-STARTED))
        atomic_npz(folder/'state.npz',**arrays);write_json(folder/'native_contacts.json',native)
        record={'generation':generation,'x_sha256':hashlib.sha256(state['x'].tobytes()).hexdigest(),'plan_sha256':plan_sha,'pid':os.getpid(),'wall_seconds':time.perf_counter()-STARTED,'state_npz':pin(folder/'state.npz'),'native_contacts':pin(folder/'native_contacts.json'),'metrics':state['record'],'creation_query_delta':state['creation_query_delta'],'original_primal':problem.primal(state),'R_trust_original_primal':problem.primal(state,R_trust=True),'counts':problem.counters()}
        write_json(folder/'record.json',record)
        archives[generation]={'record':pin(folder/'record.json'),'npz':record['state_npz'],'native_contacts':record['native_contacts']};runtime['true_state_archives']=len(archives)
        return archives[generation]
    def trace(kind,state,**extra):
        ref=archive(state)
        row={'event':len(events)+1,'kind':kind,'wall_seconds':time.perf_counter()-STARTED,'state':ref['record'],'generation':state['generation'],'original_fit_cost':state['cost'],'max_depth_mm':state['record']['max_generated_depth_mm'],'max_positive_violation_mm':state['record']['max_positive_planning_violation_mm'],'sum_positive_violation_mm':state['record']['sum_positive_planning_violation_mm'],'J_and_branch_signatures':state['record']['signature'],**extra}
        events.append(row);write_json(out/'trace_events'/f"{row['event']:06d}.json",row);return row
    def checkpoint(state,source,*,accepted=False,z=None,extra=None):
        deadline();ref=archive(state)
        stem='last_accepted' if accepted else 'latest_iterate'
        atomic_bytes(out/(stem+'.npz'),Path(ref['npz']['path']).read_bytes())
        header={'source':source,'is_algorithmically_accepted':accepted,'is_physical_qualification':False,'accepted_seq':accepted_seq if accepted else None,'pid':os.getpid(),'parent_pid':os.getppid(),'plan_sha256':plan_sha,'protocol_sha256':p['protocol']['sha256'],'retarget_variant_id':RETARGET_VARIANT_ID,'wall_seconds':time.perf_counter()-STARTED,'checkpoint_npz':pin(out/(stem+'.npz')),'state':ref,'original_primal':problem.primal(state),'R_trust_original_primal':problem.primal(state,R_trust=True),'R_augmented_z':None if z is None else np.asarray(z).tolist(),'metrics':{k:v for k,v in state['record'].items() if k!='category_branches'},'counts':problem.counters(),'runtime_counts':dict(runtime),**(extra or {})}
        write_json(out/(stem+'.json'),header)
        runtime['last_accepted_checkpoint' if accepted else 'latest_checkpoint']={k:header[k] for k in ('source','accepted_seq','pid','plan_sha256','wall_seconds','checkpoint_npz','original_primal')}
        write_json(out/'runtime.json',runtime);atomic_jsonl(out/'trace.jsonl',events)
    def stop_without_canonical(status,**extra):
        runtime['status']=status
        write_json(out/'report.json',{'status':status,'canonical_candidate':None,'latest_R_or_trial_partial':runtime.get('latest_checkpoint'),'last_accepted_partial':runtime.get('last_accepted_checkpoint'),'R_terminal':R_terminal_record,'F_stop':F_stop,'retarget_variant_id':RETARGET_VARIANT_ID,'protocol':p['protocol'],'plan_sha256':plan_sha,'counts':problem.counters() if problem else None,'physical_reference_qualified':False,**extra})
    def save_optimizer_result(path,result):
        arrays={k:np.asarray(result[k]) for k in ('x','fun','jac','multipliers') if k in result};atomic_npz(path,**arrays)
        return {'success':bool(result.success),'status':int(result.status),'message':str(result.message),'nit':int(result.nit),'nfev':int(result.nfev),'njev':int(result.njev) if 'njev'in result else None,'raw_return':pin(path)}
    def final_candidate(state):
        # A fresh reference geometry check never chooses a different state.
        final=state_at(state['x'],'final_last_accepted_fresh61',fresh=True);audit=problem.primal(final)
        trace('FINAL_FRESH61_SAME_ALGORITHMIC_STATE',final,primal=audit,accepted_seq=accepted_seq,accepted_cost_before_fresh=state['cost'],final_cost_minus_accepted_cost=final['cost']-state['cost'])
        ref=archive(final);deadline()
        atomic_bytes(out/'candidate.npz',Path(ref['npz']['path']).read_bytes());atomic_bytes(out/'native_contacts.json',Path(ref['native_contacts']['path']).read_bytes())
        if not np.array_equal(final['x'],last_accepted['x']):raise ValueError('Finalfresh illegally changed algorithmic coefficient vector')
        atomic_npz(out/'pipeline_return.npz',final_x=final['x'],last_accepted_x=last_accepted['x'],R_terminal_x=R_state_terminal['x'],F_accepted_steps=np.array(runtime['F_accepted_steps']),accepted_seq=np.array(accepted_seq),F_stop=np.array(F_stop),normal_termination=np.array(True),source_plan_sha256=np.array(plan_sha),protocol_sha256=np.array(p['protocol']['sha256']),retarget_variant_id=np.array(RETARGET_VARIANT_ID))
        canonical_provenance={'source':'normally_terminated_pipeline_final_x','normal_termination':True,'selected_phase':'F' if runtime['F_accepted_steps'] else 'R_start_F_NO_ACCEPTED_STEP','selection':'last_algorithmically_accepted_not_best_history','pipeline_return':pin(out/'pipeline_return.npz'),'final_x_sha256':hashlib.sha256(final['x'].tobytes()).hexdigest(),'last_accepted_state':archive(last_accepted),'accepted_seq':accepted_seq,'F_accepted_steps':runtime['F_accepted_steps'],'F_stop':F_stop}
        summary=problem.geometry.diagnostic_summary(final['x'])
        R_start_cost=R_state_terminal['cost']
        summary.update(schema='restoration_prefix_geometry_result:1',status='NORMAL_RESTORATION_FILTER_TERMINATION_ONLY',retarget_variant_id=RETARGET_VARIANT_ID,plan_sha256=plan_sha,protocol=p['protocol'],seed_candidate=p['seed_candidate'],candidate=pin(out/'candidate.npz'),R_terminal=R_terminal_record,F_stop=F_stop,F_accepted_steps=runtime['F_accepted_steps'],accepted_seq=accepted_seq,F_label='F_NO_ACCEPTED_STEP' if runtime['F_accepted_steps']==0 else 'F_STRICT_FEASIBLE_FIT_DESCENT_SEQUENCE',fit_sequence_had_strict_decrease=runtime['F_accepted_steps']>0,fit_improvement_claim=runtime['F_accepted_steps']>0 and final['cost']<R_start_cost,R_start_fit_cost=R_start_cost,final_cost_below_R_start=final['cost']<R_start_cost,final_original_primal=audit,finalfresh_may_fail_without_replacing_state=True,canonical_selection='Last algorithmically accepted F state, or R terminal at F0; no historical quality selection. Finalfresh retains same coefficients even if geometry qualification fails. Canonical requires a completed normal worker plus committed manifest; interrupted uncommitted files are partial only.',arithmetic_tolerance_pass_is_not_original_exact_pass=not audit['exact_raw_all_pass'] and audit['passed'],original_3mm_sampled_generated_qualification=bool(np.max(final['depth_mm'])<=3.),controller_FP32_qualification='PENDING_INDEPENDENT_EXPORT',physical_reference_qualified=False,full_motion_qualified=False,counts=problem.counters(),runtime_counts={k:v for k,v in runtime.items() if k.endswith('_calls') or k in ('F_accepted_steps','F_major_count','F_actual_trials','F_alpha_trials')})
        summary.update(source_plan=pin(planpath),canonical_candidate=canonical_provenance,pipeline_return=pin(out/'pipeline_return.npz'))
        write_json(out/'report.json',summary)
        write_json(out/'manifest.json',{'schema':'restoration_prefix_geometry_candidate:1','status':'NORMAL_ALGORITHMIC_LAST_ACCEPTED_ONLY','retarget_variant_id':RETARGET_VARIANT_ID,'protocol':p['protocol'],'target_protocol':p['target_protocol'],'parent_protocol':p['parent_protocol'],'source_identity':protocol['source_identity'],'source_plan':pin(planpath),'canonical_candidate':canonical_provenance,'pipeline_return':pin(out/'pipeline_return.npz'),'candidate':pin(out/'candidate.npz'),'report':pin(out/'report.json'),'q0_source':p['inputs']['canonical_H123_q0'],'rawFD':p['inputs']['rawFD_design_arrays'],'mapped_pose_parent':p['inputs']['frozen_full151_geometry'],'material':p['inputs']['frozen_material_anchors'],'seed_candidate':p['seed_candidate'],'linear_contract':p['linear_contract'],'source_pins':p['source_pins'],'F_accepted_steps':runtime['F_accepted_steps'],'canonical_selection':summary['canonical_selection'],'physical_reference_qualified':False,'known_contacts':False,'qd_or_u_exported':False})
        runtime['status']='completed_normal_R_F_geometry_only';return summary
    with geometry_only_guard(mujoco) as forbidden:
        runtime['forbidden_calls']=forbidden
        try:
            p,protocol,parent=checked_plan(planpath,run=args.run)
            if p.get('output') is not None and out is not None and Path(p['output']).resolve()!=out:raise ValueError('Output differs from plan')
            if out is not None:write_json(out/'plan_snapshot.json',p);write_json(out/'protocol_snapshot.json',protocol)
            deadline();problem=RestorationPrefixProblem(p['seed_candidate']['path'],p['linear_contract']['path'],parent_protocol=parent,event_sink=geometry_event,deadline=deadline)
            seed=state_at(problem.seed,'default_check_frozen_terminal_seed16_and61');z0=m.restoration_z0(problem.seed,seed['depth_mm'])
            if not args.run:
                runtime.update(status='checked_no_LP_no_optimizer',counts=problem.counters())
                result={'status':'CPU_DEFAULT_CHECK_ONLY_NO_SOLVER','seed_original_primal':problem.primal(seed),'R_seed_trust_primal':problem.primal(seed,R_trust=True),'R_z0_slack_mm':float(z0[-1]),'R_augmented_geometry_slack_min_mm':float(m.restoration_geometry_slack(z0,seed['depth_mm']).min()),'seed_original_fit_cost':seed['cost'],'counts':problem.counters(),'LP_calls':0,'R_SLSQP_calls':0,'F_QP_calls':0,'physics_steps':0,'GPU_used':False}
                if out is not None:trace('DEFAULT_CHECK_NO_SOLVER',seed);checkpoint(seed,'check_frozen_seed',z=z0);write_json(out/'check.json',result)
                print(json.dumps(result,indent=2));return
            atomic_npz(out/'R_requested_initialization.npz',z0=z0,seed_x=problem.seed,R_lower=problem.R_lower,R_upper=problem.R_upper)
            trace('R_INITIAL_SEED',seed,R_slack_mm=float(z0[-1]),R_original_primal=problem.primal(seed,R_trust=True));checkpoint(seed,'R_requested_seed',z=z0)
            runtime.update(status='R_restoration',R_SLSQP_calls=1);write_json(out/'runtime.json',runtime)
            R_first=None;A=problem.contract['A_ub'];Ad=A.toarray();b=problem.contract['b_centered'];xr=problem.contract['x0'];linear_jac=np.c_[-Ad,np.zeros(A.shape[0])]
            def R_state(z,stage):
                z=m.finite(z,(341,));z_seen.add(hashlib.sha256(z.tobytes()).hexdigest());runtime['unique_z_vectors']=len(z_seen);return state_at(z[:340],stage)
            def R_fun(z):
                nonlocal R_first
                deadline()
                if runtime['R_objective_fun_calls']>=200:raise RObjectiveBudgetReached('R200 objective calls exhausted; no terminal/canonical/F')
                runtime['R_objective_fun_calls']+=1;state=R_state(z,'R_objective_fun')
                if R_first is None:R_first=np.asarray(z).copy();atomic_npz(out/'R_solver_initialization.npz',requested_z0=z0,first_fun_z=R_first,free_optimizer_initial_shift=R_first-z0)
                trace('R_OBJECTIVE_TRIAL',state,R_call=runtime['R_objective_fun_calls'],R_slack_mm=float(z[-1]),R_augmented_geometry_min_mm=float(m.restoration_geometry_slack(z,state['depth_mm']).min()),R_original_primal=problem.primal(state,R_trust=True))
                return m.restoration_objective(z)
            def R_jac(z):
                deadline();runtime['R_objective_jac_calls']+=1;state=R_state(z,'R_objective_jac');trace('R_OBJECTIVE_JAC',state,R_slack_mm=float(z[-1]));checkpoint(state,'R_objective_jac_trial',z=z);return m.restoration_gradient(z)
            def R_constraints(z):
                deadline();runtime['R_constraint_fun_calls']+=1;state=R_state(z,'R_constraint_fun');return m.restoration_geometry_slack(z,state['depth_mm'])
            def R_Cjac(z):
                deadline();runtime['R_constraint_jac_calls']+=1;state=R_state(z,'R_constraint_jac');checkpoint(state,'R_constraint_jac_trial',z=z);return m.restoration_geometry_jacobian(state['C'])
            def R_callback(z):
                deadline();runtime['R_callbacks']+=1;state=R_state(z,'R_major_callback');trace('R_MAJOR_ITERATE_NOT_ACCEPTED',state,R_major=runtime['R_callbacks'],R_slack_mm=float(z[-1]),R_original_primal=problem.primal(state,R_trust=True));checkpoint(state,'R_major_iterate_not_feasible_accepted',z=z)
            def R_linear_fun(z):deadline();runtime['R_linear_constraint_fun_calls']+=1;return b-A@(z[:340]-xr)
            def R_linear_jac(z):deadline();runtime['R_linear_constraint_jac_calls']+=1;return linear_jac
            R_result=minimize(R_fun,z0.copy(),jac=R_jac,method='SLSQP',bounds=Bounds(np.r_[problem.R_lower,0.],np.r_[problem.R_upper,np.inf]),constraints=[{'type':'ineq','fun':R_linear_fun,'jac':R_linear_jac},{'type':'ineq','fun':R_constraints,'jac':R_Cjac}],callback=R_callback,options={'maxiter':60,'ftol':1e-9,'disp':False})
            R_meta=save_optimizer_result(out/'R_optimizer_return.npz',R_result);R_controls=np.asarray(R_result.x[:340]);R_state_terminal=state_at(R_controls,'R_terminal_independent_fresh61',fresh=True);R_primal=problem.primal(R_state_terminal,R_trust=True)
            R_terminal_record={'optimizer':R_meta,'slack_mm':float(R_result.x[-1]),'original_primal_after_fresh61':R_primal,'decision_ignores_optimizer_success_and_slack':True,'actual_creation_query_delta':R_state_terminal['creation_query_delta'],'state':archive(R_state_terminal),'max_R_initial_shift':float(np.max(np.abs(R_first-z0)))}
            write_json(out/'R_terminal.json',R_terminal_record);trace('R_NORMAL_TERMINAL_FRESH61_RAW_GATE',R_state_terminal,R_terminal=R_terminal_record);checkpoint(R_state_terminal,'R_normal_terminal_diagnostic',z=R_result.x)
            if not R_primal['passed']:
                stop_without_canonical('R_NORMAL_TERMINAL_ORIGINAL_PRIMAL_FAILED_NO_F');return
            last_accepted=R_state_terminal;accepted_seq=0;trace('R_TERMINAL_BECOMES_ALGORITHMIC_START',last_accepted,accepted_seq=0);checkpoint(last_accepted,'R_raw_feasible_start_F0',accepted=True,z=R_result.x)
            runtime['status']='F_feasible_fit_filter';write_json(out/'runtime.json',runtime)
            for major in range(1,9):
                deadline();runtime['F_major_count']=major;base=state_at(last_accepted['x'],f'F_major{major}_base')
                base_audit=problem.primal(base)
                if not base_audit['passed']:F_stop='CURRENT_BASE_PRIMAL_NOT_ESTABLISHED';trace('F_STOP_BASE_PRIMAL_FAILED',base,F_major=major,base_primal=base_audit);break
                g,H=m.gauss_newton_model(base['r'],base['J']);qp=m.qp_contract(base['x'],base['depth_mm'],base['C'],problem.contract)
                major_dir=out/'F_majors'/f'{major:02d}';major_dir.mkdir(parents=True,exist_ok=False)
                atomic_npz(major_dir/'fixed_QP_model.npz',x_base=base['x'],r=base['r'],g=g,H=H,C_data=base['C'].data,C_indices=base['C'].indices,C_indptr=base['C'].indptr,C_shape=np.array(base['C'].shape),depth_mm=base['depth_mm'],step_lower=qp['lower'],step_upper=qp['upper'],linear_rhs=qp['b_original'],geometry_rhs=qp['b_geometry'])
                base_event=trace('F_MAJOR_BEGIN_ACTUAL_BASE',base,F_major=major,original_primal=base_audit,fixed_QP_model=pin(major_dir/'fixed_QP_model.npz'))
                qpc={'fun':0,'jac':0,'callback':0};qp_local=[];QAd=qp['A_original'].toarray();QCd=qp['C_geometry'].toarray();qp_started=time.perf_counter();runtime['F_QP_calls']+=1
                def qp_log(kind,step,**extra):
                    event={'F_major':major,'kind':kind,'wall_seconds':time.perf_counter()-STARTED,'scope':'FIXED_LINEARIZATION_QP_PREDICTION_ONLY_NO_NATIVE_GEOMETRY','base_state':base_event['state'],'base_J_signature':base['record']['signature'],'step':np.asarray(step).tolist(),**extra};qp_local.append(event);qp_events.append(event)
                    write_json(major_dir/'QP_events'/f'{len(qp_local):04d}.json',event)
                def qfun(step):
                    deadline()
                    if qpc['fun']>=100:raise QPObjectiveBudgetReached('OneQP100fun budget exhausted; normalFstop/no alternativeQP')
                    qpc['fun']+=1;runtime['QP_objective_fun_calls']+=1;step=m.finite(step,(340,));pred=base['r']+base['J']@step;value=float(.5*(pred@pred)+.5*m.GN_MU*(step@step));qp_log('objective_trial',step,call=qpc['fun'],predicted_original_plus_regularizer=value);return value
                def qjac(step):
                    deadline();qpc['jac']+=1;runtime['QP_objective_jac_calls']+=1;return g+H@m.finite(step,(340,))
                def qcallback(step):deadline();qpc['callback']+=1;runtime['QP_callback_calls']+=1;qp_log('QP_major_iterate_not_geometry',step,call=qpc['callback'])
                try:
                    qresult=minimize(qfun,np.zeros(340),jac=qjac,method='SLSQP',bounds=Bounds(qp['lower'],qp['upper']),constraints=[{'type':'ineq','fun':lambda step:qp['b_original']-qp['A_original']@step,'jac':lambda step:-QAd},{'type':'ineq','fun':lambda step:qp['b_geometry']-qp['C_geometry']@step,'jac':lambda step:-QCd}],callback=qcallback,options={'maxiter':25,'ftol':1e-9,'disp':False})
                except QPObjectiveBudgetReached as exc:
                    F_stop='QP_OBJECTIVE_BUDGET_100_NORMAL_STOP';write_json(major_dir/'QP_stop.json',{'reason':F_stop,'message':str(exc),'counts':qpc});trace('F_STOP_QP_BUDGET_NO_TERMINAL_PROPOSAL',base,F_major=major,QP_counts=qpc);break
                qmeta=save_optimizer_result(major_dir/'QP_optimizer_return.npz',qresult)
                try:proposal=m.qp_direction_audit(qresult.x,g,qp)
                except ValueError as exc:proposal={'eligible':False,'reason':'NONFINITE_OR_INVALID_QP_TERMINAL','error':str(exc)}
                if proposal.get('eligible') and np.array_equal(base['x']+qresult.x,base['x']):proposal['eligible']=False;proposal['reason']='QP_X_UNCHANGED_BY_ROUNDING'
                write_json(major_dir/'QP_terminal.json',{'optimizer':qmeta,'proposal':proposal,'counts':qpc,'wall_s':time.perf_counter()-qp_started,'solver_status_is_not_proposal_qualification':True})
                trace('F_NORMAL_QP_TERMINAL_PREDICTION_AUDIT',base,F_major=major,QP_optimizer=qmeta,proposal=proposal,QP_counts=qpc)
                if not proposal['eligible']:F_stop='QP_TERMINAL_NO_RELIABLE_DESCENT_PROPOSAL';break
                accepted=False
                for trial,alpha in enumerate(m.ALPHAS,1):
                    deadline();x_trial=base['x']+alpha*qresult.x;runtime['F_alpha_trials']+=1
                    if not np.isfinite(x_trial).all() or np.array_equal(x_trial,base['x']):
                        write_json(major_dir/f'alpha_{trial:02d}_invalid.json',{'alpha':alpha,'accepted':False,'reason':'NONFINITE_OR_UNCHANGED_X_NO_GEOMETRY_QUERY'});continue
                    runtime['F_actual_trials']+=1
                    trial_state=state_at(x_trial,f'F_major{major}_actual_alpha{trial}');audit=problem.primal(trial_state)
                    decision=m.armijo_decision(base['cost'],trial_state['cost'],alpha,proposal['g_dot_step'],audit,trial_x_changed=True)
                    trace('F_ACTUAL_ALPHA_TRIAL',trial_state,F_major=major,trial_index=trial,decision=decision,actual_original_primal=audit,base_cost=base['cost'],base_state=base_event['state'])
                    checkpoint(trial_state,f'F_major{major}_trial{trial}_not_accepted',extra={'decision':decision})
                    if decision['accepted']:
                        last_accepted=trial_state;accepted_seq+=1;runtime['F_accepted_steps']+=1;accepted=True
                        trace('F_ALGORITHMIC_ACCEPT_FIRST_VALID_ALPHA',last_accepted,F_major=major,accepted_seq=accepted_seq,decision=decision,actual_original_primal=audit)
                        checkpoint(last_accepted,f'F_major{major}_accepted_alpha{trial}',accepted=True,extra={'decision':decision});break
                if not accepted:F_stop='NO_ACCEPTED_FIXED_ALPHA_NORMAL_STOP';trace('F_MAJOR_END_NO_ACCEPTED_ALPHA',base,F_major=major,base_primal=base_audit);break
                trace('F_MAJOR_END_ACCEPTED',last_accepted,F_major=major,accepted_seq=accepted_seq,original_primal=problem.primal(last_accepted))
            else:F_stop='EIGHT_MAJOR_BUDGET_NORMAL_STOP'
            summary=final_candidate(last_accepted)
            print(json.dumps({'status':summary['status'],'F_stop':F_stop,'F_accepted_steps':runtime['F_accepted_steps'],'final_original_primal':summary['final_original_primal'],'candidate':summary['candidate']},indent=2))
        except RObjectiveBudgetReached as exc:
            stop_without_canonical('R_OBJECTIVE_BUDGET_PARTIAL_NO_F_NO_CANONICAL',reason=str(exc));return
        except BaseException as exc:
            runtime.update(status='failed_engineering_partial_NO_CANONICAL',error_type=type(exc).__name__,error=str(exc))
            if out is not None:write_json(out/'failure.json',{**runtime,'last_accepted_is_partial_not_canonical':runtime.get('last_accepted_checkpoint'),'canonical_candidate':None,'no_retry':True})
            raise
        finally:
            runtime.update(forbidden_calls=forbidden,Torch_imported='torch'in sys.modules,Warp_imported='warp'in sys.modules or 'mujoco_warp'in sys.modules)
            if out is not None:
                if problem is not None:runtime['counts']=problem.counters();atomic_jsonl(out/'collision_events.jsonl',problem.geometry.events)
                atomic_jsonl(out/'trace.jsonl',events);atomic_jsonl(out/'QP_predictions.jsonl',qp_events)
                runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write_json(out/'runtime.json',runtime)
if __name__=='__main__':main()
