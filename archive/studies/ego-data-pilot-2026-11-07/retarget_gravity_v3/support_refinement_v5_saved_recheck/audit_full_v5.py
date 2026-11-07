"""Complete saved v5 consistency review; priority seed0 is consumed unchanged."""
import collections,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;RUN=ROOT/'support_refinement_v5_run1';P=ROOT.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(p):return json.loads(Path(p).read_text())
def need(x,label):
    if not x:raise AssertionError(label)
def save(p,x):
    with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)


def main():
    need(not (OUT/'report.json').exists(),'fresh full review')
    helper=OUT/'audit_seed0_priority_v5.py';need(pin(helper)['sha256']=='2ba77c065967a33be85d9d0309f7064ea0106e9d426667c10cd564f4c3fabef1','frozen arithmetic source')
    spec=importlib.util.spec_from_file_location('priority_arithmetic_v5',helper);A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
    priority=OUT/'seed0_priority_report.json';need(pin(priority)['sha256']=='d3522ec6bbeff14b7fe1b9065349f41374e13f1db2d017c87e352983641c95fb','priority report pin')
    prior=read(priority);r=read(A.checked(prior['producer_report']));c=read(A.checked(r['protocol']))
    for v in [*c['inputs'].values(),*c['source_pins'].values()]:A.checked(v)
    need((RUN/'execution_source.py').read_bytes()==A.checked(c['source_pins']['runner']).read_bytes() and read(RUN/'protocol.json')==c,'executed source/protocol')
    terminal=read(ROOT/'refine_v5_process_terminal.json') if (ROOT/'refine_v5_process_terminal.json').exists() else None
    if terminal is None:
        candidates=[p for p in ROOT.glob('*v5*terminal*.json') if p.is_file()]
        need(len(candidates)==1,'unique explicitv5 terminal receipt');terminal_path=candidates[0];terminal=read(terminal_path)
    else:terminal_path=ROOT/'refine_v5_process_terminal.json'
    need(terminal['returncode']==0,'v5 exit0');A.checked(terminal['stdout']);A.checked(terminal['stderr']);need(terminal['stderr']['bytes']==0,'empty stderr')
    parent=read(P/'hand2_holding_v1/support_preparation_v1/actual_v1/report.json')
    with np.load(A.checked(parent['outputs']['support_operands_and_controls.npz']),allow_pickle=False) as z:old={k:z[k].copy() for k in z.files}
    cases=[prior['result']]
    seed1=next(s for s in r['seed_results'] if s['seed']==1);cases.append(A.review_state(seed1,old))
    need(cases[1]['minimum_torque_slack_Nm']>=-1e-9 and max(cases[1]['normal_force_N'])<=3+1e-9
         and np.min(seed1['support_LP']['force_recovery']['lambda_normal_N'])>=-1e-9,'seed1 recovery hard bounds')
    need(cases[0]['eligible_for_bounded_dynamic_diagnostic'] and not cases[1]['eligible_for_bounded_dynamic_diagnostic'],'onlyseed0 diagnostics eligible')
    rows=[json.loads(x) for x in (RUN/'all_evaluations.jsonl').read_text().splitlines()];iters=[json.loads(x) for x in (RUN/'actual_iterations.jsonl').read_text().splitlines()]
    need(sum(x['purpose']=='base' for x in rows)==r['counts']['base_evaluations'] and sum(x['purpose']=='finite_difference' for x in rows)==r['counts']['FD_evaluations'],'base/FD counts')
    statuses=dict(collections.Counter(x['current_support_status'] for x in rows));numerical=[x for x in rows if x['current_support_status']=='PRIMARY_NUMERICAL_HARD_BOUND_FAILURE']
    primary_statuses={'OPTIMAL_PRIMARY_MOMENT_RESIDUAL','PRIMARY_FORCE_HARD_LP_INFEASIBLE','PRIMARY_NUMERICAL_HARD_BOUND_FAILURE'}
    primary_saved=sum(x['current_support_status'] in primary_statuses for x in rows)
    recovery_saved=sum(x['current_support_status']=='PRIMARY_FORCE_HARD_LP_INFEASIBLE' for x in rows)
    final_primary=sum('schema' in s.get('support_LP',{}) for s in r['seed_results']);final_recovery=sum(s.get('support_LP',{}).get('force_recovery') is not None for s in r['seed_results'])
    need(r['counts']['primary_LP_attempts']==primary_saved+final_primary,'LP primary call accounting')
    need(r['counts']['LP_calls']==primary_saved+recovery_saved+final_primary+final_recovery,'LP recovery call accounting')
    stages=[]
    for S in r['seed_results']:
        need(S['selected_evaluation'] in rows and S['selected_evaluation']['purpose']=='base','selected saved base')
        for t in S['stages']:
            rr=[x for x in rows if x['seed']==S['seed'] and x['stage']==t['stage']];ii=[x for x in iters if x['seed']==S['seed'] and x['stage']==t['stage']]
            need(len(rr)==t.get('complete_evaluations',0) and len(ii)==t.get('callback_iterations',0),'stage complete/callback counts')
            need([x['iteration'] for x in ii]==list(range(1,len(ii)+1)),'callback numbering')
            if t.get('error'):need(t['failed_attempt_not_a_valid_pose_capture'] and t['failed_attempted_state_parameters'] is not None,'failed raw parameter retention')
            stages.append({'seed':S['seed'],'stage':t['stage'],'status':t['status'],'iterations':len(ii),'complete_rows':len(rr),'error':t.get('error')})
    for x in rows:
        m=x['moment_infinity_Nm'];side=x['actual_below_line_dot_m2_s2'];f=x['force_balance_feasible']
        expected=bool(x['physical_geometry_pass'] and f and m is not None and m<=1e-4 and side is not None and side>=0)
        need(x['support_proposal_pass']==expected,'row strict proposal')
        need(x['moment_search_constraint_active']==(x['stage']=='B' and m is not None),'moment search active')
        need(x['search_constraints_pass']==(x['max_scaled_constraint_violation']==0),'search inequality flag')
        if x['stage']=='B' and not f:need(not x['search_constraints_pass'],'recovery not search pass')
        if x['current_support_status']=='PRIMARY_NUMERICAL_HARD_BOUND_FAILURE':need(not x['support_proposal_pass'],'numerical hard failure not pass')
    D=r['static_mechanics_geometry_counts'];N=D['evaluations'];need(all(D[k]==N for k in ['mj_kinematics','mj_comPos','mj_comVel','mj_rne','mj_passive','mj_transmission']),'native pipeline counter equalities')
    need(D['anchor_mj_jac']==2*N and D['actual_pair_mj_jac']==D['closest_pair_unique'],'Jacobian counter identities')
    need(D['heavy_queries_attempted']==D['heavy_queries_returned'] and D['unknown_queries']==0,'geometry query counts')
    stops=sum(bool(t.get('error')) for s in r['seed_results'] for t in s['stages']);need(N-len(rows)-len(r['seed_results'])==stops,'partial deadline native states')
    need(r['qualified_support_proposal_count']==0 and not r['holding_or_exact_statics_or_ego_policy_qualified'],'zero strict/holding qualifications')
    checks={'all_current_input_source_and_terminal_pins':True,'seed0_priority_primal_and_controller_bound':True,'seed1_recovery_primal_and_no_enrollment':True,
      'all_stage_and_base_FD_callback_counts':True,'all_primary_recovery_LP_call_accounting':True,'all_saved_row_strict_and_search_gate_logic':True,
      'native_counter_and_partial_deadline_state_consistency':True,'strict_vs_diagnostic_vs_holding_distinction':True,
      'no_model_FK_LP_solver_optimizer_or_neural_imports':not any(k in sys.modules for k in ['mujoco','torch','warp','mujoco_warp','scipy.optimize'])}
    out={'schema':'gravity_contact_v5_saved_independent_review:1','status':'PASS_SAVED_ONE_BOUNDED_DIAGNOSTIC_ELIGIBLE_NOT_HOLDING',
       'source':pin(__file__),'arithmetic_helper':pin(helper),'seed0_priority_review':pin(priority),'producer_report':prior['producer_report'],'protocol':r['protocol'],'terminal':pin(terminal_path),
       'checks':checks,'passed':sum(checks.values()),'total':len(checks),'selected_results':cases,'stage_summary':stages,'complete_rows':len(rows),'callbacks':len(iters),
       'counts':r['counts'],'native_static_counts':D,'row_support_status_counts':statuses,
       'PRIMARY_NUMERICAL_HARD_BOUND_FAILURE':{'count':len(numerical),'selected_count':sum(s['selected_evaluation']['current_support_status']=='PRIMARY_NUMERICAL_HARD_BOUND_FAILURE' for s in r['seed_results'])},
       'strict_support_proposal_count':0,'eligible_for_bounded_dynamic_diagnostic_seed_ids':[x['seed'] for x in cases if x['eligible_for_bounded_dynamic_diagnostic']],
       'holding_qualified':False,'scope':'Both final saved primal matrices/gates recomputed without model/FK/LP. Seed0 priority result consumed unchanged; seed1 force recovery remains unqualified. Intermediate lambda/dual solutions were not retained, so intermediate optimality/infeasibility is source/solver evidence, not independent re-solve. Geometry distances/Jacobians are saved source/prior-contract evidence. Bounded nominal diagnostic enrollment is neither hardware safety,2s holding nor strict static support.'}
    save(OUT/'report.json',out);print(json.dumps({'report':pin(OUT/'report.json'),'checks':f"{out['passed']}/{out['total']}",'eligible_seed_ids':out['eligible_for_bounded_dynamic_diagnostic_seed_ids'],'strict':0,'holding':False,'numerical_failures':len(numerical)}))


if __name__=='__main__':main()
