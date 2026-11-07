"""Independent source-only formal SCP gate. No production-module import."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import sys

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
OUT=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def allpins(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:yield value
        else:
            for v in value.values():yield from allpins(v)
    elif isinstance(value,list):
        for v in value:yield from allpins(v)
def samepin(value):
    actual=pin(value['path'])
    return Path(actual['path']).samefile(value['path']) and actual['sha256']==value['sha256']

source_path=ROOT/'scripts/run_contact_coupled_scp_cpu_v2.py'
old_path=ROOT/'scripts/run_contact_coupled_scp_cpu_v1.py'
proto_path=ROOT/'protocols/full151-coupled-scp-cpu-v2.json'
proto_old=ROOT/'protocols/full151-coupled-scp-cpu-v1.json'
s=source_path.read_text(encoding='utf-8-sig');old=old_path.read_text(encoding='utf-8-sig')
p=read(proto_path);pold=read(proto_old)
before={str(x):pin(x) for x in (source_path,old_path,proto_path,proto_old)}
text_before='''            accept=bool(predicted_gain>0 and actual_gain>0 and ratio is not None and ratio>=.1)
            row.update(eta_m=eta,eta_final_gate_pass=linear['eta_gate_pass'],geometry_summary=trial_summary,direct_fit_cost=direct_fit,
              predicted_phi=predicted_phi,new_actual_phi=new_phi,predicted_gain=predicted_gain,actual_gain=actual_gain,acceptance_ratio=ratio,accepted=accept)
'''
text_after='''            merit_finite=bool(np.isfinite([current_phi,predicted_phi,new_phi,predicted_gain,actual_gain]).all() and (ratio is None or np.isfinite(ratio)))
            np.savez_compressed(path/'merit_raw.npz',old_phi=current_phi,predicted_phi=predicted_phi,new_phi=new_phi,predicted_gain=predicted_gain,actual_gain=actual_gain,ratio=np.nan if ratio is None else ratio)
            accept=bool(merit_finite and predicted_gain>0 and actual_gain>0 and ratio is not None and ratio>=.1)
            def finite_number(value):return float(value) if value is not None and np.isfinite(value) else None
            row.update(eta_m=eta,eta_final_gate_pass=linear['eta_gate_pass'],geometry_summary=trial_summary,direct_fit_cost=direct_fit,
              predicted_phi=finite_number(predicted_phi),new_actual_phi=finite_number(new_phi),predicted_gain=finite_number(predicted_gain),actual_gain=finite_number(actual_gain),acceptance_ratio=finite_number(ratio),merit_finite=merit_finite,accepted=accept)
            if not merit_finite:
                stop='NONFINITE_MERIT_STOP';report['majors'].append(row);write(path/'receipt.json',row);break
'''
fatal_line="        checks['execution_without_fatal_numerical_stop']=stop not in {'LP_TERMINAL_FAILURE_NO_RETRY','RAW_LINEAR_CERTIFICATE_FAIL_NO_REPAIR','FP32_REFERENCE_CONTROL_GATE_FAIL','NONFINITE_MERIT_STOP'}\n"
expected=old.replace(text_before,text_after).replace("        checks['at_least_one_accepted_certified_step']",fatal_line+"        checks['at_least_one_accepted_certified_step']")
restored=json.loads(json.dumps(p));restored['sources']['producer']=pold['sources']['producer']
restored['inputs'].pop('local_probe_saved_independent');restored['constraints'].pop('planning_buffer_scope');restored['constraints'].pop('fatal_stop');restored.pop('prelaunch_revision')
pins=list(allpins(p));pinchecks=[{'pin':x,'match':samepin(x)} for x in pins]
probe=read(p['inputs']['local_probe_receipt']['path']);probe_audit=read(p['inputs']['local_probe_saved_independent']['path'])
checks={
    'v2_expected_guard_only_source_delta_exact':s==expected and old.count(text_before)==1,
    'v2_protocol_only_source_route_new_audit_and_explicit_scope':restored==pold,
    'all24_source_input_pins_current':len(pins)==24 and all(x['match'] for x in pinchecks),
    'producer_path_SHA_is_actualcurrent':samepin(p['sources']['producer']) and Path(p['sources']['producer']['path']).samefile(source_path),
    'localprobe_gate_and853review_actualsourcebound':probe['status']=='PASS_LOCAL_WORST_BRANCHES_TIES_DIAGNOSTIC_ONLY' and all(probe['checks'].values()) and probe_audit.get('passed')==853 and probe_audit.get('engineering_review_passed') is True,
    'fresh_output_nooverwrite': 'out.mkdir(parents=True,exist_ok=False)' in s,
    'allpins_checked_before_work_and_post_checked':s.index('for s in all_pins(p):checked(s)')<s.index('from scipy import sparse') and "checks['all_source_input_pins_unchanged']" in s,
    'prescribed_initial_P0_P1_qd_u0_guards':all(x in s for x in ("checks['fixed_initial_q0']","checks['fixed_initial_qd0']","checks['source_u0']","checks['final_P0_P1_exact']","checks['final_initial_qd0_exact']")),
    'init_and_object_arrays_are_copied_not_optimized':"np.savez_compressed(out/'init_requested.npz',**init)" in s and "refout.update({k:v for k,v in ref.items() if k.startswith('object_')})" in s and "mapped_object_pose_w=ref['mapped_object_pose_w']" in s,
    'all20_jointlinear_contracts_preserved': 'joints=[c.build_joint_lp(j,target,P0,P1,u0,tau,params,qlo,qhi) for j in range(20)]' in s,
    'single_global_LP_callsite_max8_including_reject':sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='linprog' for n in ast.walk(ast.parse(s)))==1 and 'for major in range(1,9):' in s,
    'LP_outside_geometryguard_and_attempted_before_call':"if oracle.original:raise ValueError" in s and s.index("report['cost']['LP_attempted']+=1")<s.index('result=linprog'),
    'normalfinite_rawlinear_and_postC2_precede_trialquery':s.index('if not result.success or not z.size:')<s.index("trial_grid,trial_summary=audit_grid") and s.index("if not linear['raw_primal_pass']:")<s.index("trial_grid,trial_summary=audit_grid") and s.index("if not all(post['gates'].values()):")<s.index("trial_grid,trial_summary=audit_grid"),
    'eta_not_intermediate_prerequisite': "if not linear['eta_gate_pass']" not in s and "eta_final_gate_pass=linear['eta_gate_pass']" in s,
    'directF_sum20_and_full_601_geometrymerit': 'np.mean(np.abs(q-target),axis=0).sum()+4*np.mean(np.abs(np.diff(curve,n=2,axis=0)),axis=0).sum()' in s and "new_phi=direct_fit+1e4*trial_summary['planning_max_violation_m']" in s,
    'predictedmax_from_actualwitness_step_not_solvereta':"np.max(rows['A_step']@delta-rows['rhs_m'])" in s,
    'jmajor_step_layout_correct': 'delta=(trial[2:]-P[2:]).T.reshape(3020)' in s,
    'accept_strictpositive_finite_ratio_point1': 'accept=bool(merit_finite and predicted_gain>0 and actual_gain>0 and ratio is not None and ratio>=.1)' in s,
    'accepted_current_updates_only_under_accept_rejectshrinks': 'P=trial;grid=trial_grid;gsummary=trial_summary;current_phi=new_phi;accepted_major=major;accepted_eta=eta;last_linear=linear' in s and 'else:radius*=.5' in s,
    'lastaccepted_final_no_cherryhistory':"final_grid,final_summary=audit_grid(P,out/'final_geometry')" in s and 'best' not in ''.join(n.id for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Name)),
    'fatalstop_hard_noREADY_even_if_finaldiagnostic_pass':fatal_line.strip() in s and "if all(checks.values()) else 'CPU_COUPLED_CANDIDATE_FAIL_RETAINED'" in s,
    'final_eta_independent_required':"checks['last_accepted_eta_final_gate']=accepted_eta is not None and 0<=accepted_eta<=1e-9" in s,
    'final_original3mm_only_no_false_claim_planningmargin': "checks['final_fresh_sampled_geometry']=final_summary['sampled_geometry_gate']" in s and 'not claimed as a final nonlinear achieved depth margin' in p['constraints']['planning_buffer_scope'],
    'final_FP32_old760_and_originalC2_remain_required':all(x in s for x in ("checks.update({'final_'+k:v for k,v in post['gates'].items()})","checks['final_CPU_Torch_proposal_bitexact']","checks['final_original_C2_bitexact']")),
    'grid_bound6010_and_constructor_recorded':"report['cost']['grid_audits']<=10" in s and p['constraints']['maximum_grid_queries']==6010 and p['constraints']['separate_prior_probe_queries']==164,
    'soft220_only_scope_does_not_claim_hardwatchdog': 'No claim of an independent hard OS watchdog' in p['constraints']['timing'] and 'time.perf_counter()-begin>220.' in s,
    'no_new_dynamics_or_learning_qualification':"'physical_qualified':False,'contact_intent_target_ready':False,'training_authorized':False" in s and "'continuous_geometry_qualified':False" in s,
}
report={
    'schema':'full151_SCP_driver_independent_static_review:2','status':'PASS_SOURCE_GATE_NO_REMAINING_SCIENTIFIC_BLOCKING_LIMITED_TO_DECLARED_CPU_ATTEMPT',
    'source':pin(source_path),'protocol':pin(proto_path),'old_unexecuted_source':pin(old_path),'old_unexecuted_protocol':pin(proto_old),
    'checks':checks,'all_checks_pass':all(checks.values()),'source_input_pin_checks':pinchecks,
    'formal_protocol_scope_vs_design_draft':{'planning2p999':'Linearized localmodel buffer only; no achieved nonlinear2.999mm certificate. Final is originalsample3mm. This is explicitly different from stricter draft53b34 proposal finalplanninggate.','fatal_stop':'Stops furtheroptimization. One alreadydeclared freshfinal601 may diagnose lastaccepted, but failurehardgate forbids READY.','wholeprocess':'No independenthardOSwatchdog.8LPx10s+finitegridcount andsoft220 before nextLP do not guarantee wallcap240.'},
    'nonblocking_cost_lifecycle_note':'live_state LP_RUNNING is written before LP_attempted increment. Final memorycounter countsbeforebackend correctly; if externally killed during LP, lastdurable live_state can undercount oneinflight attempt. No independentOSledger or interrupted-finalqualification is claimed.',
    'scope':{'source_JSON_AST_hash_only':True,'production_modules_imported':False,'NPZ_arrays_loaded':False,'framework_imported':any(x in sys.modules for x in ('numpy','scipy','mujoco','torch','warp')),'query_FK_solver_Env_WM_physics_GPU':False,'old_sources_modified':False},
    'review_source':pin(__file__),
}
report['review_inputs_unchanged_pre_post']=before=={str(x):pin(x) for x in (source_path,old_path,proto_path,proto_old)} and all(samepin(x) for x in pins)
diff=OUT/'v1_to_v2_source.diff'
if diff.exists():raise FileExistsError(diff)
diff.write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(old_path),tofile=str(source_path))),encoding='utf-8')
report['source_delta']=pin(diff)
target=OUT/'report.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(target),'checks':len(checks),'all_pass':all(checks.values()),'sourcepins':len(pins)}))
if not all(checks.values()) or not report['review_inputs_unchanged_pre_post']:raise SystemExit(2)
