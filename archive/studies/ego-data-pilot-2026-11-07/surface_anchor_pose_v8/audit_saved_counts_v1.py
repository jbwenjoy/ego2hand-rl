"""Saved JSON/source/count review only; no array/backend/LP/model imports."""
from pathlib import Path
import ast
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'completed_v1'
OUT=ROOT/'saved_count_review_v1'


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def checked(rec):
    p=Path(rec['path']);assert pin(p)==rec,rec['path'];return p


def main():
    assert not OUT.exists();OUT.mkdir()
    r=read(RUN/'report.json');p=read(checked(r['protocol']));terminal=read(ROOT/'run1_terminal.json')
    checks={};assert read(RUN/'protocol.json')==p
    for rec in [*p['source_pins'].values(),*p['inputs'].values()]:checked(rec)
    for k in ('source','protocol','stdout','stderr'):checked(terminal[k])
    checks['all_direct_source_input_and_terminal_hashes_current']=True
    checks['terminal_exit0_exact_source_protocol']=(terminal['returncode']==0 and terminal['pid']==23036
        and terminal['source']==p['source_pins']['runner'] and terminal['protocol']==r['protocol']
        and terminal['stderr']['bytes']==0 and terminal['automatic_retry'] is False)
    log=read(RUN/'evaluation_log.json');seeds=r['seed_results'];assert len(seeds)==2
    log_counts=[];attempts=successful_static=0
    for s in seeds:
        ls=[v for v in log if v['seed']==s['seed']];ids=[v['evaluation'] for v in ls]
        assert ids==list(range(1,len(ls)+1))
        gap=s['search_evaluations']-len(ls)
        log_counts.append({'seed':s['seed'],'name':s['name'],'search_attempts':s['search_evaluations'],
            'successful_conditional_LP_log_rows':len(ls),'unlogged_attempts':gap,
            'last_successful_evaluation':ids[-1],
            'original_purpose_counts':{key:sum(v['purpose']==key for v in ls) for key in ('base','FD')},
            'solver':s['solver'],'stop':s['stop']})
        attempts+=s['search_evaluations'];successful_static+=len(ls)
    checks['exact872_attempts871_successful_LP_logs_one_deadline_gap']=(attempts==872 and successful_static==871
        and [(v['search_attempts'],v['successful_conditional_LP_log_rows'],v['unlogged_attempts']) for v in log_counts]==[(235,235,0),(637,636,1)]
        and seeds[0]['stop'] is None and seeds[1]['stop']=="StaticBudgetExceeded('Deadline')" and seeds[1]['solver'] is None)
    source=Path(p['source_pins']['runner']['path']).read_text(encoding='utf8');tree=ast.parse(source)
    main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    eval_node=next(n for n in ast.walk(main_node) if isinstance(n,ast.FunctionDef) and n.name=='evaluate')
    def lines(text):return [n.lineno for n in ast.walk(eval_node) if text in ast.unparse(n) and isinstance(n,(ast.Assign,ast.AugAssign,ast.Expr))]
    inc=next(n.lineno for n in ast.walk(eval_node) if isinstance(n,ast.AugAssign) and ast.unparse(n.target)=='search_count')
    static=next(n.lineno for n in ast.walk(eval_node) if isinstance(n,ast.Assign) and ast.unparse(n.value).startswith('e.evaluate('))
    lp=next(n.lineno for n in ast.walk(eval_node) if isinstance(n,ast.Assign) and ast.unparse(n.value).startswith('scaled_wrench_residual_lp('))
    append=next(n.lineno for n in ast.walk(eval_node) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='logs.append')
    checks['source_attempt_counter_precedes_static_LP_and_success_log']=(inc<static<lp<append)
    final_rows=[];native=final_geo=native_LP=strict_LP=command_FW=contact_J=0;all_final_flags=True
    result_pins={}
    for s in seeds:
        assert [v['rank'] for v in s['shortlist']]==[0,1,2]
        for item in s['shortlist']:
            qpath=checked(item['result']);q=read(qpath);result_pins[qpath.name]=pin(qpath)
            assert q['seed']==s['seed'] and q['rank']==item['rank']
            g=q['final_geometry'];con=g['constraints'];values=con['values_m']
            assert len(g['state']['phone_geometry_rows'])==22 and len(g['state']['self_geometry_rows'])==189 and not g['state']['unknown']
            calculated_geo=all(v>=0 for v in values)
            anchor=q['optimizer_candidate'];closure=anchor['closure_world_m'];angles=anchor['normal_angles_rad']
            calculated_anchor=(max(math.sqrt(sum(c*c for c in row)) for row in closure)<=.0002 and max(v*180/math.pi for v in angles)<=1.)
            assert calculated_geo==g['geometry_gate_pass']==item['geometry_pass']==False
            assert calculated_anchor==q['surface_anchor_gate_pass']==item['anchor_gate']==False
            n=q['native_probe'];cur=q['current_native_LP'];strict=q['strict_native_support_LP'];repeats=q['fresh_command_probes']
            native+=n is not None;final_geo+=1;native_LP+=cur is not None;strict_LP+=strict is not None;command_FW+=len(repeats)
            contact_J+=len(n['contact_design']['phone_hand_contacts']) if n is not None else 0
            assert n is not None and not n['unknown'] and cur is not None and cur['LP_calls']==1
            all_final_flags &= (q['ideal_surface_native_support_candidate'] is False and q['holding_diagnostic_eligible'] is False
                and q['faithful_ego_or_holding_qualified'] is False and q['native_entry']=={}
                and q['candidate_state'] is None and not repeats and strict is None
                and item['ideal_native_support'] is False and item['holding_diagnostic_eligible'] is False)
            final_rows.append({'seed':s['seed'],'rank':q['rank'],'geometry_pass':False,'anchor_gate':False,
                'negative_constraints':[k for k,v in zip(con['labels'],values) if v<0],
                'surface_max_closure_m':max(math.sqrt(sum(c*c for c in row)) for row in closure),
                'surface_max_normal_angle_deg':max(v*180/math.pi for v in angles),
                'native_phone_hand_contacts':len(n['contact_design']['phone_hand_contacts']),
                'native_LP_present':True,'strict_LP_called':False,'extra_command_forwards':0,
                'holding_diagnostic_eligible':False})
    checks['all_six_full_geometry_and_anchor_gates_fail_no_candidate']=bool(all_final_flags and len(final_rows)==6 and r['holding_diagnostic_candidates']==0)
    checks['LP877_is871_logged_conditional_plus6_native_zero_strict']=(r['LP_calls']==successful_static+native_LP+strict_LP==877 and native_LP==6 and strict_LP==0)
    c=r['counts'];sc=r['static_counts']
    checks['six_forward_six_final_geometry_twelve_contact_J_no_commands_or_steps']=(native==final_geo==6 and contact_J==12 and command_FW==0
        and c['native_evaluations_attempted']==c['native_evaluations_returned']==c['native_MjData']==c['mj_forward_attempted']==c['mj_forward_returned']==6
        and c['final_geometry_evaluations']==6 and c['phone_hand_contact_jac_attempted']==c['phone_hand_contact_jac_returned']==12
        and c['contact_force_reads']==12 and c['integration_steps']==c['training_GPU']==r['new_steps_training_GPU']==0)
    checks['static878_includes872_search_attempts_plus6_final_geometry']=(sc['evaluations']==attempts+final_geo==878
        and all(sc[k]==878 for k in ('mj_kinematics','mj_comPos','mj_comVel','mj_rne','mj_passive','mj_transmission'))
        and sc['anchor_mj_jac']==1756 and sc['heavy_queries_attempted']==sc['heavy_queries_returned']==2244 and sc['unknown_queries']==0)
    checks['full_final_model_audit_passed']=(r['model_audit']['passed'] is True and r['model_audit']['counts']==c)
    assert all(checks.values()),{k:v for k,v in checks.items() if not v}
    review={'schema':'surface_anchor_pose_v8_saved_counts_review:1','status':'PASS_SAVED_COUNTS_PROVENANCE_ZERO_QUALIFIED_CANDIDATES',
        'producer_report':pin(RUN/'report.json'),'protocol':r['protocol'],'source_reviewer':pin(__file__),
        'checks':checks,'passed':sum(checks.values()),'total':len(checks),
        'seed_count_explanation':log_counts,'counts':{'search_attempts':attempts,'successful_static_conditional_LP_logs':successful_static,
            'unlogged_deadline_attempt':1,'static_pipeline_counter':878,'final_geometry_calls':6,
            'native_forwards':6,'native_contact_J':12,'native_final_LP':6,'strict_support_LP':0,
            'LP_total':877,'additional_proposed_command_forwards':0,'physical_steps':0},
        'final_six_candidates':final_rows,'source_order_lines':{'attempt_increment':inc,'static_call':static,'conditional_LP_call':lp,'successful_log_append':append},
        'scope_limits':['The omitted deadline attempt is not reconstructed. Source ordering and aggregate counters support one static-pipeline attempt that did not reach a successful LP/log row; its exact failed x14 and internal stopping point are not independently saved.',
            'Frozen geometry/LP outputs are read as saved evidence; no geometry query, force solver or model is rerun. Checks of geometry/anchor gate logic use saved scalar/vector operands only.',
            'Solver xtol status and deadline are bounded exploration outcomes, not an optimality or global feasibility certificate. All six final candidates fail geometry and anchor gates; no native support or holding success is inferred.'],
        'pins':{**result_pins,'evaluation_log':pin(RUN/'evaluation_log.json'),'algorithm_configuration':pin(RUN/'algorithm_surface_configuration.json'),'terminal':pin(ROOT/'run1_terminal.json')},
        'reviewer_NPZ_MJB_backend_forward_LP_step_calls':0,'holding_or_faithful_ego_or_method_qualified':False}
    with (OUT/'report.json').open('x',encoding='utf8') as f:json.dump(review,f,indent=2,allow_nan=False)
    print(json.dumps({'report':pin(OUT/'report.json'),'passed':review['passed'],'total':review['total'],'counts':review['counts']}))


if __name__=='__main__':main()
