"""Saved JSON/hash arithmetic only. Nullable conditional LP is descriptive."""
from pathlib import Path
import collections
import hashlib
import json
import math

OUT=Path(__file__).resolve().parent
R=OUT.parent

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def current(p):return pin(p['path'])==p
def finite(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main():
    branch_names=[('exact_source_original_hand','process_exact_source_v1'),('source1mm_opposed_side_original_hand','process_source1mm_v1')]
    checks={k:True for k in ['terminal_source_protocol_logs_current','all_direct_protocol_inputs_current','unique_attempt_return_counts_match','four_deadline_partials_retained','actual_hard10eq_le1e_minus5','complete22_189_and_reconstructed213_constraints_nonnegative','all7_final_pins_same_qpos','all7_native_LP_and_strict_status2_no_holding','model_and_custom_anchor_audits_preserved','combined_counts_match','zero_steps_training_and_reviewer_native_calls']}
    inputs={};summaries=[];records=[];deadlines=[];models=[];rawmodels=[]
    totals={k:0 for k in ['MJB_reads','native_FW','contact_J','static_evaluations','cheap_FK_returned','LP_calls','full_attempts','full_returns','final_geometry']}
    for branch,process in branch_names:
        base=R/f'{branch}_completed_v1';report=read(base/'report.json');tp=R/process/'terminal.json';term=read(tp)
        proto=read(report['protocol']['path']);algo=read(base/'algorithm_surface_configuration.json')
        inputs[branch]={'report':pin(base/'report.json'),'terminal':pin(tp),'protocol':report['protocol'],'attempts':pin(base/'attempts.jsonl'),'algorithm':pin(base/'algorithm_surface_configuration.json')}
        checks['terminal_source_protocol_logs_current'] &= (term['returncode']==0 and all(current(term[k]) for k in ['source','runner','protocol','stdout','stderr']) and term['stderr']['bytes']==0 and report['protocol']==term['protocol'] and (base/'protocol.json').read_bytes()==Path(term['protocol']['path']).read_bytes())
        checks['all_direct_protocol_inputs_current'] &= all(current(p) for p in [*proto['source_pins'].values(),*proto['inputs'].values()]) and algo['source_anchor_pin']==proto['inputs']['surface_anchors']
        checks['model_and_custom_anchor_audits_preserved'] &= report['model_audit']['passed'] is True and report['cheap_model_audit']['passed'] is True and algo['custom_anchor_compiled_checks']['status']=='PASS_SAME_MODEL_COMPILED_VERTICES_AND_OUTWARD_SUPPORT'
        models.append(report['model_audit']['model_digest'])
        log=[json.loads(l) for l in (base/'attempts.jsonl').read_text().splitlines()]
        attempts=[v for v in log if v['event']=='attempt'];returns=[v for v in log if v['event']=='returned']
        ak={(v['seed'],v['attempt']):v for v in attempts};rk={(v['seed'],v['attempt']):v for v in returns}
        missing=sorted(ak.keys()-rk.keys())
        checks['unique_attempt_return_counts_match'] &= len(ak)==len(attempts) and len(rk)==len(returns) and rk.keys()<=ak.keys() and all(ak[k]['stage']==rk[k]['stage'] for k in rk)
        for s in report['seed_results']:
            checks['unique_attempt_return_counts_match'] &= sum(k[0]==s['seed'] for k in ak)==s['full_attempts'] and sum(k[0]==s['seed'] for k in rk)==s['full_returned']
            checks['four_deadline_partials_retained'] &= 'Deadline' in s['stages']['B'].get('budget_stop','') and (s['seed'],s['full_attempts']) in missing and ak[(s['seed'],s['full_attempts'])]['stage']=='B'
            for f in s['finals']:
                d=read(f['result']['path']);c=d['candidate'];g=d['final_geometry'];state=g['state'];k=c['kinematics']
                lp=d['current_native_LP'];strict=d['strict_native_support_LP'];native=d['native_probe']
                eq=[float(x)/.001 for row in k['closure2x3'] for x in row]+[float(x) for row in k['normal_eq2x2'] for x in row]
                maxeq=max(map(abs,eq))
                checks['actual_hard10eq_le1e_minus5'] &= len(eq)==10 and all(finite(x) for x in eq) and maxeq<=1e-5 and eq==c['eq10_scaled'] and maxeq==c['max_abs_eq10_scaled'] and c['assembly_gate'] is True
                phones=state['phone_geometry_rows'];selves=state['self_geometry_rows'];lo,hi=g['new_pad_gap_hypothesis_m'];values=[];labels=[]
                for row in phones:
                    if row['is_pad']:
                        values.extend([row['lower_bound_m']-lo,hi-row['upper_bound_m']]);labels.extend([row['name']+'_lower',row['name']+'_upper'])
                    else:values.append(row['lower_bound_m']);labels.append(row['name']+'_nonpenetration')
                for row in selves:values.append(row['lower_bound_m']);labels.append('/'.join(row['names'])+'_self_nonpenetration')
                checks['complete22_189_and_reconstructed213_constraints_nonnegative'] &= (len(phones)==22 and len({x['geom_id'] for x in phones})==22 and len(selves)==189 and len({tuple(x['geoms']) for x in selves})==189 and not state['unknown'] and len(values)==213 and all(finite(x) and x>=0 for x in values) and values==g['constraints']['values_m'] and labels==g['constraints']['labels'] and g['complete22_189'] is True and g['geometry_gate_pass'] is True and g['constraints']['all_satisfied'] is True)
                checks['all7_final_pins_same_qpos'] &= current(f['result']) and s['seed']==d['seed'] and f['rank']==d['rank'] and k['q20']==state['q20_rad'] and k['qpos27']==state['qpos27'] and f['geometry_pass'] is True and f['assembly_gate'] is True
                checks['all7_native_LP_and_strict_status2_no_holding'] &= (lp is not None and strict is not None and lp['LP_calls']==1 and strict['LP_calls']==1 and strict['solver_status']==2 and strict['strict_force_moment_pass'] is False and strict['original_requested_box_infeasibility_established'] is False and d['ideal_native_support_candidate'] is False and d['holding_diagnostic_eligible'] is False and d['candidate_state'] is None and not d['fresh_command_probes'])
                rawmodels.append(native['model_parameter_digest']);agg=native['native_contact_aggregates']
                records.append({'branch':branch,'seed':s['seed'],'rank':f['rank'],'result':f['result'],
                    'max_abs_eq10_scaled':maxeq,'min_geometry_constraint_m':min(values),'phone_rows':len(phones),'self_rows':len(selves),
                    'pad_gap_bounds_m':{v['name']:[v['lower_bound_m'],v['upper_bound_m']] for v in phones if v['is_pad']},
                    'conditional_scaled_Linf':None if c['LP'] is None else c['LP']['scaled_loss'],
                    'conditional_LP_nullable_scope':'An A1 geometry pool point may not have acquired its conditional LP; do not confuse with missing final native/strict LP.',
                    'native_contacts':native['native_contact_count'],'native_phone_hand_contacts':lp['contact_count'],
                    'native_minimax_scaled_Linf':lp['scaled_loss'],'native_LP_force_residual_world_N':lp['force_residual_world_N'],
                    'native_LP_moment_residual_world_Nm':lp['moment_residual_world_Nm'],'native_LP_hard_bounds_pass':lp['hard_bounds_pass'],
                    'strict_solver_status':strict['solver_status'],'strict_status':strict['status'],'strict_force_moment_pass':False,
                    'native_initial_control_origin':d['initial_native_probe_ctrl_source'],
                    'native_PAD_Fn_N':{v:agg[v]['normal_force_signed_N'] for v in ['thumb_pad','index_pad']},
                    'native_nonpad_table_Fn_N':{v:agg[v]['normal_force_signed_N'] for v in ['phone_nonpad','phone_table']},
                    'native_COM_acceleration_world_m_s2':native['raw']['a_COM_world_m_s2'],
                    'native_alpha_world_rad_s2':native['raw']['alpha_world_rad_s2'],
                    'warnings_count':sum(native['warnings_number']),'unknown':native['unknown'],'holding_qualified':False})
        add={'MJB_reads':report['counts']['MJB_reads'],'native_FW':report['counts']['mj_forward_returned'],
             'contact_J':report['counts']['phone_hand_contact_jac_returned'],'static_evaluations':report['static_counts']['evaluations'],
             'cheap_FK_returned':report['cheap_FK_counts']['query_returned'],'LP_calls':report['LP_calls'],
             'full_attempts':len(attempts),'full_returns':len(returns),'final_geometry':report['counts']['final_geometry_evaluations']}
        for key,value in add.items():totals[key]+=value
        n=sum(len(s['finals']) for s in report['seed_results'])
        summaries.append({'branch':branch,'process_seconds':term['elapsed_s'],'producer_seconds':report['wall_seconds'],
            'counts':add,'stage_log_counts':dict(collections.Counter(f"seed{v['seed']}_{v['stage']}_{v['event']}" for v in log)),
            'deadline_without_return':[list(k) for k in missing],'final_records':n,
            'conditional_LP_runtime_inferred':report['LP_calls']-2*n,
            'LP_count_scope':'Total and inferred conditional count are runtime/source-backed. An existing full-cache state can acquire its first LP without a new full-attempt record. No independent per-LP trace.',
            'cheap_FK_count_scope':'Producer runtime/source-backed; no individual cheap-call log to independently reconstruct.'})
        deadlines.extend([{'branch':branch,'seed':k[0],'attempt':k[1]} for k in missing])
        checks['zero_steps_training_and_reviewer_native_calls'] &= report['new_steps_training_GPU']==0 and report['counts']['integration_steps']==0
    checks['four_deadline_partials_retained'] &= len(deadlines)==4
    checks['model_and_custom_anchor_audits_preserved'] &= len(set(models))==1 and len(set(rawmodels))==1
    checks['combined_counts_match'] = totals=={'MJB_reads':2,'native_FW':7,'contact_J':32,'static_evaluations':2207,'cheap_FK_returned':13944,'LP_calls':1936,'full_attempts':2200,'full_returns':2196,'final_geometry':7}
    selected=[v for v in records if v['seed']==1 and v['rank']==0]
    md='# v11 saved terminal review\n\nBoth processes exited0 with empty stderr. All7 final records pass actual hard10eq<=1e-5 and full22phone/189self geometry; the213 inequalities were reconstructed from saved rows and are nonnegative. The pad window[-0.1,+0.2]mm permits preload; this is not all-pad nonpenetration. All7 guarded strict LPs returned status2, so0 holding candidates. Tightened-box infeasibility is not original-box or global infeasibility.\n\n|Branch|Seed/rank|Hard10eq max|Min constraint m|Native scaledLinf|Thumb/index Fn N|\n|---|---|---:|---:|---:|---|\n'
    for v in records:md+=f"|{v['branch']}|{v['seed']}/{v['rank']}|{v['max_abs_eq10_scaled']:.9g}|{v['min_geometry_constraint_m']:.9g}|{v['native_minimax_scaled_Linf']:.9g}|{v['native_PAD_Fn_N']['thumb_pad']:.6g}/{v['native_PAD_Fn_N']['index_pad']:.6g}|\n"
    md+='\nCounts:2MJB/7nativeFW/32contactJ;2207static=2200attempts+7final;2196returns leave4 explicit final-deadline partials. LP1936=1922conditional(runtime/source)+7native+7strict; cached geometry can acquire LP without a new full-attempt log. Cheap13944 is runtime/source-backed, not reconstructed from individual callbacks. All source/input pins and saved model audits match;0 steps/training/GPU.\n\nSelected seed1/rank0:exact-source native scaledLinf=.254764196, Fn=.997714/1.423979N;1mm scaledLinf=.128485113, Fn=2.641127/3.457344N. The1mm actual index Fn exceeds3N; do not label it an actual3N pass. JSON keeps the native accelerations and nonpad/table loads. No instantaneous native or LP result establishes holding.\n\nScope:read-only saved hashes/JSON/log arithmetic; no backend/model/FK/query/LP/physics. Reviewer authored the helper/runner derivative, so this is a separate saved bookkeeping review of root execution, not an independent investigator reproduction of geometry or LP. Periodic/source model-audit guarantees are not per-query fresh raw model checks. The first reader assumed conditional candidate.LP non-null; its failure is preserved, while all final native/strict LP records remain mandatory.\n'
    with (OUT/'report.md').open('x',encoding='utf8') as f:f.write(md)
    review={'schema':'source_region_surface_assembly_v11_terminal_saved_review:1',
        'status':'PASS_SAVED_TERMINALS_PINS_CONSTRAINTS_COUNTS_NOT_HOLDING' if all(checks.values()) else 'FAILED_PRESERVED',
        'source':pin(__file__),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'inputs':inputs,
        'branch_summaries':summaries,'combined_counts':totals,'deadline_attempts_without_return':deadlines,
        'all7_records':records,'selected_seed1_rank0_native_descriptors':selected,
        'global_saved_model_digest':models[0],'saved_native_parameter_digest':rawmodels[0],
        'max_hard_eq10_all7':max(v['max_abs_eq10_scaled'] for v in records),
        'min_geometry_constraint_all7_m':min(v['min_geometry_constraint_m'] for v in records),
        'initial_reader_failure':pin(OUT/'readback_initial_failure_v1.json'),'markdown':pin(OUT/'report.md'),
        'reviewer_new_model_FK_geometry_LP_physics_NN_training_GPU':0,
        'scope':'Saved bookkeeping/arithmetic; no new independent geometry/LP solution. Helper/runner author reviewing root execution; no independent-investigator or holding qualification.'}
    save(OUT/'report.json',review)
    print(json.dumps({'report':pin(OUT/'report.json'),'markdown':pin(OUT/'report.md'),'checks':checks,'counts':totals}))

if __name__=='__main__':main()
