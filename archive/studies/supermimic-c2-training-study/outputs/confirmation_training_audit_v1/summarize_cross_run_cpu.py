"""Independent, read-only cross-run confirmation audit and wall-cost ledger.

Consumes nine new format103 auditor reports; does not run training/evaluation.
Original frozen sources/configs/automatic audit files are never written.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import torch

ROOT=Path('D:/workspace/supermimic-c2-training-study')
AUDIT=ROOT/'outputs/confirmation_training_audit_v1'
COST=ROOT/'outputs/confirmation_training_cost_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    assert not p.exists(),'Do not overwrite prior independent artifact: '+str(p)
    p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sec(s):return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def arrsha(a):
    return hashlib.sha256(str(a.dtype).encode()+repr(a.shape).encode()+a.tobytes()).hexdigest()
def csvwrite(p,rows):
    with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    statepath=ROOT/'outputs/confirmatory_queue_v1/state.json'
    manifestpath=ROOT/'configs/confirmatory_queue_authorized_v1.json'
    state=read(statepath);manifest=read(manifestpath)
    checks={'queue_terminal_passed_nine':state['status']=='passed' and len(state['jobs'])==9,
        'authorized_manifest_matches_queue':sha(manifestpath)==state['manifest_sha256']=='cfe02878248f0b3abfb125a4a33191b6616211d6e15f3a36960fa524ddefecd0',
        'review_manifest_remains_false':read(ROOT/'configs/confirmatory_queue_v1.json')['formal_training_authorized'] is False,
        'protocol_hash':sha(manifest['prospective_protocol_file'])==manifest['prospective_protocol_sha256'],
        'auditor_hash':sha(manifest['audit_script'])==manifest['audit_script_sha256'],
        'queue_source_hash':sha(ROOT/'scripts/run_matched_training_queue.py')==manifest['queue_runner_sha256']}
    models=[];jobs_cost=[];records=[];physics={};initial={};reference_signatures={};shared=[]
    source_reference=None;context_reference=None;reference_arrays=None;clip_arrays=None;target_spec=None;controller_domain=None
    expectedbudget={'wm':4000,'fail_head':4000,'actor':1800,'critic':7200}
    for job,planjob in zip(state['jobs'],manifest['jobs']):
        method,seed=job['method'],job['seed'];name=f'{method.lower()}_seed{seed}';run=Path(job['output'])
        reportpath=AUDIT/name/'report.json';report=read(reportpath);summary=read(run/'summary.json');attempt=summary['attempts'][-1]
        result=read(run/'compiled/result.json');runtime=read(run/'runtime.json');config=read(run/'input_config.json')
        saved=torch.load(run/'compiled/train/model_final.pt',map_location='cpu',weights_only=False)
        coverage=report['C2_actual_u_evidence'];controller=saved['c2_controller'];source=read(run/'source/inventory.json')
        metapath=run/'compiled/train/model_final.meta.json';meta=read(metapath)
        index=[json.loads(l) for l in (run/'compiled/checkpoint_index.jsonl').read_text().splitlines()]
        stateid=(job['method'],job['seed'],job['model_id']);planid=(planjob['method'],planjob['seed'],planjob['model_id'])
        checks[name+'_identity']=stateid==planid and config['method']==method and config['seed']==seed==saved['config']['seed'] and controller['training_seed']==seed and controller['trained_method']==method
        checks[name+'_new_independent_audit_passed']=report['status']=='passed' and all(report['checks'].values()) and report['audit_script_sha256']==manifest['audit_script_sha256']
        checks[name+'_full_budget']=report['actual_optimizer_steps']==attempt['actual_optimizer_steps']==result['actual_optimizer_steps']==meta['actual_optimizer_steps']==expectedbudget and report['real_simulator_transitions']==saved['sim_steps']==648192 and saved['iteration']==2000
        checks[name+'_actualu_fullprefix']=coverage['status']=='PASSED_CHECKPOINT_PREFIX' and coverage['prefix_last_collect_call']==coverage['prefix_row_count']==2001 and coverage['verified_real_transitions']==648192 and all(coverage['checks'].values())
        checks[name+'_final_index_sha']=meta in index and meta['checkpoint_sha256']==sha(run/'compiled/train/model_final.pt')==report['actual_final_checkpoint']['sha256']==job['final_checkpoint_sha256']==attempt['checkpoint_sha256']==result['checkpoint_sha256']
        checks[name+'_actual_final_vs2000_only_wall']=report['actual_final_checkpoint']['differing_payload_fields']==['wall_seconds']
        checks[name+'_config_binding']=sha(run/'input_config.json')==planjob['config_sha256']==sha(planjob['config'])==controller['config_sha256']==summary['config_sha256']
        checks[name+'_single_attempt_no_retry_cap_unreached']=len(summary['attempts'])==1 and summary['status']=='passed' and attempt['status']=='passed' and config['compile_fallback'] is False and attempt['attempt_wall_seconds']<2400 and not job['deadline']['expired'] and not job['deadline']['cleanup_unverified']
        osobs=job['actual_process_OS_observation'];command=osobs.get('command_line','').replace('\\','/').lower()
        checks[name+'_CIM_identity']=osobs['status']=='observed_exact_PID_and_identity' and osobs['pid']==job['actual_pid']==runtime['pid'] and osobs['parent_pid']==job['launcher_pid']==runtime['parent_pid'] and str(Path(planjob['config'])).replace('\\','/').lower() in command and '--run' in command
        checks[name+'_external_cap_unreached']=sec(runtime['finished_utc'])-sec(osobs['creation_date_utc'])<2700
        checks[name+'_no_warmstart']=not ({'checkpoint','model_checkpoint'} & set(config['training'])) and controller['exact_resume_supported'] is False and saved['format_version']==103
        fps=read(run/'compiled/initial_parameters.json');initial[name]=fps
        checks[name+'_initial_fingerprint_three_records_exact']=fps==result['initial_parameter_fingerprints']==controller['initial_parameter_fingerprints']
        if source_reference is None:source_reference=source['files']
        checks[name+'_native90_sources_common']=source['files']==source_reference
        currentcontext={k:v for k,v in np.load(run/'compiled/fixed_physics_context.npz').items()}
        if context_reference is None:context_reference=currentcontext
        physics[name]={k:{'exact':np.array_equal(a,context_reference[k]),'max_abs_difference':float(np.max(np.abs(a-context_reference[k]))),'array_sha256':arrsha(a)} for k,a in currentcontext.items()}
        checks[name+'_physics9_arrays_exact']=len(currentcontext)==9 and all(x['exact'] for x in physics[name].values())
        ref={k:v for k,v in np.load(run/'reference/frozen_evaluation_reference.npz').items()}
        clipped=list((run/'reference/clips').rglob('*.npz'));assert len(clipped)==1
        clips={k:v for k,v in np.load(clipped[0]).items()}
        if reference_arrays is None:reference_arrays=ref;clip_arrays=clips
        checks[name+'_reference_clip_arrays_exact']=set(ref)==set(reference_arrays) and all(np.array_equal(v,reference_arrays[k]) for k,v in ref.items()) and set(clips)==set(clip_arrays) and all(np.array_equal(v,clip_arrays[k]) for k,v in clips.items())
        reference_signatures[name]=saved['reference_signature'];checks[name+'_reference_signature_result_matches']=saved['reference_signature']==result['reference_signature']
        checks[name+'_canonical_MJB_ref']=result['nominal_model_sha256']==sha(run/'compiled/nominal_model.mjb')==manifest['nominal_model_sha256'] and sha(run/'reference/frozen_evaluation_reference.npz')==read(manifest['prospective_protocol_file'])['unchanged_inputs']['reference_npz_sha256']
        shared.append({'training':config['training'],'reference':config['reference'],'c2_controller':config['c2_controller']})
        domain=(controller['target_lower_rad'],controller['target_upper_rad'])
        if controller_domain is None:controller_domain=domain
        checks[name+'_common_C2_domain']=domain==controller_domain and controller['margin_rad']==.02
        if method!='A1L':
            if target_spec is None:target_spec=config['contact_objective']
            checks[name+'_shared_frozen_prior_spec']=config['contact_objective']==target_spec
        else:checks[name+'_A1L_no_prior']=not config.get('contact_objective')
        prefix=[json.loads(l) for l in (run/'compiled/c2_actualu_evidence.jsonl').read_text().splitlines()][:2001]
        clamped=sum(x['clamped_components'] for x in prefix);components=sum(x['total_components'] for x in prefix)
        record={'model_id':job['model_id'],'trained_method':method,'training_seed':seed,'evaluation_seed':0,'iteration':2000,'checkpoint':str(run/'compiled/train/model_final.pt'),'checkpoint_sha256':meta['checkpoint_sha256'],
            'model2000_sha256':report['checkpoint_sha256'],'checkpoint_format':103,'independent_audit':pin(reportpath),'final_vs2000_differences':report['actual_final_checkpoint']['differing_payload_fields'],
            'actual_optimizer_steps':expectedbudget,'real_transitions':648192,'actualu_collect_calls':2001,'actualu_checked_traces':len(coverage['sampled_trace_sha256']),'actualu_clamped_components':clamped,'actualu_total_components':components,
            'input_config':pin(run/'input_config.json'),'runtime':pin(run/'runtime.json'),'initial_parameters':pin(run/'compiled/initial_parameters.json'),'physical_context':pin(run/'compiled/fixed_physics_context.npz'),'reference_signature':saved['reference_signature']}
        models.append(record)
        native_started=summary['total_wall_seconds_including_failed_attempts'];phase=result['phase_seconds'];native_attempt=attempt['attempt_wall_seconds'];learn=attempt['learn_wall_seconds'];supervised=job['reserved_wall_seconds_including_CPU_health']
        row={'model_id':job['model_id'],'method':method,'training_seed':seed,'native_STARTED_seconds':native_started,'attempt_seconds':native_attempt,'learn_seconds':learn,
            'environment_init_seconds':result['environment_init_seconds'],'runner_init_seconds':result['runner_init_seconds'],'prior_setup_seconds':result.get('contact_objective_setup_seconds',0.),
            **{'phase_'+k+'_seconds':v for k,v in phase.items()},'supervised_child_including_CPU_health_seconds':supervised,
            'CIM_creation_to_runtime_finish_seconds':sec(runtime['finished_utc'])-sec(osobs['creation_date_utc']),
            'runtime_started_to_finished_seconds':sec(runtime['finished_utc'])-sec(runtime['started_utc']),
            'supervised_minus_native_STARTED_seconds':supervised-native_started,'attempt_count':len(summary['attempts']),'native_cap_seconds':2400,'OS_cap_seconds':2700,'cap_triggered':False}
        jobs_cost.append(row);records.append({'model_id':job['model_id'],'actual_PID':job['actual_pid'],'launcher_PID':job['launcher_pid'],'actual_process_OS_observation':osobs,'runtime_started_utc':runtime['started_utc'],'runtime_finished_utc':runtime['finished_utc'],'phase_calls':result['phase_calls'],'compiled_counters':result['compiled_counters'],'compiler_cache_paths':result['compiler_cache_paths'],'cache_context':summary['cache_context'],'source_result':pin(run/'compiled/result.json'),'source_summary':pin(run/'summary.json')})
        del saved
    init_by_seed={}
    for seed in [4,5,6]:
        byarm={method:initial[f'{method.lower()}_seed{seed}'] for method in ['A1L','P_ONLY','A2']}
        init_by_seed[str(seed)]={'exact_all_three_arms':byarm['A1L']==byarm['P_ONLY']==byarm['A2'],'fingerprints':byarm}
        checks[f'seed{seed}_same_initial_four_modules']=init_by_seed[str(seed)]['exact_all_three_arms']
    checks['all_nine_training_reference_controller_configs_equal']=all(x==shared[0] for x in shared)
    checks['all_nine_reference_signatures_equal']=len(set(reference_signatures.values()))==1
    checks['CPU_only_no_CUDA_initialization']=not torch.cuda.is_initialized()
    index={'status':'passed' if all(checks.values()) else 'failed','schema':'C2_020_confirmation_nine_final_index:1','authorization_manifest':pin(manifestpath),'prospective_protocol':pin(manifest['prospective_protocol_file']),'queue_state':pin(statepath),'models':models,'scope':'Full final2000 only, no physical evaluation or quality selection. training_seed4/5/6 and planned evaluation_seed0 are distinct.'}
    write(AUDIT/'model_index.json',index)
    report={'status':index['status'],'schema':'C2_020_confirmation_training_cross_run_audit:1','checks':checks,'source_script':pin(__file__),'model_index':pin(AUDIT/'model_index.json'),'initialization_same_seed_three_arm':init_by_seed,'physics9_array_cross_run':physics,'reference_signatures':reference_signatures,'canonical_context_array_hashes':{k:arrsha(v) for k,v in context_reference.items()},'scope':'CPU serialization/finiteness/budget/input/source/controller evidence only. No task quality, physical-contact prediction, actual-q safety or GPU-kernel speed claim.'}
    write(AUDIT/'report.json',report)
    indexcsv=[{k:m[k] for k in ['model_id','trained_method','training_seed','evaluation_seed','iteration','checkpoint','checkpoint_sha256','model2000_sha256','checkpoint_format','actualu_collect_calls','actualu_checked_traces']} for m in models]
    csvwrite(AUDIT/'models.csv',indexcsv)
    COST.mkdir(parents=True,exist_ok=True);csvwrite(COST/'runs.csv',jobs_cost)
    whole=sec(state['finished_utc'])-sec(state['started_utc']);sumchild=sum(x['supervised_child_including_CPU_health_seconds'] for x in jobs_cost)
    sums={k:sum(x[k] for x in jobs_cost) for k in jobs_cost[0] if isinstance(jobs_cost[0][k],(int,float)) and k not in ['training_seed','attempt_count','native_cap_seconds','OS_cap_seconds','cap_triggered']}
    bymethod={}
    for method in ['A1L','P_ONLY','A2']:
        rows=[r for r in jobs_cost if r['method']==method]
        bymethod[method]={'training_seed_N':3,'native_STARTED_seconds':{'sum':sum(r['native_STARTED_seconds'] for r in rows),'mean':statistics.mean(r['native_STARTED_seconds'] for r in rows),'min':min(r['native_STARTED_seconds'] for r in rows),'max':max(r['native_STARTED_seconds'] for r in rows)},'learn_seconds_mean':statistics.mean(r['learn_seconds'] for r in rows),'attempt_seconds_mean':statistics.mean(r['attempt_seconds'] for r in rows),'phase_seconds_sum':{k:sum(r[k] for r in rows) for k in rows[0] if k.startswith('phase_')}}
    handoff=read(ROOT/'outputs/confirmatory_queue_v1/first100_handoff.json');remaining=sec(state['finished_utc'])-sec(handoff['created_utc']);eta=handoff['remaining_queue_seconds_engineering_range']
    estimated=manifest['estimated_total_seconds_from_pilot_costs']
    cost={'status':index['status'],'schema':'C2_020_confirmation_cost_ledger:1','source_script':pin(__file__),'queue_state':pin(statepath),'authorization_manifest':pin(manifestpath),'queue_started_utc':state['started_utc'],'queue_finished_utc':state['finished_utc'],'whole_queue_wall_seconds':whole,'sum_supervised_child_with_CPU_health_seconds':sumchild,'queue_gaps_validation_loop_overhead_seconds':whole-sumchild,'summed_run_counters':sums,'per_method':bymethod,'runs':jobs_cost,'process_and_compile_records':records,
        'first100_ETA_vs_observed':{'handoff':pin(ROOT/'outputs/confirmatory_queue_v1/first100_handoff.json'),'actual_remaining_seconds':remaining,'announced_low_seconds':eta['low'],'announced_high_seconds':eta['high'],'within_announced_engineering_range':eta['low']<=remaining<=eta['high'],'initial_point_seconds':estimated,'whole_queue_minus_initial_point_seconds':whole-estimated,'whole_queue_vs_initial_point_ratio':whole/estimated,'scope':'Seed0 profiled-phase engineering range, not CI. Actual whole queue includes CPU health and queue overhead excluded from pilot point.'},
        'scopes':{'native_STARTED':'Original trainer module STARTED perf_counter through summary generation, including its startup/preparation; not literal OS CreationDate or GPU busy time.','attempt':'Native single attempt starts after imports and output creation; includes env/runner/prior setup and learn.','learn':'runner.learn wall; includes recorded collect/WM/policy/checkpoint phases plus unprofiled overhead.','phase':'Native separately accumulated timing categories; no claim all clocks exactly partition attempt/OS time.','child_supervisor':'Queue stopwatch beforePopen through final CPU health; includes interpreter/entry and first/final CPU audits while holding reserved GPU slot.','whole_queue':'Queue state started→finished; includes all child supervised spans plus preflights/gaps.','cost_comparison':'Same updates/transitions, unequal compute. No physical-task quality, GPU-utilization or algorithmic acceleration inference.'},'all_nine_single_attempt':all(len(read(Path(j['output'])/'summary.json')['attempts'])==1 for j in state['jobs']),'caps_triggered':0,'GPU_used_by_this_analysis':False}
    write(COST/'ledger.json',cost)
    text=[f"Nine independent final2000 audits: {index['status'].upper()}.",f"Each run648192 transitions; WM/fail4000 each, actor1800, critic7200;2001 complete actualu collect rows and23 predeclared sparse traces.","All final-vs2000 serialized fields equal except actual wall_seconds. Same-seed3arm initial actor/WM/fail/critic fingerprints exact; all9physics arrays, reference/signature/native source maps and C2 domain exact across9.","All nine actual PID/OS CreationDate identities recorded; one attempt each, no native/external cap triggered.",f"Whole queue {whole:.3f}s; nativeSTARTED sum {sums['native_STARTED_seconds']:.3f}s; attempt sum {sums['attempt_seconds']:.3f}s; learn sum {sums['learn_seconds']:.3f}s.",f"After first100 handoff actual remainder {remaining:.3f}s lies inside announced {eta['low']:.3f}–{eta['high']:.3f}s engineering range. Not CI or a task-quality result.","Full SHA/model metadata: model_index.json and models.csv. Task evaluation remains independent and is not examined here."]
    (AUDIT/'REPORT.txt').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    (COST/'REPORT.txt').write_text('\n\n'.join(text[3:])+'\n',encoding='utf-8')
    print(json.dumps({'status':index['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'models':len(models),'whole_queue_seconds':whole,'native_started_sum':sums['native_STARTED_seconds'],'actual_remaining_after100':remaining,'ETA_range':eta,'model_index':pin(AUDIT/'model_index.json'),'cross_run_report':pin(AUDIT/'report.json'),'cost':pin(COST/'ledger.json')},indent=2))
    raise SystemExit(index['status']!='passed')

if __name__=='__main__':main()
