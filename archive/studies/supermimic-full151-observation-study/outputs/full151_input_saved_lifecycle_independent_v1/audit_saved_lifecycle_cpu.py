"""Saved-only provenance/lifecycle audit. No array or mathematical re-solve."""
from pathlib import Path
import ast
import hashlib
import json
import math
import sys

ROOT=Path('D:/workspace/supermimic-full151-observation-study')
RUN=ROOT/'outputs/full151_cpu_input_v1'
OUT=Path(__file__).resolve().parent

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def pin(path):
    p=Path(path).resolve(strict=True)
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def pins(value):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:yield value
        else:
            for v in value.values():yield from pins(v)
    elif isinstance(value,list):
        for v in value:yield from pins(v)

report=read(RUN/'receipt.json');proto=read(report['protocol']['path'])
started=read(RUN/'execution_started.json');root=read(RUN/'root_execution_observation.json')
source=Path(proto['sources']['producer']['path']).read_text(encoding='utf-8-sig')
before={str(p):pin(p) for p in RUN.rglob('*') if p.is_file()}
protocol_current=[{'pin':p,'actual':pin(p['path']),'matches':p['sha256']==pin(p['path'])['sha256']} for p in pins(proto)]
artifact_current=[{'pin':p,'matches':pin(p['path'])==p} for p in report['artifacts']]
archived=[]
for key,p in proto['sources'].items():
    archive=RUN/'source_before'/(key+Path(p['path']).suffix)
    archived.append({'name':key,'source':p,'archive':pin(archive),'byte_exact':archive.read_bytes()==Path(p['path']).read_bytes(),'sha_matches':pin(archive)['sha256']==p['sha256']})
rows=[read(RUN/'joint_LPs'/f'joint{j:02d}_receipt.json') for j in range(20)]
joint_files={j:[RUN/'joint_LPs'/f'joint{j:02d}{suffix}' for suffix in ('_input.npz','_contract.json','_result.npz','_receipt.json')] for j in range(20)}
failed=[r['joint'] for r in rows if r['arithmetic_1e12_gate'] is False]
bad_groups={str(r['joint']):{k:v for k,v in r['row_group_max_raw_Ax_minus_b'].items() if v>1e-12} for r in rows if r['arithmetic_1e12_gate'] is False}
checks={
    'actual_protocol_v2_pin_matches':pin(report['protocol']['path'])==report['protocol'],
    'all23_declared_source_input_pins_match_current':len(protocol_current)==23 and all(x['matches'] for x in protocol_current),
    'all8_executed_source_archives_match_pinned_live_bytes':len(archived)==8 and all(x['byte_exact'] and x['sha_matches'] for x in archived),
    'all_original_saved_artifacts_match_receipt_hashes':all(x['matches'] for x in artifact_current),
    'started_and_terminal_protocol_time_identity_match':started['protocol']==report['protocol'] and started['started_UTC']==report['started_UTC'],
    'started0_calls_then_terminal20':started['cost']['LP_calls']==0 and report['cost']['LP_calls']==20,
    '20_distinct_joint_receipts_complete':len(rows)==20 and [r['joint'] for r in rows]==list(range(20)),
    'all80_perjoint_input_contract_result_receipt_files_preserved':all(p.is_file() for group in joint_files.values() for p in group),
    'aggregate_joint_rows_equal_perjoint_saved_rows':report['joint_results']==rows,
    'all20_success_status0_finite_reported_objectives':all(r['success'] is True and r['status']==0 and math.isfinite(r['objective']) for r in rows),
    '5_reported_raw_primal_failures_preserved':len(failed)==5 and report['checks']['all20_raw_arithmetic_1e12'] is False,
    'all34_gate_outcomes_only_raw_primal_false':len(report['checks'])==34 and [k for k,v in report['checks'].items() if v is not True]==['all20_raw_arithmetic_1e12'],
    'final_status_is_no_qualification':report['status']=='CPU_CANDIDATE_GATES_FAIL_NO_QUALIFICATION',
    'no_READY_manifest_written':not (RUN/'READY.json').exists(),
    'failed_candidate_arrays_are_preserved':all((RUN/k).is_file() for k in ('fit_solutions.npz','reference_bspline151.npz','planned_control150.npz','init_requested.npz','actual_PD_parameters.npz')),
    'no_reported_exception_or_fallback_selection':not any(k in report for k in ('exception','traceback','fallback','selected_best_history')),
    'solver_wall_sum_consistent_with20_receipts':abs(math.fsum(r['wall_s'] for r in rows)-report['cost']['solver_wall_s_sum'])<=1e-12,
    'worker_wall_contains_solver_wall_not_added':report['wall_s']>=report['cost']['solver_wall_s_sum']>=0,
    'source_single_LP_callsite_counted_before_call':sum(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='linprog' for n in ast.walk(ast.parse(source)))==1 and source.index("report['cost']['LP_calls']+=1")<source.index('result=linprog'),
    'source_single_fixed20_loop_no_retry_protocol': 'for j in range(20):' in source and proto['execution']['retries']==0,
    'source_failure_return1_only_READY_on_all_gates':"return 0 if report['status']=='CPU_INPUT_GATES_PASS_PHYSICS_PENDING' else 1" in source and "if report['status']=='CPU_INPUT_GATES_PASS_PHYSICS_PENDING':" in source,
    'root_tool_exit1_once_noretry_explicitly_scoped':root['exit_code']==1 and root['observed_launches_by_root_this_turn']==1 and root['retry_performed_by_root'] is False and 'not an independently sampled OS process ledger' in root['source'],
    'root_tool_pins_actual_protocol_producer_receipt':root['protocol']==report['protocol'] and root['producer']==proto['sources']['producer'] and root['receipt']==pin(RUN/'receipt.json'),
    'Root_launch_cuda_minus1_matches_worker_record':root['launch_environment']['CUDA_VISIBLE_DEVICES']=='-1' and report['runtime']['CUDA_VISIBLE_DEVICES']=='-1',
    'reported_cost0_physics_query_WMtrainingGPU':all(report['cost'][k]==0 for k in ('physics_steps','geometry_queries','WM_forwards','training_updates')) and report['cost']['GPU_used'] is False,
    'PID_creation_absence_is_not_fabricated':root['PID_and_creation_time']=='NOT_SAMPLED_SHORT_FOREGROUND_CPU_COMMAND',
}
result={
    'schema':'full151_input_saved_lifecycle_independent_review:1',
    'status':'PASS_SOURCE_AND_FAILURE_PRESERVATION_CPU_CANDIDATE_REMAINS_FAILED',
    'receipt':pin(RUN/'receipt.json'),'root_execution_observation':pin(RUN/'root_execution_observation.json'),
    'protocol':report['protocol'],'checks':checks,'all_checks_pass':all(checks.values()),
    'protocol_pin_checks':protocol_current,'source_archives':archived,'original_artifact_hash_checks':artifact_current,
    'failed_joint_indices_from_saved_receipts':failed,'failed_row_groups_from_saved_receipts':bad_groups,
    'cost':{'solver_sum_s':report['cost']['solver_wall_s_sum'],'worker_wall_s':report['wall_s'],'Root_tool_whole_wall_s':root['tool_wall_time_s'],'nested_not_additive':True,'PID_Creation_OS_lifecycle':'NOT_SAMPLED; no independent OS qualification claimed'},
    'scope':{'provenance_failure_status_and_JSON_consistency_independently_recomputed':True,'array_LP_raw_numerics':'Not recomputed here; Contact independent mathematical audit is separate','solver_success_not_primal_certificate':True,'all_postC2_init_physics_related_gate_values':'Saved execution claims only, not independently rederived here','Root_once_exitcode_evidence':'Pinned Root tool observation, not independently sampledOS','no_Array_import_load_model_query_solver_Env_physics_GPU':True,'frameworks_imported':any(x in sys.modules for x in ('numpy','scipy','mujoco','torch','warp'))},
    'review_source':pin(__file__),
}
result['all_run_artifacts_unchanged_pre_post']=before=={str(p):pin(p) for p in RUN.rglob('*') if p.is_file()}
target=OUT/'report.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'report':pin(target),'checks':len(checks),'all_pass':all(checks.values()),'failed_joints':failed,'run_unchanged':result['all_run_artifacts_unchanged_pre_post']}))
if not all(checks.values()) or not result['all_run_artifacts_unchanged_pre_post']:raise SystemExit(2)
