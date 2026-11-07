"""Archive new source and immutable observations of existing live processes."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, statistics, subprocess

P=Path('D:/workspace/ego-data-pilot-2026-11-07')
T=P/'phone_tripod_supermimic_v1'
E=P/'phone_tripod_policy_evaluation_v1'
S=P/'outputs/tripod_online_SOURCE_live_snapshot_v16'
OUT=P/'outputs/goal_extension_tripod_online_loop_SOURCE_milestone_v16.json'
assert not S.exists() and not OUT.exists(), 'Immutable output already exists'

def pin(path):
    path=Path(path);raw=path.read_bytes()
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def read(path):return json.loads(Path(path).read_text('utf8'))

def write(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def complete_prefix(path):
    raw=path.read_bytes();return raw[:raw.rfind(b'\n')+1]

stamp=datetime.now(timezone.utc)
command=('Get-CimInstance Win32_Process -Filter "ProcessId=28104 OR ProcessId=24420 OR ProcessId=41220" '
         '| Select-Object ProcessId,ParentProcessId,@{Name="CreationDate";Expression={$_.CreationDate.ToUniversalTime().ToString("o")}},CommandLine | ConvertTo-Json -Depth 3')
procs=json.loads(subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,encoding='utf8'))
assert {v['ProcessId'] for v in procs}=={28104,24420,41220}
assert any(v['ProcessId']==28104 and 'True_resume1.json' in v['CommandLine'] for v in procs)
assert any(v['ProcessId']==41220 and 'execute_final_native_after_training_v1.py --run' in v['CommandLine'] for v in procs)
prefix=complete_prefix(T/'gpu_resume1/paired_updates.jsonl')
rows=[json.loads(v) for v in prefix.splitlines()]
assert [v['paired_round'] for v in rows]==list(range(101,rows[-1]['paired_round']+1))
assert not (T/'process_resume1/terminal.json').exists()
assert not (E/'learned_process_v2_run1/supervisor_failure.json').exists()
speed=statistics.mean(v['round_seconds'] for v in rows[-25:])
gpu=[v.strip() for v in subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True).strip().split(',')]
S.mkdir();(S/'training_log_prefix.jsonl').write_bytes(prefix)
(S/'evaluation_wait_prefix.jsonl').write_bytes(complete_prefix(E/'learned_process_v2_run1/wait_observations.jsonl'))
status=dict(created_UTC=stamp.isoformat(),processes=procs,training_session=95471,evaluation_supervisor_session=42698,
    latest_training_row=rows[-1],mean_last25_round_seconds=speed,
    update_only_remaining_seconds=(1000-rows[-1]['paired_round'])*speed,
    estimated_update_finish_Asia_Shanghai=(stamp+timedelta(seconds=(1000-rows[-1]['paired_round'])*speed)).astimezone(timezone(timedelta(hours=8))).isoformat(),
    whole_machine_GPU=dict(name=gpu[0],utilization_pct=float(gpu[1]),memory_used_MiB=float(gpu[2]),memory_total_MiB=float(gpu[3])),
    training_stderr_bytes=(T/'process_resume1/stderr.log').stat().st_size,
    training_terminal_exists=False,policy_evaluation_started=False,
    evaluation_supervisor_waiting_existing_worker=True,
    scope='Point-in-time OS and complete log observations. Mutable live logs are not pinned; immutable prefix copies are pinned.')
hp=T/'gpu_resume1/health_0500.json'
if hp.exists():
    h=read(hp);status['checkpoint500_health']=dict(passed=h['passed'],total=h['total'],checkpoint=h['checkpoint'])
write(S/'status.json',status)
files={
 'previous_handoff':P/'outputs/goal_extension_tripod_GPU_resume_milestone_v15.json',
 'plan':Path('D:/workspace/supermimic-chord-study/docs/research-plan-v0.35.md'),
 'offline_trainer':T/'train_shared_WM_A1_A2.py',
 'live_training_protocol':T/'protocol_CPU_GPU_True_resume1.json',
 'evaluation_supervisor_source':E/'execute_final_native_after_training_v1.py',
 'evaluation_supervisor_start':E/'learned_process_v2_run1/supervisor_start.json',
 'online_learning_delivery':P/'phone_tripod_online_learning_v1/SOURCE_DELIVERY_v1.json',
 'online_learning_source':P/'phone_tripod_online_learning_v1/train_online_matched_v1.py',
 'online_learning_pending_protocol':P/'phone_tripod_online_learning_v1/protocol_CPUFalse_v1.json',
 'online_learning_recipe':P/'phone_tripod_online_learning_v1/recipe_v1.json',
 'online_learning_metadata':P/'phone_tripod_online_learning_v1/metadata_CPUFalse_v1.json',
 'online_pool_numpy_contract':P/'phone_tripod_online_learning_v1/pool_merge_numpy_proof_v1.json',
 'online_collector_delivery':P/'phone_tripod_online_collection_v1/SOURCE_DELIVERY_v3.json',
 'online_collector_contract':P/'phone_tripod_online_collection_v1/source_contract_v3.json',
 'WM_diagnostic_source':P/'phone_tripod_final_WM_diagnostics_v1/compare_final_WM_v2.py',
 'WM_diagnostic_pending_protocol':P/'phone_tripod_final_WM_diagnostics_v1/protocol_CPUFalse_v2.json',
 'WM_diagnostic_metadata':P/'phone_tripod_final_WM_diagnostics_v1/metadata_CPUFalse_v2.json',
 'WM_diagnostic_revision':P/'phone_tripod_final_WM_diagnostics_v1/source_revision_v2.json',
 'status_snapshot':S/'status.json','immutable_training_log_prefix':S/'training_log_prefix.jsonl',
 'immutable_evaluation_wait_prefix':S/'evaluation_wait_prefix.jsonl','freezer':Path(__file__)}
pins={k:pin(v) for k,v in files.items()}
assert pins['offline_trainer']['sha256']=='a1bcf9f820e00d7da4ac515dd81d45d39a2c59abd07c85319d1346db384b2439'
assert pins['online_learning_source']['sha256']=='ab7a9fde8ddbf58d2dbf447605cb597f8fc8a8bb839f0beaa0356ae88961519b'
write(OUT,dict(schema='tripod_online_loop_SOURCE_goal_milestone:16',created_UTC=stamp.isoformat(),
    previous_goal_turn_classification='progress',this_goal_turn_classification='progress',
    native_goal_status_last_verified='active',original_full_goal_complete=False,
    current_live_status=status,offline_fixed1000_completed=False,
    original4case_policy_evaluation_supervisor_live=True,actual_learned_Actor_native_evaluation_started=False,
    online_actual_collection_or_learning_started=False,new_learning_experiment_started_this_turn=False,
    new_online_SOURCE_covers='Three real collect/update cycles; current cycle100 Actor supplies next collection; frozen norms/scales; fresh5Adam atnewstage; exact ownresume no recollection; first10 healthy+ETA exits.',
    online_pure_merge_checks=dict(passed=8,total=8),collector_pure_contract_checks=dict(passed=8,total=8),
    source_peer_scope='Read-only full runner/collector and Root full source plus final join deltas; no known blocker. SOURCE checks do not qualify real runtime.',
    new_WM_diagnostic_actual_NN_calls=0,new_stage_actual_NN_or_native_calls=0,
    standing_first_new_training_health_ETA_pause_still_required=True,goal_pause_due=False,
    next_action='Observe the same28104 and41220, never restart on observation timeout. After1000 producer health/terminal, waiting supervisor binds/runs all4 native cases once. Inspect results, run fixed finalWM diagnostics, then bind/review/execute newonline stage to first10 healthy+ETA and pause per user rule.',
    full_goal_remaining='Actual native policy success and online learning verification, contact/gradient response limits, calibrated robust assets/dynamics and source gravity, independent ego/RGB hand retarget input, full61frame execution, matched multi-trajectory shared cost and hardware observation/control contract. No stable CHORD benefit or full completion established.',
    evidence_pins=pins))
print(json.dumps(dict(output=pin(OUT),round=rows[-1]['paired_round'],remaining_minutes=status['update_only_remaining_seconds']/60,live_processes=[v['ProcessId'] for v in procs]),ensure_ascii=False))
