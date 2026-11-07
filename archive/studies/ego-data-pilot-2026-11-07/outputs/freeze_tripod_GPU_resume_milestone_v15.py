"""Freeze an observation of the existing worker; never launch/stop training."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib, json, statistics, subprocess

P = Path('D:/workspace/ego-data-pilot-2026-11-07')
T = P/'phone_tripod_supermimic_v1'
OUT = P/'outputs/goal_extension_tripod_GPU_resume_milestone_v15.json'
SNAP = P/'outputs/tripod_GPU_resume_live_snapshot_v15'
assert not OUT.exists() and not SNAP.exists(), 'Immutable output already exists'

def pin(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

now = datetime.now(timezone.utc)
protocol = json.loads((T/'protocol_CPU_GPU_True_resume1.json').read_text('utf-8'))
assert protocol['device'] == 'cuda:0'
raw = (T/'gpu_resume1/paired_updates.jsonl').read_bytes()
prefix = raw[:raw.rfind(b'\n')+1]
rows = [json.loads(line) for line in prefix.splitlines()]
assert rows and rows[0]['paired_round'] == 101
assert [row['paired_round'] for row in rows] == list(range(101, rows[-1]['paired_round']+1))
latest = rows[-1]
speed = statistics.mean(row['round_seconds'] for row in rows[-25:])
remaining_s = (1000-latest['paired_round'])*speed
procs = json.loads(subprocess.check_output([
    'powershell', '-NoProfile', '-Command',
    'Get-CimInstance Win32_Process -Filter "ProcessId=28104 OR ProcessId=24420" | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3'
], text=True, encoding='utf-8'))
assert {p['ProcessId'] for p in procs} == {28104, 24420}, 'Inspect worker status before any further action'
assert any('train_shared_WM_A1_A2.py' in p['CommandLine'] and 'True_resume1.json' in p['CommandLine'] for p in procs)
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], text=True).strip()
name, util, used, total = [v.strip() for v in gpu.split(',')]
assert not (T/'process_resume1/terminal.json').exists(), 'Worker already finished; inspect terminal'

SNAP.mkdir()
(SNAP/'paired_updates_prefix.jsonl').write_bytes(prefix)
status = dict(created_UTC=now.isoformat(), observed_processes=procs, exec_session_id=95471,
              latest_training_row=latest, round_seconds_mean_last25=speed,
              update_only_remaining_seconds=remaining_s,
              estimated_finish_Asia_Shanghai=(now+timedelta(seconds=remaining_s)).astimezone(timezone(timedelta(hours=8))).isoformat(),
              whole_machine_GPU=dict(name=name, utilization_pct=float(util), memory_used_MiB=float(used), memory_total_MiB=float(total), memory_capacity_used_pct=100*float(used)/float(total)),
              stderr_bytes=(T/'process_resume1/stderr.log').stat().st_size,
              terminal_exists=False,
              scope='Point-in-time observation. Copied complete log prefix is immutable; live log paths and future progress are not hash-pinned.')
save(SNAP/'status.json', status)
evidence = {
    'previous_handoff':P/'outputs/goal_extension_tripod_matched_learning_milestone_v14.json',
    'plan':Path('D:/workspace/supermimic-chord-study/docs/research-plan-v0.34.md'),
    'trainer':T/'train_shared_WM_A1_A2.py',
    'resume_false_protocol':T/'protocol_CPU_GPUFalse_resume1.json',
    'resume_actual_protocol':T/'protocol_CPU_GPU_True_resume1.json',
    'resume_launcher':T/'execute_GPU_resume_once_v1.py',
    'resume_source_review':T/'resume_SOURCE_review_v1/report.json',
    'policy_eval_source':P/'phone_tripod_policy_evaluation_v1/evaluate_learned_actors_v2.py',
    'policy_eval_pending_protocol':P/'phone_tripod_policy_evaluation_v1/learned_CPUFalse_v2.json',
    'policy_eval_source_delivery':P/'phone_tripod_policy_evaluation_v1/learned_SOURCE_delivery_v2.json',
    'policy_eval_peer':P/'phone_tripod_policy_evaluation_v1/source_review_windows_v2/report.json',
    'tracking_numpy_check':P/'phone_tripod_policy_evaluation_v1/tracking_numpy_check_v2/report.json',
    'online_source_delivery':P/'phone_tripod_online_collection_v1/SOURCE_DELIVERY_v1.json',
    'live_status_snapshot':SNAP/'status.json',
    'immutable_training_log_prefix':SNAP/'paired_updates_prefix.jsonl',
    'freezer_source':Path(__file__),
}
pins = {key:pin(path) for key,path in evidence.items()}
assert pins['trainer']['sha256'] == 'a1bcf9f820e00d7da4ac515dd81d45d39a2c59abd07c85319d1346db384b2439'
save(OUT, dict(schema='tripod_GPU_resume_final_policy_SOURCE_goal_milestone:15',
    created_UTC=now.isoformat(), this_goal_turn_classification='progress',
    native_goal_status_last_verified='active', original_full_goal_complete=False,
    same_experiment_resumed=True, new_learning_experiment_started=False,
    standing_first_checkpoint_pause_already_served=True, goal_pause_due=False,
    background_training_verified_live=True, live_worker_PID=28104, supervisor_PID=24420,
    live_exec_session_id=95471, live_status=status,
    final1000_completed=False, actual_learned_Actor_native_evaluation_executed=False,
    policy_eval_SOURCE_checks=dict(passed=14,total=14), tracking_numpy_checks=dict(passed=26,total=26),
    online_SOURCE_contract_checks=dict(passed=8,total=8), actual_online_collection_or_learning_executed=False,
    next_action='Observe existing worker through fixed1000; do not restart on session loss. Verify500/1000 health and terminal0, then bind true resume protocol/final artifacts and run all four native policy cases. Continue separately versioned collection/replay learning consumer.',
    full_goal_remaining='Real contact dynamics/asset calibration and source gravity; successful native policy qualification; online learning loop; WM response limits; independent ego and RGB hand reconstruction; full61frame execution; matched multi-trajectory cost evidence and hardware contracts remain. No stable CHORD/holding benefit established.',
    evidence_pins=pins))
print(json.dumps(dict(output=pin(OUT), round=latest['paired_round'], remaining_minutes=remaining_s/60, GPU=status['whole_machine_GPU']), ensure_ascii=False))
