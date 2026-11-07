"""Wait for the existing worker, then execute the four fixed native cases once.

This supervisor never trains, restarts training, selects a checkpoint, or
changes physics. Default inspects only source/JSON and the OS process table.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, os, subprocess, sys, time, traceback

H = Path(__file__).resolve().parent
T = H.parent/'phone_tripod_supermimic_v1'
BASE = H/'learned_CPUFalse_v2.json'
REVIEW = H/'source_review_windows_v2/report.json'
ACTUAL_TRAINING = T/'protocol_CPU_GPU_True_resume1.json'
TRAIN_OUT = T/'gpu_resume1'
TRAIN_TERMINAL = T/'process_resume1/terminal.json'
EXPECTED_PID = 28104
EXPECTED_CREATION = '2026-11-07T19:17:48.751826+08:00'
PROCESS = H/'learned_process_v2_run1'
BOUND_FALSE = H/'learned_bound_CPUFalse_v2_run1.json'
BOUND_TRUE = H/'learned_CPUTrue_v2_run1.json'

def need(ok, message):
    if not ok: raise ValueError(message)

def read(path): return json.loads(Path(path).read_text(encoding='utf8'))

def pin(path):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def checked(value):
    need(pin(value['path']) == value, 'Current immutable input/source pin')
    return Path(value['path'])

def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()

def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()

def existing_worker():
    command = ('Get-CimInstance Win32_Process -Filter "ProcessId=28104 OR ProcessId=24420" '
               '| Select-Object ProcessId,ParentProcessId,@{Name="CreationDate";Expression={$_.CreationDate.ToUniversalTime().ToString("o")}},CommandLine | ConvertTo-Json -Depth 3')
    result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True, encoding='utf8', timeout=15, check=True)
    items = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(items, dict): items = [items]
    workers = [item for item in items if item['ProcessId'] == EXPECTED_PID]
    if workers:
        item = workers[0]
        observed = datetime.datetime.fromisoformat(item['CreationDate'].replace('Z', '+00:00'))
        expected = datetime.datetime.fromisoformat(EXPECTED_CREATION)
        need(abs((observed-expected).total_seconds()) < .000001 and item['ParentProcessId'] == 24420,
             'Same already-running worker, not reused PID')
        need('train_shared_WM_A1_A2.py' in item['CommandLine'] and 'protocol_CPU_GPU_True_resume1.json' in item['CommandLine'], 'Exact existing training command')
    return items

def preflight():
    p = read(BASE)
    need(pin(BASE)['sha256'] == '6965b1df10d9a225066407dd077db3d3e6d3584c5253346154ff1ab10ecf15cc', 'Frozen pending evaluator protocol')
    need(p['source']['sha256'] == 'bf154e3810bfc4a587c03e70281ff250a5079fb5ff382c54a733662663fa2c43', 'Frozen evaluator v2')
    review = read(REVIEW)
    need(review['passed'] == review['total'] == 14 and all(value is True for value in review['checks'].values()) and review['source'] == p['source'], 'Positive exact-source peer review')
    for value in [p['source'], *p['sources'].values()]: checked(value)
    for value in p['inputs'].values():
        if value is not None: checked(value)
    train = read(ACTUAL_TRAINING)
    need(train['device'] == 'cuda:0' and train['sources']['trainer'] == p['sources']['trainer'] and train['recipe']['paired_rounds'] == 1000, 'Same fixed training target/source')
    need(all(p['inputs'][key] is None for key in ['training_final_protocol', 'final_result', 'final_health', 'final_terminal']), 'No fabricated final inputs')
    need(not any(p[key] for key in ['CPU_run_authorized', 'NN_forward_authorized', 'native_execution_authorized', 'GPU_run_authorized', 'training_authorized']), 'Pending protocol has no executed backend')
    need(not any(path.exists() for path in [PROCESS, BOUND_FALSE, BOUND_TRUE, Path(p['output'])]), 'Fresh single execution namespace')
    return p

def execute():
    p = preflight()
    initial = existing_worker()
    need(TRAIN_TERMINAL.exists() or any(v['ProcessId'] == EXPECTED_PID for v in initial), 'Observe live original worker or its terminal; never relaunch')
    PROCESS.mkdir()
    began = time.perf_counter()
    start = utc()
    write(PROCESS/'supervisor_start.json', dict(started_UTC=start, existing_processes=initial, expected_worker_pid=EXPECTED_PID, expected_creation=EXPECTED_CREATION, launcher=pin(__file__), base_protocol=pin(BASE), peer_review=pin(REVIEW), no_new_learning=True, training_restarts=0))
    print(json.dumps(dict(supervisor_pid=os.getpid(), waiting_for_existing_worker=EXPECTED_PID, process_directory=str(PROCESS), new_learning=False)), flush=True)
    phase = 'wait_existing_training'
    rc = None
    child = None
    try:
        deadline = time.monotonic()+3*3600
        with (PROCESS/'wait_observations.jsonl').open('x', encoding='utf8') as observations:
            while not TRAIN_TERMINAL.exists():
                need(time.monotonic() < deadline, 'Wait deadline reached; inspect same process, no restart')
                try:
                    items = existing_worker()
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError) as error:
                    observations.write(json.dumps(dict(UTC=utc(), transient_observation_error=repr(error)))+'\n')
                    observations.flush()
                    time.sleep(10)
                    continue
                observations.write(json.dumps(dict(UTC=utc(), processes=items))+'\n')
                observations.flush()
                if not items:
                    time.sleep(2)
                    need(TRAIN_TERMINAL.exists(), 'Original processes terminal without terminal artifact; inspect, no restart')
                else:
                    time.sleep(15)
        phase = 'bind_fixed_final1000'
        terminal = read(TRAIN_TERMINAL)
        need(terminal['pid'] == EXPECTED_PID and terminal['returncode'] == 0 and terminal['protocol'] == pin(ACTUAL_TRAINING) and terminal['source'] == p['sources']['trainer'], 'Original worker completed successfully')
        final_path = TRAIN_OUT/'result.json'
        final = read(final_path)
        need(final['status'] == 'COMPLETED_FIXED_OFFLINE_MATCHED1000_NOT_NATIVE_POLICY_QUALIFICATION' and final['paired_round'] == 1000 and final['shared_WM_updates'] == 1100, 'Only fixed final1000')
        checked(final['checkpoint'])
        for round_id in [500, 1000]:
            health = read(TRAIN_OUT/f'health_{round_id:04d}.json')
            need(health['paired_round'] == round_id and health['shared_WM_updates'] == 100+round_id and health['checks'] and all(value is True for value in health['checks'].values()), 'Actual saved checkpoint health')
            checked(health['checkpoint'])
        need(read(TRAIN_OUT/'health_1000.json')['checkpoint'] == final['checkpoint'], 'Final health joins exact model')
        p['inputs'].update(training_final_protocol=pin(ACTUAL_TRAINING), final_result=pin(final_path), final_health=pin(TRAIN_OUT/'health_1000.json'), final_terminal=pin(TRAIN_TERMINAL))
        p['root_review'] = dict(base_protocol=pin(BASE), peer_review=pin(REVIEW), launcher=pin(__file__), scope='Execute all four fixed final1000 native cases once after original training terminal0 and saved500/1000 health. No learning, checkpoint selection, physics edits, retry or stage launch.')
        write(BOUND_FALSE, p)
        phase = 'bound_metadata_preflight'
        meta = subprocess.run([sys.executable, '-X', 'utf8', p['source']['path'], '--protocol', str(BOUND_FALSE)], capture_output=True, text=True, encoding='utf8')
        write(PROCESS/'bound_metadata_preflight.json', dict(returncode=meta.returncode, stdout=meta.stdout, stderr=meta.stderr, protocol=pin(BOUND_FALSE)))
        need(meta.returncode == 0, 'Final metadata preflight must pass before native execution')
        for key in ['CPU_run_authorized', 'NN_forward_authorized', 'native_execution_authorized']: p[key] = True
        p['root_review']['bound_false_protocol'] = pin(BOUND_FALSE)
        write(BOUND_TRUE, p)
        phase = 'native_four_case_evaluation'
        command = [sys.executable, '-X', 'utf8', p['source']['path'], '--protocol', str(BOUND_TRUE), '--run']
        with (PROCESS/'stdout.log').open('x', encoding='utf8') as stdout, (PROCESS/'stderr.log').open('x', encoding='utf8') as stderr:
            child = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            print(json.dumps(dict(phase=phase, pid=child.pid, UTC=utc())), flush=True)
            rc = child.wait()
        write(PROCESS/'terminal.json', dict(schema='phone_tripod_final_Actor_evaluation_supervisor_terminal:1', returncode=rc, pid=child.pid, command=command, protocol=pin(BOUND_TRUE), source=p['source'], launcher=pin(__file__), started_UTC=start, ended_UTC=utc(), elapsed_s=time.perf_counter()-began, training_terminal=pin(TRAIN_TERMINAL), stdout=pin(PROCESS/'stdout.log'), stderr=pin(PROCESS/'stderr.log'), automatic_retry=False, new_learning_launched=False))
        print(json.dumps(dict(returncode=rc, terminal=pin(PROCESS/'terminal.json'))), flush=True)
        return rc
    except BaseException as error:
        write(PROCESS/'supervisor_failure.json', dict(phase=phase, error=repr(error), traceback=traceback.format_exc(), UTC=utc(), child_pid=child.pid if child else None, child_returncode=rc, automatic_retry=False, training_restarts=0))
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.run: raise SystemExit(execute())
    p = preflight()
    print(json.dumps(dict(status='SOURCE_METADATA_NO_NN_NATIVE_OR_WAIT', current_processes=existing_worker(), output=p['output'], maximum_cases=4, maximum_native_steps=38400, maximum_Actor_FW=400, new_learning=False)))
