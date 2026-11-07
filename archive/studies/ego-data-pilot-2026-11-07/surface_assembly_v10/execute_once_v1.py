"""Single owned subprocess, preserved terminal; no retry or process deadline."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import subprocess
import time

R = Path(__file__).resolve().parent
PYTHON = Path('D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe')

def pin(p):
    p = Path(p).resolve()
    b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def save(p, x):
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, indent=2, allow_nan=False)
        f.write('\n')

def main():
    assert not (R / 'run1_started.json').exists()
    assert not (R / 'completed_v1').exists()
    source = R / 'solve_assembly_support_v10.py'
    assert pin(source)['sha256'] == '44abfc1099d0359856003847b1577874c31eeeab5ea0d25badf902452dcd65ca'
    p = json.loads((R / 'protocol_metadata_v1.json').read_text(encoding='utf-8'))
    assert p['CPU_run_authorized'] is False
    p['CPU_run_authorized'] = True
    p['metadata_preflight'] = pin(R / 'metadata_preflight_v1' / 'report.json')
    p['source_derivation'] = pin(R / 'source_derivation_v1.json')
    protocol_path = R / 'protocol_run_v1.json'
    save(protocol_path, p)
    for rec in [*p['source_pins'].values(), *p['inputs'].values()]:
        assert pin(rec['path']) == rec
    cmd = [str(PYTHON), '-X', 'utf8', str(source), '--protocol', str(protocol_path), '--run']
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    t = time.perf_counter()
    with (R / 'run1_stdout.log').open('xb') as stdout, (R / 'run1_stderr.log').open('xb') as stderr:
        child = subprocess.Popen(cmd, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        start = {'schema': 'surface_assembly_process_started:10', 'pid': child.pid,
                 'command': cmd, 'started_UTC': started, 'source': pin(source),
                 'protocol': pin(protocol_path), 'executor': pin(__file__),
                 'automatic_retry': False, 'new_steps_training_GPU': 0}
        save(R / 'run1_started.json', start)
        print(json.dumps(start), flush=True)
        code = child.wait()
    terminal = {'schema': 'surface_assembly_process_terminal:10', 'pid': child.pid,
                'returncode': code, 'command': cmd, 'started_UTC': started,
                'ended_UTC': dt.datetime.now(dt.timezone.utc).isoformat(),
                'elapsed_s': time.perf_counter() - t, 'source': pin(source),
                'protocol': pin(protocol_path), 'stdout': pin(R / 'run1_stdout.log'),
                'stderr': pin(R / 'run1_stderr.log'), 'automatic_retry': False}
    if (R / 'completed_v1' / 'report.json').exists():
        terminal['report'] = pin(R / 'completed_v1' / 'report.json')
    save(R / 'run1_terminal.json', terminal)
    print(json.dumps(terminal), flush=True)
    raise SystemExit(code)

if __name__ == '__main__':
    main()
