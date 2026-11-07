"""Retained hidden precision worker; default CPU metadata, explicit GPU once."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from full151_native.io import read, sha
from full151_native.metadata import write

DRIVER = Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py')
DRIVER_SHA = 'f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475'
ENTRY = ROOT / 'scripts/execute_full151_precision_v1.py'
HIDDEN = (subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == 'nt' else 0
WORKER_PATTERN = r'[\\/](execute_full151_(?:a0|learning|policy_eval|gradient|precision)_v[0-9]+|train_rough_prefix_pen(?:_v[0-9]+)?|track_prefix_warp|train_controlled_pen|evaluate_pen_checkpoint|evaluate_confirmation_checkpoint|evaluate_c2_portfolio|probe_world_model_gradients|rest_hold_warp)\.py(?:[\s\"\x27]|$)'


def driver():
    if sha(DRIVER) != DRIVER_SHA: raise ValueError('Reviewed identity driver changed')
    spec = importlib.util.spec_from_file_location('full151_precision_reused_v3_identity', DRIVER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.ROOT = ROOT; module.WORKER = ENTRY
    return module


def preflight(config, python):
    p = read(config)
    for value in p['source_pins'].values():
        if sha(value['path']) != value['sha256']: raise ValueError('Frozen source changed')
    for path in (Path(__file__), ENTRY, DRIVER):
        if not any(value['sha256'] == sha(path) and Path(value['path']).samefile(path) for value in p['source_pins'].values()):
            raise ValueError('Supervisor/entry/identity dependency unpinned')
    process = subprocess.Popen([str(python), str(ENTRY), '--config', str(config)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, creationflags=HIDDEN,
                               env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'})
    try:
        stdout, stderr = process.communicate(timeout=20)
    except BaseException as exc:
        cleanup = {'status': 'UNVERIFIED', 'retained_CPU_launcher_pid': process.pid}
        try: cleanup['owned_tree_cleanup'] = driver().lifecycle().stop_owned(process)
        except BaseException as error: cleanup['cleanup_error'] = repr(error)
        try: process.communicate(timeout=1)
        except BaseException:
            if process.stdout: process.stdout.close()
            if process.stderr: process.stderr.close()
        raise RuntimeError('CPU preflight interrupted/timeout; GPU NOT_RUN; retained owned cleanup ' + repr(cleanup)) from exc
    if process.returncode: raise ValueError('CPU preflight failed ' + stderr + stdout)
    proof = json.loads(stdout)
    if (proof.get('status') != 'CPU_INPUT_SOURCE_PREFLIGHT_ONLY' or proof.get('Torch_imported') is not False
            or proof.get('Warp_imported') is not False or proof.get('GPU_started') is not False
            or proof.get('config_sha256') != sha(config) or proof.get('entry_sha256') != sha(ENTRY)):
        raise ValueError('Default CPU source/import boundary failed')
    return proof


def reject_other_workers():
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' -and $_.ProcessId -ne " + str(os.getpid()) + " -and $_.CommandLine -match '" + WORKER_PATTERN.replace("'", "''") + "' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', query],
                            capture_output=True, text=True, timeout=10, check=True, creationflags=HIDDEN)
    if result.stdout.strip(): raise ValueError('Another project GPU worker exists ' + result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--python', type=Path, default=Path('D:/workspace/dex-supermimic-research/.venv/Scripts/python.exe'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(); config = args.config.resolve(strict=True)
    d = driver(); identity_python = d.selected_python_identity(args.python)
    proof = preflight(config, args.python); proof['selected_python_identity'] = identity_python
    if not args.run:
        print(json.dumps(proof, indent=2)); return
    p = read(config)
    if p['GPU_run_authorized'] is not True: raise ValueError('Separate Root GPU authorization absent')
    output = Path(p['output']).resolve()
    if not output.is_relative_to((ROOT / 'outputs').resolve()) or output.exists():
        raise ValueError('Owned fresh output required')
    logs = output.with_name(output.name + '_watchdog'); logs.mkdir(parents=True, exist_ok=False)
    write(logs / 'preflight.json', proof); shutil.copy2(config, logs / 'config_snapshot.json')
    shutil.copy2(__file__, logs / 'supervisor_source.py')
    life = d.lifecycle(); watch_error = None; watch_invoked = False
    try:
        with life.gpu_lock():
            reject_other_workers()
            if sha(config) != proof['config_sha256'] or sha(ENTRY) != proof['entry_sha256']:
                raise ValueError('Config/entry changed after CPU preflight')
            command = [str(args.python), '-u', str(ENTRY), '--config', str(config), '--output', str(output), '--run']
            observer = lambda pid, parent: life.observe_actual_pid(pid, parent, ENTRY, config, output)
            watch_invoked = True
            try: result = life.watch(command, output, logs, hard_seconds=p['hard_process_seconds'], observer=observer)
            except BaseException as exc:
                watch_error = exc
                result = read(logs / 'state.json') if (logs / 'state.json').is_file() else {'status': 'MISSING_TERMINAL_METADATA'}
            post = {'status': 'UNVERIFIED', 'queried_pids': [], 'actual_CIM_rows': None}
            try: post = d.terminal_processes([result.get('launcher_pid'), result.get('actual_pid')])
            except BaseException as exc: post['error'] = repr(exc)
            write(logs / 'post_terminal_CIM.json', post)
            try: runtime = read(output / 'runtime.json')
            except BaseException as exc: runtime = {'read_error': repr(exc)}
            identity = d.qualify_terminal_identity(result, runtime, post, plan=config, output=output, command=command,
                       plan_sha=proof['config_sha256'], worker_sha=proof['entry_sha256'], python_identity=identity_python)
            identity.update(supervisor_sha256=sha(__file__), identity_source_sha256=DRIVER_SHA)
            write(logs / 'engineering_identity.json', identity)
    except BaseException as exc:
        if not watch_invoked:
            write(logs / 'state.json', {'status': 'failed_no_child_started', 'supervisor_pid': os.getpid(),
                'finished_utc': datetime.now(timezone.utc).isoformat(), 'watch_invoked': False, 'Popen_attempted': False,
                'config_sha256': proof['config_sha256'], 'error': repr(exc), 'automatic_retry': False})
            write(logs / 'engineering_identity.json', {'status': 'NO_CHILD_IDENTITY_NOT_APPLICABLE', 'engineering_identity_qualified': False,
                'physical_outcome': 'NOT_RUN', 'watch_invoked': False,
                'PID_or_CIM_absence_claim': 'No child launch attempted; no PID/CIM emptiness fabricated'})
        raise
    print(json.dumps({'watchdog_status': result.get('status'), **identity}, indent=2))
    if watch_error is not None: raise watch_error
    raise SystemExit(0 if identity['engineering_identity_qualified'] else 1)


if __name__ == '__main__': main()
