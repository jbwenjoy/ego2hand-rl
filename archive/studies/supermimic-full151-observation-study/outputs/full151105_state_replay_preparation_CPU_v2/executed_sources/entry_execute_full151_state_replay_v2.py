"""Bounded native CUDA B1 Reader replay. Default is metadata only, no GPU."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'vendor'))
from full151_native.io import read, pin, sha, checked, jsonable
from full151_native.metadata import write

SCHEMA = 'full151105_matched_native_GPU_reader_replay_run:1'
ROLE = 'full151105_native_GPU_state_replay'
KEYS = {'schema', 'role', 'protocol', 'output', 'source_pins', 'CPU_ready', 'CPU_contract_report',
        'GPU_run_authorized', 'training_authorized', 'Env_or_physics_authorized', 'automatic_retry',
        'compile_fallback', 'soft_worker_seconds', 'hard_process_seconds', 'evaluation_seed', 'device', 'allow_tf32'}
PROTOCOL = ROOT / 'protocols/full151105-matched-native-GPU-reader-replay-prereg-v1.json'
PROTOCOL_SHA = 'fddcf970d7a8e44732fe93e6c83d1918fe017d4d5df159ae5c851a77e15fca6c'
MODULE = ROOT / 'src/full151_gradient/state_replay105_v2.py'
SUPERVISOR = ROOT / 'scripts/run_full151_state_replay_v2.py'
CHECKER = ROOT / 'scripts/check_full151_state_replay_cpu_v2.py'
DRIVER = Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py')
DRIVER_SHA = 'f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475'
LIFECYCLE = Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
LIFECYCLE_SHA = '40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'
IO_SHA = '159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def same_pin(left, right):
    return left['sha256'] == right['sha256'] and checked(left).samefile(checked(right))


def archived(value, sources):
    return any(item['sha256'] == value['sha256'] and same_pin(item, value) for item in sources.values())


def validate_front_metadata(c):
    """Exact namespace and declared scope; callable with small CPU fixtures."""
    if type(c) is not dict or set(c) != KEYS or c['schema'] != SCHEMA or c['role'] != ROLE:
        raise ValueError('Exact native GPU Reader replay namespace/keyset required')
    if type(c['evaluation_seed']) is not int or c['evaluation_seed'] != 0:
        raise ValueError('Fixed evaluation_seed0 required')
    if c['device'] != 'cuda:0' or c['allow_tf32'] is not True:
        raise ValueError('Fixed CUDA0/source-intended TF32-allowed condition required')
    if type(c['GPU_run_authorized']) is not bool or c['training_authorized'] is not False or c['Env_or_physics_authorized'] is not False:
        raise ValueError('Explicit GPU bool with noTraining/noEnv/noPhysics required')
    if c['automatic_retry'] is not False or c['compile_fallback'] is not False:
        raise ValueError('No retry or compile fallback')
    if type(c['soft_worker_seconds']) is not int or type(c['hard_process_seconds']) is not int or (c['soft_worker_seconds'], c['hard_process_seconds']) != (90, 120):
        raise ValueError('Fixed90soft/120hard owned lifecycle required')
    if type(c['output']) is not str or not Path(c['output']).is_absolute():
        raise ValueError('Absolute owned fresh output required')
    out = Path(c['output']).resolve()
    if out.parent != (ROOT / 'outputs').resolve() or out.exists():
        raise ValueError('Exact study output parent and fresh directory required')
    if type(c['source_pins']) is not dict or not c['source_pins']:
        raise ValueError('Flat immutable source closure required')


def validate_CPU_proof(c):
    ready, report = read(checked(c['CPU_ready'])), read(checked(c['CPU_contract_report']))
    checks = report.get('checks')
    if ready.get('schema') != 'full151105_state_replay_CPU_READY:1' or report.get('schema') != 'full151105_state_replay_CPU_contract:1' or report.get('status') != 'PASS_PURE_CPU_CONTRACT_NO_GPU':
        raise ValueError('Actual current-source state-replay CPU proof required')
    if type(checks) is not dict or not checks or not all(type(value) is bool and value for value in checks.values()):
        raise ValueError('All pure CPU contract checks must pass')
    if type(report.get('passed')) is not int or type(report.get('total')) is not int or report['passed'] != report['total'] or report['total'] != len(checks):
        raise ValueError('CPU proof counts differ')
    if not same_pin(ready['cpu_report'], c['CPU_contract_report']) or ready.get('cpu_checks') != {'passed': report['passed'], 'total': report['total']}:
        raise ValueError('READY/report count or identity mismatch')
    if not same_pin(ready['protocol'], c['protocol']) or not same_pin(report['protocol'], c['protocol']):
        raise ValueError('CPU proof/new replay preregistration mismatch')
    for key in ('actual_GPU_calls', 'native_state_calls', 'NPZ_input_reads', 'MJB_loads', 'model_loads'):
        if type(report.get(key)) is not int or report[key] != 0:
            raise ValueError('Synthetic CPU proof elevated execution scope: ' + key)
    expected = {'entry': Path(__file__), 'supervisor': SUPERVISOR, 'module': MODULE, 'checker': CHECKER}
    for key, path in expected.items():
        value = ready['source_pins'][key]
        if not same_pin(value, report['source_pins'][key]) or not same_pin(value, pin(path)):
            raise ValueError('Actual CPU-tested execution source differs: ' + key)
    for value in [c['CPU_ready'], c['CPU_contract_report'], *ready['source_pins'].values()]:
        if not archived(value, c['source_pins']):
            raise ValueError('CPU proof/source dependency not archived')


def preflight(path):
    if any(key in sys.modules for key in ('torch', 'warp', 'mujoco', 'mujoco_warp')):
        raise ValueError('Fresh noTorch/noWarp metadata process required')
    c = read(path)
    validate_front_metadata(c)
    for value in c['source_pins'].values():
        checked(value)
    if not same_pin(c['protocol'], {'path': str(PROTOCOL), 'sha256': PROTOCOL_SHA}):
        raise ValueError('Frozen matched B1 full58 replay protocol required')
    protocol = read(checked(c['protocol']))
    if protocol.get('schema') != 'full151105_matched_native_GPU_reader_replay_preregistration:1' or protocol.get('fixed_mode') != 'NATIVE_FP32_NO_AUTOCAST_CUDA_BATCH1':
        raise ValueError('Wrong replay scalar/backend/batch condition')
    if any(protocol.get(key) is not False for key in ('GPU_run_authorized', 'CPU_conversion_authorized', 'proxy_scoring_authorized', 'training_authorized')):
        raise ValueError('Preregisration is source-only; Root config grants separate GPU scope only')
    for value in protocol['inputs'].values():
        checked(value)
    for path_, digest in [(DRIVER, DRIVER_SHA), (LIFECYCLE, LIFECYCLE_SHA), (ROOT / 'vendor/runtime_io.py', IO_SHA)]:
        if sha(path_) != digest:
            raise ValueError('Reviewed lifecycle/atomic source changed')
    for path_ in (Path(__file__), SUPERVISOR, MODULE, CHECKER, DRIVER, LIFECYCLE, ROOT / 'vendor/runtime_io.py'):
        if not archived(pin(path_), c['source_pins']):
            raise ValueError('Execution dependency not archived: ' + str(path_))
        ast.parse(path_.read_text(encoding='utf-8-sig'))
    validate_CPU_proof(c)
    module = importlib.import_module('full151_gradient.state_replay105_v2')
    if not same_pin(pin(module.__file__), pin(MODULE)):
        raise ValueError('Unexpected imported replay core')
    module.validate_config(c)
    if any(key in sys.modules for key in ('torch', 'warp', 'mujoco', 'mujoco_warp')):
        raise AssertionError('Metadata preflight crossed backend boundary')
    return c, {'status': 'CPU_INPUT_SOURCE_PREFLIGHT_ONLY', 'config_sha256': sha(path), 'entry_sha256': sha(__file__),
               'schema': SCHEMA, 'role': ROLE, 'planned_counts': protocol['expected_counts'], 'source_pin_count': len(c['source_pins']),
               'Torch_imported': False, 'Warp_imported': False, 'GPU_started': False, 'Env_created': False, 'physics_steps': 0,
               'scope': 'Metadata/source qualification only; no NPZ transformation or native state/model/GPU execution'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    path = args.config.resolve(strict=True)
    if not args.run:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    c, proof = preflight(path)
    if not args.run:
        print(json.dumps(proof, indent=2))
        return
    if c['GPU_run_authorized'] is not True:
        raise ValueError('Root explicit fresh GPU authorization required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
        raise ValueError('Explicit GPU replay cannot inherit CPU-hidden GPU visibility')
    out = Path(c['output']).resolve()
    if args.output is None or args.output.resolve() != out or out.exists():
        raise ValueError('Fresh exact owned output; no overwrite/resume/retry')
    out.mkdir(parents=True, exist_ok=False)
    runtime = {'status': 'initializing', 'pid': os.getpid(), 'parent_pid': os.getppid(), 'started_utc': utc(),
               'plan_sha256': sha(path), 'source_sha256': sha(__file__), 'role': ROLE, 'automatic_retry': False,
               'optimizer_updates': 0, 'Env_created': False, 'physics_steps': 0, 'physical_gradient_qualified': False}
    write(out / 'runtime.json', runtime)
    body_error = None
    result_validation = None
    try:
        shutil.copy2(path, out / 'config.json')
        write(out / 'CPU_preflight.json', proof)
        source_folder = out / 'source'
        source_folder.mkdir()
        with zipfile.ZipFile(source_folder / 'execution_sources.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for index, value in enumerate(c['source_pins'].values()):
                archive.write(checked(value), '%04d_%s' % (index, Path(value['path']).name))
        write(source_folder / 'inventory.json', c['source_pins'])

        def emit(**updates):
            runtime.update(updates, worker_elapsed_seconds=time.perf_counter()-STARTED)
            write(out / 'runtime.json', runtime)

        core = importlib.import_module('full151_gradient.state_replay105_v2')
        result = core.run_replay(path, out, emit, started_perf=STARTED)
        write(out / 'result.json', jsonable(result))
        result_validation = core.validate_result(result)
        if not same_pin(result['config'], pin(path)):
            raise ValueError('Replay report actual config differs')
        runtime.update(status='completed_full151_native_GPU_state_replay', result_status=result['status'],
                       execution_complete=result_validation['execution_complete'],
                       all_original_gates_passed=result_validation['all_original_gates_passed'],
                       future_scoring_inputs_qualified=result_validation['future_scoring_inputs_qualified'],
                       exit_code=result_validation['exit_code'])
        write(out / 'completion.json', {'schema': 'full151105_native_GPU_state_replay_worker_completion:1',
              'config': pin(path), 'result_status': result['status'], 'result_validation': result_validation,
              'physical_gradient_qualified': False, 'whole_hidden_state_qualified': False,
              'factory_or_scoring_calls': 0, 'scope': 'Native Reader algebra replay only; separate saved review and future scoring authorization required'})
    except BaseException as exc:
        body_error = exc
        runtime.update(status='failed_full151_native_GPU_state_replay', error=repr(exc))
        try:
            write(out / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(), 'no_retry': True})
        except BaseException as saving:
            exc.add_note('Secondary failure-metadata write: '+repr(saving))
        raise
    finally:
        runtime.update(finished_utc=utc(), wall_seconds=time.perf_counter()-STARTED)
        try:
            write(out / 'runtime.json', runtime)
        except BaseException as saving:
            if body_error is None:
                raise
            body_error.add_note('Secondary terminal-runtime write: '+repr(saving))
    print(json.dumps({'status': result['status'], 'output': str(out), **result_validation,
                      'GPU_used': True, 'Env_created': False, 'physics_steps': 0, 'physical_gradient_qualified': False}, indent=2))
    raise SystemExit(result_validation['exit_code'])


if __name__ == '__main__':
    main()
