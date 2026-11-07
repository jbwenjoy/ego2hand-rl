"""Bounded105 GPU precision entry. Default is saved metadata only."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
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

SCHEMA = 'full151105_GPU_precision_diagnostic_run:1'
ROLE = 'full151105_GPU_precision_diagnostic'
MODES = ['FP32_STORAGE_TF32_ALLOWED', 'BF16_NET_FP32_INTEGRATE_TF32_ALLOWED']
COUNTS = {'strict_CPU_WM_loader_calls': 1, 'actual_checkpoint_reads': 2,
          'WM_forward_calls': 2, 'predicted_rows': 16, 'u_autograd_calls': 2,
          'physical_comparison_rows_per_mode': 64, 'dependent_direction_groups': 32,
          'Actor_forwards': 0, 'physics_steps': 0, 'optimizer_updates': 0}
DRIVER = Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py')
DRIVER_SHA = 'f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475'
LIFECYCLE = Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py')
LIFECYCLE_SHA = '40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa'
MODULE_SHA = 'b292630b290dd2d05f2987c3f0156bb2d301968680e8282ae086f66fb498964f'
PROTOCOL_SHA = '3a217f419784a71e9350589126fd738fc3d6bce2876c0316aed874abaf4ce5bc'
IO_SHA = '159c0d11e50017a4b5fab511e6594efb290955c5ee4a543f67374b7ed19675d1'


def utc():
    return datetime.now(timezone.utc).isoformat()


def same_pin(left, right):
    return left['sha256'] == right['sha256'] and checked(left).samefile(checked(right))


def validate_front_metadata(c):
    """Pure namespace/budget guard, independently testable without datasets."""
    if c.get('schema') != SCHEMA or c.get('role') != ROLE or c.get('modes') != MODES:
        raise ValueError('Exact105 precision namespace and fixed mode order required')
    if type(c.get('evaluation_seed')) is not int or c['evaluation_seed'] != 0:
        raise ValueError('Fixed evaluation_seed0 required')
    if c.get('device') != 'cuda:0' or c.get('allow_tf32') is not True:
        raise ValueError('Singlecuda0 TF32-allowed diagnostic required')
    if type(c.get('GPU_run_authorized')) is not bool or c.get('training_authorized') is not False or c.get('Env_or_physics_authorized') is not False:
        raise ValueError('Explicit GPU authorization and noEnv/noTraining scope required')
    if c.get('automatic_retry') is not False or c.get('compile_fallback') is not False:
        raise ValueError('No retry/compile fallback')
    if type(c.get('soft_worker_seconds')) is not int or type(c.get('hard_process_seconds')) is not int or (c['soft_worker_seconds'], c['hard_process_seconds']) != (60, 90):
        raise ValueError('Fixed60soft/90hard bounded worker required')
    if not Path(c['output']).resolve().is_relative_to((ROOT / 'outputs').resolve()):
        raise ValueError('Study-owned output required')


def validate_result(result):
    """Engineering counts/invariants only, never derivative-accuracy selection."""
    if result.get('schema') != 'full151105_GPU_precision_diagnostic_report:1' or result.get('status') != 'COMPLETED_GPU_PRECISION_DIAGNOSTIC' or result.get('modes') != MODES:
        raise ValueError('Complete exact105 GPU precision result required')
    for key, expected in COUNTS.items():
        if type(result.get(key)) is not int or result[key] != expected:
            raise ValueError('Actual precision count differs: ' + key)
    for key, expected in {'GPU_used': True, 'Env_created': False, 'compile_used': False,
                          'physical_gradient_qualified': False, 'whole_hidden_state_qualified': False}.items():
        if result.get(key) is not expected: raise ValueError('Result scope differs: ' + key)
    if set(result.get('models', {})) != set(MODES): raise ValueError('Both fixed modes must be retained')
    values = result.get('invariants')
    if type(values) is not dict or not values or not all(type(v) is bool and v for v in values.values()):
        raise ValueError('Precision invariants must all pass')


def validate_protocol_metadata(protocol):
    if protocol.get('schema') != 'full151105_GPU_precision_preregistration:1' or protocol.get('execution_authorized') is not False:
        raise ValueError('Exact source-only precision preregistration required')
    for key, value in {'modes': MODES, 'allow_tf32': True, 'device': 'cuda:0', 'evaluation_seed': 0,
                       'soft_worker_seconds': 60, 'hard_process_seconds': 90, 'automatic_retry': False,
                       'compile_fallback': False, 'training_authorized': False, 'Env_or_physics_authorized': False}.items():
        if json.dumps(protocol.get(key)) != json.dumps(value): raise ValueError('Protocol semantic differs: ' + key)
    expected = {**COUNTS, 'batch_rows_per_forward': 8, 'Env_created': False, 'compile_used': False}
    if json.dumps(protocol.get('expected_counts'), sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('Predeclared precision counts differ')


def validate_CPU_proof(c):
    ready = read(checked(c['CPU_ready'])); report = read(checked(c['CPU_contract_report']))
    checks = report.get('checks')
    if (report.get('schema') != 'full151105_GPU_precision_CPU_contract:1'
            or report.get('status') != 'PASS_PURE_CPU_CONTRACT_NO_MODEL'
            or type(checks) is not dict or not checks or not all(type(v) is bool and v for v in checks.values())
            or type(report.get('passed')) is not int or report['passed'] != report.get('total') or report['total'] != len(checks)):
        raise ValueError('Actual pure precision CPU contract incomplete')
    if not same_pin(ready['cpu_report'], c['CPU_contract_report']) or ready.get('cpu_checks') != {'passed': report['passed'], 'total': report['total']}:
        raise ValueError('CPU READY/report alias or counts differ')
    if report.get('actual_model_loads') != 0 or report.get('actual_model_forwards') != 0 or report.get('physics_steps') != 0 or report.get('GPU_used') is not False:
        raise ValueError('Preparation is pure CPU only')
    for key in ('CPU_ready', 'CPU_contract_report'):
        if not any(same_pin(c[key], value) for value in c['source_pins'].values() if value['sha256'] == c[key]['sha256']):
            raise ValueError('Actual CPU receipt not archived')
    for key in ('entry', 'supervisor', 'module', 'checker'):
        value = ready['source_pins'][key]
        if not same_pin(value, report['source_pins'][key]): raise ValueError('READY/report tested source differs: ' + key)
        expected = {'entry': Path(__file__), 'supervisor': ROOT / 'scripts/run_full151_precision_v1.py',
                    'module': ROOT / 'src/full151_gradient/precision105_v1.py',
                    'checker': ROOT / 'scripts/check_full151_precision_front_cpu_v1.py'}[key]
        if not same_pin(value, pin(expected)): raise ValueError('Executed CPU proof source drifted: ' + key)
    for value in ready['source_pins'].values():
        if not any(value['sha256'] == item['sha256'] and same_pin(value, item) for item in c['source_pins'].values()):
            raise ValueError('CPU-tested source dependency unarchived')


def preflight(path):
    if any(k in sys.modules for k in ('torch', 'warp', 'mujoco_warp')):
        raise ValueError('Fresh noTorch/noWarp metadata process required')
    c = read(path); validate_front_metadata(c)
    pins = c.get('source_pins')
    if type(pins) is not dict or not pins: raise ValueError('Flat frozen source closure required')
    for value in pins.values(): checked(value)
    module = ROOT / 'src/full151_gradient/precision105_v1.py'
    if sha(module) != MODULE_SHA: raise ValueError('Frozen precision module changed; new version/review required')
    required = (Path(__file__), ROOT / 'scripts/run_full151_precision_v1.py', module,
                ROOT / 'src/full151_gradient/analysis105_v1.py', DRIVER, LIFECYCLE, ROOT / 'vendor/runtime_io.py')
    for source in required:
        actual = pin(source)
        if not any(same_pin(value, actual) for value in pins.values() if value['sha256'] == actual['sha256']):
            raise ValueError('Actual execution dependency absent/drifted: ' + str(source))
        ast.parse(source.read_text(encoding='utf-8-sig'))
    if sha(DRIVER) != DRIVER_SHA or sha(LIFECYCLE) != LIFECYCLE_SHA or sha(ROOT / 'vendor/runtime_io.py') != IO_SHA:
        raise ValueError('Reviewed identity/lifecycle/atomicIO source changed')
    validate_CPU_proof(c)
    if c['protocol']['sha256'] != PROTOCOL_SHA: raise ValueError('Frozen precision protocol changed')
    protocol = read(checked(c['protocol'])); validate_protocol_metadata(protocol)
    for key, alias in (('CPU_prerequisite_config', 'CPU_analysis_config'), ('CPU_report', 'CPU_analysis_report'),
                       ('CPU_saved_review', 'CPU_saved_independent_review')):
        if not same_pin(c[key], protocol['inputs'][alias]): raise ValueError('Protocol/config actual CPU prerequisite differs: ' + key)
    from full151_gradient.precision105_v1 import preflight as module_preflight
    actual, p, old = module_preflight(path)
    if actual != c: raise ValueError('Front/module actual config differs')
    if any(k in sys.modules for k in ('torch', 'warp', 'mujoco_warp')):
        raise AssertionError('Metadata preflight crossed framework boundary')
    proof = {'status': 'CPU_INPUT_SOURCE_PREFLIGHT_ONLY', 'config_sha256': sha(path), 'entry_sha256': sha(__file__),
             'role': ROLE, 'schema': SCHEMA, 'modes': list(MODES), 'planned_counts': dict(COUNTS),
             'source_pin_count': len(pins), 'Torch_imported': False, 'Warp_imported': False,
             'GPU_started': False, 'Env_created': False, 'physics_steps': 0,
             'scope': 'Saved input/source qualification only; no model, precision forward or GPU run'}
    return c, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args(); path = args.config.resolve(strict=True)
    if not args.run: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    c, proof = preflight(path)
    if not args.run:
        print(json.dumps(proof, indent=2)); return
    if c['GPU_run_authorized'] is not True:
        raise ValueError('Root explicit GPU authorization required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') == '-1':
        raise ValueError('Explicit GPU run may not inherit hidden GPU visibility')
    out = Path(c['output']).resolve()
    if args.output is None or args.output.resolve() != out or out.exists():
        raise ValueError('Fresh exact owned output only; no overwrite/resume/retry')
    out.mkdir(parents=True, exist_ok=False)
    runtime = {'status': 'initializing', 'pid': os.getpid(), 'parent_pid': os.getppid(), 'started_utc': utc(),
               'plan_sha256': sha(path), 'source_sha256': sha(__file__), 'role': ROLE,
               'automatic_retry': False, 'optimizer_updates': 0, 'Env_created': False, 'physics_steps': 0,
               'physical_gradient_qualified': False}
    write(out / 'runtime.json', runtime)
    body_error = None
    try:
        shutil.copy2(path, out / 'config.json'); write(out / 'CPU_preflight.json', proof)
        folder = out / 'source'; folder.mkdir()
        with zipfile.ZipFile(folder / 'execution_sources.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for i, value in enumerate(c['source_pins'].values()): archive.write(checked(value), '%04d_%s' % (i, Path(value['path']).name))
        write(folder / 'inventory.json', c['source_pins'])
        def emit(**updates):
            runtime.update(updates, worker_elapsed_seconds=time.perf_counter() - STARTED)
            write(out / 'runtime.json', runtime)
        from full151_gradient.precision105_v1 import run
        result = run(path, out, emit)
        write(out / 'result.json', jsonable(result)); validate_result(result)
        if not same_pin(result['config'], pin(path)): raise ValueError('Module result config differs')
        runtime.update(status='completed_full151_GPU_precision', result_status=result['status'],
                       actual_counts={key: result[key] for key in COUNTS})
        write(out / 'completion.json', {'schema': 'full151105_GPU_precision_worker_completion:1', 'config': pin(path),
              'result_status': result['status'], 'actual_counts': runtime['actual_counts'], 'invariants': result['invariants'],
              'physical_gradient_qualified': False, 'scope': 'Numerical precision diagnostic only; independent saved review required'})
    except BaseException as exc:
        body_error = exc; runtime.update(status='failed_full151_GPU_precision', error=repr(exc))
        try: write(out / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(), 'no_retry': True})
        except BaseException as saving: exc.add_note('Secondary failure-metadata write: ' + repr(saving))
        raise
    finally:
        runtime.update(finished_utc=utc(), wall_seconds=time.perf_counter() - STARTED)
        try: write(out / 'runtime.json', runtime)
        except BaseException as saving:
            if body_error is None: raise
            body_error.add_note('Secondary terminal-runtime write: ' + repr(saving))
    print(json.dumps({'status': result['status'], 'output': str(out), 'actual_counts': runtime['actual_counts'],
                      'all_invariants': all(result['invariants'].values()), 'GPU_used': True,
                      'Env_created': False, 'physics_steps': 0, 'physical_gradient_qualified': False}, indent=2))


if __name__ == '__main__': main()
