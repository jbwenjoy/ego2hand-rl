"""Saved-only A1L primary initialization pairing and launch-record recovery review."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151_v3_A1L_primary_smoke_initial_pairing_v1'
CASES = {'A1L_primary': 'full151_v3_primary2000_A1L_seed0_v1',
         'A1L_smoke': 'full151_v2_smoke40_A1L_seed0_v1',
         'A2_primary': 'full151_v3_primary2000_A2_seed0_v1'}
CONFIGS = {'A1L_primary': 'full151-v3-primary2000-A1L-seed0-v1.json',
           'A1L_smoke': 'full151-v2-smoke40-A1L-seed0-v1.json',
           'A2_primary': 'full151-v3-primary2000-A2-seed0-v1.json'}
MODULES = ('wm', 'fail_head', 'actor', 'critic')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def pin(p):
    p = Path(p).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    OUT.mkdir(exist_ok=False)
    own = pin(__file__); (OUT / 'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    import torch
    torch.set_num_threads(1)
    checks = {}; raws = {}; fps = {}; cfg = {}; artifacts = {}; layouts = {}
    def need(k, v): checks[k] = bool(v)
    for name, folder in CASES.items():
        run = ROOT / 'outputs' / folder
        cp = ROOT / 'configs' / CONFIGS[name]; cfg[name] = read(cp)
        raw = torch.load(run / 'initial_parameters.pt', map_location='cpu', weights_only=False)
        raws[name] = raw; expected = read(run / 'initial_parameters.json'); fps[name] = {}; layouts[name] = {}
        need(name + '_actual_schema2_seed0_and_capture',
             cfg[name]['schema'] == 'full151_knownrough_native_training_run:2'
             and cfg[name]['seed'] == 0 and raw['schema'] == 'full151_native_initial_parameters:1'
             and raw['capture_point'] == 'after_native_DexRunner_init_before_first_collect_or_optimizer'
             and set(raw['modules']) == set(MODULES))
        for module in MODULES:
            m = raw['modules'][module]; values = m['parameters']; names = m['parameter_names']; h = hashlib.sha256()
            valid = len(values) == len(names) == len(set(names)) and bool(values)
            valid = valid and m['shapes'] == [list(v.shape) for v in values] and m['dtypes'] == [str(v.dtype) for v in values]
            for value in values:
                valid = valid and value.device.type == 'cpu' and bool(torch.isfinite(value).all())
                h.update(str((str(value.dtype), tuple(value.shape))).encode())
                h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            fps[name][module] = {'sha256': h.hexdigest(), 'tensors': len(values), 'parameters': sum(v.numel() for v in values)}
            layouts[name][module] = {'names': names, 'shapes': m['shapes'], 'dtypes': m['dtypes']}
            need(name + '_' + module + '_raw_matches_initial_JSON', valid and fps[name][module] == expected[module])
        artifacts[name] = {'raw_initial': pin(run / 'initial_parameters.pt'),
                           'initial_JSON': pin(run / 'initial_parameters.json'), 'config': pin(cp)}
    need('proper_case_methods_roles_and_common_controller_input',
         cfg['A1L_primary']['method'] == cfg['A1L_smoke']['method'] == 'A1L'
         and cfg['A2_primary']['method'] == 'A2'
         and cfg['A1L_primary']['role'] == cfg['A2_primary']['role'] == 'primary_matched'
         and cfg['A1L_smoke']['role'] == 'engineering_smoke'
         and all(cfg[k]['input_bundle'] == cfg['A1L_primary']['input_bundle']
                 and cfg[k]['controller'] == cfg['A1L_primary']['controller'] for k in CASES))
    same = True
    for other in ('A1L_smoke', 'A2_primary'):
        need(other + '_capture_source_identity_exact', raws[other]['source_pins'] == raws['A1L_primary']['source_pins'])
        for module in MODULES:
            a = raws['A1L_primary']['modules'][module]; b = raws[other]['modules'][module]
            exact = (layouts['A1L_primary'][module] == layouts[other][module]
                     and len(a['parameters']) == len(b['parameters'])
                     and all(x.shape == y.shape and x.dtype == y.dtype
                             and x.contiguous().view(torch.uint8).numpy().tobytes()
                             == y.contiguous().view(torch.uint8).numpy().tobytes()
                             for x, y in zip(a['parameters'], b['parameters'])))
            need(other + '_' + module + '_vs_A1L_primary_name_order_shape_dtype_bytes', exact)
            same = same and exact
    ld = ROOT / 'outputs/full151_v3_primary2000_A1L_launch_v1'
    launch = read(ld / 'launch.json'); recovery = read(ld / 'launch_record_recovery.json')
    initial = read(ld / 'initial_CIM_recovered.json')
    bad = Path(recovery['failed_source']); text = bad.read_text(encoding='utf-8-sig')
    need('failed_launch_source_preserved_and_error_after_one_Start_Process',
         pin(bad)['sha256'] == recovery['failed_source_sha256']
         and text.count('Start-Process -FilePath') == 1
         and '-Algorithm SHA1L56' in text
         and text.index('Start-Process -FilePath') < text.index('-Algorithm SHA1L56')
         and recovery['relaunch_performed'] is False and recovery['process_modified'] is False)
    need('recovered_record_pins_actual_config_worker_supervisor',
         launch['schema'] == 'full151_owned_hidden_supervisor_launch:2'
         and launch['config_sha256'] == artifacts['A1L_primary']['config']['sha256']
         and launch['worker_sha256'] == pin(ROOT / 'scripts/execute_full151_learning_v3.py')['sha256']
         and launch['supervisor_sha256'] == pin(ROOT / 'scripts/run_full151_learning_v3.py')['sha256']
         and pin(launch['initial_CIM']['path'])['sha256'] == launch['initial_CIM']['sha256']
         and pin(launch['failure_recovery']['path'])['sha256'] == launch['failure_recovery']['sha256'])
    ids = [29736, 34456, 36500, 16096]
    need('recovered_exact_original_four_PID_chain', [v['pid'] for v in initial] == recovery['original_four_PID_chain'] == ids
         and launch['retained_launcher_pid'] == recovery['actual_original_retained_launcher_pid'] == ids[0]
         and all(initial[i]['parent_pid'] == ids[i - 1] for i in (1, 2, 3)))
    tails = [['-u', launch['supervisor'], '--config', launch['config'], '--run'],
             ['-u', str(ROOT / 'scripts/execute_full151_learning_v3.py'), '--config', launch['config'],
              '--output', str(ROOT / 'outputs' / CASES['A1L_primary']), '--run']]
    need('recovered_raw_commands_and_executable_samefile',
         all([v.strip('"') for v in shlex.split(row['command_line'], posix=False)][1:]
             == tails[0 if i < 2 else 1]
             and os.path.samefile(shlex.split(row['command_line'], posix=False)[0].strip('"'), row['executable_path'])
             for i, row in enumerate(initial)))
    need('recovered_timestamp_explicitly_OS_creation_not_invocation',
         launch['launched_utc'] == initial[0]['creation_utc']
         and launch['launch_timestamp_source'] == 'Retained original launcher OS CreationDate; precise script invocation UTC was not saved before recording failure'
         and launch['launch_record_recovered_after_source_error'] is True)
    for key, value in raws['A1L_primary']['source_pins'].items():
        need('initial_capture_source_current_' + key, pin(value['path'])['sha256'] == value['sha256'])
    need('source_and_saved_initial_artifacts_unchanged', pin(__file__) == own
         and all(pin(v['path']) == v for table in artifacts.values() for v in table.values()))
    need('no_GPU_Env_native_model_forward', not torch.cuda.is_initialized()
         and not any(k.split('.')[0] in ('warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic') for k in sys.modules))
    ok = all(checks.values())
    report = {'schema': 'full151_A1L_primary_smoke_crossprimary_initial_pairing:1',
              'status': 'PASS_SAVED_RAW_INITIAL_PAIRING_AND_RECOVERED_LAUNCH_SCOPE' if ok else 'FAIL_SAVED_INITIAL_OR_RECOVERY_REVIEW',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'same_initial_four_parameter_fingerprints': same, 'raw_initial_fingerprints': fps,
              'parameter_layout': layouts, 'artifact_pins': artifacts,
              'source_pins': {'auditor': own, 'failed_launch_source': pin(bad), **raws['A1L_primary']['source_pins']},
              'launch_evidence': {k: pin(ld / k) for k in ('launch.json', 'launch_record_recovery.json', 'initial_CIM_recovered.json')},
              'scope': ['Three raw initial CPU artifacts loaded, no final model/weights/forward.',
                        'Saved recovered CIM establishes the specified original fourPID chain and exact commands. No new OS query, current liveness or terminal absence is asserted here.',
                        'One Start-Process precedes the invalid post-launch SHA1L56 expression in retained source; no relaunch/process change is Root recovery evidence, not a new machine-wide audit.',
                        'Original script-invocation UTC was lost; recovered launched_utc is explicitly original launcher OS CreationDate. Metadata failure is not retraining or a task-quality result.'],
              'execution': {'raw_initial_CPU_loads': 3, 'final_weights_loads': 0, 'model_forward': 0, 'Env': 0,
                            'GPU': 0, 'physics': 0, 'process_operations': 0}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'passed': report['passed'], 'total': report['total'],
                      'same_initial': same, 'status': report['status']}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
