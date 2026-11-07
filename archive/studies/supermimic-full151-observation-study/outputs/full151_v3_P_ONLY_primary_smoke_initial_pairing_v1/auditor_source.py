"""Saved-only four-case P_ONLY primary initialization pairing."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151_v3_P_ONLY_primary_smoke_initial_pairing_v1'
CASES = {'P_ONLY_primary':'full151_v3_primary2000_P_ONLY_seed0_v1',
         'P_ONLY_smoke':'full151_v2_smoke40_P_ONLY_seed0_v1',
         'A1L_primary':'full151_v3_primary2000_A1L_seed0_v1',
         'A2_primary':'full151_v3_primary2000_A2_seed0_v1'}
CONFIGS = {'P_ONLY_primary':'full151-v3-primary2000-P_ONLY-seed0-v1.json',
           'P_ONLY_smoke':'full151-v2-smoke40-P_ONLY-seed0-v1.json',
           'A1L_primary':'full151-v3-primary2000-A1L-seed0-v1.json',
           'A2_primary':'full151-v3-primary2000-A2-seed0-v1.json'}
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
         cfg['P_ONLY_primary']['method']==cfg['P_ONLY_smoke']['method']=='P_ONLY'
         and cfg['A1L_primary']['method']=='A1L' and cfg['A2_primary']['method']=='A2'
         and all(cfg[k]['role']==('engineering_smoke' if k=='P_ONLY_smoke' else 'primary_matched') for k in CASES)
         and all(cfg[k]['input_bundle']==cfg['P_ONLY_primary']['input_bundle']
                 and cfg[k]['controller']==cfg['P_ONLY_primary']['controller'] for k in CASES))
    same = True
    for other in ('P_ONLY_smoke','A1L_primary','A2_primary'):
        need(other + '_capture_source_identity_exact', raws[other]['source_pins'] == raws['P_ONLY_primary']['source_pins'])
        for module in MODULES:
            a = raws['P_ONLY_primary']['modules'][module]; b = raws[other]['modules'][module]
            exact = (layouts['P_ONLY_primary'][module] == layouts[other][module]
                     and len(a['parameters']) == len(b['parameters'])
                     and all(x.shape == y.shape and x.dtype == y.dtype
                             and x.contiguous().view(torch.uint8).numpy().tobytes()
                             == y.contiguous().view(torch.uint8).numpy().tobytes()
                             for x, y in zip(a['parameters'], b['parameters'])))
            need(other + '_' + module + '_vs_P_ONLY_primary_name_order_shape_dtype_bytes', exact)
            same = same and exact
    for key, value in raws['P_ONLY_primary']['source_pins'].items():
        need('initial_capture_source_current_' + key, pin(value['path'])['sha256'] == value['sha256'])
    need('source_and_saved_initial_artifacts_unchanged', pin(__file__) == own
         and all(pin(v['path']) == v for table in artifacts.values() for v in table.values()))
    need('no_GPU_Env_native_model_forward', not torch.cuda.is_initialized()
         and not any(k.split('.')[0] in ('warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic') for k in sys.modules))
    ok = all(checks.values())
    report = {'schema':'full151_P_ONLY_primary_smoke_crossprimary_initial_pairing:1',
              'status': 'PASS_SAVED_FOUR_CASE_RAW_INITIAL_PAIRING_NOT_TASK_QUALITY' if ok else 'FAIL_SAVED_INITIAL_OR_RECOVERY_REVIEW',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'same_initial_four_parameter_fingerprints': same, 'raw_initial_fingerprints': fps,
              'parameter_layout': layouts, 'artifact_pins': artifacts,
              'source_pins': {'auditor':own, **raws['P_ONLY_primary']['source_pins']},
              'scope':['Four raw initial CPU artifacts loaded; no training/final weights or model forward.',
                       'P_ONLY primary equals its schema2 smoke and both new A1L/A2 primaries by exact parameter name/order/shape/dtype/bytes.',
                       'No live runtime/CIM, training health, ETA, task quality, process control, restart or experimental intervention is assessed.'],
              'execution': {'raw_initial_CPU_loads':4, 'final_weights_loads': 0, 'model_forward': 0, 'Env': 0,
                            'GPU': 0, 'physics': 0, 'process_operations': 0}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'passed': report['passed'], 'total': report['total'],
                      'same_initial': same, 'status': report['status']}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
