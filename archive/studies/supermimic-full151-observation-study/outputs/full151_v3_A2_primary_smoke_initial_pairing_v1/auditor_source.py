"""CPU saved-only A2 primary/smoke raw initialization comparison; no model."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / 'outputs/full151_v3_primary2000_A2_seed0_v1'
SMOKE = ROOT / 'outputs/full151_v2_smoke40_A2_seed0_v1'
OUT = ROOT / 'outputs/full151_v3_A2_primary_smoke_initial_pairing_v1'
CONFIG = ROOT / 'configs/full151-v3-primary2000-A2-seed0-v1.json'
SMOKE_CONFIG = ROOT / 'configs/full151-v2-smoke40-A2-seed0-v1.json'
KEYS = ('wm', 'fail_head', 'actor', 'critic')


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    OUT.mkdir(exist_ok=False)
    own = pin(__file__)
    (OUT / 'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    artifacts = {name: pin(path) for name, path in {
        'primary_raw': PRIMARY / 'initial_parameters.pt',
        'primary_fingerprint': PRIMARY / 'initial_parameters.json',
        'smoke_raw': SMOKE / 'initial_parameters.pt',
        'smoke_fingerprint': SMOKE / 'initial_parameters.json',
        'primary_config': CONFIG, 'smoke_config': SMOKE_CONFIG,
        'smoke_pair_review': ROOT / 'outputs/full151_v2_three_arm_smoke_pairing_v2/report.json'
    }.items()}
    import torch
    torch.set_num_threads(1)
    checks = {}; fingerprints = {}; layout = {}; raws = {}
    def need(name, value): checks[name] = bool(value)
    primary_cfg = read(CONFIG); smoke_cfg = read(SMOKE_CONFIG)
    need('same_seed0_A2_new_schema2_fromscratch_roles',
         primary_cfg['schema'] == smoke_cfg['schema'] == 'full151_knownrough_native_training_run:2'
         and primary_cfg['method'] == smoke_cfg['method'] == 'A2'
         and type(primary_cfg['seed']) is int and primary_cfg['seed'] == smoke_cfg['seed'] == 0
         and primary_cfg['role'] == 'primary_matched' and smoke_cfg['role'] == 'engineering_smoke')
    need('same_input_controller_and_frozen_objective',
         all(primary_cfg[k] == smoke_cfg[k] for k in ('input_bundle', 'controller', 'contact_binding', 'evaluation_gates')))
    for name, run in (('primary', PRIMARY), ('smoke', SMOKE)):
        raw = torch.load(run / 'initial_parameters.pt', map_location='cpu', weights_only=False)
        fp = read(run / 'initial_parameters.json'); raws[name] = raw
        need(name + '_raw_native_capture_before_first_collect_or_optimizer',
             raw['schema'] == 'full151_native_initial_parameters:1'
             and raw['capture_point'] == 'after_native_DexRunner_init_before_first_collect_or_optimizer'
             and set(raw['modules']) == set(KEYS))
        fingerprints[name] = {}; layout[name] = {}
        for module in KEYS:
            m = raw['modules'][module]; values = m['parameters']; names = m['parameter_names']
            h = hashlib.sha256()
            valid = (len(names) == len(values) == len(set(names)) and bool(values)
                     and m['shapes'] == [list(v.shape) for v in values]
                     and m['dtypes'] == [str(v.dtype) for v in values])
            for value in values:
                valid = valid and value.device.type == 'cpu' and bool(torch.isfinite(value).all())
                h.update(str((str(value.dtype), tuple(value.shape))).encode())
                h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
            fingerprints[name][module] = {'sha256': h.hexdigest(), 'tensors': len(values),
                                           'parameters': sum(v.numel() for v in values)}
            layout[name][module] = {'names': names, 'shapes': m['shapes'], 'dtypes': m['dtypes']}
            need(name + '_' + module + '_raw_recomputed_matches_JSON', valid and fingerprints[name][module] == fp[module])
    need('raw_capture_sources_and_point_exact', raws['primary']['source_pins'] == raws['smoke']['source_pins']
         and raws['primary']['capture_point'] == raws['smoke']['capture_point'])
    for key, value in raws['primary']['source_pins'].items():
        need('capture_source_current_' + key, pin(value['path'])['sha256'] == value['sha256']
             and any(v['sha256'] == value['sha256'] and Path(v['path']).resolve() == Path(value['path']).resolve()
                     for v in primary_cfg['source_pins'].values())
             and any(v['sha256'] == value['sha256'] and Path(v['path']).resolve() == Path(value['path']).resolve()
                     for v in smoke_cfg['source_pins'].values()))
    same = True
    for module in KEYS:
        a = raws['primary']['modules'][module]; b = raws['smoke']['modules'][module]
        exact = (layout['primary'][module] == layout['smoke'][module]
                 and len(a['parameters']) == len(b['parameters'])
                 and all(x.shape == y.shape and x.dtype == y.dtype
                         and x.contiguous().view(torch.uint8).numpy().tobytes()
                         == y.contiguous().view(torch.uint8).numpy().tobytes()
                         for x, y in zip(a['parameters'], b['parameters'])))
        need(module + '_crossrun_name_order_shape_dtype_bytes_exact', exact)
        same = same and exact
    prior_pair = read(artifacts['smoke_pair_review']['path'])
    need('A2_smoke_anchor_bound_to_actual_three_arm_pairing',
         prior_pair['status'] == 'PASS_SAVED_THREE_ARM_SMOKE_PAIRING_NOT_TASK_QUALITY'
         and prior_pair['same_initial_four_parameter_fingerprints'] is True
         and prior_pair['raw_initial_fingerprints']['A2'] == fingerprints['smoke']
         and prior_pair['artifact_pins']['A2']['initial_parameters.pt'] == artifacts['smoke_raw'])
    need('all_source_and_initial_artifacts_unchanged', pin(__file__) == own
         and all(pin(v['path'])['sha256'] == v['sha256'] for v in artifacts.values()))
    need('no_CUDA_Env_native_model_forward_imports', not torch.cuda.is_initialized()
         and not any(k.split('.')[0] in ('warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic') for k in sys.modules))
    ok = bool(checks) and all(checks.values())
    report = {'schema': 'full151_primary_smoke_raw_initial_pairing:1',
              'status': 'PASS_SAVED_RAW_INITIAL_EQUALITY_NOT_TRAINING_HEALTH_OR_TASK_QUALITY' if ok else 'FAIL_RAW_INITIAL_PAIRING',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'same_initial_four_parameter_fingerprints': same,
              'raw_initial_fingerprints': fingerprints, 'parameter_layout': layout,
              'source_pins': {'auditor': own, **raws['primary']['source_pins']},
              'artifact_pins': artifacts,
              'scope': ['Only two raw initial CPU tensor artifacts loaded, no final weights or model construction/forward.',
                        'New metadata-only worker3/index does not change seed0 A2 original parameter initialization for this pair.',
                        'Primary is live; no runtime/checkpoint/health/ETA status was assessed and no process was queried or operated.',
                        'Equality does not imply trained-policy benefit or physical task quality.'],
              'execution': {'raw_initial_CPU_loads': 2, 'final_loads': 0, 'forward': 0, 'Env': 0, 'GPU': 0, 'physics': 0}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'passed': report['passed'],
                      'total': report['total'], 'same_initial': same, 'status': report['status']}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
