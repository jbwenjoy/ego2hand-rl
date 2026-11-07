"""Two strict105 CPU WM loads, two fixed-input forwards and two u AD calls."""
from pathlib import Path
import hashlib
import json
import os
import pickle
import random
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151_schema2_pair_WM_loader_CPU_v1'
LOADER = ROOT / 'src/full151_native/world_model_checkpoint.py'
LOADER_SHA = '119fa7cebc0e9b0727b0160660b1a47822deee2a0e0e9fa5be69465ac5578052'
AUDITOR = ROOT / 'scripts/audit_full151_checkpoint_cpu_v2.py'
AUDITOR_SHA = '5950b9947b00ed0cfce6a8403a65877db6d71be1d7b01b8c3ad1626bdd5de792'
A0 = ROOT / 'outputs/full151_native_A0_v1'
METHODS = ('A1L', 'A2')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def rng(torch):
    return {'Python': hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
            'NumPy': hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
            'TorchCPU': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            'CUDA_initialized': torch.cuda.is_initialized()}


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    OUT.mkdir(exist_ok=False); own = pin(__file__)
    (OUT / 'checker_source.py').write_bytes(Path(__file__).read_bytes())
    checks = {}; results = {}; saved = {}; denied = {}; errors = {}
    counters = {'strict_WM_loader_calls': 0, 'checkpoint_torch_load_calls_internal': 0,
                'WM_forward_calls': 0, 'u_AD_calls': 0}
    source_pins = {'checker': own, 'strict_WM_loader': pin(LOADER),
                   'strict_Actor_loader': pin(ROOT / 'src/full151_native/checkpoint.py'),
                   'bundle_loader': pin(ROOT / 'src/full151_native/bundle.py'),
                   'health_auditor': pin(AUDITOR),
                   'native_networks': pin(Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/networks.py')),
                   'native_state': pin(Path('D:/workspace/dex-supermimic-research/src/dex_supermimic/rl/dex_supermimic/state.py')),
                   'native_reader_capture': pin(ROOT / 'src/full151_native/probe.py')}
    artifact_pins = {'saved_A0_transitions': pin(A0 / 'native_reward_transitions.json'),
                     'saved_A0_control_events': pin(A0 / 'control_events.json'), 'saved_A0_config': pin(A0 / 'config.json')}
    def need(name, value): checks[name] = bool(value)
    need('frozen119_loader_and5950_health_source', source_pins['strict_WM_loader']['sha256'] == LOADER_SHA
         and source_pins['health_auditor']['sha256'] == AUDITOR_SHA)
    if not all(checks.values()): raise ValueError('Frozen source changed before execution')
    transitions = read(A0 / 'native_reward_transitions.json'); t0 = transitions[0]
    events = read(A0 / 'control_events.json'); a0cfg = read(A0 / 'config.json')
    x_np = np.asarray(t0['x0'], dtype=np.float32).reshape(1, -1)
    u_np = np.asarray(t0['actualu'], dtype=np.float32).reshape(1, -1)
    need('actual_native_saved_phase0_state58_absoluteu20_not_guessed',
         len(transitions) == 150 and t0['phase'] == 0 and x_np.shape == (1, 58) and u_np.shape == (1, 20)
         and np.isfinite(x_np).all() and np.isfinite(u_np).all()
         and np.asarray(t0['memory']['time']).item() == 0
         and np.array_equal(u_np, np.asarray(events[0]['actualu'], np.float32).reshape(1, 20)))
    if not all(checks.values()): raise ValueError('Exact saved phase0 input is unavailable')

    import torch
    import mujoco
    torch.set_num_threads(1)
    original_load = torch.load
    def observe_load(*args, **kwargs):
        counters['checkpoint_torch_load_calls_internal'] += 1
        if kwargs.get('map_location') != 'cpu': raise ValueError('Only map_location CPU permitted')
        return original_load(*args, **kwargs)
    def deny(name):
        def call(*args, **kwargs):
            denied[name] += 1; raise AssertionError('Forbidden native physical/query API: ' + name)
        return call
    native_names = ('MjData', 'mj_step', 'mj_step1', 'mj_step2', 'mj_forward', 'mj_inverse',
                    'mj_kinematics', 'mj_comPos', 'mj_collision', 'mj_geomDistance', 'mj_contactForce', 'mj_jac')
    patches = []
    for name in native_names:
        denied[name] = 0; patches.append(patch.object(mujoco, name, deny(name)))
    patches.append(patch.object(torch, 'load', observe_load))
    for item in patches: item.start()
    sys.path.insert(0, str(ROOT / 'src'))
    started = time.perf_counter()
    try:
        from full151_native.bundle import load_bundle
        from full151_native.world_model_checkpoint import load_world_model_checkpoint
        from full151_native.checkpoint import _tensor_digest
        for method in METHODS:
            if time.perf_counter() - started > 40: raise TimeoutError('Bounded CPU validation40s, no retry')
            hp = ROOT / f'outputs/full151_v3_primary_{method}_final2000_health_v1/report.json'
            h = read(hp); cfg = read(h['config']['path']); run = Path(h['run_dir'])
            for key, value in {'health': pin(hp), 'config': h['config'], 'checkpoint': h['checkpoint']}.items():
                artifact_pins[method + '__' + key] = value
            need(method + '_actual_schema2_primary_final2000_health',
                 cfg['schema'] == 'full151_knownrough_native_training_run:2' and cfg['method'] == method
                 and cfg['role'] == 'primary_matched' and cfg['input_bundle'] == a0cfg['input_bundle']
                 and h['status'] == 'PASS_SAVED_ENGINEERING_HEALTH' and h['final_iteration_scope'] is True
                 and h['iteration'] == 2000 and h['auditor'] == source_pins['health_auditor']
                 and h['passed'] == h['total'] == len(h['checks']) and all(v is True for v in h['checks'].values()))
            bundle = load_bundle(cfg['input_bundle']['path'], expected_sha256=cfg['input_bundle']['sha256'])
            before_rng = rng(torch)
            try:
                counters['strict_WM_loader_calls'] += 1
                module, receipt = load_world_model_checkpoint(h['checkpoint']['path'], bundle, method, run, hp,
                    expected_health_auditor=source_pins['health_auditor'], expected_iteration=2000)
                (OUT / (method + '_load_receipt.json')).write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                before = {k: v.detach().clone() for k, v in module.state_dict().items()}
                digest_before = _tensor_digest(module.state_dict())
                x = torch.from_numpy(x_np.copy()); u = torch.from_numpy(u_np.copy()).requires_grad_(True)
                x_before = x.clone(); u_before = u.detach().clone()
                counters['WM_forward_calls'] += 1
                prediction, self15 = module(x, u)
                counters['u_AD_calls'] += 1
                du, = torch.autograd.grad(prediction.sum() + self15.sum(), u)
                need(method + '_one_FP32_forward58_self15_and_connected_finite_uAD20',
                     prediction.shape == (1, 58) and self15.shape == (1, 15) and du.shape == (1, 20)
                     and prediction.dtype == self15.dtype == du.dtype == torch.float32
                     and torch.isfinite(prediction).all() and torch.isfinite(self15).all() and torch.isfinite(du).all()
                     and prediction.requires_grad and self15.requires_grad)
                need(method + '_parameter_frozen_eval_no_grad_accumulation',
                     not module.training and not module.wm.training and module.wm.amp is False
                     and all(not p.requires_grad and p.grad is None for p in module.parameters()))
                need(method + '_weights_all_norm_buffers_inputs_and_RNG_unchanged',
                     all(torch.equal(v, module.state_dict()[k]) for k, v in before.items())
                     and digest_before == _tensor_digest(module.state_dict())
                     and torch.equal(x, x_before) and torch.equal(u.detach(), u_before)
                     and rng(torch) == before_rng and not torch.cuda.is_initialized())
                need(method + '_strict_receipt_native58_xi46_dyn278_total298_self15',
                     receipt['status'] == 'PASS_FROZEN_CPU_WORLD_MODEL_LOAD'
                     and receipt['state_dim'] == 58 and receipt['physical_xi_dim'] == 46
                     and receipt['dynamics_dim'] == 278 and receipt['WM_total_input_dim'] == 298
                     and receipt['delta_dim'] == 52 and receipt['self_collision_logits_dim'] == 15
                     and receipt['parameter_requires_grad'] is False and receipt['Env_or_Warp_created'] is False)
                results[method] = {'status': 'STRICT_LOAD_ONE_FORWARD_ONE_AD_COMPLETED',
                    'load_receipt': pin(OUT / (method + '_load_receipt.json')),
                    'wrapper_state_sha256_before': digest_before, 'wrapper_state_sha256_after': _tensor_digest(module.state_dict()),
                    'RNG_digest_before': before_rng, 'RNG_digest_after': rng(torch),
                    'u_AD_nonzero_components': int(torch.count_nonzero(du)), 'u_AD_absmax': float(du.abs().max()),
                    'prediction_shape': list(prediction.shape), 'self_collision_logits_shape': list(self15.shape)}
                saved[method + '__prediction58'] = prediction.detach().numpy()
                saved[method + '__self15_logits'] = self15.detach().numpy()
                saved[method + '__mixed_sum_u_AD20'] = du.detach().numpy()
                del module, prediction, self15, du
            except BaseException as exc:
                checks[method + '_actual_strict_loader_engineering'] = False
                errors[method] = {'error': repr(exc), 'traceback': traceback.format_exc()}
                results[method] = {'status': 'FAILURE_PRESERVED_NO_RETRY'}
        need('bounded_two_load_two_forward_two_AD_calls', counters['strict_WM_loader_calls'] == 2
             and counters['WM_forward_calls'] == counters['u_AD_calls'] == 2
             and counters['checkpoint_torch_load_calls_internal'] == 4)
        need('no_CUDA_Env_Warp_or_native_physical_query', not torch.cuda.is_initialized()
             and not any(k.split('.')[0] in ('warp', 'mujoco_warp', 'mjlab') for k in sys.modules)
             and not any(denied.values()))
        need('all_checkpoint_health_inputs_and_frozen_source_files_unchanged',
             all(pin(v['path'])['sha256'] == v['sha256'] for v in [*source_pins.values(), *artifact_pins.values()]))
    finally:
        for item in reversed(patches): item.stop()
    np.savez_compressed(OUT / 'one_step_engineering_cases.npz', saved_phase0_x58=x_np,
                        saved_phase0_actualu20=u_np, **saved)
    ok = bool(checks) and all(checks.values())
    report = {'schema': 'full151_schema2_pair_WM_loader_CPU_engineering:1',
              'status': 'PASS_TWO_ACTUAL_SCHEMA2_FINAL_WM_LOADS_ONE_FORWARD_AD_EACH' if ok else 'FAILURE_OR_PARTIAL_PRESERVED',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks), 'source_pins': source_pins,
              'artifact_pins': artifact_pins, 'models': results, 'errors': errors, 'actual_calls': counters,
              'denied_native_physical_query_API_calls': denied,
              'cases': pin(OUT / 'one_step_engineering_cases.npz'),
              'input_scope': 'Both models use the same predeclared saved nativeReader A0 phase0 x0 and executed absoluteu, no state guess or C2 repetition.',
              'limits': ['One-step/AD checks are interface/autograd connectivity/finiteness only, not prediction accuracy, physical Jacobian, hand-object/contact-force, native task reward gradient or useful control direction qualification.',
                         'The scalar sum(x1[58])+sum(self-collision logits[15]) mixes units intentionally only as a finite engineering graph probe. It is not a task scalar or scientific gradient endpoint.',
                         'self15 is native self-collision logits; there is no hand-object label/force or positive-class capability test.',
                         'Frozen119 internally loads each checkpoint twice and constructs/discards a strict CPU Actor plus passive MjModel. No Actor forward, Env, Warp, optimizer or physical operation was invoked.',
                         'Source/normalizer/weight and RNG equality is independently compared before/after these two actual CPU loader+forward+AD calls; no prior GPU/restore/physical-gradient qualification is inherited.'],
              'execution': {'GPU': False, 'Env_created': False, 'physics_steps': 0,
                            'Actor_forward_calls': 0, 'optimizer_steps': 0,
                            'passive_MjModel_reads_source_backed': counters['strict_WM_loader_calls']}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'passed': report['passed'], 'total': report['total'],
                      'status': report['status'], 'actual_calls': counters}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
