"""Three fixed WM-only checkpoint100 payloads; no model or forward creation."""
from pathlib import Path
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151_schema2_three_arm_prepolicy100_comparison_CPU_v1'
METHODS = ('A1L', 'P_ONLY', 'A2')
MODULES = ('wm', 'fail_head', 'actor', 'critic')


def pin(p):
    p = Path(p).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    OUT.mkdir(exist_ok=False); own = pin(__file__)
    (OUT / 'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    import torch
    torch.set_num_threads(1)
    def raw(t): return t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    def digest(value):
        h = hashlib.sha256()
        def visit(v):
            if isinstance(v, torch.Tensor):
                h.update(str(('tensor', str(v.dtype), tuple(v.shape))).encode()); h.update(raw(v))
            elif isinstance(v, np.ndarray):
                h.update(str(('ndarray', str(v.dtype), v.shape)).encode()); h.update(v.tobytes())
            elif isinstance(v, dict):
                h.update(b'dict')
                for k in sorted(v, key=lambda x: (type(x).__name__, repr(x))): visit(k); visit(v[k])
            elif isinstance(v, (list, tuple)):
                h.update(type(v).__name__.encode())
                for x in v: visit(x)
            elif v is None or isinstance(v, (str, bool, int, float, np.generic)):
                h.update(str((type(v).__name__, repr(v))).encode())
            else: raise TypeError(type(v).__name__)
        visit(value); return h.hexdigest()
    def compare(a, b):
        keys_same = list(a) == list(b)
        compatible = keys_same and all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype for k in a)
        if not compatible: return {'same_key_order_shape_dtype': False, 'byte_exact': False}
        exact = all(raw(a[k]) == raw(b[k]) for k in a)
        max_abs = 0.; squared = 0.; elements = 0; changed = 0
        for k in a:
            delta = a[k].double() - b[k].double()
            if delta.numel(): max_abs = max(max_abs, float(delta.abs().max())); squared += float(delta.square().sum())
            av = np.frombuffer(raw(a[k]), dtype=np.uint8).reshape(-1, a[k].element_size())
            bv = np.frombuffer(raw(b[k]), dtype=np.uint8).reshape(-1, b[k].element_size())
            changed += int(np.any(av != bv, axis=1).sum()); elements += a[k].numel()
        return {'same_key_order_shape_dtype': True, 'byte_exact': exact,
                'max_abs_numeric_difference': max_abs, 'L2_numeric_difference': math.sqrt(squared),
                'bitwise_different_elements': changed, 'elements': elements}

    pair_path = ROOT / 'outputs/full151_v3_P_ONLY_primary_smoke_initial_pairing_v1/report.json'
    pair = read(pair_path)
    states = {}; summaries = {}; inputs = {'initial_pair43': pin(pair_path)}; checks = {}
    def need(k, v): checks[k] = bool(v)
    need('original_all_primary_initials_previously_rawbyte_matched', pair['same_initial_four_parameter_fingerprints'] is True
         and pair['passed'] == pair['total'] == 43)
    loads = 0
    for method in METHODS:
        run = ROOT / f'outputs/full151_v3_primary2000_{method}_seed0_v1'
        cp = run / 'train/model_100.pt'
        hp = ROOT / f'outputs/full151_v3_primary_{method}_ckpt100_health_v1/report.json'
        h = read(hp); cfg = read(h['config']['path']); inputs[method + '__checkpoint100'] = pin(cp)
        inputs[method + '__health100'] = pin(hp); inputs[method + '__config'] = h['config']
        need(method + '_fixed100_positive_WM_only_health_and_schema2',
             h['status'] == 'PASS_SAVED_ENGINEERING_HEALTH' and h['iteration'] == 100
             and h['final_iteration_scope'] is False and h['passed'] == h['total'] == len(h['checks'])
             and all(v is True for v in h['checks'].values())
             and cfg['schema'] == 'full151_knownrough_native_training_run:2' and cfg['method'] == method
             and cfg['role'] == 'primary_matched' and cfg['training']['wm_only_iters'] == 200
             and h['checkpoint']['sha256'] == inputs[method + '__checkpoint100']['sha256'])
        loads += 1; s = torch.load(cp, map_location='cpu', weights_only=False); states[method] = s
        need(method + '_format105_iteration100_no_policy_optimizer_or_priorforward',
             type(s['format_version']) is int and s['format_version'] == 105 and s['iteration'] == 100
             and s['sim_steps'] == 40192 and s['config']['wm_only_iters'] == 200
             and all(s['full151_prior_evidence'][k] == 0 for k in ('policy_update_calls', 'objective_forward_calls', 'dt_applications'))
             and all(not s['optimizers'][k]['state'] for k in ('actor', 'critic')))
        layout = pair['parameter_layout'][method + '_primary']; module_data = {}; opts = {}
        for name in MODULES:
            module = s['modules'][name]; names = layout[name]['names']
            params = {k: module[k] for k in names}; other = {k: v for k, v in module.items() if k not in params}
            normalizers = {k: v for k, v in other.items() if 'norm.' in k}
            h0 = hashlib.sha256()
            for value in params.values():
                h0.update(str((str(value.dtype), tuple(value.shape))).encode()); h0.update(raw(value))
            fp = {'sha256': h0.hexdigest(), 'tensors': len(params), 'parameters': sum(v.numel() for v in params.values())}
            original = pair['raw_initial_fingerprints'][method + '_primary'][name]
            module_data[name] = {'parameter_keys_in_order': names, 'parameter_fingerprint': fp,
                'original_parameter_fingerprint': original, 'parameters_still_original': fp == original,
                'full_module_digest': digest(module), 'normalizer_keys': list(normalizers),
                'normalizer_digest': digest(normalizers),
                'normalizer_scalar_counts': {k: float(v) for k, v in normalizers.items() if k.endswith('count') and v.numel() == 1}}
            need(method + '_' + name + '_all_state_finite_CPU',
                 all(v.device.type == 'cpu' and bool(torch.isfinite(v).all()) for v in module.values()))
            if name in ('actor', 'critic'): need(method + '_' + name + '_parameters_still_byte_original', fp == original)
            opt = s['optimizers'][name]
            steps = [float(v['step']) for v in opt['state'].values()]
            opts[name] = {'digest': digest(opt), 'state_entries': len(opt['state']),
                          'stored_step_values': sorted(set(steps)), 'param_group_digest': digest(opt['param_groups'])}
            need(method + '_' + name + '_optimizer_steps_match_WM_only',
                 (not steps if name in ('actor', 'critic') else bool(steps) and all(v == 200 for v in steps)))
        rng_keys = ('torch_rng', 'cuda_rng', 'numpy_rng', 'python_rng')
        cap = s['full151_capacity_evidence']
        summaries[method] = {'modules': module_data, 'optimizers': opts,
            'saved_RNG_fields': {k: {'present': k in s, 'digest': digest(s[k]) if k in s else None} for k in rng_keys},
            'capacity_prefix': {'record_count': cap['record_count'], 'before_reset_read_count': cap['before_reset_read_count'],
                                'status': cap['status'], 'finalized': cap['finalized']},
            'replay_transition_counts_source_reported': s['full151_runtime_evidence']['replay_counts'],
            'prior_forward_calls': s['full151_prior_evidence']['objective_forward_calls'],
            'initial_fingerprint_metadata_bound': s['full151_initial_parameter_fingerprints'] == pair['raw_initial_fingerprints'][method + '_primary']}
    pairs = {}
    for left, right in (('A1L', 'P_ONLY'), ('A1L', 'A2'), ('P_ONLY', 'A2')):
        value = {'modules': {}, 'optimizers_digest_equal': {}, 'saved_RNG_digest_equal': {}}
        for name in MODULES:
            names = summaries[left]['modules'][name]['parameter_keys_in_order']
            am = states[left]['modules'][name]; bm = states[right]['modules'][name]
            value['modules'][name] = {'parameters': compare({k: am[k] for k in names}, {k: bm[k] for k in names}),
                                     'normalizers': compare({k: v for k, v in am.items() if 'norm.' in k}, {k: v for k, v in bm.items() if 'norm.' in k}),
                                     'full_module': compare(am, bm)}
            value['optimizers_digest_equal'][name] = summaries[left]['optimizers'][name]['digest'] == summaries[right]['optimizers'][name]['digest']
        for k in summaries[left]['saved_RNG_fields']:
            value['saved_RNG_digest_equal'][k] = summaries[left]['saved_RNG_fields'][k] == summaries[right]['saved_RNG_fields'][k]
        pairs[left + '__' + right] = value
    need('no_model_forward_environment_or_CUDA_initialization', not torch.cuda.is_initialized()
         and not any(k.split('.')[0] in ('mujoco', 'warp', 'mujoco_warp', 'mjlab', 'dex_supermimic') for k in sys.modules))
    need('all_checkpoint100_files_and_reports_unchanged', all(pin(v['path'])['sha256'] == v['sha256'] for v in inputs.values())
         and pin(__file__) == own)
    ok = bool(checks) and all(checks.values())
    report = {'schema': 'full151_three_arm_prepolicy100_saved_comparison:1',
              'status': 'COMPLETED_CPU_DESCRIPTIVE_PREPOLICY_COMPARISON' if ok else 'FAIL_OR_MALFORMED_PREFIX_EVIDENCE',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'source': own, 'input_pins': inputs, 'methods': summaries, 'pairwise': pairs,
              'WM_parameters_byte_identical_all3': all(p['modules']['wm']['parameters']['byte_exact'] for p in pairs.values()),
              'failure_parameters_byte_identical_all3': all(p['modules']['fail_head']['parameters']['byte_exact'] for p in pairs.values()),
              'Actor_Critic_parameters_original_and_identical_all3': all(summaries[m]['modules'][n]['parameters_still_original'] for m in METHODS for n in ('actor', 'critic')),
              'interpretation': ['This is a fixed checkpoint100 diagnostic, not selection or retraining. Equal initial parameters and budgets remain valid independently of equality of later trajectories.',
                                 'At iteration100 no Actor/Critic optimizer step or prior forward has occurred. P_ONLY/A2 factories were already constructed before collection, with separately saved RNG-preservation checks.',
                                 'If WM/failure/normalizer/RNG states differ, record before-prior-forward trajectories/numerics as not identical. These payloads alone do not establish a unique RNG, implementation, factory, reset or backend cause.',
                                 'Saved RNG states are checkpoint-time native snapshots, compared as evidence only; none is restored into this process. Reset/capacity counts and replay summaries are source-reported, not independently replayed dynamics.',
                                 'Parameter/buffer numeric differences are tensor engineering summaries, not physical error measures, method-benefit statistics or a strict common-training-trajectory causal comparison.'],
              'execution': {'checkpoint100_CPU_loads': loads, 'model_constructs': 0, 'forward': 0, 'optimizer_steps': 0, 'Env': 0, 'physics': 0, 'GPU': 0}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'checks': [report['passed'], report['total']],
                      'WM_same': report['WM_parameters_byte_identical_all3'],
                      'failure_same': report['failure_parameters_byte_identical_all3'],
                      'Actor_Critic_original': report['Actor_Critic_parameters_original_and_identical_all3']}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
