"""Read ONLY three trusted new boundary-component checkpoints on explicit --run.

Default is stdlib metadata: no Torch/native model/NPZ/checkpoint reads. Pure
payload validators also serve a separately tagged Tiny synthetic fixture; that
fixture cannot qualify the actual native fourteen-plus-two parameter contract.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'full151105_boundary_impulse_WM_component:1'
INITIAL_SCHEMA = 'full151105_boundary_impulse_matched_initial:1'
PAIR_SCHEMA = 'full151105_boundary_impulse_matched_pair:1'
PROTOCOL_SCHEMA = 'full151105_boundary_impulse_matched_training_CPU_protocol:1'
ROLE = 'matched_public_aux_vs_cross_target_consistency'
RECIPE_SCHEMA = 'full151105_boundary_impulse_matched_training_recipe:1'
ARMS = ('shared_public_losses', 'shared_public_plus_cross_target_consistency')
COUNTS = ('native_forward_attempted', 'native_forward_returned',
          'feature_hook_calls', 'aux_head_forward_returned', 'backward_calls', 'optimizer_steps')
LOSS_NAMES = ('state', 'mode', 'impulse', 'consistency', 'public', 'total')
FIXED_RECIPE = {
    'iterations': 1000, 'checkpoint_iterations': [100, 1000], 'fullbatch_rows': 152,
    'cpu_threads': 4, 'device': 'cpu', 'dtype': 'float32', 'amp': False,
    'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': 0.0,
    'betas': [0.9, 0.999], 'optimizer_eps': 1e-8, 'optimizer_foreach': False,
    'clip_norm': 1.0, 'lambda_state': 1.0, 'lambda_mode': 1.0, 'lambda_imp': 1.0,
    'lambda_cons_by_arm': {'shared_public_losses': 0.0, 'shared_public_plus_cross_target_consistency': 1.0},
    'native_log_chart_margin_rad': 0.01, 'normalizers_frozen': True,
    'physical_context_frozen': True, 'fixed_fullbatch_order': True,
    'compute_consistency_both_arms': True, 'initial_parity_from_first_training_forward': True,
    'save_pre_post_clip_named_gradients': True, 'log_every_iteration': True, 'resume_supported': False,
}
# Source-native named_parameters order: three Linear/LayerNorm hidden blocks,
# final Linear512->67, then the two independently copied auxiliary Parameters.
ACTUAL_PARAMETER_SPECS = tuple(
    (f'wm.net.{i}.{field}', shape)
    for i, shapes in ((0, ((512, 298), (512,))), (1, ((512,), (512,))),
                      (3, ((512, 512), (512,))), (4, ((512,), (512,))),
                      (6, ((512, 512), (512,))), (7, ((512,), (512,))),
                      (9, ((67, 512), (67,))))
    for field, shape in zip(('weight', 'bias'), shapes)
) + (('aux_head.weight', (14, 512)), ('aux_head.bias', (14,)))
CONTEXT_SPECS = {
    'mass_actual_FP32': ((), 'float32'), 'I_body_F64': ((3, 3), 'float64'),
    'gravity_palm_F64': ((3,), 'float64'), 'DeltaT_F64': ((), 'float64'),
    'palm_rot_FP32': ((3, 3), 'float32'), 'effective_s52_FP32': ((52,), 'float32'),
    'physical_xi46_FP32': ((46,), 'float32'), 'I_body_antisymmetry_maxabs_F64': ((), 'float64')}
PARENT_SHA = 'ea051225b7279fc69fbd97725eb15462faaf4b3a6706881f8fe4a05b658a14bf'
COMPONENT = ROOT/'src/full151_gradient/boundary_impulse_component105_v1.py'
COMPONENT_SHA = '003aac369bcac69dff9312c79837a8faf5a583473898fdb64515aa7608b60317'
HELPER = ROOT/'scripts/audit_WM_component_checkpoint_cpu_v2.py'
HELPER_SHA = '64c38dbed909b98381086cbd0c2cbfad89f8f790108fd07bbe4675ec23b2d47e'
RECIPE_FILE = ROOT/'protocols/full151105_boundary_impulse_matched_training_recipe_v1.json'
RECIPE_SHA = 'ed5e6cc378f80d94e1771607b8fd4ed1198ce31eec80e17c0465d9933ae4f855'
OWN = {'auditor': Path(__file__), 'checker': ROOT/'scripts/check_boundary_impulse_checkpoint_auditor_cpu_v1.py',
       'component': COMPONENT, 'saved_value_helpers': HELPER,
       'recipe_proposal': ROOT/'docs/full151_boundary_impulse_matched_training_recipe_proposal_v1.md', 'frozen_recipe': RECIPE_FILE}
PREP = ROOT/'outputs/full151105_boundary_impulse_checkpoint_auditor_preparation_CPU_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def pin(path):
    return {'path': str(Path(path).absolute()), 'sha256': sha(path)}


def checked(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'} and
            type(value['path']) is str and Path(value['path']).is_absolute() and
            type(value['sha256']) is str and len(value['sha256']) == 64, 'Exact absolute pin')
    p = Path(value['path']).resolve(strict=True)
    require(sha(p) == value['sha256'], 'Immutable pin changed')
    return p


def same(a, b):
    return a['sha256'] == b['sha256'] and checked(a).samefile(checked(b))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load_helpers():
    """Borrow saved-value functions, never the old namespace loader/audit."""
    require(sha(HELPER) == HELPER_SHA, 'Frozen raw RNG/Adam helper source')
    spec = importlib.util.spec_from_file_location('_boundary_saved_value_helpers', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _map_same(a, b):
    return type(a) is dict and type(b) is dict and set(a) == set(b) and all(same(a[k], b[k]) for k in a)


def recipe_matches(value, expected=FIXED_RECIPE):
    if type(expected) is bool:
        return type(value) is bool and value is expected
    if type(expected) in (int, float):
        return type(value) in (int, float) and math.isfinite(value) and value == expected
    if type(expected) is dict:
        return type(value) is dict and set(value) == set(expected) and all(recipe_matches(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return type(value) is list and len(value) == len(expected) and all(recipe_matches(a, b) for a, b in zip(value, expected))
    return type(value) is type(expected) and value == expected


def _tensor(value, shape, dtype, torch, name):
    require(isinstance(value, torch.Tensor) and value.device.type == 'cpu' and
            value.dtype == dtype and tuple(value.shape) == tuple(shape) and
            not value.requires_grad and bool(torch.isfinite(value).all()), 'Finite detached CPU '+name)


def _state_tensors(payload, specs, torch, helper):
    names = [name for name, _ in specs]
    require(payload['ordered_parameter_names'] == names, 'Ordered parameter-name contract')
    wm, head = payload['WM_state'], payload['aux_head_state']
    expected_wm = {name[3:] for name in names if name.startswith('wm.')}
    expected_head = {name[9:] for name in names if name.startswith('aux_head.')}
    require({k for k in wm if k.startswith('net.')} == expected_wm and set(head) == expected_head,
            'No missing/extra WM or head parameter states')
    values = []
    for name, shape in specs:
        value = wm[name[3:]] if name.startswith('wm.') else head[name[9:]]
        _tensor(value, shape, torch.float32, torch, name)
        values.append(value)
    require(helper.finite_tree(wm, torch) and helper.finite_tree(head, torch), 'All saved native/head values finite')
    return values


def gradient_readback(g, specs, iteration, torch, helper):
    names = [name for name, _ in specs]
    require(g['iteration'] == iteration and type(g['iteration']) is int and g['tag'] == 'PRE_OPTIMIZER_I' and
            g['weights_after_updates'] == iteration-1 and type(g['weights_after_updates']) is int and
            g['parameter_names'] == names and type(g['max_norm']) in (int, float) and g['max_norm'] == 1.0,
            'Raw gradient PRE-I phase / weights after I-1 / max norm1')
    pre, post = g['pre_clip'], g['post_clip']
    require(set(pre) == set(post) == set(names), 'All named gradients; never fill missing gradients with zero')
    for name, shape in specs:
        _tensor(pre[name], shape, torch.float32, torch, name+' pre-gradient')
        _tensor(post[name], shape, torch.float32, torch, name+' post-gradient')
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(pre[name], 2) for name in names]), 2)
    post_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(post[name], 2) for name in names]), 2)
    _tensor(g['pre_clip_returned_norm_FP32'], (), torch.float32, torch, 'returned clip norm')
    _tensor(g['post_clip_total_norm_FP32'], (), torch.float32, torch, 'post clip norm')
    require(helper.tensor_equal(norm, g['pre_clip_returned_norm_FP32'], torch) and
            helper.tensor_equal(post_norm, g['post_clip_total_norm_FP32'], torch), 'Same-order raw gradient norm evidence')
    # This is the pinned Torch clip_grad_norm_ coefficient, not a physics floor.
    coefficient = torch.clamp(1.0/(norm+1e-6), max=1.0)
    require(all(helper.tensor_equal(post[name], pre[name]*coefficient, torch) for name in names),
            'Raw post-gradient equals pinned foreach=False clip operation')
    return {'parameter_gradients': len(names), 'pre_clip_norm': float(norm),
            'post_clip_norm': float(post_norm), 'clip_coefficient': float(coefficient),
            'phase': 'PRE_OPTIMIZER_I; checkpoint is AFTER_OPTIMIZER_I'}


def _loss_row(losses, arm, torch):
    require(type(losses) is dict and set(losses) == set(LOSS_NAMES) and
            all(type(v) in (int, float) and math.isfinite(v) for v in losses.values()), 'All common loss components finite')
    # Preserve component_losses left-associative FP32 state+mode then F64 impulse.
    public = (torch.tensor(losses['state'], dtype=torch.float32)+
              torch.tensor(losses['mode'], dtype=torch.float32)).double()+torch.tensor(losses['impulse'], dtype=torch.float64)
    total = public+FIXED_RECIPE['lambda_cons_by_arm'][arm]*torch.tensor(losses['consistency'], dtype=torch.float64)
    require(float(public) == losses['public'] and float(total) == losses['total'], 'Fixed public1/1/1 and consistency0/1 arithmetic')


def validate_payload_pair(initial, payloads, logs, iteration, protocol_pin, recipe_pin,
                          input_pins, source_pins, torch, helper, *, fixture_specs=None):
    """No file loads/model calls. fixture_specs is private to tagged Tiny proof.

    The real audit never supplies it: actual16 names/shapes/order are mandatory.
    """
    specs = ACTUAL_PARAMETER_SPECS if fixture_specs is None else tuple(fixture_specs)
    fixture = fixture_specs is not None
    require(type(iteration) is int and iteration in (100, 1000), 'Only fixed checkpoint100/1000')
    require(initial['schema'] == INITIAL_SCHEMA and initial['pair_schema'] == PAIR_SCHEMA and
            'arm' not in initial and 'iteration' not in initial, 'Own shared initial namespace, not a relabeled checkpoint')
    if not fixture:
        require(len(initial['ordered_parameter_names']) == 16 and tuple(initial['ordered_parameter_names']) ==
                tuple(name for name, _ in ACTUAL_PARAMETER_SPECS), 'Actual16 parameter contract; Tiny is not native qualification')
    require(set(payloads) == set(ARMS), 'Both fixed arms')
    require(set(initial['training_counts']) == set(ARMS) and
            all(set(v) == set(COUNTS) and all(type(n) is int and n == 0 for n in v.values()) for v in initial['training_counts'].values()),
            'Shared initial starts before any new paired training update')
    for k in ('fit_input_digest', 'fit_label_digest'):
        require(type(initial[k]) is str and len(initial[k]) == 64 and all(c in '0123456789abcdef' for c in initial[k]), 'Raw fit digest header')
    require(same(initial['protocol'], protocol_pin) and same(initial['recipe'], recipe_pin) and
            _map_same(initial['input_pins'], input_pins) and _map_same(initial['source_pins'], source_pins) and
            same(initial['source_parent'], input_pins['initial_component']), 'Shared initial exact input/source/protocol binding')
    initial_values = _state_tensors(initial, specs, torch, helper)
    require(all(helper.tensor_equal(v, torch.zeros_like(v), torch) for v in initial['aux_head_state'].values()), 'Shared head raw positive-zero initialization')
    ctx = initial['physical_context_state']
    require(set(ctx) == set(CONTEXT_SPECS), 'All frozen registered physical buffers')
    for k, (shape, dtype) in CONTEXT_SPECS.items():
        _tensor(ctx[k], shape, getattr(torch, dtype), torch, k)
    scale = torch.maximum(torch.sqrt(initial['WM_state']['delta_norm.var']+
                          float(initial['normalizer_host_configuration']['delta_norm']['eps'])**2), initial['WM_state']['error_floor'])
    require(helper.tensor_equal(scale, ctx['effective_s52_FP32'], torch) and bool((scale > 0).all()), 'Initial effective52, not raw std or new label scales')
    construction = initial['construction_receipt']
    require(all(construction[k] is True for k in ('raw_WM_unchanged', 'disjoint_model_head_buffer_storage',
                    'constructor_Python_TorchCPU_RNG_unchanged', 'feature_hooks_independent')) and
            set(construction['raw_clone_state_identity']) == set(ARMS) and all(v is True for v in construction['raw_clone_state_identity'].values()),
            'Source-backed separate raw clones/hook/cache construction evidence')
    checks = {'shared_initial_namespace_inputs_zero_head_and_context': True}
    details = {'fixture_only': fixture, 'actual16_parameter_contract_verified': not fixture,
               'parameter_count': len(specs), 'optimizer': {}, 'gradients': {}, 'raw_RNG': {}}
    reference_rng = initial['RNG_this_run']
    helper.raw_rng_comparison(reference_rng, reference_rng, torch)
    for arm, payload in payloads.items():
        require(payload['schema'] == SCHEMA and payload['pair_schema'] == PAIR_SCHEMA and payload['arm'] == arm and
                type(payload['iteration']) is int and payload['iteration'] == iteration and payload['checkpoint_tag'] == 'AFTER_OPTIMIZER_I',
                'Own checkpoint identity and AFTER-I phase')
        require(same(payload['protocol'], protocol_pin) and same(payload['recipe'], recipe_pin) and
                _map_same(payload['input_pins'], input_pins) and _map_same(payload['source_pins'], source_pins), 'Per-arm exact provenance')
        require(all(payload[k] == initial[k] for k in ('fit_input_digest', 'fit_label_digest')), 'Fit inputs/labels unchanged within this run')
        lp = payload['log_prefix']
        require(type(lp) is dict and set(lp) == {'path', 'records', 'bytes', 'sha256'} and
                type(lp['path']) is str and Path(lp['path']).is_absolute() and type(lp['records']) is int and
                lp['records'] == iteration and type(lp['bytes']) is int and lp['bytes'] > 0 and
                type(lp['sha256']) is str and len(lp['sha256']) == 64, 'Exact immutable update-log prefix claim')
        values = _state_tensors(payload, specs, torch, helper)
        require(set(payload['WM_state']) == set(initial['WM_state']) and
                set(helper.norm_state(payload['WM_state'])) == set(helper.norm_state(initial['WM_state'])) and
                all(helper.tensor_equal(v, payload['WM_state'][k], torch) for k, v in helper.norm_state(initial['WM_state']).items()) and
                payload['normalizer_host_configuration'] == initial['normalizer_host_configuration'] and
                set(payload['physical_context_state']) == set(ctx) and
                all(helper.tensor_equal(v, payload['physical_context_state'][k], torch) for k, v in ctx.items()),
                'Normalizer raw buffers/host fields/effective scales/xi/physical context unchanged')
        require(set(payload['counts']) == set(COUNTS) and all(type(v) is int and v == iteration for v in payload['counts'].values()), 'Actual paired training counters equal checkpoint index')
        state = payload['optimizer_state']
        details['optimizer'][arm] = helper.optimizer_readback(state, iteration, torch, values)
        require(all(g.get('foreach') is False for g in state['param_groups']) and helper.finite_tree(state, torch), 'Explicit finite foreach=False AdamW')
        details['gradients'][arm] = gradient_readback(payload['gradient_snapshot'], specs, iteration, torch, helper)
        details['raw_RNG'][arm] = helper.raw_rng_comparison(reference_rng, payload['RNG'], torch)
        require(details['raw_RNG'][arm]['all_raw_values_equal'], 'Checkpoint RNG equals THIS run initial raw values, not historical re-pickle hash')
        checks[arm+'_saved_params_buffers_Adam_grads_counts_RNG'] = True
    require(helper.raw_rng_comparison(payloads[ARMS[0]]['RNG'], payloads[ARMS[1]]['RNG'], torch)['all_raw_values_equal'], 'Both raw RNG histories sampled equally')
    la, lb = (payloads[a]['log_prefix'] for a in ARMS)
    require(Path(la['path']).samefile(lb['path']) and all(la[k] == lb[k] for k in ('records', 'bytes', 'sha256')),
            'Both fixed checkpoints bind one update prefix by file identity')
    require(len(logs) == iteration, 'Full fixed update prefix, no selected rows')
    for i, row in enumerate(logs, 1):
        require(type(row['iteration']) is int and row['iteration'] == i and set(row['arms']) == set(ARMS), 'All ordered lockstep log rows')
        for arm, r in row['arms'].items():
            require(set(r['counts']) == set(COUNTS) and all(type(v) is int and v == i for v in r['counts'].values()) and
                    type(r['update_seconds']) in (int, float) and math.isfinite(r['update_seconds']) and r['update_seconds'] >= 0, 'One FW/head/backward/step per arm per update')
            _loss_row(r['losses'], arm, torch)
    checks['all_lockstep_update_prefix_and_identical_public_weights'] = True
    details['initial_WM_sha256'] = helper.tensor_digest(initial['WM_state'], torch)
    details['fit_input_digest'] = initial['fit_input_digest']
    details['fit_label_digest'] = initial['fit_label_digest']
    details['label_order_scope'] = 'Within-run saved digest/source evidence and full NPZ pins; no independent internal training-label ordering reconstruction.'
    details['self15_scope'] = 'No native self15 loss: initial function identity does not preserve post-fit self15 accuracy.'
    details['disjoint_history_scope'] = 'Construction receipt is runtime source evidence. Saved tensors/Adam histories do not independently prove training-process storage identity.'
    return checks, details


def _ready():
    ready = read(PREP/'READY.json');report = read(checked(ready['cpu_report']))
    require(ready['schema'] == 'full151105_boundary_impulse_checkpoint_auditor_CPU_READY:1' and
            report['schema'] == 'full151105_boundary_impulse_checkpoint_auditor_CPU_contract:1' and
            report['status'] == 'PASS_SYNTHETIC_PAYLOAD_AND_RAW_HEALTH_NO_REAL_LOAD' and
            ready['cpu_checks'] == {'passed': report['passed'], 'total': report['total']} and
            report['passed'] == report['total'] == len(report['checks']) and all(v is True for v in report['checks'].values()) and
            report['actual_counts'] and all(type(v) is int and v == 0 for v in report['actual_counts'].values()), 'Executed focused synthetic proof')
    require(set(ready['source_pins']) == set(report['source_pins']) == set(OWN) and
            all(same(pin(p), ready['source_pins'][k]) and same(ready['source_pins'][k], report['source_pins'][k]) for k, p in OWN.items()), 'Current proof source closure')
    return pin(PREP/'READY.json'), ready['cpu_report']


def metadata_preflight(protocol_path, run_dir, iteration):
    require(type(iteration) is int and iteration in (100, 1000), 'Fixed checkpoint cadence100/1000')
    p = read(protocol_path);recipe = read(checked(p['recipe']))
    require(p['schema'] == PROTOCOL_SCHEMA and p['role'] == ROLE and p['CPU_run_authorized'] is True and
            p['GPU_run_authorized'] is False and p['automatic_retry'] is False, 'Explicit isolated Root CPU training protocol')
    require(sha(RECIPE_FILE) == RECIPE_SHA and same(p['recipe'], pin(RECIPE_FILE)) and
            recipe['schema'] == RECIPE_SCHEMA and recipe_matches(recipe['recipe']), 'Root frozen canonical recipe, not old unfrozen draft')
    require(Path(p['output']).resolve(strict=True).samefile(Path(run_dir).resolve(strict=True)), 'Actual training output route')
    inputs, sources = p['inputs'], p['source_pins']
    for value in [*inputs.values(), *sources.values()]:
        checked(value)
    require({'initial_component', 'initial_parity_report', 'initial_parity_saved_review', 'target_report', 'target_arrays', 'target_saved8'} <= set(inputs), 'Necessary initial/label qualification inputs')
    require(inputs['initial_component']['sha256'] == PARENT_SHA and sha(COMPONENT) == COMPONENT_SHA, 'Fixed original A2 initial and source-tested ungated module')
    parity = read(checked(inputs['initial_parity_report']));target = read(checked(inputs['target_report']));saved = read(checked(inputs['target_saved8']))
    parity_saved = read(checked(inputs['initial_parity_saved_review']))
    require(parity['schema'] == 'full151105_boundary_component_initial_parity_CPU_analysis:1' and
            parity['status'] == 'COMPLETED_REAL_INITIAL_FUNCTION_PARITY_NOT_TRAINING_QUALIFICATION' and
            parity['function_identity_qualified'] is True and parity['engineering_qualified'] is True and
            same(parity['initial_component'], inputs['initial_component']) and
            parity['passed'] == parity['total'] == len(parity['checks']) and all(v is True for v in parity['checks'].values()) and
            same(parity['input_pins']['target_report'], inputs['target_report']) and same(parity['input_pins']['target_saved8'], inputs['target_saved8']), 'Actual initial function and all152 input binding')
    require(parity_saved['schema'] == 'full151105_boundary_initial_parity_saved_independent:1' and
            parity_saved['status'] == 'PASS_SAVED_REAL_INITIAL_FUNCTION_PARITY' and parity_saved['engineering_qualified'] is True and
            parity_saved['function_identity_qualified'] is True and same(parity_saved['producer_report'], inputs['initial_parity_report']) and
            parity_saved['passed'] == parity_saved['total'] == len(parity_saved['checks']) and all(v is True for v in parity_saved['checks'].values()) and
            same(recipe['source_targets'], inputs['target_arrays']), 'Independent saved initial parity and frozen target array enrollment')
    require(target['schema'] == 'full151105_saved_boundary_impulse_target_dataset_CPU_analysis:1' and target['enrollment_qualified'] is True and
            all(v is True for v in target['guards'].values()) and saved['schema'] == 'full151105_boundary_impulse_dataset_saved_independent:1' and
            saved['status'] == 'PASS_SAVED_ALL152_ALL608_CONDITIONAL_TARGET_ARITHMETIC' and saved['engineering_qualified'] is True and
            saved['passed'] == saved['total'] == len(saved['checks']) and all(v is True for v in saved['checks'].values()) and same(saved['producer_report'], inputs['target_report']), 'Conditional all152 labels independently reviewed')
    ready, proof = _ready()
    for path in OWN.values():
        require(any(same(pin(path), value) for value in sources.values()), 'Actual training source archive enrolls health dependencies')
    return p, {'initial_parity': parity, 'cpu_ready': ready, 'cpu_report': proof}


def declaration():
    return {'schema': 'full151105_boundary_impulse_checkpoint_auditor_declaration:1',
            'status': 'METADATA_ONLY_NO_CHECKPOINT_OR_TORCH_READ', 'protocol_schema': PROTOCOL_SCHEMA,
            'role': ROLE, 'artifact_schema': SCHEMA, 'pair_schema': PAIR_SCHEMA,
            'checkpoint_iterations': [100, 1000], 'actual_parameter_count': 16,
            'actual_checkpoint_loads_model_FW_AD_MJB_Env_physics_GPU': 0}


def audit(protocol_path, run_dir, iteration, output):
    out = Path(output);out.mkdir(exist_ok=False);loads = []
    try:
        p, meta = metadata_preflight(protocol_path, run_dir, iteration)
        import torch
        import numpy as np
        import random
        torch.set_num_threads(4);require(not torch.cuda.is_initialized(), 'Fresh CPU health process, never CUDA')
        helper = load_helpers()
        def rng():
            return {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch_CPU': torch.get_rng_state().clone(), 'CUDA_initialized': False}
        before_rng = rng();run = Path(run_dir).resolve(strict=True)
        pair_path = run/f'paired_checkpoint_{iteration:04d}.json';pair_pin = pin(pair_path);pair = read(pair_path)
        require(pair['schema'] == PAIR_SCHEMA and type(pair['iteration']) is int and pair['iteration'] == iteration and
                set(pair['checkpoints']) == set(ARMS) and same(pair['protocol'], pin(protocol_path)), 'Fixed producer pair manifest, no best checkpoint selection')
        def own_load(name, expected):
            path = (run/name).resolve(strict=True)
            require(path.parent.samefile(run) and same(pin(path), expected), 'Exactly owned producer-pinned new component route')
            before = pin(path);loads.append({'checkpoint': before, 'returned': False})
            value = torch.load(path, map_location='cpu', weights_only=False)
            require(same(before, pin(path)), 'Immutable checkpoint changed during load')
            loads[-1]['returned'] = True
            return value
        initial = own_load('initial_pair_state.pt', pair['initial_pair_state'])
        payloads = {a: own_load(f'{a}_update_{iteration:04d}.pt', pair['checkpoints'][a]) for a in ARMS}
        logs = [];prefix = hashlib.sha256();prefix_bytes = 0
        with (run/'updates.jsonl').open('rb') as f:
            for _ in range(iteration):
                line = f.readline();require(line.endswith(b'\n'), 'Complete immutable requested log prefix')
                prefix.update(line);prefix_bytes += len(line);logs.append(json.loads(line))
        for payload in payloads.values():
            lp = payload['log_prefix']
            require(Path(lp['path']).samefile(run/'updates.jsonl') and lp['records'] == iteration and
                    lp['bytes'] == prefix_bytes and lp['sha256'] == prefix.hexdigest(), 'Actual immutable log prefix matches fixed checkpoint, not current whole-file hash')
        checks, details = validate_payload_pair(initial, payloads, logs, iteration, pin(protocol_path), p['recipe'],
                                               p['inputs'], p['source_pins'], torch, helper)
        require(details['initial_WM_sha256'] == meta['initial_parity']['initial_WM_sha256'] and
                details['fit_input_digest'] == meta['initial_parity']['input_digest_before'], 'Actual shared initial WM and fit order equal trusted parity digest')
        runtime_rng = helper.raw_rng_comparison(before_rng, rng(), torch)
        require(runtime_rng['all_raw_values_equal'] and not torch.cuda.is_initialized() and len(loads) == 3 and
                all(v['returned'] for v in loads) and not any(k in sys.modules for k in ('warp', 'mujoco', 'mujoco_warp', 'dex_supermimic', 'mjlab')), 'Three own reads / no runtime RNG repair or models/native')
        checks['trusted_initial_digest_and_readback_only_runtime_scope'] = True
        details.update(raw_runtime_RNG_preserved=runtime_rng, log_prefix_sha256=prefix.hexdigest(), log_prefix_rows=iteration, log_prefix_bytes=prefix_bytes,
                       gradient_scope='Saved PRE-I raw gradients, not gradients recomputed from AFTER-I checkpoint.',
                       RNG_scope='THIS run initial and checkpoint raw values at saved samples; no historical-process RNG comparison or pickle equivalence.',
                       backend_scope='NumPy/Torch imports and thread setup precede auditor RNG capture; no whole-process initialization claim.')
        for row in loads:
            checked(row['checkpoint'])
        checked(pair_pin)
        metadata_preflight(protocol_path, run_dir, iteration)
        report = {'schema': 'full151105_boundary_impulse_checkpoint_saved_review:1',
                  'status': 'PASS_SAVED_PAIRED_BOUNDARY_COMPONENT_HEALTH_NOT_ACCURACY', 'engineering_qualified': True,
                  'iteration': iteration, 'final_iteration_scope': iteration == 1000,
                  'checks': checks, 'passed': len(checks), 'total': len(checks), 'auditor': pin(__file__),
                  'protocol': pin(protocol_path), 'recipe': p['recipe'], 'run_dir': str(run),
                  'producer_pair_manifest': pair_pin, 'producer_inmemory_health': pair['producer_health'],
                  'initial_pair': loads[0]['checkpoint'], 'checkpoints': {a: loads[i+1]['checkpoint'] for i, a in enumerate(ARMS)},
                  'own_CPU_checkpoint_loads': 3, 'load_trace': loads, 'cpu_ready': meta['cpu_ready'], 'cpu_report': meta['cpu_report'],
                  'source_pins': {k: pin(v) for k, v in OWN.items()}, 'details': details,
                  'models_FW_AD_MJB_Env_physics_GPU': 0, 'native105_policy_or_Actor_qualified': False,
                  'accuracy_physical_force_gradient_wholehidden_or_ego_qualified': False}
        write(out/'report.json', report)
        print(json.dumps({'status': report['status'], 'report': pin(out/'report.json'), 'passed': report['passed'], 'total': report['total']}))
        return 0
    except BaseException as error:
        if not (out/'failure.json').exists():
            write(out/'failure.json', {'error': repr(error), 'auditor': pin(__file__), 'actual_load_trace': loads,
                                     'actual_saved_review_completed': False, 'source_checkpoint_files_unmodified': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path);parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--iteration', type=int, default=100);parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true');args = parser.parse_args()
    if not args.run:
        print(json.dumps(declaration()));return 0
    require(None not in (args.protocol, args.run_dir, args.output), 'Explicit Root protocol/run/output for real readback')
    return audit(args.protocol, args.run_dir, args.iteration, args.output)


if __name__ == '__main__':
    raise SystemExit(main())
