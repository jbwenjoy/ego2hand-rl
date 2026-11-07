"""DRAFT: isolated WM-only common-pre finite-response calibration.

No backend is imported by the declaration/metadata path. Execution is a NEW
component experiment, never native policy training/resume or a replacement105
checkpoint. The unchanged strict119f loader validates the parent first. Only
deepcopies of its native WM become trainable. Root owns protocol freeze/run.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
SCHEMA = 'full151105_WM_common_pre_calibration_CPU_protocol:1'
ROLE = 'WM_only_common_pre_finite_response_calibration'
ARTIFACT_SCHEMA = 'full151105_WM_common_pre_component_calibration:1'
OUTPUT_PREFIX = 'full151105_WM_common_pre_calibration_CPU_'
ARMS = ('pointwise', 'pointwise_plus_common_pre_response')
OWN = {'adapter': ROOT/'src/full151_gradient/response_calibration105_v1.py',
       'CLI': ROOT/'scripts/calibrate_full151_WM_common_pre_cpu_v1.py'}
INPUTS = {'input_bundle', 'fixed_A2_training_config', 'fixed_A2_checkpoint',
          'fixed_A2_final_health', 'fixed_A2_health_auditor',
          'fixed_A2_fixed_physics_context', 'response_config', 'response_report',
          'response_output_pins', 'response_raw152', 'response_metadata152',
          'response_pairs64', 'response_guard_summary', 'response_saved_independent22'}
RECIPE = {'iterations': 1000, 'checkpoint_interval': 100, 'first_checkpoint': 100,
          'fullbatch_rows': 152, 'pairs': 64, 'cpu_threads': 4,
          'dtype': 'float32', 'device': 'cpu', 'amp': False,
          'optimizer': 'AdamW', 'lr': 1e-4, 'weight_decay': 0.,
          'betas': [0.9, 0.999], 'optimizer_eps': 1e-8,
          'grad_clip_norm': 1., 'lambda_pair': 1., 'control_reference_rad': 1.,
          'chart_margin_rad': 0.01, 'normalizers_frozen': True,
          'physical_xi_frozen': True, 'seed': 0,
          'donor_rows_used_for_fit': 0, 'H8_used': False}
FROZEN_WM_LOADER_SHA = '119fa7cebc0e9b0727b0160660b1a47822deee2a0e0e9fa5be69465ac5578052'
FROZEN_CHECKPOINT_SHA = '70d0a21d5b2f0238ab745264bf24db08515fef65f072cbe00fb6893a33512a0c'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    p = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return {'path': str(p), 'sha256': h.hexdigest()}


def checked(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact pin')
    require(type(value['path']) is str and Path(value['path']).is_absolute(), 'Absolute pin')
    sha = value['sha256']
    require(type(sha) is str and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha), 'SHA256')
    p = Path(value['path']).resolve(strict=True)
    require(pin(p)['sha256'] == sha, 'Changed pinned file: ' + str(p))
    return p


def same(a, b):
    return a['sha256'] == b['sha256'] and checked(a).samefile(checked(b))


def allchecks(report):
    checks = report.get('checks')
    require(type(checks) is dict and checks and all(type(v) is bool and v for v in checks.values()), 'All-true parent checks')
    require(type(report.get('passed')) is int and type(report.get('total')) is int and report['passed'] == report['total'] == len(checks), 'Parent check counts')


def canonical_sources(values):
    """Preserve first recorded spelling; SHA alone never merges physical copies."""
    result = []; seen = {}
    for value in values:
        p = checked(value); st = p.stat()
        key = (st.st_dev, st.st_ino) if st.st_ino else ('resolved', str(p).casefold())
        if key in seen:
            require(same(value, seen[key]), 'Conflicting pins for one actual file')
        else:
            seen[key] = dict(value); result.append(dict(value))
    return {'file_%04d__%s' % (i, Path(v['path']).name): v for i, v in enumerate(result)}


def validate_protocol(p, require_authorized=False):
    require(type(p) is dict and p.get('schema') == SCHEMA and p.get('role') == ROLE, 'New WM-only protocol/schema')
    for name in ('execution_authorized', 'CPU_run_authorized', 'GPU_run_authorized', 'background_continue_to_final'):
        require(type(p.get(name)) is bool, 'Explicit protocol bool: ' + name)
    require(p['GPU_run_authorized'] is False and type(p.get('cpu_threads')) is int and p['cpu_threads'] == 4, 'CPU4 only')
    require(type(p.get('inputs')) is dict and set(p['inputs']) == INPUTS, 'Exact semantic input aliases')
    require(type(p.get('source_pins')) is dict and p['source_pins'], 'Current source closure')
    require(type(p.get('recipe')) is dict and set(p['recipe']) == set(RECIPE), 'Exact fixed recipe fields')
    # JSON types are significant: True is not iteration1 or lambda1.
    for k, v in RECIPE.items():
        a = p['recipe'][k]
        require(type(a) is type(v) and a == v, 'Frozen recipe mismatch: ' + k)
    require(p.get('output_prefix') == OUTPUT_PREFIX, 'Scoped output prefix')
    if require_authorized:
        require(p['execution_authorized'] and p['CPU_run_authorized'], 'Root explicit CPU calibration authorization required')
    return p


def declaration():
    require(not any(m in sys.modules for m in ('torch', 'mujoco', 'warp')), 'Metadata-only fresh process')
    return {'schema': 'full151105_WM_common_pre_calibration_declaration:1',
            'status': 'SOURCE_DRAFT_NOT_FROZEN_NOT_EXECUTED', 'protocol_schema': SCHEMA,
            'role': ROLE, 'input_aliases': sorted(INPUTS), 'recipe': RECIPE,
            'output_prefix': OUTPUT_PREFIX, 'source_pins': {k: pin(v) for k, v in OWN.items()},
            'actual_NPZ_reads': 0, 'actual_model_loads_forwards_AD_optimizer_physics_GPU': 0}


def metadata_preflight(protocol_path, output=None):
    """JSON/hash only, with no dataset decoding, model/backend or output writes."""
    require(not any(m in sys.modules for m in ('torch', 'mujoco', 'warp')), 'Fresh metadata-only process')
    p = validate_protocol(read(protocol_path)); i = p['inputs']
    for v in i.values(): checked(v)
    h = read(checked(i['fixed_A2_final_health'])); allchecks(h)
    require(h['schema'] == 'full151105_checkpoint_CPU_health:1' and h['status'] == 'PASS_SAVED_ENGINEERING_HEALTH', 'Actual105 initial health')
    require(h['method'] == 'A2' and h['iteration'] == 2000 and h['seed'] == 0 and h['final_iteration_scope'] is True, 'Fixed A2final2000')
    require(same(h['checkpoint'], i['fixed_A2_checkpoint']) and i['fixed_A2_checkpoint']['sha256'] == FROZEN_CHECKPOINT_SHA, 'Fixed initial checkpoint')
    require(same(h['auditor'], i['fixed_A2_health_auditor']) and same(h['config'], i['fixed_A2_training_config']), 'Initial health producer/config')
    # strict119f internally also checks these literal recorded aliases; never rewrite its inputs.
    require(h['checkpoint'] == i['fixed_A2_checkpoint'] and h['auditor'] == i['fixed_A2_health_auditor'], 'Preserve strict119f literal aliases')
    t = read(checked(i['fixed_A2_training_config'])); run = Path(t['output']).resolve(strict=True)
    require(run.samefile(h['run_dir']) and checked(i['fixed_A2_checkpoint']).samefile(run/'train/model_final.pt'), 'Actual final route')
    require(t['schema'] == 'full151_knownrough_native_training_run:2' and t['method'] == 'A2' and t['role'] == 'primary_matched', 'Parent schema2 primary A2')
    require(same(t['input_bundle'], i['input_bundle']) and checked(i['fixed_A2_fixed_physics_context']).samefile(run/'fixed_physics_context.npz'), 'Bundle/xi provenance')
    review = read(checked(i['response_saved_independent22'])); allchecks(review)
    require(review['schema'] == 'full151105_saved_action_response_saved_independent:1' and review['status'] == 'PASS_SAVED_ALL152_RESPONSE_REVIEW_NOT_JACOBIAN' and review['engineering_qualified'] is True, 'Actual response22 required')
    require(same(review['config'], i['response_config']) and same(review['producer_report'], i['response_report']), 'Response proof aliases')
    response = read(checked(i['response_report'])); response_run = checked(i['response_report']).parent
    require(Path(review['run_dir']).resolve(strict=True).samefile(response_run) and response['fixed_visible_state_interpretation_permitted'] is True, 'Same visible-state source run')
    require(same(response['config'], i['response_config']) and checked(i['response_output_pins']).samefile(response_run/'output_pins.json'), 'Response output manifest')
    outpins = read(checked(i['response_output_pins']))
    for alias, name in (('response_raw152', 'all152_branch_raw.npz'), ('response_metadata152', 'all152_branch_metadata.json'),
                        ('response_pairs64', 'all64_pair_records.json'), ('response_guard_summary', 'guard_summary.json')):
        require(same(i[alias], outpins[name]), 'Response immutable artifact: ' + alias)
    guards = read(checked(i['response_guard_summary']))
    require(guards and all(v['all_passed'] is True and v['failed_elements'] == 0 for v in guards.values()), 'Original full visible/control guards')
    require(response['physical_gradient_qualified'] is False and response['whole_hidden_state_qualified'] is False, 'Do not upgrade parent scope')
    values = list(p['source_pins'].values()) + [pin(v) for v in OWN.values()]
    values += list(t['source_pins'].values())
    values += list(read(checked(i['response_config']))['source_pins'].values())
    sources = canonical_sources(values)
    loader = ROOT/'src/full151_native/world_model_checkpoint.py'
    require(pin(loader)['sha256'] == FROZEN_WM_LOADER_SHA and any(same(v, pin(loader)) for v in sources.values()), 'Frozen119f source membership')
    for own in OWN.values():
        require(any(same(v, pin(own)) for v in p['source_pins'].values()), 'Protocol lacks current own source')
    out = None if output is None else Path(output).resolve()
    if out is not None:
        require(out.parent == ROOT/'outputs' and out.name.startswith(OUTPUT_PREFIX) and not out.exists(), 'Fresh scoped output only')
    require(not any(m in sys.modules for m in ('torch', 'mujoco', 'warp')), 'Metadata imported backend')
    return {'status': 'PASS_HASH_JSON_ONLY_NOT_CALIBRATED', 'protocol': p,
            'protocol_pin': pin(protocol_path), 'sources': sources, 'initial_training_run': str(run),
            'output': None if out is None else str(out), 'Torch_imported': False,
            'NPZ_opened': 0, 'strict_loads': 0, 'WM_forwards_AD_optimizer_steps': 0}


def _write_json(path, value, *, replace=False):
    """Small local artifact IO only; new experiment files never touch parents."""
    path = Path(path)
    if replace:
        tmp = path.with_name(path.name + '.partial')
        require(not tmp.exists(), 'Stale local temporary file')
        with tmp.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    else:
        with path.open('x', encoding='utf-8') as f:
            json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def _tensor_digest(value, torch):
    h = hashlib.sha256()
    for key, tensor in sorted(value.items()):
        require(isinstance(tensor, torch.Tensor), 'Tensor-state map required')
        a = tensor.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str((str(a.dtype), tuple(a.shape))).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _norm_state(wm):
    return {k: v for k, v in wm.state_dict().items()
            if k.startswith(('norm.', 'delta_norm.')) or k == 'error_floor'}


def _norm_config(wm):
    # RunningNorm's host count and arithmetic eps are not state_dict buffers.
    return {name: {k: getattr(getattr(wm, name), k) for k in ('_host_count', 'eps', 'max_count')}
            for name in ('norm', 'delta_norm')}


def _rng(torch, np):
    require(not torch.cuda.is_initialized(), 'CPU experiment must never initialize CUDA')
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_CPU': torch.random.get_rng_state().clone(), 'CUDA_initialized': False}


def _rng_digest(value):
    # Use raw CPU RNG tensor bytes, not torch's pickle storage identity.
    v = dict(value); v['torch_CPU'] = value['torch_CPU'].numpy().tobytes()
    return hashlib.sha256(pickle.dumps(v, protocol=5)).hexdigest()


def _finite_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.device.type == 'cpu' and (not value.is_floating_point() or bool(torch.isfinite(value).all()))
    if isinstance(value, dict): return all(_finite_tree(v, torch) for v in value.values())
    if isinstance(value, (list, tuple)): return all(_finite_tree(v, torch) for v in value)
    if type(value) is float: return math.isfinite(value)
    return value is None or type(value) in (int, bool, str)


def adam_step_evidence(optimizer, expected_iteration, torch):
    """Inspect actual Adam states, independently of the adapter's call counter."""
    parameters = [q for group in optimizer.param_groups for q in group['params']]
    values = []
    require(len(parameters) > 0 and len(optimizer.state) == len(parameters), 'Every WM parameter needs Adam state')
    for parameter in parameters:
        require(parameter in optimizer.state and 'step' in optimizer.state[parameter], 'Missing actual Adam step')
        step = optimizer.state[parameter]['step']
        require(isinstance(step, torch.Tensor) and step.device.type == 'cpu' and step.numel() == 1
                and bool(torch.isfinite(step).all()), 'Finite scalar CPU Adam step')
        value = float(step)
        require(value == expected_iteration, 'Actual Adam step differs from checkpoint iteration')
        values.append(value)
    return {'parameter_states': len(parameters), 'actual_step_values': sorted(set(values)),
            'expected_iteration': expected_iteration, 'all_actual_steps_equal_index': True,
            'scope': 'IN_MEMORY_NOT_SERIALIZED_READBACK'}


def _prepare_data(raw, metadata, pairs, np):
    """Narrow new adapter guards; parent22 supplies the full old provenance proof."""
    shapes = {'pre_visible152_FP32': (8,19,58), 'target_y152_FP32': (8,19,58),
              'actual_u152_FP32': (8,19,20), 'original_snapshot_visible58_FP32': (8,58),
              'original_snapshot_u_FP32': (8,20)}
    for key, shape in shapes.items():
        a = raw[key]
        require(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all(), 'Finite source FP32: ' + key)
    require(type(metadata) is list and len(metadata) == 152 and type(pairs) is list and len(pairs) == 64, 'All152/64 source records')
    x = raw['pre_visible152_FP32']; common = raw['original_snapshot_visible58_FP32']
    require(np.array_equal(x[:,:,:40].view(np.uint32), np.broadcast_to(common[:,None,:40], (8,19,40)).view(np.uint32)), 'Shared visible q/qd bits')
    require(np.array_equal(x[:,:,40:], np.broadcast_to(common[:,None,40:], (8,19,18))), 'Shared visible numeric18')
    u = raw['actual_u152_FP32']; center = raw['original_snapshot_u_FP32']
    require(np.array_equal(u[:,:3].view(np.uint32), np.broadcast_to(center[:,None], (8,3,20)).view(np.uint32)), 'Original baseline u bits')
    for row_id, row in enumerate(metadata):
        require(type(row['row_id']) is int and row['row_id'] == row_id and row['snapshot_id'] == row_id//19, 'Mechanical source branch order')
    minus = []; plus = []; du = []
    for pair_id, row in enumerate(pairs):
        s = pair_id//8; j = (pair_id%8)//2; ei = pair_id%2
        require(row['pair_id'] == pair_id and row['snapshot_id'] == s and row['j'] == j and row['epsilon_index'] == ei, 'Original pair order')
        coordinate = (0,4,8,12)[j]; off = 3 + 8*ei + 2*j
        mi, pl = s*19+off, s*19+off+1
        require(row['minus_branch_id'] == mi and row['plus_branch_id'] == pl and row['joint_index'] == coordinate, 'Original endpoint coordinate join')
        for idx in (mi, pl):
            r = metadata[idx]
            require(r['type'] == 'direction' and r['joint_index'] == coordinate and r['snapshot_id'] == s, 'Pair/source metadata join')
            mask = np.arange(20) != coordinate
            require(np.array_equal(u.reshape(152,20)[idx,mask].view(np.uint32), center[s,mask].view(np.uint32)), 'Non-j controls unchanged')
        actual = float(u.reshape(152,20)[pl,coordinate].astype(np.float64) - u.reshape(152,20)[mi,coordinate].astype(np.float64))
        require(math.isfinite(actual) and actual > 0 and actual == row['actual_coordinate_denominator_rad'], 'Nonzero actual F64 control denominator; no repair')
        minus.append(mi); plus.append(pl); du.append(actual)
    return {'pre': x.reshape(152,58).copy(), 'target': raw['target_y152_FP32'].reshape(152,58).copy(),
            'u': u.reshape(152,20).copy(), 'common_pre': np.broadcast_to(common[:,None], (8,19,58)).reshape(152,58).copy(),
            'minus': np.asarray(minus, np.int64), 'plus': np.asarray(plus, np.int64),
            'actual_du_rad_F64': np.asarray(du, np.float64),
            'same_pre_bits18_diagnostic': (x[:,:,40:].view(np.uint32) == common[:,None,40:].view(np.uint32))}


def chart_guard(a, b, delta52, margin_rad, torch):
    """Guard the ACTUAL native quaternion-log chart, not an invented acos log.

    Native so3_log -> mat_to_quat -> quat_to_rotvec is unchanged. The detached
    raw trace cosine is diagnostic only; no R projection/repair is inserted.
    """
    require(bool(torch.isfinite(delta52).all()), 'Nonfinite native52/chart output')
    angle = torch.linalg.vector_norm(delta52[...,43:46].detach().double(), dim=-1)
    require(bool(torch.isfinite(angle).all()) and bool((angle < math.pi-margin_rad).all()), 'Native log chart near cut locus; stop whole run')
    ra = a[...,43:52].detach().double().reshape(-1,3,3)
    rb = b[...,43:52].detach().double().reshape(-1,3,3)
    raw_cos = ((ra*rb).sum((-2,-1))-1.)*.5
    return {'native_log_rotvec_norm_max_rad': float(angle.max()),
            'raw_trace_cosine_min': float(raw_cos.min()), 'raw_trace_cosine_max': float(raw_cos.max()),
            'raw_trace_cosine_outside_unit_interval': int(((raw_cos < -1) | (raw_cos > 1)).sum()),
            'additional_rotation_projection': False}


def common_pre_losses(pred, target, common_pre, s52, target_common52,
                      minus, plus, actual_du_rad_F64, difference, torch, *, diagnostics=None):
    """One pred152 is reused. Native pointwise stays FP32; pair subtracts in F64."""
    point_delta = difference(pred, target)
    predicted_common52 = difference(common_pre, pred)
    charts = {'pointwise': chart_guard(pred, target, point_delta, RECIPE['chart_margin_rad'], torch),
              'common_pre_model': chart_guard(common_pre, pred, predicted_common52, RECIPE['chart_margin_rad'], torch)}
    point = (point_delta/s52).square().mean()
    model_response = predicted_common52[plus].double()-predicted_common52[minus].double()
    physical_response = target_common52[plus].double()-target_common52[minus].double()
    h = actual_du_rad_F64/RECIPE['control_reference_rad']
    require(bool(torch.isfinite(h).all()) and bool((h != 0).all()), 'Undefined actual control denominator')
    pair = ((model_response-physical_response)/s52.double()/h[:,None]).square().mean()
    require(bool(torch.isfinite(point)) and bool(torch.isfinite(pair)), 'Nonfinite loss; no subset or fallback')
    if diagnostics is not None:
        diagnostics.update(point_delta52_FP32=point_delta.detach(),
                           pred_common52_FP32=predicted_common52.detach())
    return point, pair, charts


def _make_trainable(frozen, torch):
    class OwnTrainableWorldModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wm = copy.deepcopy(frozen.wm)  # Never unfreeze the original wrapper.
            self.wm.eval().requires_grad_(True)
            self.wm.amp = False
            self.register_buffer('physical_xi', frozen.physical_xi.detach().clone())
            require(self.physical_xi.shape == (46,), 'Frozen physical xi46')
            require(not any(isinstance(m, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm)) for m in self.wm.modules()), 'No stochastic/moving-stat layer allowed')
        def forward(self, x, u):
            require(x.dtype == u.dtype == torch.float32 and x.device.type == u.device.type == 'cpu', 'FP32 CPU only')
            dynamics = torch.cat((x, self.physical_xi[None].expand(x.shape[0], -1), x.new_zeros((x.shape[0],174))), -1)
            return self.wm(x, u, dynamics)
    return OwnTrainableWorldModel()


def _atomic_torch_save(path, payload, torch):
    path = Path(path); tmp = path.with_name(path.name+'.partial')
    require(not path.exists() and not tmp.exists(), 'Fresh component checkpoint only')
    try:
        with tmp.open('xb') as f:
            torch.save(payload, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        # Keep any partial file as evidence; never mask the body exception.
        raise


def run_calibration(protocol_path, output, emit=None):
    """Future explicit Root CPU run. This DRAFT has not been executed/frozen."""
    started = time.perf_counter(); m = metadata_preflight(protocol_path, output)
    p = validate_protocol(m['protocol'], require_authorized=True); i = p['inputs']
    out = Path(m['output']); out.mkdir()
    emit = emit or (lambda event: None)
    with (out/'protocol.json').open('xb') as f:
        f.write(checked(m['protocol_pin']).read_bytes())
    _write_json(out/'source_inventory.json', m['sources'])
    with zipfile.ZipFile(out/'execution_sources.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for key, value in m['sources'].items(): z.write(checked(value), key)
    _write_json(out/'progress.json', {'status': 'STARTING_STRICT_INITIAL_VALIDATION', 'iteration': 0,
        'Goal_pause_controls_this_process': False, 'background_continue_to_final': p['background_continue_to_final']})
    phase = 'backend_import'; counts = {name: {'WM_forwards': 0, 'backward_calls': 0, 'optimizer_steps': 0} for name in ARMS}
    torch = None; checkpoint_pins = []; validation_forwards = 0
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        import numpy as np
        import torch
        torch.set_num_threads(4)
        require(not torch.cuda.is_initialized(), 'No CUDA initialized')
        from full151_native.bundle import load_bundle
        from full151_native.world_model_checkpoint import load_world_model_checkpoint
        from dex_supermimic.rl.dex_supermimic.state import difference
        original_rng = _rng(torch, np); rng_before = _rng_digest(original_rng)
        phase = 'strict105_initial_model_load'
        bundle = load_bundle(checked(i['input_bundle']))
        old_load = torch.load; load_calls = 0
        def counted_load(*args, **kwargs):
            nonlocal load_calls
            load_calls += 1
            return old_load(*args, **kwargs)
        torch.load = counted_load
        try:
            frozen, initial_receipt = load_world_model_checkpoint(
                checked(i['fixed_A2_checkpoint']), bundle, 'A2', m['initial_training_run'],
                checked(i['fixed_A2_final_health']), expected_health_auditor=i['fixed_A2_health_auditor'], expected_iteration=2000)
        finally:
            torch.load = old_load
        require(load_calls == 2 and _rng_digest(_rng(torch, np)) == rng_before, 'Strict load count/RNG changed')
        require(all(not parameter.requires_grad for parameter in frozen.parameters()), 'Original strict wrapper must remain frozen')
        initial_sha = _tensor_digest(frozen.wm.state_dict(), torch)
        norm_sha = _tensor_digest(_norm_state(frozen.wm), torch)
        norm_config = _norm_config(frozen.wm)
        xi_sha = _tensor_digest({'xi': frozen.physical_xi}, torch)
        require(initial_sha == initial_receipt['WM_state_sha256'], 'Strict native initial state identity')
        phase = 'saved_training_data_only'
        with np.load(checked(i['response_raw152']), allow_pickle=False) as f:
            raw = {k: f[k].copy() for k in ('pre_visible152_FP32', 'target_y152_FP32', 'actual_u152_FP32',
                                          'original_snapshot_visible58_FP32', 'original_snapshot_u_FP32')}
        data = _prepare_data(raw, read(checked(i['response_metadata152'])), read(checked(i['response_pairs64'])), np)
        np.savez_compressed(out/'matched_fit_data152.npz', **data)
        x, target, u, common_pre = (torch.from_numpy(data[k]) for k in ('pre', 'target', 'u', 'common_pre'))
        minus, plus = (torch.from_numpy(data[k]) for k in ('minus', 'plus'))
        du = torch.from_numpy(data['actual_du_rad_F64'])
        scale = torch.maximum(frozen.wm.delta_norm.std(), frozen.wm.error_floor).detach().clone()
        require(scale.shape == (52,) and bool(torch.isfinite(scale).all()) and bool((scale > 0).all()), 'Frozen native s52')
        target_common = difference(common_pre, target).detach()
        target_chart = chart_guard(common_pre, target, target_common, RECIPE['chart_margin_rad'], torch)
        # One separately counted frozen forward, on exactly the later fit batch.
        # Do not substitute archived batch452 outputs for this bit-parity gate.
        with torch.no_grad():
            strict_pred, strict_self15 = frozen(x, u)
        validation_forwards += 1
        require(_tensor_digest(frozen.wm.state_dict(), torch) == initial_sha
                and _norm_config(frozen.wm) == norm_config, 'Frozen validation changed parent state')
        np.savez_compressed(out/'strict_initial_samebatch152_validation.npz',
                            predicted152_FP32=strict_pred.numpy(), self15_FP32=strict_self15.numpy())
        own = {name: _make_trainable(frozen, torch) for name in ARMS}
        optim = {name: torch.optim.AdamW(own[name].wm.parameters(), lr=RECIPE['lr'],
                     betas=tuple(RECIPE['betas']), eps=RECIPE['optimizer_eps'], weight_decay=0., foreach=False) for name in ARMS}
        for name in ARMS:
            require(_tensor_digest(own[name].wm.state_dict(), torch) == initial_sha, 'Two-arm initial parameters/normalizers differ')
            require(_tensor_digest({'xi': own[name].physical_xi}, torch) == xi_sha, 'Two-arm physical xi differs')
            require(_norm_config(own[name].wm) == norm_config, 'Two-arm normalizer host configuration differs')
        require(_rng_digest(_rng(torch, np)) == rng_before, 'Import/build/data path changed RNG; do not restore to conceal it')
        _atomic_torch_save(out/'initial_component_state.pt', {'schema': ARTIFACT_SCHEMA,
            'parent_checkpoint105': i['fixed_A2_checkpoint'], 'WM': copy.deepcopy(frozen.wm.state_dict()),
            'frozen_s52': scale, 'physical_xi46': frozen.physical_xi,
            'normalizer_host_configuration': norm_config, 'RNG': original_rng}, torch)
        initial_parity = {}
        _write_json(out/'initial_validation.json', {'strict_loader': initial_receipt, 'strict_loader_calls': 1,
            'strict_frozen_validation_forwards': validation_forwards,
            'two_trainable_first_iteration_parity': 'PENDING_UNTIL_EACH_FIRST_UPDATE',
            'actual_torch_load_calls': load_calls, 'source_accounted_passive_MJB_loads': {'caller_bundle': 1, 'strict_loader_internal': 2, 'total': 3},
            'initial_WM_sha256': initial_sha, 'frozen_norm_sha256': norm_sha, 'xi46_sha256': xi_sha,
            'normalizer_host_configuration': norm_config,
            'RNG_sha256': rng_before, 'target_chart': target_chart,
            'Actor_forwards': 0, 'Actor_validation_constructors_source_accounted': 1,
            'data_rows': 152, 'pairs': 64, 'donor_rows_for_fit': 0, 'new_Env_physics_GPU': 0})
        phase = 'lockstep_fullbatch_calibration'; fit_started = time.perf_counter(); iteration = 0
        with (out/'updates.jsonl').open('x', encoding='utf-8') as log:
            for iteration in range(1, RECIPE['iterations']+1):
                rows = {}
                for name in ARMS:
                    tick = time.perf_counter(); optimizer = optim[name]
                    optimizer.zero_grad(set_to_none=True)
                    prediction, self15 = own[name](x, u); counts[name]['WM_forwards'] += 1
                    require(prediction.shape == (152,58) and self15.shape == (152,15) and bool(torch.isfinite(prediction).all()) and bool(torch.isfinite(self15).all()), 'Finite one fullbatch WM result')
                    if iteration == 1:
                        initial_parity[name] = {
                            'prediction58_bitexact': torch.equal(prediction.detach().contiguous().view(torch.int32), strict_pred.contiguous().view(torch.int32)),
                            'self15_bitexact': torch.equal(self15.detach().contiguous().view(torch.int32), strict_self15.contiguous().view(torch.int32))}
                        require(all(initial_parity[name].values()), 'Initial deepcopy first samebatch prediction differs from strict frozen validation')
                    diagnostics = {} if iteration in (1, 100) else None
                    lp, lr, charts = common_pre_losses(prediction, target, common_pre, scale, target_common, minus, plus, du, difference, torch, diagnostics=diagnostics)
                    loss = lp if name == ARMS[0] else lp + RECIPE['lambda_pair']*lr
                    if diagnostics is not None:
                        # All are from this single update's PRE-optimizer forward.
                        # iteration100 means model after99 updates, NOT checkpoint100.
                        np.savez_compressed(out/('%s_iteration_%04d_preupdate_loss_evidence.npz' % (name, iteration)),
                            pred152_FP32=prediction.detach().numpy(),
                            point_delta52_FP32=diagnostics['point_delta52_FP32'].numpy(),
                            pred_common52_FP32=diagnostics['pred_common52_FP32'].numpy(),
                            target_common52_FP32=target_common.numpy(), scale52_FP32=scale.numpy(),
                            actual_du_rad_F64=du.numpy(), minus_indices=minus.numpy(), plus_indices=plus.numpy())
                    loss.backward(); counts[name]['backward_calls'] += 1
                    require(all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in own[name].wm.parameters()), 'Nonfinite parameter gradient')
                    gradient_norm = torch.nn.utils.clip_grad_norm_(own[name].wm.parameters(), RECIPE['grad_clip_norm'])
                    require(bool(torch.isfinite(gradient_norm)), 'Nonfinite norm; do not skip update')
                    optimizer.step(); counts[name]['optimizer_steps'] += 1
                    require(_finite_tree(own[name].wm.state_dict(), torch) and _finite_tree(optimizer.state_dict(), torch), 'Nonfinite parameters/optimizer')
                    require(_tensor_digest(_norm_state(own[name].wm), torch) == norm_sha and _tensor_digest({'xi': own[name].physical_xi}, torch) == xi_sha, 'Frozen statistics/context changed')
                    require(_norm_config(own[name].wm) == norm_config, 'Frozen normalizer host configuration changed')
                    rows[name] = {'pointwise_loss': float(lp.detach()), 'common_pre_pair_loss': float(lr.detach()),
                        'optimized_loss': float(loss.detach()), 'pre_clip_gradient_norm': float(gradient_norm),
                        'gradient_clipped': bool(gradient_norm > RECIPE['grad_clip_norm']), 'chart': charts,
                        'update_seconds': time.perf_counter()-tick, 'counts': dict(counts[name])}
                require(_tensor_digest(frozen.wm.state_dict(), torch) == initial_sha, 'Original frozen119f WM changed')
                require(_norm_config(frozen.wm) == norm_config, 'Original frozen normalizer host configuration changed')
                require(all(not q.requires_grad for q in frozen.parameters()), 'Original wrapper was unfrozen')
                record = {'iteration': iteration, 'arms': rows, 'elapsed_fit_seconds': time.perf_counter()-fit_started}
                log.write(json.dumps(record, allow_nan=False)+'\n'); log.flush()
                if iteration % RECIPE['checkpoint_interval'] == 0:
                    phase = 'checkpoint_'+str(iteration)
                    rng_now = _rng(torch, np)
                    actual_adam = {name: adam_step_evidence(optim[name], iteration, torch) for name in ARMS}
                    checks = {'both_initial_WM_equal': True,
                        'both_initial_samebatch_predictions_bitexact_strict_validation': all(all(row.values()) for row in initial_parity.values()) and set(initial_parity) == set(ARMS),
                        'actual_update_counts_equal_index': all(counts[name] == {'WM_forwards': iteration, 'backward_calls': iteration, 'optimizer_steps': iteration} for name in ARMS),
                        'actual_Adam_state_steps_equal_index': all(v['all_actual_steps_equal_index'] for v in actual_adam.values()),
                        'all_parameters_optimizers_finite': all(_finite_tree(own[name].wm.state_dict(), torch) and _finite_tree(optim[name].state_dict(), torch) for name in ARMS),
                        'all_frozen_normalizers_unchanged': all(_tensor_digest(_norm_state(own[name].wm), torch) == norm_sha for name in ARMS),
                        'all_normalizer_host_configuration_unchanged': all(_norm_config(own[name].wm) == norm_config for name in ARMS),
                        'all_physical_xi_unchanged': all(_tensor_digest({'xi': own[name].physical_xi}, torch) == xi_sha for name in ARMS),
                        'original_strict_wrapper_unchanged': _tensor_digest(frozen.wm.state_dict(), torch) == initial_sha,
                        'RNG_unchanged_without_repair': _rng_digest(rng_now) == rng_before,
                        'CUDA_never_initialized': not torch.cuda.is_initialized()}
                    require(all(checks.values()), 'Checkpoint component health failed')
                    this = {}
                    for name in ARMS:
                        filename = '%s_update_%04d.pt' % (name, iteration)
                        payload = {'schema': ARTIFACT_SCHEMA, 'arm': name, 'iteration': iteration,
                            'parent_checkpoint105': i['fixed_A2_checkpoint'], 'parent_health': i['fixed_A2_final_health'],
                            'protocol': m['protocol_pin'], 'recipe': RECIPE, 'initial_WM_sha256': initial_sha,
                            'frozen_norm_sha256': norm_sha, 'physical_xi46': own[name].physical_xi.detach().clone(),
                            'normalizer_host_configuration': norm_config,
                            'frozen_s52': scale.clone(), 'data_pins': {k:i[k] for k in ('response_raw152','response_metadata152','response_pairs64','response_saved_independent22')},
                            'WM_state': copy.deepcopy(own[name].wm.state_dict()), 'optimizer_state': copy.deepcopy(optim[name].state_dict()),
                            'counts': dict(counts[name]), 'RNG': rng_now, 'source_pins': m['sources'],
                            'actual_Adam_step_evidence': actual_adam[name],
                            'native105_checkpoint_or_policy_resume_compatible': False,
                            'fresh_test_evaluated': False}
                        _atomic_torch_save(out/filename, payload, torch); this[name] = pin(out/filename)
                    elapsed = time.perf_counter()-fit_started
                    remaining = (RECIPE['iterations']-iteration)*elapsed/iteration
                    health = {'schema': 'full151105_WM_component_checkpoint_health:1', 'iteration': iteration,
                        'status': 'PASS_COMPONENT_HEALTH_NOT_ACCURACY', 'checks': checks, 'passed': len(checks), 'total': len(checks),
                        'health_verification_scope': 'IN_MEMORY_BEFORE_SAVE_NOT_SERIALIZED_READBACK',
                        'serialized_checkpoint_readback_verified': False,
                        'actual_Adam_step_evidence': actual_adam,
                        'initial_samebatch_bitparity': initial_parity,
                        'strict_frozen_validation_forwards': validation_forwards,
                        'checkpoints': this, 'counts': counts, 'first_checkpoint_user_pause_condition': iteration == 100,
                        'Goal_pause_controls_this_process': False, 'background_continue_to_final': p['background_continue_to_final'],
                        'ETA': {'elapsed_fit_seconds_including_prior_saves': elapsed, 'remaining_updates_per_arm': RECIPE['iterations']-iteration,
                            'remaining_seconds_linear_heuristic': remaining, 'planning_range_seconds_not_CI': [remaining*.5, remaining*1.8],
                            'unmeasured_finalization_overhead_seconds': 20., 'not_a_GPU_or_native_training_ETA': True},
                        'fresh_test': 'SEALED_NOT_READ_NOT_EVALUATED', 'accuracy_or_physical_gradient_qualified': False}
                    _write_json(out/('checkpoint_%04d_health.json' % iteration), health)
                    checkpoint_pins.append(pin(out/('checkpoint_%04d_health.json' % iteration)))
                    _write_json(out/'progress.json', health, replace=True)
                    emit({'stage': 'checkpoint_health', 'iteration': iteration, 'health': checkpoint_pins[-1],
                          'status': health['status'], 'background_continue_to_final': p['background_continue_to_final']})
                    if iteration == 100 and not p['background_continue_to_final']:
                        break
                    phase = 'lockstep_fullbatch_calibration'
        for value in i.values(): checked(value)
        for value in m['sources'].values(): checked(value)
        require(same(pin(protocol_path), m['protocol_pin']), 'Protocol changed during calibration')
        complete = iteration == RECIPE['iterations']
        result = {'schema': 'full151105_WM_component_calibration_result:1',
            'status': 'COMPLETED_COMPONENT_CALIBRATION_NO_TEST' if complete else 'STOPPED_AT_FIRST_COMPONENT_CHECKPOINT_NO_TEST',
            'completed_iterations_per_arm': iteration, 'counts': counts, 'checkpoints': checkpoint_pins,
            'strict_frozen_validation_forwards': validation_forwards,
            'initial_samebatch_bitparity': initial_parity,
            'protocol': m['protocol_pin'], 'inputs': i, 'initial_WM_sha256': initial_sha,
            'frozen_norm_sha256': norm_sha, 'xi46_sha256': xi_sha,
            'background_continue_to_final': p['background_continue_to_final'], 'Goal_pause_controls_this_process': False,
            'fresh_test_evaluated': False, 'component_accuracy_confirmed': False,
            'policy_training_Env_physics_GPU': 0, 'elapsed_total_seconds': time.perf_counter()-started}
        _write_json(out/'result.json', result)
        _write_json(out/'output_pins.json', {f.name: pin(f) for f in out.iterdir() if f.is_file()})
        return result
    except BaseException as error:
        failure = {'schema': 'full151105_WM_component_calibration_failure:1',
            'status': 'FAILED_COMPONENT_CALIBRATION_NO_REPAIR_NO_TEST', 'phase': phase, 'error': repr(error),
            'counts': counts, 'protocol': m['protocol_pin'], 'checkpoints': checkpoint_pins,
            'strict_frozen_validation_forwards': validation_forwards,
            'fresh_test_evaluated': False, 'no_automatic_retry': True}
        try:
            if torch is not None and 'own' in locals():
                partial = {'schema': ARTIFACT_SCHEMA, 'status': failure['status'],
                           'parent_checkpoint105': i['fixed_A2_checkpoint'],
                           'counts': counts, 'protocol': m['protocol_pin'],
                           'WM_states': {name: value.wm.state_dict() for name, value in own.items()},
                           'optimizer_states': {name: value.state_dict() for name, value in optim.items()}}
                _atomic_torch_save(out/'failure_component_state.pt', partial, torch)
        except BaseException as save_error:
            failure['partial_checkpoint_save_error'] = repr(save_error)
        try:
            _write_json(out/'failure.json', failure)
        except BaseException as save_error:
            try: error.add_note('Failure evidence write also failed: '+repr(save_error))
            except BaseException: pass
        raise
