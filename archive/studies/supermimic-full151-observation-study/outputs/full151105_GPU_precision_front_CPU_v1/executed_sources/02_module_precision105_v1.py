"""Explicit GPU precision diagnostic on the frozen full05 eight-state input.

Default preflight is saved metadata only. The strict CPU consumer stays frozen;
its validated native core is transferred into this separately authorized ABI.
No environment, Warp, physics, optimizer or Actor forward is used.
"""
from __future__ import annotations
import os
import hashlib
from pathlib import Path
import pickle
import random
import time
import numpy as np
from . import analysis105_v1 as a

MODES = ('FP32_STORAGE_TF32_ALLOWED', 'BF16_NET_FP32_INTEGRATE_TF32_ALLOWED')
SCHEMA = 'full151105_GPU_precision_diagnostic_run:1'
FIXED_CPU = {
    'CPU_prerequisite_config': 'c7f84bfbff5e3bd62f8f75f4f297917948cc208bc7d8b7f2796c96c0f76f7cbf',
    'CPU_report': '588bff73f32abce3ff756d3bfaf2f503d294f70c57b253766e87246da0131502',
    'CPU_saved_review': 'f382bb580efe865c0d9818197ff1bdfbf678868eefd0a72845fe35d3ccb90404',
}


def validate_modes(config):
    a.require(config.get('schema') == SCHEMA and config.get('role') == 'full151105_GPU_precision_diagnostic', 'GPU precision namespace required')
    a.require(config.get('modes') == list(MODES), 'Exactly two fixed precision modes required')
    a.require(config.get('allow_tf32') is True and config.get('device') == 'cuda:0', 'Fixed single-device source-intended TF32 context required')
    a.require(config.get('evaluation_seed') == 0 and type(config['evaluation_seed']) is int, 'Seed0 only')
    a.require(type(config.get('GPU_run_authorized')) is bool and config.get('training_authorized') is False and config.get('Env_or_physics_authorized') is False, 'Separate GPU authorization; no training or physics')
    a.require(config.get('automatic_retry') is False and config.get('compile_fallback') is False, 'No retry/fallback')
    a.require((config.get('soft_worker_seconds'), config.get('hard_process_seconds')) == (60, 90), 'Fixed bounded execution budget')


def preflight(config_path):
    a.require('torch' not in __import__('sys').modules and 'warp' not in __import__('sys').modules, 'Fresh metadata process required')
    c = a.read(config_path); validate_modes(c)
    for key, value in FIXED_CPU.items(): a.require(c[key]['sha256'] == value, 'Fixed CPU evidence differs: ' + key)
    for value in c['source_pins'].values(): a.checked(value)
    a.require(a.pin(__file__) in c['source_pins'].values(), 'Current precision source must be archived')
    p = a.metadata_preflight(a.checked(c['CPU_prerequisite_config']))
    cpu = a.read(a.checked(c['CPU_report']))
    a.require(cpu['schema'] == 'full151105_physical_gradient_CPU_analysis:1' and cpu['status'] == 'COMPLETED_CPU_DIAGNOSTIC', 'Actual CPU comparison required')
    a.require(a.same(cpu['config'], c['CPU_prerequisite_config']) and all(cpu['invariants'].values()), 'CPU prerequisite source/config/invariants differ')
    a.require(cpu['comparison_rows'] == 64 and cpu['dependent_direction_groups'] == 32 and cpu['WM_forward_calls'] == cpu['u_autograd_calls'] == 1, 'CPU actual counts differ')
    saved = a.read(a.checked(c['CPU_saved_review'])); a._all_checks(saved, 'CPU saved audit incomplete')
    a.require(saved['schema'] == 'full151105_WM_analysis_saved_independent:1', 'Exact saved analysis schema required')
    a.require(a.same(saved['config'], c['CPU_prerequisite_config']) and a.same(saved['analysis_report'], c['CPU_report'])
              and a.same(saved['analysis_arrays'], c['CPU_predictions']), 'Saved CPU review joins differ')
    a.require(all(saved.get(k) is False for k in ('physical_gradient_qualified', 'whole_hidden_state_qualified', 'HO_gradient_qualified')), 'CPU evidence cannot upgrade qualification')
    # Rebind every original CPU output, not just its producer report.
    pins = a.read(a.checked(c['CPU_output_pins']))
    for value in pins.values(): a.checked(value)
    a.require(a.same(pins['report.json'], c['CPU_report']) and a.same(pins['WM_predictions_gradients.npz'], c['CPU_predictions']), 'CPU predictions/report route differs')
    protocol = a.read(a.checked(c['protocol']))
    a.require(protocol['schema'] == 'full151105_GPU_precision_preregistration:1' and protocol['execution_authorized'] is False, 'Frozen source-only preregistration required')
    for value in protocol['source_pins'].values(): a.checked(value)
    for value in protocol['source_pins'].values():
        a.require(any(value['sha256'] == v['sha256'] and a.checked(value).samefile(a.checked(v)) for v in c['source_pins'].values()), 'Preregistration source unarchived')
    with np.load(a.checked(c['CPU_predictions']), allow_pickle=False) as z: old = {k: z[k].copy() for k in z.files}
    a.require(old['gradient_u'].shape == (8, 20) and old['predicted_x'].shape == (8, 58), 'CPU arrays dimensions differ')
    a.require(a.bitexact(old['snapshot_x'], p['arrays']['snapshot_x']) and a.bitexact(old['actualu'], p['arrays']['snapshot_u']), 'GPU and CPU physical input centers differ')
    return c, p, old


def projected_rows(gradient, physical_rows):
    g = np.asarray(gradient)
    a.require(g.shape == (8, 20) and g.dtype == np.float32 and np.isfinite(g).all(), 'Finite FP32 gradient8x20 required')
    rows = []
    for row in physical_rows:
        v = dict(row)
        v['WM_AD_effective'] = float(g[row['snapshot_id']].astype(np.float64) @ np.asarray(row['d_eff'], np.float64))
        v['WM_AD_nominal'] = float(g[row['snapshot_id'], row['joint_index']])
        rows.append(v)
    a.require(len(rows) == 64, 'Keep all64 epsilon rows')
    return rows


def difference_summary(left, right):
    left, right = np.asarray(left, np.float64), np.asarray(right, np.float64)
    a.require(left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all(), 'Comparable finite arrays required')
    delta = left - right
    return {'maximum_absolute': float(abs(delta).max()), 'mean_absolute': float(abs(delta).mean()),
            'RMS': float(np.sqrt(np.mean(delta * delta)))}


def run(config_path, output, emit):
    began = time.perf_counter(); c, p, old = preflight(config_path)
    a.require(c['GPU_run_authorized'] is True and os.environ.get('CUDA_VISIBLE_DEVICES') != '-1', 'Explicit GPU run required')
    out = Path(output); base = p['config']; a.require(out.resolve() == Path(c['output']).resolve(), 'Exact output required')
    import torch
    a.require(not torch.cuda.is_initialized() and 'warp' not in __import__('sys').modules, 'Fresh isolated GPU process required')
    torch.set_num_threads(4); torch.manual_seed(0)
    from full151_native.bundle import load_bundle
    from full151_native.world_model_checkpoint import load_world_model_checkpoint
    from full151_native.checkpoint import _tensor_digest
    from full151_gradient.serial_probe105_v2 import controlled_object_score_FP32
    bundle = load_bundle(a.checked(base['input_bundle']), expected_sha256=base['input_bundle']['sha256'])
    emit(status='loading_strict_CPU_model', phase='strict_load')
    cpu_torch_rng_before = torch.get_rng_state().clone()
    host_state_before = (random.getstate(), np.random.get_state())
    cpu_rng_before = pickle.dumps(host_state_before, protocol=5) + cpu_torch_rng_before.numpy().tobytes()
    original_load, reads = torch.load, []
    def counted_load(*args, **kwargs):
        reads.append(str(args[0])); return original_load(*args, **kwargs)
    torch.load = counted_load
    try:
        wrapper, receipt = load_world_model_checkpoint(a.checked(base['checkpoint']), bundle, 'A2', base['training_run'], a.checked(base['health']), expected_health_auditor=base['health_auditor'], expected_iteration=2000)
    finally: torch.load = original_load
    a.require(len(reads) == 2 and all(Path(v).samefile(a.checked(base['checkpoint'])) for v in reads), 'One strict CPU validation with original two checkpoint reads')
    core = wrapper.wm
    a.require(core.amp is False and not core.training and all(not v.requires_grad for v in core.parameters()), 'Frozen CPU-loaded native core required')
    weights_before = _tensor_digest(core.state_dict())
    norm_before = _tensor_digest({k: v for k, v in core.state_dict().items() if 'norm.' in k})
    xi = wrapper.physical_xi.detach().clone()
    a.require(not torch.cuda.is_initialized(), 'Strict loader must not initialize CUDA')
    device = torch.device(c['device']); torch.cuda.init()
    a.require(torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported(), 'One visible BF16-capable GPU required')
    torch.backends.cuda.matmul.allow_tf32 = True
    core.to(device); physical_xi = xi.to(device)
    a.require(_tensor_digest(core.state_dict()) == weights_before, 'Transfer changed weight/normalizer bytes')
    gpu_rng_before = [v.clone() for v in torch.cuda.get_rng_state_all()]
    x = torch.from_numpy(p['arrays']['snapshot_x'].copy()).to(device)
    center = torch.from_numpy(p['arrays']['snapshot_u'].copy()).to(device)
    gp = torch.from_numpy(p['arrays']['goal_p'].copy()).to(device); gR = torch.from_numpy(p['arrays']['goal_R'].copy()).to(device)
    inputs_before = _tensor_digest({'x': x, 'u': center, 'xi': physical_xi, 'gp': gp, 'gR': gR})
    physical_rows = a.read(Path(a.checked(c['CPU_report'])).parent / 'comparisons.json')
    results = {}; raw = {}; captures = {}; flags = []; flags_after = []
    def current_flags():
        def optional_matmul(name):
            try: return getattr(torch.backends.cuda.matmul, name)
            except (AttributeError, RuntimeError) as exc: return {'status': 'UNAVAILABLE', 'reason': repr(exc)}
        return dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32, float32_matmul_precision=torch.get_float32_matmul_precision(),
                    allow_bf16_reduced_precision_reduction=optional_matmul('allow_bf16_reduced_precision_reduction'),
                    allow_bf16_reduced_precision_reduction_split_k=optional_matmul('allow_bf16_reduced_precision_reduction_split_k'),
                    allow_fp16_reduced_precision_reduction=optional_matmul('allow_fp16_reduced_precision_reduction'),
                    allow_fp16_accumulation=optional_matmul('allow_fp16_accumulation'),
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(), deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
                    CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                    NVIDIA_TF32_OVERRIDE=os.environ.get('NVIDIA_TF32_OVERRIDE'),
                    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'),
                    CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    for mode, amp in zip(MODES, (False, True)):
        a.require(time.perf_counter() - began < c['soft_worker_seconds'], 'Soft precision deadline; no retry')
        emit(status='running_GPU_precision', phase=mode)
        core.amp = amp; norm_values = []; net_dtypes = []
        def capture_norm(module, args, result): norm_values.append(result.detach().cpu().numpy().copy())
        def capture_net(module, args, result): net_dtypes.append(str(result.dtype))
        handles = [core.norm.register_forward_hook(capture_norm), core.net.register_forward_hook(capture_net)]
        try:
            flags.append(current_flags()); u = center.detach().clone().requires_grad_(True)
            dynamics = torch.cat((x, physical_xi[None].expand(8, -1), x.new_zeros((8, 174))), -1)
            pred, contact = core(x, u, dynamics)
            terms, score = controlled_object_score_FP32(pred, gp, gR)
            gradient, = torch.autograd.grad(score.sum(), u)
            torch.cuda.synchronize()
            flags_after.append(current_flags())
        finally:
            for handle in handles: handle.remove()
        a.require(len(norm_values) == len(net_dtypes) == 1 and net_dtypes[0] == ('torch.bfloat16' if amp else 'torch.float32'), 'Actual native net output precision differs')
        values = {'predicted_x': pred, 'self15_logits': contact, 'gradient_u': gradient, 'predicted_score': score, 'score_terms': terms}
        values = {k: v.detach().cpu().numpy().copy() for k, v in values.items()}
        a.require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in values.values()), 'Nonfinite/nonFP32 native outputs')
        rows = projected_rows(values['gradient_u'], physical_rows)
        metrics = a._metric_groups(values['predicted_x'], p['arrays']['donor_next_x'])
        metrics['score_RMS'] = difference_summary(values['predicted_score'], p['arrays']['donor_object_score_k'])['RMS']
        results[mode] = {'agreement': a.agreement(rows), 'prediction_metrics': metrics,
                         'CPU_gradient_difference': difference_summary(values['gradient_u'], old['gradient_u']),
                         'CPU_state_difference_separate_units': a._metric_groups(values['predicted_x'], old['predicted_x']),
                         'native_net_output_dtype': net_dtypes[0], 'amp': amp}
        captures[mode] = norm_values[0]
        for key, value in values.items(): raw[mode + '__' + key] = value
        raw[mode + '__normalized_features'] = norm_values[0]
        a._write(out, mode + '_comparisons.json', rows)
    core.amp = False
    host_state_after = (random.getstate(), np.random.get_state())
    cpu_rng_after = pickle.dumps(host_state_after, protocol=5) + torch.get_rng_state().numpy().tobytes()
    gpu_rng_after = torch.cuda.get_rng_state_all()
    invariants = dict(weights_and_normalizers_unchanged=_tensor_digest(core.state_dict()) == weights_before,
        normalizers_unchanged=_tensor_digest({k: v for k, v in core.state_dict().items() if 'norm.' in k}) == norm_before,
        CPU_RNG_unchanged=cpu_rng_before == cpu_rng_after,
        GPU_RNG_unchanged=all(torch.equal(v, w) for v, w in zip(gpu_rng_before, gpu_rng_after)) and len(gpu_rng_before) == len(gpu_rng_after),
        inputs_unchanged=_tensor_digest({'x': x, 'u': center, 'xi': physical_xi, 'gp': gp, 'gR': gR}) == inputs_before,
        flags_held_between_modes=a.same(flags[0], flags[1]), normalized_features_bitexact=a.bitexact(captures[MODES[0]], captures[MODES[1]]),
        flags_held_within_modes=all(a.same(v, w) for v, w in zip(flags, flags_after)),
        parameters_have_no_grad=all(v.grad is None for v in core.parameters()), no_Warp='warp' not in __import__('sys').modules)
    a.require(all(invariants.values()), 'GPU precision invariance failure')
    for value in c['source_pins'].values(): a.checked(value)
    for key in ('CPU_report', 'CPU_saved_review', 'CPU_output_pins', 'CPU_predictions', 'CPU_prerequisite_config', 'protocol'): a.checked(c[key])
    weight_after = _tensor_digest(core.state_dict())
    rng_arrays = {'torch_CPU_before': cpu_torch_rng_before.numpy().copy(),
                  'torch_CPU_after': torch.get_rng_state().numpy().copy()}
    for i, value in enumerate(gpu_rng_before): rng_arrays['CUDA_before_%d' % i] = value.cpu().numpy()
    for i, value in enumerate(gpu_rng_after): rng_arrays['CUDA_after_%d' % i] = value.cpu().numpy()
    np.savez_compressed(out / 'precision_arrays.npz', **raw)
    np.savez_compressed(out / 'RNG_arrays.npz', **rng_arrays)
    with (out / 'RNG_host_states.pkl').open('xb') as file: pickle.dump({'before': host_state_before, 'after': host_state_after}, file, protocol=5)
    report = dict(schema='full151105_GPU_precision_diagnostic_report:1', status='COMPLETED_GPU_PRECISION_DIAGNOSTIC', config=a.pin(config_path),
        modes=list(MODES), strict_CPU_WM_loader_calls=1, actual_checkpoint_reads=2, WM_forward_calls=2, predicted_rows=16, u_autograd_calls=2,
        physical_comparison_rows_per_mode=64, dependent_direction_groups=32, models=results, invariants=invariants,
        mode_pair_gradient_difference=difference_summary(raw[MODES[1] + '__gradient_u'], raw[MODES[0] + '__gradient_u']),
        actual_flags_per_mode=flags, actual_flags_after_modes=flags_after,
        weights_before=weights_before, weights_after=weight_after, normalizers_before=norm_before,
        normalizers_after=_tensor_digest({k: v for k, v in core.state_dict().items() if 'norm.' in k}),
        CPU_RNG_before_sha256=hashlib.sha256(cpu_rng_before).hexdigest(), CPU_RNG_after_sha256=hashlib.sha256(cpu_rng_after).hexdigest(),
        GPU_device_name=torch.cuda.get_device_name(device), GPU_capability=list(torch.cuda.get_device_capability(device)), torch_version=torch.__version__, CUDA_runtime=torch.version.cuda,
        source_intended_training_allow_tf32=True, historical_training_runtime_flags_observed=False,
        GPU_used=True, Env_created=False, physics_steps=0, optimizer_updates=0, Actor_forwards=0, compile_used=False,
        physical_gradient_qualified=False, whole_hidden_state_qualified=False, elapsed_seconds=time.perf_counter() - began,
        scope='Two single GPU passes with native net AMP toggle and TF32 allowed. No repeatability estimate; numerical mode differences cannot be uniquely attributed to precision alone. CPU-vs-GPU differences also conflate backend/TF32. Finite physical secants are unchanged; no true unique Jacobian, HO gradient, ego or hardware certificate.')
    a._write(out, 'strict_CPU_loader_receipt.json', receipt)
    a._write(out, 'report.json', report)
    a._write(out, 'output_pins.json', {f.name: a.pin(f) for f in out.iterdir() if f.is_file() and f.name != 'runtime.json'})
    return report
