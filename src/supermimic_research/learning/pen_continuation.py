"""Explicit continuation of the frozen first100 Fn6 pen experiment.

The parent entry point and its source pins remain unchanged. This stage never
refits scales, resets an optimizer, changes data roles, or trains an Actor.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..artifacts import file_pin, verify_file_pin
from .checkpoint import _adam_valid, _finite_tree
from .gpu_monitor import GPUMonitor
from .impulse_stage import update_impulse
from .pen_response import INPUT_SCHEMA, OUTPUT_SCHEMA, pen_families, validate_arm_schema, training_cache
from .pen_stage import restore_arm, metrics, write
from .world_model import atomic_checkpoint, capture_rng, raw_tree_equal


def validate_parent(saved, health, prep):
    """Reject lineage/schema/cursor drift before creating an output directory."""
    if not saved['arms']:raise ValueError('Explicit paired parent required')
    cfg=next(iter(saved['arms'].values()))['config'];modes=pen_families(cfg);version=cfg.get('schema',1)
    if (saved['schema'] != f'controlled_pen_response_pair_checkpoint:{version}'
            or saved['update'] != 100 or saved['planned_updates'] != 2000
            or health['schema'] != f'controlled_pen_first100_health:{version}'
            or health['update'] != 100 or health['training_running']
            or not health['checks'] or not all(health['checks'].values())
            or health['passed'] != health['total']
            or health['total'] != len(health['checks'])
            or saved['dataset_pins'] != prep['outputs']
            or set(saved['arms']) != set(modes)):
        raise ValueError('Healthy first100 Fn6 pair with unchanged data required')
    configs = []
    for mode, arm in saved['arms'].items():
        validate_arm_schema(arm)
        cfg = arm['config']; configs.append(cfg)
        if (arm['schema'] != f'controlled_pen_response_arm:{version}'
                or arm['input_schema'] != INPUT_SCHEMA
                or arm['output_schema'] != OUTPUT_SCHEMA
                or arm['mode'] != mode or arm['coupled'] != (mode == 'IMPULSE')
                or arm['update'] != 100 or arm['physics'] != prep['physics']
                or cfg['first_checkpoint_update'] != 100
                or cfg['planned_updates'] != 2000
                or len(saved['logs'][mode]) != 100):
            raise ValueError('Explicit matched arm schema, physics and cursor required')
    if configs[0] != configs[1] or not raw_tree_equal(
            saved['arms'][modes[0]]['sampler'], saved['arms'][modes[1]]['sampler']):
        raise ValueError('Matched hyperparameters and sampler cursors required')
    return configs[0]


def continuation_payload(arm, parent, update, device, gradient):
    return {**parent, 'update': update, 'model': arm['model'].state_dict(),
            'optimizer': arm['optimizer'].state_dict(), 'sampler': arm['sampler'].get_state(),
            'rng': capture_rng(device=device), 'gradient_snapshot': gradient}


def run_continuation(root, checkpoint_path, health_path, out, *, run=False):
    root, cp, hp, out = [Path(p).resolve() for p in (root, checkpoint_path, health_path, out)]
    health = json.loads(hp.read_text(encoding='utf8'))
    if verify_file_pin(health['checkpoint']) != cp:
        raise ValueError('Health must bind the exact requested parent checkpoint')
    saved = torch.load(cp, map_location='cpu', weights_only=False)
    prep_path = verify_file_pin(saved['prepared']); prep = json.loads(prep_path.read_text())
    review_path = verify_file_pin(saved['review']); review = json.loads(review_path.read_text())
    cfg = validate_parent(saved, health, prep)
    modes=pen_families(cfg);version=cfg.get('schema',1)
    if not all(prep['checks'].values()) or review['passed'] != review['total'] or verify_file_pin(review['prepared']) != prep_path:
        raise ValueError('Unchanged healthy preparation/review required')
    pins = saved['sources'] + prep['sources'] + [saved['config'], review['source'], prep['scales'], prep['head_scales']]
    for pin in pins: verify_file_pin(pin)
    data = {role: dict(np.load(verify_file_pin(pin))) for role, pin in saved['dataset_pins'].items()}
    summary = dict(training_started=False, parent_update=100, target_update=2000,
                   additional_updates_per_arm=1900, Actor_updates=0,
                   valid_rows={r: int(d['valid'].sum()) for r, d in data.items()})
    if not run: return summary
    if out.exists() or not out.is_relative_to(root / 'runs') or not torch.cuda.is_available():
        raise ValueError('Fresh unified CUDA output required')
    device = torch.device(cfg['device'])
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    sources = [file_pin(root / p) for p in (
        'tools/continue_pen_response.py', 'src/supermimic_research/learning/pen_continuation.py')]
    lineage = dict(parent_checkpoint=file_pin(cp), parent_health=file_pin(hp),
                   prepared=saved['prepared'], review=saved['review'], config=saved['config'],
                   parent_sources=saved['sources'], sources=sources, dataset_pins=saved['dataset_pins'])
    arms = {mode: restore_arm(a, device) for mode, a in saved['arms'].items()}
    prechecks = {}
    for mode, arm in arms.items():
        parent = saved['arms'][mode]
        prechecks.update({mode + '_restored_' + k: v for k, v in dict(
            model=raw_tree_equal(arm['model'].state_dict(), parent['model']),
            optimizer=raw_tree_equal(arm['optimizer'].state_dict(), parent['optimizer']),
            sampler=raw_tree_equal(arm['sampler'].get_state(), parent['sampler']),
            Adam100=_adam_valid(arm['optimizer'], arm['model'], 100, torch)).items()})
    if not all(prechecks.values()): raise ValueError('Exact continuation restoration failed')
    d = data['TRAIN']; mask = d['valid']
    cache = training_cache(d,device,modes)
    check = data['CHECK']; mask = check['valid']
    cx = torch.as_tensor(check['x'][mask], dtype=torch.float32, device=device)
    cy = torch.as_tensor(check['y'][mask], dtype=torch.float64, device=device)
    curve = {'100': {m: metrics(a['model'], cx, cy)[0] for m, a in arms.items()}}
    prechecks['parent_CHECK_metrics_exact'] = curve['100'] == health['CHECK_metrics']
    if not all(prechecks.values()): raise ValueError('Restored parent evaluation changed')
    out.mkdir(); write(out / 'intent.json', dict(schema=f'pen_response_continuation:{version}', **lineage,
        **summary, restored=prechecks, optimizer_reset=False, scales_refitted=False,
        CHECK_used_for_optimizer_or_selection=False,
        scope='Fixed 2000-update paired component experiment; no policy training or ego validation.'))
    logs = {m: [] for m in arms}; timers = []; gradients = {}; last_batch = {}
    monitor = GPUMonitor(); monitor.start(); torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    try:
        for update in range(101, 2001):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for mode, arm in arms.items():
                value, indices, grad = update_impulse(arm, cache, cfg, capture_gradient=update == 2000)
                logs[mode].append(value)
                if update == 2000: gradients[mode] = grad; last_batch[mode] = indices.detach().clone()
            end.record(); timers.append((begin, end))
            if update in (500, 1000, 2000):
                curve[str(update)] = {m: metrics(a['model'], cx, cy)[0] for m, a in arms.items()}
        torch.cuda.synchronize(device); wall = time.perf_counter() - start
    finally:
        monitor.stop()
    milliseconds = [a.elapsed_time(b) for a, b in timers]
    peak = dict(allocated=torch.cuda.max_memory_allocated(device), reserved=torch.cuda.max_memory_reserved(device))
    full_logs = {m: saved['logs'][m] + torch.stack(v).cpu().tolist() for m, v in logs.items()}
    children = {m: continuation_payload(a, saved['arms'][m], 2000, device, gradients[m]) for m, a in arms.items()}
    final_cp = out / 'pen_response_checkpoint_2000.pt'
    atomic_checkpoint(final_cp, dict(schema=saved['schema'], update=2000, planned_updates=2000,
        arms=children, **lineage, logs=full_logs))
    read = torch.load(final_cp, map_location='cpu', weights_only=False)
    checks = dict(prechecks); parity = {}
    for mode, arm in arms.items():
        child = read['arms'][mode]; clone = restore_arm(child, device)
        _, p = metrics(arm['model'], cx, cy); _, q = metrics(clone['model'], cx, cy)
        local = dict(model_exact=raw_tree_equal(arm['model'].state_dict(), clone['model'].state_dict()),
            optimizer_exact=raw_tree_equal(arm['optimizer'].state_dict(), clone['optimizer'].state_dict()),
            sampler_exact=raw_tree_equal(arm['sampler'].get_state(), clone['sampler'].get_state()),
            CHECK_prediction_exact=raw_tree_equal(p, q), Adam2000=_adam_valid(clone['optimizer'], clone['model'], 2000, torch),
            finite_model_optimizer_gradient=_finite_tree([child['model'], child['optimizer'], gradients[mode]], torch),
            finite_full_logs=bool(np.isfinite(full_logs[mode]).all()),
            frozen_buffers=all(raw_tree_equal(v, saved['arms'][mode]['model'][k]) for k, v in arm['model'].named_buffers()),
            Fn6_finite_nonnegative=bool(torch.isfinite(p).all() and (p[:,84:] >= 0).all()),
            cursor=child['update'] == 2000 and len(full_logs[mode]) == 2000)
        checks.update({mode + '_' + k: bool(v) for k, v in local.items()})
        a, b = restore_arm(child, device), restore_arm(child, device)
        la, ia, ga = update_impulse(a, cache, cfg, capture_gradient=True)
        lb, ib, gb = update_impulse(b, cache, cfg, capture_gradient=True)
        parity[mode] = dict(loss_exact=raw_tree_equal(la, lb), batch_exact=raw_tree_equal(ia, ib),
            gradients_exact=raw_tree_equal(ga, gb), model_exact=raw_tree_equal(a['model'].state_dict(), b['model'].state_dict()),
            optimizer_exact=raw_tree_equal(a['optimizer'].state_dict(), b['optimizer'].state_dict()),
            sampler_exact=raw_tree_equal(a['sampler'].get_state(), b['sampler'].get_state()))
        checks.update({mode + '_clone_next_' + k: v for k, v in parity[mode].items()})
        del clone, a, b
    checks.update(matched_last_batch=raw_tree_equal(last_batch[modes[0]], last_batch[modes[1]]),
        matched_sampler_cursor=raw_tree_equal(read['arms'][modes[0]]['sampler'], read['arms'][modes[1]]['sampler']),
        saved_cursor=read['update'] == 2000)
    for pin in pins + sources + [lineage['parent_checkpoint'], lineage['parent_health']]: verify_file_pin(pin)
    result = dict(schema=f'pen_response_continuation_health:{version}', **lineage,
        checkpoint=file_pin(final_cp), checks=checks, passed=sum(checks.values()), total=len(checks),
        CHECK_curve=curve, update=2000, additional_live_updates_per_arm=1900,
        extra_clone_optimizer_updates=4, Actor_optimizer_updates=0, training_running=False,
        paired_compute_and_metric_wall_s=wall, CUDA_ms_per_paired_update=milliseconds,
        training_peak_bytes=peak, GPU_telemetry=dict(samples=monitor.samples, errors=monitor.errors),
        hardware=dict(GPU_name=torch.cuda.get_device_name(device), Torch_version=torch.__version__),
        scope='Fixed-dataset WM fit only. CHECK curves do not establish physical gradients, rollout accuracy, Actor benefit or ego generalization.')
    write(out / 'health_2000.json', result)
    if not all(checks.values()): raise ValueError('Keep failed artifacts; no downstream use')
    return dict(checkpoint=file_pin(final_cp), health=file_pin(out / 'health_2000.json'),
        passed=result['passed'], total=result['total'], training_running=False,
        paired_compute_and_metric_wall_s=wall, CHECK_metrics=curve['2000'])
