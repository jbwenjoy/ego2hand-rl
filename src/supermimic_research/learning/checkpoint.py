"""Own-checkpoint readback checks shared by offline and online stages.

The caller owns experiment cursors, lineage and pause policy. This module
constructs five readback networks and makes ten diagnostic forwards; it never
fits a normalizer, runs an optimizer or restores RNG as a way to hide drift.
"""
import json
import hashlib
import math
from pathlib import Path

import numpy as np

from . import world_model as WM, imagination as IM, update_step as U


HEALTH_SCHEMA = 'supermimic_research.checkpoint_health:1'


def _finite_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(v, torch) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(v, torch) for v in value)
    return True


def _log_prefix_valid(prefix):
    """Only the immutable prefix is bound; later appended rows are allowed."""
    size, records, span = prefix['bytes'], prefix['records'], prefix['range']
    if (type(size) is not int or size < 1 or type(records) is not int
            or records < 1 or len(span) != 2
            or any(type(v) is not int for v in span)
            or span[0] < 1 or span[1] - span[0] + 1 != records):
        return False
    with Path(prefix['path']).open('rb') as stream:
        raw = stream.read(size)
    if (len(raw) != size or not raw.endswith(b'\n')
            or hashlib.sha256(raw).hexdigest() != prefix['sha256']):
        return False
    rows = [json.loads(row) for row in raw.splitlines()]
    return (len(rows) == records
            and [row['update'] for row in rows] == list(range(span[0], span[1] + 1)))


def _optimizer_modules(context):
    modules = {'WM': context['model']}
    for mode, arm in context['arms'].items():
        modules[mode + '_Actor'] = arm['actor']
        modules[mode + '_Critic'] = arm['critic']
    return modules


def _adam_valid(optimizer, module, expected_step, torch):
    parameters = tuple(module.parameters())
    grouped = tuple(p for group in optimizer.param_groups for p in group['params'])
    if (not parameters or len(grouped) != len(parameters)
            or any(a is not b for a, b in zip(grouped, parameters))
            or len(optimizer.state) != len(parameters)
            or set(optimizer.state) != set(parameters)):
        return False
    for parameter in parameters:
        state = optimizer.state[parameter]
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys():
            return False
        step = state['step']
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or not _finite_tree(step, torch) or float(step) != expected_step):
            return False
        for name in ['exp_avg', 'exp_avg_sq']:
            moment = state[name]
            if (not isinstance(moment, torch.Tensor)
                    or moment.shape != parameter.shape or moment.dtype != parameter.dtype
                    or not _finite_tree(moment, torch)):
                return False
        if 'max_exp_avg_sq' in state and not _finite_tree(state['max_exp_avg_sq'], torch):
            return False
    return True


def _gradients_valid(snapshot, module, torch):
    named = dict(module.named_parameters())
    if (not isinstance(snapshot, dict) or not named
            or not isinstance(snapshot.get('tag'), str)
            or not snapshot['tag'].startswith('PRE')
            or set(snapshot.get('pre_clip', {})) != set(named)
            or set(snapshot.get('post_clip', {})) != set(named)):
        return False
    norm = snapshot.get('clip_returned_norm')
    if (not isinstance(norm, torch.Tensor) or norm.numel() != 1
            or not _finite_tree(norm, torch) or float(norm) < 0):
        return False
    for name, parameter in named.items():
        for phase in ['pre_clip', 'post_clip']:
            value = snapshot[phase][name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not _finite_tree(value, torch)):
                return False
        if (parameter.grad is None
                or not WM.raw_tree_equal(snapshot['post_clip'][name], parameter.grad)):
            return False
    return True


def checkpoint_health(path, context, optimizers, gradients, counts, probeF, probeO,
                      *, schema, extra_checks=None):
    """Read back an own artifact and return ``(health, raw_probe_arrays)``.

    ``schema`` is the expected checkpoint schema, not the returned generic
    health schema. Wrappers may give the returned health a stage-specific
    schema. ``extra_checks(ck)`` supplies Python-bool cursor/pool/lineage gates;
    this helper does not infer an offline common-update or online root offset.
    Counts describe this stage's Adam history: WM_updates and paired_updates.
    The standard raw gradient field is ``gradient_snapshots``.
    """
    s = context
    torch = s['torch']
    device = torch.device(s['device'])
    if not isinstance(schema, str) or not schema:
        raise ValueError('Explicit expected own-checkpoint schema')
    if len(s['arms']) != 2:
        raise ValueError('Exactly two matched policy arms for five-network health')
    modules = _optimizer_modules(s)
    if set(optimizers) != set(modules) or set(gradients) != set(modules):
        raise ValueError('Exact five optimizer and gradient snapshot keys')
    if any(type(counts.get(k)) is not int or counts[k] < 1
           for k in ['WM_updates', 'paired_updates']):
        raise ValueError('Positive own-stage WM/paired Adam counters')
    rng = WM.capture_rng(device=device)
    model, arms = s['model'], s['arms']
    with torch.no_grad():
        expected = {'WM': model(probeF)}
        expected.update({mode + '_' + key: arm[key](probeO)
                         for mode, arm in arms.items() for key in ['actor', 'critic']})
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck.get('schema') != schema:
        raise ValueError('Unexpected own-checkpoint artifact schema')
    # Constructor RNG is temporary; observed prediction calls may not draw RNG.
    with torch.random.fork_rng(devices=[] if device.type == 'cpu' else [device.index]):
        cfg = s['c']['model']
        clone = WM.make_model(WM.WorldModelConfig(
            hidden=tuple(cfg['hidden']), geometry_chunk_rows=cfg.get('geometry_chunk_rows', 4)),
            device=device)
        clone.load_state_dict(ck['shared_WM'], strict=True)
        clone.train(model.training)
        restored = {}
        for mode, arm in arms.items():
            pair = IM.make_fresh_actor_critic(seed=s['c']['training']['seed'], device=device)
            for key, field in [('actor', 'stateactor'), ('critic', 'statecritic')]:
                pair[key].load_state_dict(ck['arms'][mode][field], strict=True)
                pair[key].train(arm[key].training)
            restored[mode] = pair
    with torch.no_grad():
        actual = {'WM': clone(probeF)}
        actual.update({mode + '_' + key: arm[key](probeO)
                       for mode, arm in restored.items() for key in ['actor', 'critic']})
    current_states = {'WM': model.state_dict(), 'arms': U.arm_states(arms)}
    saved_states = {'WM': ck['shared_WM'], 'arms': ck['arms']}
    checks = dict(
        artifact_schema=ck['schema'] == schema,
        restored_predictions=all(WM.raw_tree_equal(expected[k], actual[k])
                                 and _finite_tree(actual[k], torch) for k in expected),
        all_networks_and_norms=WM.raw_tree_equal(saved_states, current_states),
        all_weights_and_norms_finite=_finite_tree(saved_states, torch)
                                    and _finite_tree(current_states, torch),
        all_optimizers=WM.raw_tree_equal(ck['optimizers'],
                                       {k: opt.state_dict() for k, opt in optimizers.items()}),
        all_Adam_steps_and_moments=all(_adam_valid(opt, modules[k],
            counts['WM_updates'] if k == 'WM' else counts['paired_updates'], torch)
            for k, opt in optimizers.items()),
        saved_gradients=WM.raw_tree_equal(ck['gradient_snapshots'], gradients),
        named_pre_post_gradients_finite_and_current=all(
            _gradients_valid(gradients[k], module, torch) for k, module in modules.items()),
        frozen_normalizers=WM.raw_tree_equal(ck['normalizers'], s['norm'])
                           and _finite_tree(ck['normalizers'], torch),
        loss_scales=WM.raw_tree_equal(ck['loss_scales'], s['scales'])
                    and _finite_tree(ck['loss_scales'], torch),
        physical_context=WM.raw_tree_equal(ck['physical_context'], s['physical'])
                         and _finite_tree(ck['physical_context'], torch),
        sample_generator=WM.raw_tree_equal(ck['sample_generator'], s['generator'].get_state()),
        log_prefix=_log_prefix_valid(ck['log_prefix'])
                   and all(_log_prefix_valid(v) for v in ck['parent_log_prefixes']),
        saved_RNG=WM.raw_tree_equal(ck['RNG'], rng),
        raw_RNG_unchanged=WM.raw_tree_equal(rng, WM.capture_rng(device=device)),
        configuration_and_bindings=ck['configuration'] == s['c']
                                   and ck['provenance'] == s['provenance'],
        counters=ck['counts'] == counts,
    )
    if extra_checks is not None:
        extra = extra_checks(ck)
        if (not isinstance(extra, dict) or any(not isinstance(k, str) or not k
                or type(v) is not bool for k, v in extra.items()) or set(extra) & set(checks)):
            raise ValueError('Extra health checks must be uniquely named Python bools')
        checks.update(extra)
    # Include any caller-supplied gate's RNG effects, too.
    checks['raw_RNG_unchanged'] = WM.raw_tree_equal(rng, WM.capture_rng(device=device))
    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError('Checkpoint health failed; preserve files, no retry: ' + ', '.join(failed))
    evidence = {}
    for key in expected:
        evidence['expected_' + key] = expected[key].detach().cpu().numpy().copy()
        evidence['restored_' + key] = actual[key].detach().cpu().numpy().copy()
    health = dict(schema=HEALTH_SCHEMA, checkpoint_schema=schema, checks=checks,
                  passed=sum(checks.values()), total=len(checks), counts=counts.copy(),
                  extra_NN_FW=10, extra_NN_constructors=5, own_CPU_checkpoint_reads=1,
                  parameter_counts={k: len(tuple(module.parameters())) for k, module in modules.items()},
                  evidence_keys=list(evidence))
    return health, evidence
