"""Saved full152 -> frozen105 A2 CPU diagnostic. Default path imports no Torch.

Finite-interval secants are empirical responses, not unique physical Jacobians.
Graph execution scratch remains unknown; native self15 is not a HO force head.
This module never creates an environment, restores state, runs physics or trains.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
import zipfile
import numpy as np

SCHEMA = 'full151105_physical_gradient_CPU_analysis_run:1'
PHASES = (0, 49, 99, 149)
JOINTS = (0, 4, 8, 12)
EPS = (.002, .001)
FIXED = {
    'input_bundle': '59406c6214a085afba7efcc2b33880d5c82b409ee4f6485ec71c073fec19dc76',
    'proposal': 'eedde4fff0fbf687ffcf32cc7e826a4c5190f2c4971281f6802f38162734d6b0',
    'checkpoint': '70d0a21d5b2f0238ab745264bf24db08515fef65f072cbe00fb6893a33512a0c',
    'health': '13b705b7421402091fde5bb431e835360f71185d80beceb72ccf4cefff92f574',
    'health_auditor': '5950b9947b00ed0cfce6a8403a65877db6d71be1d7b01b8c3ad1626bdd5de792',
    'loader': '119fa7cebc0e9b0727b0160660b1a47822deee2a0e0e9fa5be69465ac5578052',
}
ARTIFACTS = ('producer_report', 'physical_arrays', 'branch_index', 'snapshot_records',
             'donor_records', 'baseline_gates', 'capacity_ledgers', 'physical_samples',
             'reward_snapshots', 'runtime', 'producer_config', 'source_inventory',
             'source_zip', 'watch_state', 'engineering_identity', 'post_terminal_CIM', 'saved_review')
ARRAYS = {
    'snapshot_x': (8, 58), 'snapshot_u': (8, 20), 'donor_next_x': (8, 58), 'donor_self15': (8, 15),
    'goal_p': (8, 3), 'goal_R': (8, 3, 3), 'goal_p_next': (8, 3), 'goal_R_next': (8, 3, 3),
    'branch_x_next': (152, 58), 'branch_self15': (152, 15), 'branch_u': (152, 20),
    'branch_object_terms_k': (152, 2), 'branch_object_score_k': (152,),
    'branch_object_terms_k_CPU_secondary': (152, 2), 'branch_object_terms_kplus1': (152, 2),
    'donor_object_terms_k': (8, 2), 'donor_object_score_k': (8,),
    'donor_object_terms_k_CPU_secondary': (8, 2), 'donor_object_terms_kplus1': (8, 2),
    'branch_native_reward_terms_15ms_secondary': (152, 2), 'branch_native_reward_score_15ms_secondary': (152,),
    'donor_native_reward_terms_15ms_secondary': (8, 2), 'donor_native_reward_score_15ms_secondary': (8,),
    'direction__plus': (8, 4, 2, 20), 'direction__minus': (8, 4, 2, 20), 'direction__centers': (8, 20),
}
BOOLS = {'snapshot_completed': (8,), 'completed': (152,), 'restore_inventory_bitexact': (152,),
         'restore_qualified': (152,), 'direction__clipped': (8, 4, 2)}
F64S = {'direction__d_eff': (8, 4, 2, 20), 'direction__center_offset': (8, 4, 2, 20)}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    p = Path(path).resolve(strict=True)
    return {'path': str(p), 'sha256': digest(p)}


def checked(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact path/SHA pin required')
    p = Path(value['path']).resolve(strict=True)
    require(type(value['sha256']) is str and digest(p) == value['sha256'], 'Pin mismatch: ' + str(p))
    return p


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def same(a, b):
    """JSON types matter, including bool vs integer."""
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def bitexact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def tree_array_hash(arrays):
    return hashlib.sha256(json.dumps({k: array_hash(v) for k, v in sorted(arrays.items())}, sort_keys=True).encode()).hexdigest()


def object_terms_f64(x, goal_p, goal_R):
    """Independent same-x20 arithmetic diagnostic, never substitutes GPU primary."""
    x, goal_p, goal_R = (np.asarray(v, np.float64) for v in (x, goal_p, goal_R))
    pos, R = x[..., 40:43], x[..., 43:52].reshape(-1, 3, 3)
    off, goff = .1 * R[..., :, 0], .1 * goal_R[..., :, 0]
    endpoint = (np.linalg.norm(pos + off - goal_p - goff, axis=-1) + np.linalg.norm(pos - off - goal_p + goff, axis=-1)) * .5
    return np.stack((2 * np.exp(-(np.linalg.norm(pos - goal_p, axis=-1) / .05) ** 2), 8 * np.exp(-(endpoint / .04) ** 2)), -1)


def baseline_summary(a):
    """Independent replay tolerances and score floor, all scores FP32 -> FP64."""
    records = []
    for s in range(8):
        ids = np.arange(s * 19, s * 19 + 3)
        delta = abs(a['branch_x_next'][ids].astype(np.float64) - a['donor_next_x'][s].astype(np.float64))
        errors = {n: float(delta[:, lo:hi].max()) for n, lo, hi, tol in
                  [('q', 0, 20, 2e-5), ('qd', 20, 40, 2e-4), ('p', 40, 43, 2e-5),
                   ('R', 43, 52, 2e-5), ('v', 52, 55, 2e-4), ('omega', 55, 58, 2e-4)]}
        scores = a['branch_object_score_k'][ids].astype(np.float64)
        donor = float(a['donor_object_score_k'][s])
        span = float(np.ptp(scores)); err = float(abs(scores - donor).max())
        passed = all(errors[n] <= tol for n, tol in [('q', 2e-5), ('qd', 2e-4), ('p', 2e-5),
                      ('R', 2e-5), ('v', 2e-4), ('omega', 2e-4)]) and err <= 2e-5
        records.append(dict(snapshot_id=s, errors=errors, baseline_range=span,
                            baseline_donor_error=err, signal_floor=max(1e-5, 10 * max(span, err)), passed=bool(passed)))
    return records


def classify_pair(score_minus, score_plus, baseline_range, baseline_donor_error):
    """Two eps, one dependent direction group. Clipping is an orthogonal flag."""
    minus, plus = np.asarray(score_minus), np.asarray(score_plus)
    require(minus.shape == plus.shape == (2,) and np.isfinite(minus).all() and np.isfinite(plus).all(), 'Two finite epsilon scores required')
    require(np.isfinite([baseline_range, baseline_donor_error]).all() and min(baseline_range, baseline_donor_error) >= 0, 'Nonnegative baseline noise required')
    delta = plus.astype(np.float64) - minus.astype(np.float64)
    slope = delta / (2 * np.asarray(EPS, np.float64))
    floor = max(1e-5, 10 * max(float(baseline_range), float(baseline_donor_error)))
    signal = abs(delta) > floor
    rel = float(abs(slope[0] - slope[1]) / max(abs(slope[0]), abs(slope[1]), 1e-12))
    stable = bool(signal.all() and np.sign(slope[0]) == np.sign(slope[1]) and rel <= .5)
    label = 'STABLE_NONFLAT' if stable else ('FLAT_OR_BELOW_NOISE' if not signal.all() else 'NONFLAT_UNSTABLE')
    return dict(label=label, stable_nonflat=stable, signal_floor=floor, delta_score=delta.tolist(),
                slopes=slope.tolist(), relative_disagreement=rel, both_above_noise=bool(signal.all()))


def effective_direction(center, minus, plus, epsilon):
    require(epsilon in EPS, 'Frozen epsilon required')
    c, m, p = (np.asarray(x) for x in (center, minus, plus))
    require(c.shape == m.shape == p.shape == (20,) and all(x.dtype == np.float32 and np.isfinite(x).all() for x in (c, m, p)), 'FP32 control20 endpoints required')
    return (p.astype(np.float64) - m.astype(np.float64)) / (2 * epsilon), (p.astype(np.float64) + m.astype(np.float64)) * .5 - c.astype(np.float64)


def agreement(rows):
    """Descriptive only: no absent-class success, accuracy gate or CI."""
    out = {}
    for ei, epsilon in enumerate(EPS):
        selected = [r for r in rows if r['epsilon_index'] == ei and r['stable_nonflat']]
        p = np.array([r['physical_slope'] for r in selected]); w = np.array([r['WM_AD_effective'] for r in selected])
        n = len(selected)
        out[str(epsilon)] = dict(eligible=n, sign_correct=int((np.sign(p) == np.sign(w)).sum()),
            sign_fraction=None if n == 0 else float((np.sign(p) == np.sign(w)).mean()),
            pearson=None if n < 2 or np.ptp(p) == 0 or np.ptp(w) == 0 else float(np.corrcoef(p, w)[0, 1]),
            mean_absolute_disagreement=None if n == 0 else float(abs(p - w).mean()),
            clipped_eligible=sum(bool(r['clipped']) for r in selected))
    return out


def _all_checks(report, message):
    checks = report.get('checks')
    require(type(checks) is dict and checks and all(type(v) is bool and v for v in checks.values()), message)
    require(type(report.get('passed')) is int and report['passed'] == report.get('total') == len(checks), message + ' counts')


def validate_saved_review(review, config):
    """Inventory-only completion can never authorize this model consumer."""
    p = config['artifact_pins']
    require(review.get('schema') == 'full151105_full_serial_saved_independent:1', 'Actual full-serial saved audit required; inventory-only is not eligible')
    _all_checks(review, 'Saved audit has incomplete checks')
    expected = {'engineering_qualified': True, 'probe_mode': 'serial_full', 'completed_inventory_snapshots': 8,
                'completed_branches': 152, 'baseline_count': 24, 'episode_step_returned': 1808,
                'physical_gradient_qualified': False, 'whole_hidden_state_qualified': False}
    for k, v in expected.items(): require(same(review.get(k), v), 'Saved audit scope mismatch: ' + k)
    for k, v in {'input_bundle': config['input_bundle'], 'config': p['producer_config'],
                 'producer_report': p['producer_report'], 'source_inventory': p['source_inventory']}.items():
        require(same(review.get(k), v), 'Saved audit provenance mismatch: ' + k)
    require(Path(review['run_dir']).resolve() == Path(config['physical_run']).resolve(), 'Saved audit run route mismatch')


def _validate_capacity(ledgers):
    require(type(ledgers) is list and len(ledgers) == 2 and [v['path'] for v in ledgers] == ['A0', 'A2'], 'Two separate actual donor ledgers required')
    for item in ledgers:
        ledger = item['ledger']
        require(ledger.get('engineering_capacity_qualified') is True and ledger.get('finalized') is True and ledger.get('status') == 'OK', 'Capacity not finalized OK')
        # Recompute zero/unknown from actual raw records, not report booleans.
        records = ledger.get('records'); require(type(records) is list and records, 'Capacity raw records missing')
        require(ledger.get('cumulative_OR_bitmask') == 0 and ledger.get('unknown_read_count') == 0 and ledger.get('final_read_count') == 1 and ledger.get('record_count') == len(records), 'Capacity aggregate differs')
        final = [r for r in records if r.get('event') == 'final']
        require(len(final) == 1 and final[0] is records[-1], 'Exactly one terminal capacity read required')
        for index, row in enumerate(records):
            require(row.get('read_status') == row.get('overall_status') == 'OK', 'Capacity read UNKNOWN/error')
            raw = row.get('raw_flags_values'); require(type(raw) is list, 'Raw capacity flags absent')
            a = np.asarray(raw); require(a.dtype.kind in 'iu' and a.size > 0 and not a.any(), 'Nonzero or malformed raw capacity flag')
            require(list(a.shape) == row.get('raw_flags_shape') and np.dtype(row['raw_flags_dtype']).kind in 'iu', 'Capacity dtype/shape metadata mismatch')
            require(row.get('sequence') == index + 1 and row.get('raw_bitmask') == row.get('cumulative_OR_bitmask') == row.get('cumulative_unknown_reads') == 0, 'Capacity OR/UNKNOWN progression differs')
            require(row.get('flags_cleared_by_observer') is False and row.get('flag_shape_is_world_count') is False, 'Capacity flags may not be cleared or attributed to worlds')


def _validate_lifecycle(c, docs):
    state, identity, post, runtime = (docs[k] for k in ('watch_state', 'engineering_identity', 'post_terminal_CIM', 'runtime'))
    require(state.get('status') == 'completed' and state.get('exit_code') == 0 and state.get('timeout') is False, 'Actual supervisor terminal failure')
    require(identity.get('engineering_identity_qualified') is True and type(identity.get('checks')) is dict and len(identity['checks']) == 12 and all(v is True for v in identity['checks'].values()), 'Actual OS identity must pass all12')
    obs = state['actual_process_OS_observation']; cmd = state['command']
    require(obs.get('status') == 'exact_OS_identity_observed' and type(obs.get('pid')) is int and obs['pid'] > 0, 'Actual worker OS observation unknown')
    require(len(cmd) == 8 and cmd[3] == '--config' and cmd[5] == '--output' and cmd[7] == '--run', 'Unexpected actual worker command ABI')
    require(Path(cmd[4]).resolve() == checked(c['artifact_pins']['producer_config']) and Path(cmd[6]).resolve() == Path(c['physical_run']).resolve(), 'Actual command config/output route differs')
    require(runtime['pid'] == obs['pid'] == identity['actual_pid'] and runtime['parent_pid'] == obs['parent_pid'], 'Runtime PID/PPID differs')
    require(runtime['plan_sha256'] == c['artifact_pins']['producer_config']['sha256'], 'Runtime plan differs')
    require(post.get('actual_CIM_rows') == [] and obs['pid'] in post.get('queried_pids', []), 'Postterminal actual owned PID not confirmed absent')


def _validate_data(a, rows, snapshots, baselines, proposal):
    require(set(a) == set(ARRAYS) | set(BOOLS) | set(F64S), 'Unexpected/missing physical array ABI')
    for name, shape in ARRAYS.items(): require(a[name].dtype == np.float32 and a[name].shape == shape and np.isfinite(a[name]).all(), 'Nonfinite/uncompleted FP32 ABI: ' + name)
    for name, shape in F64S.items(): require(a[name].dtype == np.float64 and a[name].shape == shape and np.isfinite(a[name]).all(), 'F64 direction ABI: ' + name)
    for name, shape in BOOLS.items():
        require(a[name].dtype == np.bool_ and a[name].shape == shape, 'Boolean mask ABI: ' + name)
        if name != 'direction__clipped': require(a[name].all(), 'Uncompleted/unqualified physical row: ' + name)
    for name in ('donor_self15', 'branch_self15'): require(np.isin(a[name], [0, 1]).all(), 'Self15 must be native binary labels')
    for prefix in ('branch', 'donor'):
        require(bitexact(a[prefix + '_object_terms_k'].sum(-1, dtype=np.float32), a[prefix + '_object_score_k']), 'Primary GPU two-term FP32 sum differs')
        ids = np.arange(152) // 19 if prefix == 'branch' else np.arange(8)
        states = a['branch_x_next'] if prefix == 'branch' else a['donor_next_x']
        diagnostic = object_terms_f64(states, a['goal_p'][ids], a['goal_R'][ids])
        bound = 8 * np.finfo(np.float32).eps * np.maximum(1., abs(diagnostic))
        require(np.all(abs(a[prefix + '_object_terms_k'].astype(np.float64) - diagnostic) <= bound), 'Same-phase F64 arithmetic bound failed; never a15/20ms phase tolerance')
    require(type(rows) is list and len(rows) == 152 and type(snapshots) is list and len(snapshots) == 8, 'Full8/152 JSON rows required')
    for s in range(8):
        path, k = ('A0', 'A2')[s // 4], PHASES[s % 4]
        snap = snapshots[s]
        require(snap['snapshot_id'] == s and snap['path'] == path and snap['k'] == k and snap['baseline3_qualified'] is True and snap['empirical_visible_state_baseline_qualified'] is True and snap['foreign_review_qualified'] is True, 'Snapshot actual baseline/source scope mismatch')
        require(bitexact(a['snapshot_u'][s], a['direction__centers'][s]), 'Direction center differs from actual selected target')
        slots = [(None, None, None, 0, b) for b in range(3)] + [(j, ei, eps, sign, None) for ei, eps in enumerate(EPS) for j in range(4) for sign in (-1, 1)]
        for offset, (j, ei, eps, sign, repeat) in enumerate(slots):
            i = s * 19 + offset; r = rows[i]
            expected = dict(row_id=i, snapshot_id=s, path=path, k=k, j=j, epsilon_index=ei, epsilon_rad=eps, sign=sign, baseline_repeat=repeat,
                            joint_index=None if j is None else JOINTS[j], type='baseline' if j is None else 'direction', scheduled=True,
                            status='COMPLETED', reason=None, restorequalified=True, source_frame=200 + k, terminal_observation=k + 1,
                            valid_next_policy_decision=k + 1 < 150, no_u150=True, scalar_kind='matched_post20_gpu_score')
            for key, val in expected.items(): require(same(r.get(key), val), 'Branch order/status mismatch: %d/%s' % (i, key))
            require(bitexact(np.array(r['actualu'], np.float32), a['branch_u'][i]) and float(r['r_k']) == float(a['branch_object_score_k'][i]), 'JSON actual target/primary score differs')
            wanted = a['snapshot_u'][s] if j is None else a['direction__plus' if sign == 1 else 'direction__minus'][s, j, ei]
            require(bitexact(wanted, a['branch_u'][i]), 'Branch target differs from planned physical endpoint')
        for j, joint in enumerate(JOINTS):
            for ei, eps in enumerate(EPS):
                c, m, p = a['snapshot_u'][s], a['direction__minus'][s, j, ei], a['direction__plus'][s, j, ei]
                d, off = effective_direction(c, m, p, eps)
                require(bitexact(d, a['direction__d_eff'][s, j, ei]) and bitexact(off, a['direction__center_offset'][s, j, ei]), 'Saved effective direction/offset differs')
                unit = np.eye(20, dtype=np.float32)[joint]
                clipped = not (bitexact(p, c + np.float32(eps) * unit) and bitexact(m, c - np.float32(eps) * unit))
                require(bool(a['direction__clipped'][s, j, ei]) == clipped, 'Clip flag differs from actual FP32 endpoints')
    computed = baseline_summary(a); require(all(r['passed'] for r in computed), 'Independent three-baseline visible-state gate failed')
    require(len(baselines) == 24 and all(r.get('passed') is True for r in baselines), 'Saved24 baseline gate rows required')
    require(same(proposal['snapshot_phases'], list(PHASES)) and same(proposal['eps_rad'], list(EPS)) and same(proposal['direction_joint_indices'], list(JOINTS)), 'Proposal schedule changed')
    return computed


def _jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def _validate_captures(a, rewards, samples):
    require(len(rewards) == 452 and len(samples) == 1808, 'Actual reward/solver capture count differs')
    reward_map = {}
    for r in rewards:
        key = (r['path'], r['k'], r['branch_id'])
        require(key not in reward_map, 'Duplicate reward capture')
        reward_map[key] = r
        require(same(r['scalar_contract']['primary'], 'matched_post20_gpu_score'), 'Wrong reward scalar source')
    for s in range(8):
        key = (('A0', 'A2')[s // 4], PHASES[s % 4], None)
        r = reward_map[key]
        require(bitexact(np.array(r['physics'], np.float32).reshape(58), a['donor_next_x'][s]), 'Donor reward-snapshot state differs')
        require(bitexact(np.array(r['contact'], np.float32).reshape(15), a['donor_self15'][s]), 'Donor native self15 differs')
        require(float(r['matched_post20_GPU_score_FP32']) == float(a['donor_object_score_k'][s]), 'Donor primary score capture differs')
    for i in range(152):
        s = i // 19; r = reward_map[(('A0', 'A2')[s // 4], PHASES[s % 4], i)]
        require(bitexact(np.array(r['physics'], np.float32).reshape(58), a['branch_x_next'][i]), 'Branch reward state differs')
        require(bitexact(np.array(r['contact'], np.float32).reshape(15), a['branch_self15'][i]), 'Branch native self15 differs')
        require(float(r['matched_post20_GPU_score_FP32']) == float(a['branch_object_score_k'][i]), 'Branch primary GPU score differs')
    groups = {}
    for r in samples:
        key = (r['path'], r['phase'], r['branch_id']); groups.setdefault(key, []).append(r)
    require(len(groups) == 452, 'Every complete interval needs four solver samples')
    for (path, k, bid), values in groups.items():
        require([r['substep'] for r in values] == [1, 2, 3, 4], 'Solver order must be four5ms preintegration captures')
        for r in values:
            require(r['canonical_pre_tick'] == 4 * k + r['substep'] - 1 and r['canonical_post_tick'] == 4 * k + r['substep'], 'Solver pre/post phase differs')
            require(r['direct_torque_included_once'] is True and r['external_force_max'] == 0, 'Force/exogenous capture scope differs')
    return groups


def metadata_preflight(config_path):
    """Read saved data and independently qualify prerequisites, never import WM."""
    require('torch' not in sys.modules and 'warp' not in sys.modules, 'Default metadata preflight must be a fresh noTorch/noWarp process')
    c = read(config_path)
    require(c.get('schema') == SCHEMA and c.get('method') == 'A2' and type(c.get('iteration')) is int and c['iteration'] == 2000, 'Fixed A2final2000 analysis config required')
    require(type(c.get('CPU_run_authorized')) is bool and c.get('GPU_authorized') is False and c.get('training_authorized') is False and c.get('Env_or_physics_authorized') is False, 'CPU-only explicit authorizer flags required')
    require(c.get('cpu_threads') == 4 and type(c['cpu_threads']) is int, 'Fixed CPU thread budget4 required')
    for key, sha in FIXED.items(): require(c[key]['sha256'] == sha and checked(c[key]), 'Frozen input source differs: ' + key)
    require(set(c['artifact_pins']) == set(ARTIFACTS), 'Exact artifact pin keyset required')
    require(type(c.get('source_pins')) is dict and c['source_pins'], 'CPU source closure missing')
    for value in c['source_pins'].values(): checked(value)
    require(any(v['sha256'] == digest(__file__) and checked(v) == Path(__file__).resolve() for v in c['source_pins'].values()), 'Actual analysis source not in closure')
    score_source = Path(__file__).with_name('serial_probe105_v2.py')
    require(digest(score_source) == 'a24c89da47f4aad7acfc9d9dfd1f053518cdc878230716bb2c857127ec13e592' and pin(score_source) in c['source_pins'].values(), 'Frozen matched post20 score implementation missing/drifted')
    require(all(c[k] in c['source_pins'].values() for k in ('loader', 'health_auditor')), 'Strict loader/auditor dependency not in CPU source closure')
    files = {k: checked(v) for k, v in c['artifact_pins'].items()}
    run = Path(c['physical_run']).resolve(strict=True)
    for key in ('producer_report', 'physical_arrays', 'branch_index', 'snapshot_records', 'donor_records', 'baseline_gates', 'capacity_ledgers', 'physical_samples', 'reward_snapshots', 'runtime'):
        require(files[key].parent == run, 'Artifact outside declared actual physical run: ' + key)
    docs = {k: read(v) for k, v in files.items() if v.suffix == '.json'}
    proposal = read(checked(c['proposal'])); producer = docs['producer_report']; pc = docs['producer_config']
    health = read(checked(c['health'])); _all_checks(health, 'FinalA2 health not qualified')
    require(health.get('schema') == 'full151105_checkpoint_CPU_health:1' and health.get('status') == 'PASS_SAVED_ENGINEERING_HEALTH' and health.get('method') == 'A2' and health.get('iteration') == 2000 and health.get('final_iteration_scope') is True and same(health.get('checkpoint'), c['checkpoint']) and same(health.get('auditor'), c['health_auditor']), 'Fixed actual final A2health/checkpoint mismatch')
    require(Path(health['run_dir']).resolve() == Path(c['training_run']).resolve() and checked(c['checkpoint']) == Path(c['training_run']).resolve() / 'train/model_final.pt', 'Final training/checkpoint route differs')
    checked(health['config'])
    require(producer.get('schema') == 'full151105_serial_physical_probe_worker_report:3' and producer.get('status') == 'COMPLETED_PHYSICAL_PROBE' and producer.get('probe_mode') == 'serial_full', 'Full serial physical completion required, not inventory/partial')
    for key in ('engineering_qualified', 'baseline_all_passed', 'capacity_finalized', 'capacity_qualified', 'foreign_inventory_qualified', 'restore_inventory_qualified', 'full_serial_qualified', 'longclock_qualified'):
        require(producer.get(key) is True, 'Physical prerequisite false: ' + key)
    for key in ('gradient_qualified', 'whole_graph_hidden_state_qualified'): require(producer.get(key) is False, 'Do not upgrade physical derivative/hidden state qualification')
    for key, n in {'episode_step_attempted': 1808, 'episode_step_returned': 1808, 'control_interval_attempted': 452, 'control_interval_returned': 452,
                   'transaction_begin_attempted': 152, 'transaction_begin_returned': 152, 'transaction_finish_attempted': 152, 'transaction_finish_returned': 152,
                   'native_snapshot_restore_attempted': 152, 'native_snapshot_restore_returned': 152, 'integration_set_attempted': 8, 'integration_set_returned': 8,
                   'integration_get_attempted': 168, 'integration_get_returned': 168, 'baseline_gate_count': 24}.items():
        require(type(producer.get(key)) is int and producer[key] == n, 'Physical actual count differs: ' + key)
    require(pc['probe_mode'] == 'serial_full' and same(pc['input_bundle'], c['input_bundle']) and same(pc['proposal'], c['proposal']) and same(producer['scalar_contract'], proposal['scalar_contract']), 'Producer scalar/bundle/proposal differs')
    require(digest(run / 'config.json') == c['artifact_pins']['producer_config']['sha256'], 'Actual command plan vs saved copy bytes differ')
    validate_saved_review(docs['saved_review'], c)
    _validate_lifecycle(c, docs); _validate_capacity(docs['capacity_ledgers'])
    inventory = docs['source_inventory']; require(same(inventory, pc['source_pins']), 'Execution source inventory differs from actual config')
    for value in inventory.values(): checked(value)
    with zipfile.ZipFile(files['source_zip']) as archive:
        names = []
        for i, value in enumerate(inventory.values()):
            name = '%04d_%s' % (i, Path(value['path']).name); names.append(name)
            require(hashlib.sha256(archive.read(name)).hexdigest() == value['sha256'], 'Archived executed source bytes differ: ' + name)
        require(archive.namelist() == names, 'Execution ZIP contents/order differs')
    with np.load(files['physical_arrays'], allow_pickle=False) as z: a = {k: z[k].copy() for k in z.files}
    bases = _validate_data(a, docs['branch_index'], docs['snapshot_records'], docs['baseline_gates'], proposal)
    rewards, samples = _jsonl(files['reward_snapshots']), _jsonl(files['physical_samples'])
    groups = _validate_captures(a, rewards, samples)
    require('torch' not in sys.modules and 'warp' not in sys.modules, 'Metadata path imported a model/backend')
    return dict(config=c, config_pin=pin(config_path), arrays=a, baseline=bases, docs=docs,
                solver_groups=groups, rewards=rewards, input_array_sha256=tree_array_hash(a),
                status='PASS_CPU_METADATA_FULL_SERIAL_PREREQUISITES', Torch_imported=False, Warp_imported=False)


def _metric_groups(pred, truth):
    d = pred.astype(np.float64) - truth.astype(np.float64)
    R, T = pred[:, 43:52].astype(np.float64).reshape(-1, 3, 3), truth[:, 43:52].astype(np.float64).reshape(-1, 3, 3)
    # Native WM outputs rotation matrices; use principal relative rotation angle.
    angle = np.arccos(np.clip((np.einsum('bij,bij->b', R, T) - 1) * .5, -1, 1))
    rmsnorm = lambda v: float(np.sqrt(np.mean(np.sum(v * v, axis=-1))))
    return dict(q_component_RMSE_mrad=float(np.sqrt(np.mean(d[:, :20] ** 2)) * 1000),
                qd_component_RMSE_rad_s=float(np.sqrt(np.mean(d[:, 20:40] ** 2))),
                p_vector_RMS_mm=rmsnorm(d[:, 40:43]) * 1000, R_geodesic_RMS_degrees=float(np.rad2deg(np.sqrt(np.mean(angle ** 2)))),
                v_vector_RMS_m_s=rmsnorm(d[:, 52:55]), omega_vector_RMS_rad_s=rmsnorm(d[:, 55:58]),
                scope='52delta groups have separate units; no single mixed-unit aggregate')


def _self_metrics(logits, truth):
    z, y = logits.astype(np.float64), truth.astype(np.float64)
    p = np.exp(-np.logaddexp(0., -z)); pred = p >= .5
    tp, fp, fn, tn = (int(v.sum()) for v in (pred & (y == 1), pred & (y == 0), (~pred) & (y == 1), (~pred) & (y == 0)))
    return dict(labels=int(y.size), positive_truth=int(y.sum()), positive_prediction=int(pred.sum()),
                BCE=float((np.logaddexp(0., z) - y * z).mean()), Brier=float(((p - y) ** 2).mean()),
                TP=tp, FP=fp, FN=fn, TN=tn, precision=None if tp + fp == 0 else tp / (tp + fp),
                recall=None if tp + fn == 0 else tp / (tp + fn), F1=None if tp + fn == 0 or 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn),
                label_phase='last solver preintegration15ms; state prediction ispost20ms',
                scope='Native self-collision15 only, not hand-object force/contact or human labels')


def _contact_diagnostics(groups):
    out = []
    for (path, k, bid), values in groups.items():
        signatures = []
        for row in values:
            signatures.append(_contact_signature(row))
        out.append(dict(path=path, k=k, branch_id=bid, solver_preintegration_offset_ms=[0, 5, 10, 15],
            postintegration_state_offset_ms=[5, 10, 15, 20],
            contact_signature=signatures, contact_signature_changed_within_interval=any(v != signatures[0] for v in signatures[1:]),
            force_on_object_w_N=[r['hand_object_force_on_object_w_N'] for r in values],
            moment_about_COM_w_Nm=[r['hand_object_moment_about_COM_w_Nm'] for r in values],
            direct_torque_included_once=True, scope='Measured preintegration solver diagnostics; no WM HO force head'))
    return out


def _contact_signature(row):
    """Discrete solver diagnostic; address/order changes need not be kinks."""
    raw = row['raw_wp_contact']
    return dict(raw_nacon=row['nacon'], raw_nefc=row['nefc'],
                CPU_contact_geom_active=sorted((tuple(c['geom']), c['efc_address'] >= 0) for c in row['contacts']),
                WP_contact_geom=raw['geom'], WP_contact_type=raw['type'], WP_contact_efc=raw['efc_address'],
                efc_type=row['raw_efc_type'])


def _write(out, name, value):
    with (out / name).open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)


def run_analysis(config_path):
    """Explicit authorized fresh CPU process. One strict loader, one batched WM AD."""
    start = time.perf_counter(); p = metadata_preflight(config_path); c = p['config']
    require(c['CPU_run_authorized'] is True, 'Root CPU authorization required for actual model load/inference')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'Set CUDA_VISIBLE_DEVICES=-1 before starting the CPU analysis process')
    out = Path(c['output']).resolve(); require(not out.exists(), 'Fresh output required; no retry/overwrite')
    out.mkdir(parents=True)
    from full151_native.bundle import load_bundle
    bundle = load_bundle(checked(c['input_bundle']), expected_sha256=c['input_bundle']['sha256'])
    import torch
    require(not torch.cuda.is_initialized() and 'warp' not in sys.modules, 'Fresh CPU-only noWarp consumer required')
    torch.set_num_threads(c['cpu_threads'])
    def rng_digest():
        return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()), protocol=5) + torch.get_rng_state().numpy().tobytes()).hexdigest()
    rng_before = rng_digest(); input_before = tree_array_hash(p['arrays']); checkpoint_before = digest(checked(c['checkpoint']))
    from full151_native.world_model_checkpoint import load_world_model_checkpoint
    from full151_native.checkpoint import _tensor_digest
    from full151_gradient.serial_probe105_v2 import controlled_object_score_FP32
    original_load = torch.load; load_calls = []
    def counted_load(*args, **kwargs):
        load_calls.append(str(args[0])); return original_load(*args, **kwargs)
    torch.load = counted_load
    try:
        wm, receipt = load_world_model_checkpoint(checked(c['checkpoint']), bundle, 'A2', c['training_run'], checked(c['health']),
            expected_health_auditor=c['health_auditor'], expected_iteration=2000)
    finally:
        torch.load = original_load
    require(len(load_calls) == 2 and all(Path(v).resolve() == checked(c['checkpoint']) for v in load_calls), 'One strict WM loader must perform its two original checkpoint reads')
    require(not wm.training and all(not v.requires_grad for v in wm.parameters()), 'Parameters must be frozen; input autograd remains allowed')
    weight_before = _tensor_digest(wm.state_dict()); norm_before = _tensor_digest({k: v for k, v in wm.state_dict().items() if 'norm.' in k})
    a = p['arrays']; x = torch.from_numpy(a['snapshot_x'].copy()); u = torch.from_numpy(a['snapshot_u'].copy()).requires_grad_(True)
    gp = torch.from_numpy(a['goal_p'].copy()); gR = torch.from_numpy(a['goal_R'].copy())
    input_tensors_before = _tensor_digest({'x': x, 'u': u, 'goal_p': gp, 'goal_R': gR})
    predicted, logits = wm(x, u)
    terms, score = controlled_object_score_FP32(predicted, gp, gR)
    gradient, = torch.autograd.grad(score.sum(), u)
    require(bool(torch.isfinite(gradient).all()) and all(v.grad is None for v in wm.parameters()), 'Finite input derivative only; no parameter gradients')
    pred, log, grad, scores = (v.detach().numpy().copy() for v in (predicted, logits, gradient, score))
    rows, classifications = [], []
    for s in range(8):
        for j, joint in enumerate(JOINTS):
            ids = [(s * 19 + 3 + ei * 8 + j * 2, s * 19 + 4 + ei * 8 + j * 2) for ei in range(2)]
            minus = np.array([a['branch_object_score_k'][i] for i, _ in ids], np.float32)
            plus = np.array([a['branch_object_score_k'][i] for _, i in ids], np.float32)
            base = p['baseline'][s]; cl = classify_pair(minus, plus, base['baseline_range'], base['baseline_donor_error'])
            classifications.append(dict(snapshot_id=s, path=('A0', 'A2')[s // 4], k=PHASES[s % 4], joint_index=joint, **cl))
            for ei, eps in enumerate(EPS):
                d, off = a['direction__d_eff'][s, j, ei], a['direction__center_offset'][s, j, ei]
                key = (('A0', 'A2')[s // 4], PHASES[s % 4])
                minus_samples = p['solver_groups'][(*key, ids[ei][0])]
                plus_samples = p['solver_groups'][(*key, ids[ei][1])]
                donor_samples = p['solver_groups'][(*key, None)]
                changed_pm = any(not same(_contact_signature(m), _contact_signature(v)) for m, v in zip(minus_samples, plus_samples))
                changed_center = any(not same(_contact_signature(v), _contact_signature(t)) for values in (minus_samples, plus_samples) for v, t in zip(values, donor_samples))
                rows.append(dict(snapshot_id=s, path=('A0', 'A2')[s // 4], k=PHASES[s % 4], joint_index=joint,
                    epsilon_index=ei, epsilon_rad=eps, minus_row=ids[ei][0], plus_row=ids[ei][1], physical_slope=cl['slopes'][ei],
                    WM_AD_effective=float(grad[s].astype(np.float64) @ d), WM_AD_nominal=float(grad[s, joint]),
                    d_eff=d.tolist(), center_offset=off.tolist(), center_offset_maxabs=float(abs(off).max()),
                    clipped=bool(a['direction__clipped'][s, j, ei]), stable_nonflat=cl['stable_nonflat'], physical_label=cl['label'],
                    contact_signature_change_plus_minus=changed_pm, contact_signature_change_vs_fresh_donor=changed_center,
                    contact_signature_scope='Orthogonal diagnostic only; ordering/address changes are not proof of a nonsmooth derivative',
                    center_offset_nonzero=bool(off.any()), central_unique_derivative_qualified=False,
                    signal_floor=cl['signal_floor'], relative_epsilon_disagreement=cl['relative_disagreement'], units='controlled score/rad'))
    invariants = dict(checkpoint_unchanged=digest(checked(c['checkpoint'])) == checkpoint_before,
        WM_weights_unchanged=_tensor_digest(wm.state_dict()) == weight_before,
        normalizers_unchanged=_tensor_digest({k: v for k, v in wm.state_dict().items() if 'norm.' in k}) == norm_before,
        input_arrays_unchanged=tree_array_hash(a) == input_before,
        input_tensors_unchanged=_tensor_digest({'x': x, 'u': u, 'goal_p': gp, 'goal_R': gR}) == input_tensors_before,
        RNG_unchanged=rng_digest() == rng_before, CUDA_uninitialized=not torch.cuda.is_initialized(), Warp_not_imported='warp' not in sys.modules)
    require(all(invariants.values()), 'CPU diagnostic invariance failure')
    for value in c['source_pins'].values(): checked(value)
    for value in c['artifact_pins'].values(): checked(value)
    metrics = _metric_groups(pred, a['donor_next_x']); metrics['score_RMS'] = float(np.sqrt(np.mean((scores.astype(float) - a['donor_object_score_k'].astype(float)) ** 2)))
    counts = {label: sum(v['label'] == label for v in classifications) for label in ('STABLE_NONFLAT', 'NONFLAT_UNSTABLE', 'FLAT_OR_BELOW_NOISE')}
    report = dict(schema='full151105_physical_gradient_CPU_analysis:1', status='COMPLETED_CPU_DIAGNOSTIC',
        config=p['config_pin'], source_pins=c['source_pins'], input_pins=c['artifact_pins'], checkpoint=c['checkpoint'], health=c['health'],
        method='A2', iteration=2000, strict_WM_loader_calls=1, actual_torch_load_calls=len(load_calls), WM_forward_calls=1,
        predicted_rows=8, u_autograd_calls=1, comparison_rows=64, dependent_direction_groups=32, physical_direction_branches=128,
        baseline_branches=24, classification_counts=counts, invariants=invariants, physical_gradient_qualified=False,
        whole_graph_hidden_state_qualified=False, Env_created=False, physics_steps=0, GPU_used=False, elapsed_seconds=time.perf_counter() - start,
        scope='One source seed0; eight dependent states, four directions, two eps per group. Fresh donor action-source replay, not archived closed-loop reproduction. Finite secants are not unique physical Jacobians; graph hidden scratch UNKNOWN. FP32CPU WM derivative is not GPUAMP or physical-gradient qualification. Self15 is not HO force/contact.',
        classification_quality_gate=None, agreement=agreement(rows), prediction_metrics=metrics)
    np.savez_compressed(out / 'WM_predictions_gradients.npz', predicted_x=pred, self15_logits=log, gradient_u=grad,
                        predicted_score=scores, truth_x=a['donor_next_x'], truth_score=a['donor_object_score_k'], snapshot_x=a['snapshot_x'], actualu=a['snapshot_u'])
    for name, value in [('loader_receipt.json', receipt), ('baselines_independent.json', p['baseline']), ('classifications.json', classifications),
                        ('comparisons.json', rows), ('prediction_metrics.json', metrics), ('self15_metrics.json', _self_metrics(log, a['donor_self15'])),
                        ('contact_force_diagnostics.json', _contact_diagnostics(p['solver_groups'])), ('report.json', report)]: _write(out, name, value)
    fields = ['snapshot_id', 'path', 'k', 'joint_index', 'epsilon_rad', 'physical_slope', 'WM_AD_effective', 'WM_AD_nominal', 'clipped', 'center_offset_maxabs', 'physical_label', 'signal_floor']
    with (out / 'comparisons.csv').open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    _write(out, 'output_pins.json', {f.name: pin(f) for f in out.iterdir() if f.is_file()})
    return report
