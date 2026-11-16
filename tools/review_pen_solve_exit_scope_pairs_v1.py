"""Quantify all saved observer/baseline pairs without imposing neutrality."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.strict_json import write_json


def read(pin):
    return json.loads(verify_file_pin(pin).read_text())


def difference(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Paired trace layout changed')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite paired trace')
    return dict(bytes_equal=a.tobytes() == b.tobytes(),
                max_abs=float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.)


def main():
    out = ROOT / 'runs/pen_solve_exit_scope_pairs_v1'
    if out.exists():
        raise ValueError('Fresh output required')
    inputs = []
    rows = []
    fields = {}
    for replica in ['A', 'B']:
        pp = file_pin(ROOT / f'runs/pen_solve_exit_replay_v2/{replica}/report.json')
        vp = file_pin(ROOT / f'runs/pen_solve_exit_saved_review_v2/{replica}/report.json')
        p, v = read(pp), read(vp)
        inputs += [pp, vp]
        if not all(p['checks'].values()) or not all(v['checks'].values()) or v['producer'] != pp:
            raise ValueError('Complete independently reviewed producer required')
        pairs = {}
        for slot in p['rows']:
            q = read(slot['report']); inputs.append(slot['report'])
            d = q['declaration']; c = read(q['case']); inputs.append(q['case'])
            key = (d['repeat'], d['row_index'])
            pairs.setdefault(key, {})[d['scope']] = (q, c)
        if len(pairs) != 30:
            raise ValueError('All 30 pairs per process required')
        for (repeat, ri), pair in sorted(pairs.items()):
            if set(pair) != {'PACKET_BASELINE', 'PACKET_PLUS_SOLVE_EXIT'}:
                raise ValueError('Both registered scopes required')
            (b, bc), (o, oc) = pair['PACKET_BASELINE'], pair['PACKET_PLUS_SOLVE_EXIT']
            inputs += [bc['trace'], oc['trace']]
            with np.load(verify_file_pin(bc['trace']), allow_pickle=False) as z:
                bz = {k: z[k] for k in z.files}
            with np.load(verify_file_pin(oc['trace']), allow_pickle=False) as z:
                oz = {k: z[k] for k in z.files}
            if set(bz) != set(oz):
                raise ValueError('Trace field inventory changed')
            diffs = {k: difference(oz[k], bz[k]) for k in bz}
            if diffs != o['differences_between_scopes']:
                raise ValueError('Stored field differences disagree with arrays')
            row = dict(replica=replica, repeat=repeat, row_index=ri,
                       name=o['declaration']['original_name'],
                       all_trace_bytes_equal=all(x['bytes_equal'] for x in diffs.values()),
                       baseline_impulse_valid=b['new_impulse_valid'],
                       observed_impulse_valid=o['new_impulse_valid'],
                       physical_masks_equal=b['new_physical_valid'] == o['new_physical_valid'],
                       impulse_masks_equal=b['new_impulse_valid'] == o['new_impulse_valid'],
                       task_flags_equal=(bc['terminated'], bc['truncated'], bc['termination_reasons']) ==
                                        (oc['terminated'], oc['truncated'], oc['termination_reasons']),
                       reward_abs_difference=abs(bc['source_reward'] - oc['source_reward']),
                       peak_impulse_abs_difference_Ns=abs(
                           bc['physical_audit']['decoder_metrics']['max_linear_impulse_residual_N_s'] -
                           oc['physical_audit']['decoder_metrics']['max_linear_impulse_residual_N_s']))
            for k, d in diffs.items():
                fields.setdefault(k, []).append(d)
                row[k + '_max_abs'] = d['max_abs']
            rows.append(row)
    summary_pin = file_pin(ROOT / 'runs/pen_solve_exit_summary_v2/report.json')
    summary = read(summary_pin); inputs.append(summary_pin)
    checks = dict(all60=len(rows) == 60,
                  all_fields60=all(len(v) == 60 for v in fields.values()),
                  complete_reviewed_inputs=True,
                  byte_count_matches=sum(r['all_trace_bytes_equal'] for r in rows) == summary['scope_pairs_byte_equal'],
                  summary_all120=summary['queries'] == 120 and summary['passed'] == summary['total'])
    for pin in inputs:
        verify_file_pin(pin)
    out.mkdir()
    receipt = freeze_sources([Path(__file__)] + sorted((ROOT / 'src').rglob('*.py')),
                             out / 'source_snapshot', ROOT,
                             subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write_json(out / 'source_receipt.json', receipt)
    with (out / 'all60_pairs.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    result = dict(schema='pen_solve_exit_scope_pairs:1', inputs=inputs,
                  source_receipt=file_pin(out / 'source_receipt.json'), checks=checks,
                  passed=sum(checks.values()), total=len(checks), pairs=len(rows),
                  all_trace_bytes_equal=sum(r['all_trace_bytes_equal'] for r in rows),
                  physical_masks_equal=sum(r['physical_masks_equal'] for r in rows),
                  impulse_masks_equal=sum(r['impulse_masks_equal'] for r in rows),
                  task_flags_equal=sum(r['task_flags_equal'] for r in rows),
                  max_reward_abs_difference=max(r['reward_abs_difference'] for r in rows),
                  max_peak_impulse_abs_difference_Ns=max(r['peak_impulse_abs_difference_Ns'] for r in rows),
                  fields={k: dict(equal_pairs=sum(x['bytes_equal'] for x in ds),
                                  max_abs=max(x['max_abs'] for x in ds)) for k, ds in fields.items()},
                  csv=file_pin(out / 'all60_pairs.csv'), actual_native_integrations=0,
                  optimizer_steps=0, normalizer_fits=0,
                  scope='All registered saved scope pairs, directly recomputed trace differences. '
                        'No neutrality threshold or physical gate changed. Equality counts are observations, '
                        'not success requirements. Unmatched contact/EFC row differences cannot be read as '
                        'aggregate force errors. No attribution of nondeterminism to observation alone.')
    verify_source_snapshot(receipt, live=True)
    write_json(out / 'report.json', result)
    print(json.dumps({k: result[k] for k in ['passed', 'total', 'pairs', 'all_trace_bytes_equal',
                                          'physical_masks_equal', 'impulse_masks_equal', 'task_flags_equal',
                                          'max_reward_abs_difference', 'max_peak_impulse_abs_difference_Ns']}))
    print(json.dumps(result['fields']))
    if not all(checks.values()):
        raise ValueError('Pair review coverage discrepancy')


if __name__ == '__main__':
    main()
