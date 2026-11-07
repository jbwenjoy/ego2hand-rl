"""Independent stdlib-only full3600 prior-log audit. No learned module imports.

Frozen equations: A-weighted mean=fsum(A_i*m_i)/sum(A_i); exact P_mean>0
counts updates, not individual rows. Derived penalty=-.1P; support=S for A2
and zero for P_ONLY after the all151 unit-phase-weight health check. Apply
dt=.02 once. CSV/report arithmetic comparisons use abs=rel=1e-10; counters,
coverage, byte hashes and metadata joins remain exact. No tuned thresholds.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

METHODS = ('P_ONLY', 'A2')
COUNT = ('policy_update_calls', 'objective_forward_calls', 'active_rows',
         'inactive_rows', 'dt_applications')
PREFIX = 'prior/geometry_intent_'
COMMON = {'prior/bonus_unscaled'} | {PREFIX+s for s in (
    'support_match', 'penetration_prior', 'phase_weight', 'bonus_unscaled',
    'min_sdf_m', 'reference_frame', 'phase')}
WEIGHTED = ('P_raw_active_mean', 'S_raw_active_mean',
            'weighted_penalty_active_mean', 'weighted_support_active_mean',
            'bonus_logged_active_mean', 'dt_scaled_bonus_active_mean',
            'dt_scaled_penalty_active_mean', 'dt_scaled_support_active_mean')
TOL = 1e-10


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(p):
    return {'path': str(Path(p).resolve(strict=True)), 'sha256': sha(p)}


def checked(p):
    path = Path(p['path']).resolve(strict=True)
    if sha(path) != p['sha256']:
        raise ValueError('Pin mismatch: '+str(path))
    return path


def same(a, b):
    return a['sha256'] == b['sha256'] and Path(a['path']).samefile(b['path'])


def dump(p, v):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(v, f, indent=2, allow_nan=False)


def num(v):
    return type(v) in (int, float) and math.isfinite(v)


def equivalent(a, b):
    if a is None or b is None:
        return a is b
    if type(a) is int:
        return type(b) is int and a == b
    return num(a) and num(b) and math.isclose(a, b, abs_tol=TOL, rel_tol=TOL)


def audit(summary_path, out):
    if out.exists():
        raise ValueError('Fresh output required')
    out.mkdir(parents=True)
    source = pin(__file__)
    (out/'executed_checker_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'source_freeze_before_numeric_read.json', {
        'schema': 'full151_training_prior_log_independent_freeze:1',
        'checker': source, 'UTC': datetime.now(timezone.utc).isoformat(),
        'equations': __doc__, 'summary': pin(summary_path),
        'numeric_logs_or_summary_statistics_read': False})
    checks, results, issues = {}, {}, []

    def gate(name, ok):
        checks[name] = bool(ok)
        if not ok:
            issues.append(name)

    report = read(summary_path)
    contract = read(checked(report['contract']))
    frozen = read(checked(report['output_pins']['source_freeze_before_data.json']))
    source_pins = contract['source_pins']
    gate('reader_contract_and_predata_source_binding',
         report['schema'] == 'full151_trained_prior_saved_log_summary:1'
         and report['status'] == 'COMPLETED_SAVED_ONLY_FULL_LOG_ANALYSIS'
         and contract['status'] == 'FROZEN_BEFORE_NUMERIC_LOG_READ'
         and contract['methods'] == list(METHODS)
         and same(report['reader'], frozen['reader'])
         and same(report['contract'], frozen['contract'])
         and frozen['numeric_log_read_started'] is False
         and frozen['equations_and_checklist_frozen'] is True)
    all_pins = [report['reader'], report['contract'], *source_pins.values(),
                *report['output_pins'].values()]
    all_pins.extend(p for m in METHODS for p in report['input_pins'][m].values())
    for p in all_pins:
        checked(p)
    gate('all_declared_source_input_output_pins_exact', True)
    csv_path = checked(report['output_pins']['all3600_prior_updates.csv'])
    csv_errors = 0
    csv_max_abs = 0.
    total = 0
    with csv_path.open(newline='', encoding='utf-8-sig') as cf:
        cr = csv.DictReader(cf)
        for method in METHODS:
            pins = report['input_pins'][method]
            h = read(checked(pins['health']))
            final = read(checked(pins['result.json']))
            initial = read(checked(pins['prior_metadata.json']))
            marker = read(checked(pins['training_completion.json']))
            cfg = read(checked(pins['config.json']))
            eff = read(checked(pins['effective_training_config.json']))
            evidence = final['full151_prior_evidence']
            static = {k: v for k, v in evidence.items() if k not in COUNT}
            ok_health = (h['schema'] == 'full151105_checkpoint_CPU_health:1'
                         and h['status'] == 'PASS_SAVED_ENGINEERING_HEALTH'
                         and h['passed'] == h['total'] == len(h['checks'])
                         and all(v is True for v in h['checks'].values())
                         and h['method'] == method and h['role'] == 'primary_matched'
                         and h['iteration'] == 2000 and h['final_iteration_scope'] is True
                         and same(h['auditor'], source_pins['health_auditor'])
                         and same(h['config'], pins['actual_root_training_config'])
                         and same(h['checkpoint'], final['checkpoint'])
                         and h['checks']['prior_target_actual151_unknownhuman_declared_only'] is True)
            gate(method+'_final_health_config_checkpoint_binding', ok_health)
            gate(method+'_method_budget_and_initial_final_metadata',
                 cfg['method'] == method and cfg['schema'] == 'full151_knownrough_native_training_run:2'
                 and eff['max_iterations'] == 2000 and eff['wm_only_iters'] == 200
                 and eff['horizon'] == 8 and eff['pi_batch'] == eff['num_envs'] == 128
                 and marker['result'] == final
                 and marker['config_sha256'] == h['config']['sha256']
                 and same(evidence['spec_pin'], pins['objective_spec'])
                 and same(evidence['source_pins']['prior_hook'], source_pins['prior_hook'])
                 and evidence['dt_s'] == .02 and evidence['logging_errors'] == []
                 and evidence['imagined_soft_required'] is True
                 and evidence['real_collect_shaped'] is evidence['WMfit_shaped'] is False
                 and all(type(initial[k]) is int and initial[k] == 0 for k in COUNT)
                 and {k: v for k, v in initial.items() if k not in COUNT} == static)
            metric_hash = hashlib.sha256()
            metrics = []
            with checked(pins['metrics.jsonl']).open('rb') as f:
                for line in f:
                    metric_hash.update(line)
                    metrics.append(json.loads(line))
            metric_ok = (len(metrics) == 2000
                         and all(type(v['iteration']) is int and v['iteration'] == i for i, v in enumerate(metrics))
                         and not any(k.startswith('prior/') for r in metrics[:200] for k in r)
                         and metric_hash.hexdigest() == h['details']['metrics_prefix_sha256'])
            sums = {k: 0 for k in COUNT}
            values = []
            raw_fields = None
            row_ok = mean_ok = cumulative_ok = True
            log_hash = hashlib.sha256()
            nbytes = 0
            with checked(pins['full151_prior_evidence.jsonl']).open('rb') as stream:
                for i, line in enumerate(stream):
                    log_hash.update(line)
                    nbytes += len(line)
                    r = json.loads(line)
                    active, inactive = r['active_rows'], r['inactive_rows']
                    means = r['active_means']
                    expected_keys = COMMON | ({PREFIX+'support_coefficient', PREFIX+'penetration_coefficient'} if method == 'P_ONLY' else set())
                    row_ok &= (r['schema'] == 'full151_imagination_prior_policy_log:1'
                               and r['method'] == method and r['complete'] is True
                               and all(type(r[k]) is int for k in ('policy_update_call', 'iteration', *COUNT[1:]))
                               and r['policy_update_call'] == i+1 and r['iteration'] == i+200
                               and active >= 0 and inactive >= 0 and active+inactive == 1024
                               and r['objective_forward_calls'] == r['dt_applications'] == 8
                               and r['dt_s'] == .02)
                    for k in COUNT:
                        sums[k] += 1 if k == 'policy_update_calls' else r[k]
                    cumulative_ok &= all(type(r['cumulative'][k]) is int and r['cumulative'][k] == sums[k] for k in COUNT)
                    cumulative_ok &= {k: v for k, v in r['cumulative'].items() if k not in COUNT} == static
                    mean_ok &= (set(means) == expected_keys and all(num(v) for v in means.values())
                                and (active > 0 or all(v == 0 for v in means.values()))
                                and means[PREFIX+'phase_weight'] == (1 if active else 0))
                    metric_ok &= all(metrics[i+200].get(k) == v for k, v in means.items())
                    P, S = means[PREFIX+'penetration_prior'], means[PREFIX+'support_match']
                    bonus = means['prior/bonus_unscaled']
                    mean_ok &= P >= 0 and means[PREFIX+'bonus_unscaled'] == bonus
                    penalty, support = -.1*P, S if method == 'A2' else 0.
                    expected = dict(method=method, policy_update_call=i+1, raw_iteration=i+200,
                        completed_training_iteration=i+201, objective_forward_calls=8, dt_applications=8,
                        active_rows=active, inactive_rows=inactive, P_raw_active_mean=P, S_raw_active_mean=S,
                        phase_weight_active_mean=means[PREFIX+'phase_weight'], weighted_penalty_active_mean=penalty,
                        weighted_support_active_mean=support, bonus_logged_active_mean=bonus,
                        dt_scaled_bonus_active_mean=.02*bonus, dt_scaled_penalty_active_mean=.02*penalty,
                        dt_scaled_support_active_mean=.02*support, bonus_identity_arithmetic_residual=bonus-penalty-support, **means)
                    c = next(cr)
                    for k in cr.fieldnames:
                        if k not in expected:
                            csv_errors += c[k] != ''
                        elif isinstance(expected[k], str):
                            csv_errors += c[k] != expected[k]
                        else:
                            actual = float(c[k])
                            if type(expected[k]) is int:
                                csv_errors += actual != expected[k]
                            else:
                                csv_max_abs = max(csv_max_abs, abs(actual-expected[k]))
                                csv_errors += not math.isclose(actual, expected[k], abs_tol=TOL, rel_tol=TOL)
                    if not set(expected) <= set(cr.fieldnames):
                        csv_errors += 1
                    raw_fields = expected_keys
                    values.append(expected)
            prefix = h['details']['prior_checkpoint_prefix']
            gate(method+'_all1800_complete_H8_iteration_dt_rows', row_ok and len(values) == 1800)
            gate(method+'_cumulative_static_and_final_counters', cumulative_ok and sums == {k: evidence[k] for k in COUNT}
                 and sums['policy_update_calls'] == 1800 and sums['objective_forward_calls'] == sums['dt_applications'] == 14400)
            gate(method+'_allmean_fields_finite_phase_and_metric_joins', mean_ok and metric_ok)
            gate(method+'_health_exact_log_byteprefix', prefix['prefix_SHA256'] == log_hash.hexdigest()
                 and prefix['prefix_byte_count'] == nbytes and prefix['prefix_policy_calls'] == 1800
                 and prefix['later_complete_rows_ignored'] == 0 and prefix['checkpoint_counters'] == sums)
            denom = sums['active_rows']
            weighted = {k: math.fsum(r['active_rows']*r[k] for r in values)/denom if denom else None
                        for k in sorted(set(WEIGHTED) | raw_fields)}
            derived = dict(policy_updates=len(values), raw_iteration_range=[200, 1999],
                completed_training_iteration_range=[201, 2000], cumulative=sums,
                candidate_imagined_rows=1800*1024, active_weighted_denominator=denom,
                positive_P_mean_updates=sum(r['P_raw_active_mean'] > 0 for r in values),
                nonzero_S_mean_updates=sum(r['S_raw_active_mean'] != 0 for r in values),
                nonzero_bonus_mean_updates=sum(r['bonus_logged_active_mean'] != 0 for r in values),
                negative_penalty_mean_updates=sum(r['weighted_penalty_active_mean'] < 0 for r in values),
                zeroactive_updates=sum(r['active_rows'] == 0 for r in values),
                P_mean_maximum=max(r['P_raw_active_mean'] for r in values),
                P_mean_first_positive_completed_iteration=next((r['completed_training_iteration'] for r in values if r['P_raw_active_mean'] > 0), None),
                maximum_abs_bonus_identity_arithmetic_residual=max(abs(r['bonus_identity_arithmetic_residual']) for r in values))
            declared = report['methods'][method]
            summary_ok = all((equivalent(v, declared[k]) if v is None or num(v) else v == declared[k]) for k, v in derived.items())
            summary_ok &= all(equivalent(weighted[k], declared['active_row_weighted_logged_means'][k]) for k in WEIGHTED)
            summary_ok &= declared['health']['sha256'] == pins['health']['sha256'] and declared['health']['passed'] == declared['health']['total'] == h['total']
            summary_ok &= declared['health_prefix_log_sha256'] == log_hash.hexdigest() and declared['metrics_sha256'] == metric_hash.hexdigest()
            gate(method+'_report_summary_and_all_eight_weighted_means', summary_ok)
            results[method] = {**derived, 'all_active_row_weighted_means': weighted,
                'original_log': pins['full151_prior_evidence.jsonl'], 'health': pins['health'],
                'log_bytes': nbytes, 'log_SHA256': log_hash.hexdigest()}
            total += len(values)
        gate('CSV_exact3600_rows_allfields_with_frozen_roundoff', next(cr, None) is None and total == 3600 and csv_errors == 0)
    gate('summary_whole_cohort_and_saved_only_scope', report['total_policy_rows'] == total == 3600
         and report['model_loads'] == report['objective_builds'] == report['WM_forwards'] == report['gradients'] == report['physics_steps'] == 0
         and report['Env_created'] is report['GPU_used'] is report['causal_method_comparison'] is report['physical_contact_or_force_qualification'] is False)
    for p in all_pins:
        checked(p)
    gate('all_pins_and_checker_unchanged_after_stream', pin(__file__) == source)
    gate('no_model_native_or_GPU_module_imported', not {'torch', 'warp', 'mujoco', 'scipy'} & set(sys.modules))
    result = dict(schema='full151_training_prior_logs_saved_independent:1',
        status='PASS_SAVED_ONLY_LOG_REVIEW' if all(checks.values()) else 'FAILED_SAVED_ONLY_LOG_REVIEW',
        engineering_qualified=all(checks.values()), passed=sum(checks.values()), total=len(checks), checks=checks,
        checker=source, reviewed_report=pin(summary_path), reader=report['reader'], contract=report['contract'],
        source_pins=source_pins, methods=results, rows=total, CSV_maximum_abs_numeric_difference=csv_max_abs,
        comparison_tolerance={'absolute': TOL, 'relative': TOL, 'counter_and_provenance': 'exact'}, issues=issues,
        limitations=['Aggregation of logged rounded FP32 active means; not reconstruction of individual H8 rows.',
            'Positive-P-mean updates do not measure per-row positive fraction, survival-weighted Actor loss or gradient.',
            'Existing final-health/source/RNG assertions are bound, not newly independently replayed.',
            'Sampled imagined geometry proxy activation is not physical contact, penetration, force or method benefit.'],
        model_loads=0, native_queries=0, physics_steps=0, GPU_used=False)
    dump(out/'report.json', result)
    dump(out/'README.json', {'conclusion': result['status'], 'passed': result['passed'], 'total': result['total'],
        'report': pin(out/'report.json'), 'interpretation': result['limitations']})
    print(json.dumps({'report': pin(out/'report.json'), 'passed': result['passed'], 'total': result['total'],
        'methods': {m: {'P': results[m]['all_active_row_weighted_means']['P_raw_active_mean'],
                       'positive_updates': results[m]['positive_P_mean_updates'],
                       'active_rows': results[m]['active_weighted_denominator']} for m in METHODS}}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(audit(args.summary, args.output))
    except Exception as exc:
        if args.output.exists() and not (args.output/'failure.json').exists():
            dump(args.output/'failure.json', {'error': repr(exc), 'checker': pin(__file__),
                'saved_only': True, 'no_retry': True})
        raise
