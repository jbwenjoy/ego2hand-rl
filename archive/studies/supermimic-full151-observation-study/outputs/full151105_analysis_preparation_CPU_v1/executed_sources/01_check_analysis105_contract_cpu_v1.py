"""Pure closed-form and metadata contract checks; no WM/dataset/model/backend."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from full151_gradient import analysis105_v1 as a


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    args = p.parse_args(); out = Path(args.output).resolve()
    if out.exists(): raise ValueError('Fresh CPU proof output only')
    sources = [Path(a.__file__).resolve(), Path(__file__).resolve(), ROOT / 'scripts/analyze_full151105_physical_gradient_cpu_v1.py', ROOT / 'scripts/plot_full151105_physical_gradient_saved_v1.py']
    before = {str(v): a.pin(v) for v in sources}
    # Freeze exact executed bytes before importing/testing any runtime/model.
    out.mkdir(parents=True); snapshot = out / 'executed_sources'; snapshot.mkdir()
    for i, file in enumerate(sources): (snapshot / ('%02d_' % i + file.name)).write_bytes(file.read_bytes())
    (out / 'source_before.json').write_text(json.dumps(before, indent=2), encoding='utf-8')
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
        if not checks[name]: raise AssertionError(name)
    def rejects(name, function):
        try: function()
        except (ValueError, KeyError): check(name, True)
        else: check(name, False)
    try:
        for source in sources: ast.parse(source.read_text(encoding='utf-8'))
        check('all_sources_parse', True)
        check('fresh_process_noTorch_Warp', 'torch' not in sys.modules and 'warp' not in sys.modules)
        # A nonsymmetric full quadratic (symmetric Hessian) tests clipped midpoint
        # bias exactly; this is not a duplicate implementation of the consumer.
        H = np.diag(np.arange(1, 21, dtype=np.float64) * .3)
        H[0, 4] = H[4, 0] = .7
        b = np.linspace(-.4, .9, 20); center = np.linspace(-.2, .3, 20).astype(np.float32)
        c = center.astype(np.float64)
        f = lambda x: float(1.2 + b @ x + .5 * x @ H @ x)
        analytic = b + H @ c
        for i, eps in enumerate(a.EPS):
            minus = center.copy(); plus = center.copy()
            minus[4] -= np.float32(eps); plus[4] += np.float32(eps)
            d, offset = a.effective_direction(center, minus, plus, eps)
            finite = (f(plus.astype(float)) - f(minus.astype(float))) / (2 * eps)
            exact = analytic @ d + d @ H @ offset
            check('quadratic_secant_uses_nominal_2eps_%d' % i, abs(finite - exact) < 1e-11)
            check('effective_direction_is_actual_FP32_gain_%d' % i, d[4] == (float(plus[4]) - float(minus[4])) / (2 * eps))
            plus[4] = center[4] + np.float32(eps * .25)  # asymmetric upper clip
            d, offset = a.effective_direction(center, minus, plus, eps)
            finite = (f(plus.astype(float)) - f(minus.astype(float))) / (2 * eps)
            check('clipped_quadratic_midpoint_bias_explicit_%d' % i, abs(finite - (analytic @ d + d @ H @ offset)) < 1e-11 and abs(offset[4]) > 0)
            check('clipped_effective_projection_not_nominal_gradient_%d' % i, abs(float(analytic @ d) - analytic[4]) > .01)
        rejects('reject_unfrozen_epsilon', lambda: a.effective_direction(center, center, center, .003))
        rejects('reject_FP64_control_endpoint', lambda: a.effective_direction(c, center, center, .002))
        rejects('reject_nonfinite_control_endpoint', lambda: a.effective_direction(center, center, np.full(20, np.nan, np.float32), .002))
        # Force FP32 subtraction overflow: cast-before-difference remains finite.
        plus = np.array([3e38, 1.5e38], np.float32); minus = -plus
        result = a.classify_pair(minus, plus, 0., 0.)
        check('FP32scores_cast_F64_before_subtraction', np.isfinite(result['slopes']).all() and result['stable_nonflat'])
        result = a.classify_pair(np.array([0., 0.]), np.array([.004, .002]), 0, 0)
        check('same_slope_twoeps_stable', result['stable_nonflat'] and result['slopes'] == [1., 1.])
        check('frozen_signal_floor_1e_5', result['signal_floor'] == 1e-5)
        result = a.classify_pair(np.zeros(2), np.array([.004, -.002]), 0, 0)
        check('opposite_sign_nonflat_unstable', result['label'] == 'NONFLAT_UNSTABLE')
        result = a.classify_pair(np.zeros(2), np.array([.004, .0002]), 0, 0)
        check('relative_eps_disagreement_unstable', result['label'] == 'NONFLAT_UNSTABLE' and result['relative_disagreement'] > .5)
        result = a.classify_pair(np.zeros(2), np.array([1e-5, 1e-5]), 0, 0)
        check('strict_above_floor_not_equal', result['label'] == 'FLAT_OR_BELOW_NOISE')
        result = a.classify_pair(np.zeros(2), np.array([.00035, .00035]), 2e-5, 3e-5)
        check('noise_uses_max_not_sum', result['signal_floor'] == 10 * 3e-5 and result['both_above_noise'])
        rejects('reject_negative_baseline_noise', lambda: a.classify_pair([0, 0], [1, 1], -1., 0.))
        rejects('reject_NaN_score', lambda: a.classify_pair([0, 0], [float('nan'), 1], 0., 0.))
        empty = a.agreement([])
        check('no_stable_rows_fraction_and_correlation_null', all(v['eligible'] == 0 and v['sign_fraction'] is None and v['pearson'] is None for v in empty.values()))
        require_fields = dict(artifact_pins={}, input_bundle={'path': 'synthetic', 'sha256': 'synthetic'}, physical_run='synthetic')
        rejects('inventory750_not_full_qualification', lambda: a.validate_saved_review({'schema': 'full151105_inventory_discovery_all_saved_independent:1', 'engineering_qualified': True}, require_fields))
        # An internally consistent full-serial authority is accepted only with
        # exact counts and config/report/source pins; this is not real saved data.
        pins = {k: {'path': 'synthetic_' + k, 'sha256': 'synthetic'} for k in ('producer_config', 'producer_report', 'source_inventory')}
        cfg = dict(artifact_pins=pins, input_bundle={'path': 'synthetic_bundle', 'sha256': 'synthetic'}, physical_run='synthetic_run')
        review = dict(schema='full151105_full_serial_saved_independent:1', engineering_qualified=True, probe_mode='serial_full',
                      completed_inventory_snapshots=8, completed_branches=152, baseline_count=24, episode_step_returned=1808,
                      physical_gradient_qualified=False, whole_hidden_state_qualified=False, checks={'synthetic_guard': True}, passed=1, total=1,
                      input_bundle=cfg['input_bundle'], config=pins['producer_config'], producer_report=pins['producer_report'], source_inventory=pins['source_inventory'], run_dir='synthetic_run')
        a.validate_saved_review(review, cfg); check('strict_full_saved_schema_positive_synthetic', True)
        for key, bad in [('completed_branches', 151), ('whole_hidden_state_qualified', True), ('config', {'path': 'wrong', 'sha256': 'synthetic'})]:
            rejects('full_saved_reject_' + key, lambda key=key, bad=bad: a.validate_saved_review({**review, key: bad}, cfg))
        rejects('boolean_integer_JSON_types_distinct', lambda: a.require(a.same(True, 1), 'distinct'))
        # Scalar two endpoints: hand-calculated endpoint distances + rotation.
        x = np.zeros((1, 58)); x[:, 43:52] = np.eye(3).reshape(1, 9)
        goal = np.array([[.01, 0, 0]]); R = np.eye(3)[None]
        terms = a.object_terms_f64(x, goal, R)[0]
        check('same_x20_closed_form_score_2plus8', np.allclose(terms, [2 * np.exp(-.2 ** 2), 8 * np.exp(-.25 ** 2)], atol=1e-14, rtol=0))
        # Negative/zero labels do not invent positive-class recall or F1.
        metrics = a._self_metrics(np.full((8, 15), -20.), np.zeros((8, 15)))
        check('allnegative_self15_F1_recall_null', metrics['F1'] is None and metrics['recall'] is None and metrics['Brier'] > 0)
        check('self15_is120labels_not_HO', metrics['labels'] == 120 and 'not hand-object' in metrics['scope'])
        # Source contract, not a claim that a real loader was called here.
        tree = ast.parse(Path(a.__file__).read_text(encoding='utf-8'))
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_analysis')
        loader_calls = [n for n in ast.walk(run) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'load_world_model_checkpoint']
        check('exactly_one_declared_strict_loader_call', len(loader_calls) == 1)
        check('old_state15_reward_not_WM_target', 'native_reward_score_15ms_secondary' not in ast.get_source_segment(Path(a.__file__).read_text(encoding='utf-8'), run))
        for path, value in before.items(): check('source_unchanged_' + Path(path).name, a.pin(path) == value)
        check('after_tests_noTorch_Warp', 'torch' not in sys.modules and 'warp' not in sys.modules)
        report = dict(schema='full151105_analysis_CPU_contract:1', status='PASS_PURE_METADATA_AND_CLOSED_FORM_ONLY', passed=sum(checks.values()), total=len(checks), checks=checks,
                      source_pins={Path(k).name: v for k, v in before.items()}, actual_WM_loads=0, actual_WM_forwards=0, Env_created=False, physics_steps=0, GPU_used=False,
                      actual_full152_data_read=False, scope='Synthetic closed-form algebra/negative metadata only; no physical/WM derivative or actual dataset qualification')
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        ready = dict(schema='full151105_analysis_CPU_preparation_READY:1', status='READY_SOURCE_AND_PURE_CPU_CONTRACT_ONLY',
                     source_pins=report['source_pins'], cpu_report=a.pin(out / 'report.json'), cpu_checks={'passed': report['passed'], 'total': report['total']},
                     api={'entry': str(ROOT / 'scripts/analyze_full151105_physical_gradient_cpu_v1.py'), 'module': str(Path(a.__file__).resolve()),
                          'default': 'metadata_preflight(config_path)', 'authorized': 'CUDA_VISIBLE_DEVICES=-1; --config CONFIG --run', 'config_schema': a.SCHEMA,
                          'required_artifact_pin_keys': list(a.ARTIFACTS), 'saved_only_plot': str(ROOT / 'scripts/plot_full151105_physical_gradient_saved_v1.py')}, model_or_physical_run_authorized=False)
        (out / 'READY.json').write_text(json.dumps(ready, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'passed': report['passed'], 'total': report['total'], 'report': a.pin(out / 'report.json'), 'READY': a.pin(out / 'READY.json')}, indent=2))
    except BaseException as exc:
        (out / 'failure.json').write_text(json.dumps({'error': repr(exc), 'checks_completed': checks, 'source_before': before}, indent=2, allow_nan=False), encoding='utf-8')
        raise


if __name__ == '__main__': main()
