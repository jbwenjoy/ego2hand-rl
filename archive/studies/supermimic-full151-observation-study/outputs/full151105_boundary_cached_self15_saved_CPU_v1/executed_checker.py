"""One Tiny closed-form cached-label fixture; no actual NPZ/model/metrics."""
import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py'
OUT = ROOT/'outputs/full151105_cached_self15_saved_preparation_CPU_v1'


def main():
    OUT.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('_tiny_cached_self15', CORE)
    core = importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
    pins = {k: core.pin(path) for k, path in core.OWN.items()}
    for k, path in core.OWN.items():
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'));(OUT/('frozen_'+k+'.py')).write_bytes(path.read_bytes())
    core.write(OUT/'FREEZE.json', {'source_pins': pins, 'actual_parent_arrays_read': False, 'fixture_rows': 2})
    checks = {};details = {}
    try:
        d = subprocess.run([sys.executable, str(core.OWN['CLI'])], check=True, text=True, capture_output=True)
        checks['default_no_numpy_torch_or_saved_reads'] = json.loads(d.stdout)['status'] == 'METADATA_ONLY_NO_SAVED_ARRAY_OR_METRICS' and 'numpy' not in sys.modules and 'torch' not in sys.modules
        import numpy as np
        npz_attempts = [];original_load = np.load
        def deny_load(*args, **kwargs):
            npz_attempts.append(True);raise AssertionError('No actual/synthetic file reads in closed-form fixture')
        np.load = deny_load
        try:
            mapping = {'middle': core.MIDDLE, 'distal': core.DISTAL, 'tips': core.TIPS, 'native_sensor': 'finger_collision',
                       'state_phase': 'postintegration20ms', 'sensor_phase': 'lastsolver preintegration15ms before Managerforward'}
            plan = [{'row_id': bid, 'snapshot_id': 0, 'path': 'A0', 'mode': 'baseline', 'k': 0, 'branch_id': bid,
                     'group_id': 150+bid, 'physical_rows': [4*(150+bid)+j for j in range(4)], 'terminal_observation': 1} for bid in (0, 1)]
            raw0 = [0]*25;raw0[0] = 7;raw0[1] = 2;raw0[4] = 1
            y0 = [0.]*15;y0[0] = 1.;y0[10] = 1.
            rewards = [{'path': 'A0', 'k': 0, 'branch_id': bid, 'mode': 'baseline', 'source_frame': 200, 'terminal_observation': 1,
                        'valid_next_policy_decision': True, 'no_u150': True, 'label_time': 'lastsolver_preintegration~kdt+.015',
                        'state_time': 'postintegration_kdt+.020', 'capture': 'after_original_reward_compute_before_Manager_forward',
                        'contact': y0 if bid == 0 else [0.]*15, 'raw_found': [raw0 if bid == 0 else [0]*25]} for bid in (0, 1)]
            target, joined = core.join_labels(plan, rewards, {'A0': mapping, 'A2': mapping})
            target = np.asarray(target, dtype=np.float32)
            checks['cached15_join_raw25_OR_mapping_and_phase'] = np.array_equal(target[0], np.asarray(y0, np.float32)) and not target[1].any() and len(joined) == 2
            l = np.float32(math.log(3.));correct = np.where(target > 0, l, -l).astype(np.float32)
            logits = {core.ROLES[0]: np.zeros((2, 15), np.float32), core.ROLES[1]: correct, core.ROLES[2]: -correct}
            scores = {role: core.score_matrix(a, target, np) for role, a in logits.items()}
            summaries = {role: core.summary(s, [0, 1], np) for role, s in scores.items()}
            a, b, c = (summaries[r] for r in core.ROLES)
            expected = [{'TP': 0, 'FP': 0, 'FN': 2, 'TN': 28, 'zero_logit_tie': 30, 'tie_positive_target': 2, 'tie_negative_target': 28},
                        {'TP': 2, 'FP': 0, 'FN': 0, 'TN': 28, 'zero_logit_tie': 0, 'tie_positive_target': 0, 'tie_negative_target': 0},
                        {'TP': 0, 'FP': 28, 'FN': 2, 'TN': 0, 'zero_logit_tie': 0, 'tie_positive_target': 0, 'tie_negative_target': 0}]
            for observed, wanted in zip((a, b, c), expected):core.require(observed['confusion_total'] == wanted, 'Closed-form strict>0 confusion/ties')
            tol = 32*np.finfo(np.float64).eps
            np.testing.assert_allclose(a['BCE_mean_entry_F64'], math.log(2.), rtol=0, atol=tol)
            np.testing.assert_allclose(a['Brier_mean_entry_F64'], .25, rtol=0, atol=tol)
            np.testing.assert_allclose(b['BCE_mean_entry_F64'], math.log1p(math.exp(-float(l))), rtol=0, atol=tol)
            np.testing.assert_allclose(c['BCE_mean_entry_F64'], math.log1p(math.exp(float(l))), rtol=0, atol=tol)
            np.testing.assert_allclose(scores[core.ROLES[0]]['proxy_penalty_error_F64'], [-.11, -.15], rtol=0, atol=tol)
            extreme = np.asarray([[-1000., 0., 1000.]+[0.]*12], dtype=np.float32);ey = np.asarray([[0., 1., 1.]+[0.]*12], dtype=np.float32)
            e = core.score_matrix(extreme, ey, np)
            core.require(np.isfinite(e['BCE15_F64']).all(), 'Stable finite extremes')
            np.testing.assert_allclose(e['BCE15_F64'][0, :3], [0., math.log(2.), 0.], rtol=0, atol=tol)
            checks['three_roles_stable_F64_BCE_Brier_confusion_ties_rewarddt_proxy'] = True
            # Plan-only synthetic metadata: no labels or logits scored for152.
            fullplan = []
            for bid in range(152):
                s, off = divmod(bid, 19);pi, si = divmod(s, 4);k = (0, 49, 99, 149)[si];g = 226*pi+150+19*si+off
                fullplan.append({'row_id': bid, 'snapshot_id': s, 'path': ('A0', 'A2')[pi], 'mode': 'baseline' if off < 3 else 'direction',
                                 'k': k, 'branch_id': bid, 'group_id': g, 'physical_rows': [4*g+j for j in range(4)], 'terminal_observation': k+1})
            for row in fullplan:core.validate_plan_row(row)
            subsets = core.cohorts(fullplan)
            checks['exact13_cohort_metadata_not_path_mode_replacement'] = list(subsets) == ['ALL', 'A0', 'A2', 'A0/k0', 'A0/k49', 'A0/k99', 'A0/k149', 'A2/k0', 'A2/k49', 'A2/k99', 'A2/k149', 'baseline', 'direction'] and [len(v) for v in subsets.values()] == [152, 76, 76]+[19]*8+[24, 128]
            rejection = {}
            def reject(call):
                try:call()
                except (ValueError, KeyError):return True
                return False
            for label in ('contact_mismatch', 'phase_mismatch', 'duplicate_reward', 'short_raw_found'):
                bad = copy.deepcopy(rewards)
                if label == 'contact_mismatch':bad[0]['contact'][0] = 0.
                elif label == 'phase_mismatch':bad[0]['label_time'] = 'post20_geometry'
                elif label == 'duplicate_reward':bad.append(copy.deepcopy(bad[0]))
                else:bad[0]['raw_found'][0] = bad[0]['raw_found'][0][:-1]
                rejection[label] = reject(lambda bad=bad: core.join_labels(plan, bad, {'A0': mapping, 'A2': mapping}))
            checks['one_fixture_source_identity_label_phase_mutations_rejected'] = all(rejection.values()) and not npz_attempts
            core.write(OUT/'tiny_closed_form_fixture.json', {'scope': 'SYNTHETIC_TWO_ROWS_ONLY_NOT_ACTUAL_LABEL_EVALUATION', 'plan': plan,
                        'reward_records': rewards, 'logits': {k: v.tolist() for k, v in logits.items()}, 'expected_confusion': expected})
            details = {'fixture_rows': 2, 'actual152_label_qualification': False, 'synthetic_plan_only_cohort_counts': {k: len(v) for k, v in subsets.items()}, 'rejected_mutations': rejection}
        finally:
            np.load = original_load
        core.require(all(checks.values()) and all(core.same(core.pin(path), pins[k]) for k, path in core.OWN.items()), 'Frozen focused fixture passed')
        report = {'schema': 'full151105_cached_self15_saved_CPU_contract:1', 'status': 'PASS_TINY_CACHED_LABEL_JOIN_AND_F64_SCORE_NO_ACTUAL_DATA',
                  'checks': checks, 'passed': sum(checks.values()), 'total': len(checks), 'source_pins': pins, 'details': details,
                  'actual_counts': {'actual_NPZ_reads': 0, 'actual_label_or_prediction_evaluation': 0, 'weights_NN_FW_AD_MJB_Env_physics_GPU': 0}}
        core.write(OUT/'report.json', report)
        ready = {'schema': 'full151105_cached_self15_saved_CPU_READY:1', 'status': report['status'], 'source_pins': pins,
                 'cpu_report': core.pin(OUT/'report.json'), 'cpu_checks': {'passed': report['passed'], 'total': report['total']},
                 'api': {'entry': str(core.OWN['CLI']), 'default': '--protocol PATH --output FRESH', 'actual': '--protocol PATH --output FRESH --execute'},
                 'actual152_label_evaluation_qualified': False}
        core.write(OUT/'READY.json', ready)
        print(json.dumps({'status': report['status'], 'passed': report['passed'], 'total': report['total'], 'READY': core.pin(OUT/'READY.json')}))
        return 0
    except BaseException as error:
        core.write(OUT/'failure.json', {'error': repr(error), 'source_pins': pins, 'actual_data_models_FW_AD_physics': 0})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
