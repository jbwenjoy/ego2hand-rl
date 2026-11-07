"""Saved predictions only: descriptive, source-dependent training-set metrics."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

ROLES = ('unwrapped_baseline', 'shared_public_losses', 'shared_public_plus_cross_target_consistency')
BLOCKS = {'q': (0, 20, 'rad'), 'qd': (20, 40, 'rad/s'), 'p': (40, 43, 'm'),
          'native_LogR': (43, 46, 'rad'), 'v': (46, 49, 'm/s'), 'omega': (49, 52, 'rad/s')}


def pin(p):
    p = Path(p).absolute()
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def arrays(p):
    with np.load(p, allow_pickle=False) as f:
        return {k: f[k].copy() for k in f.files}


def write(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def vector_stats(value, unit):
    x = np.asarray(value, dtype=np.float64)
    assert x.ndim == 2 and len(x) and np.isfinite(x).all()
    return {'unit': unit, 'component_RMSE': float(np.sqrt(np.mean(x*x))),
            'vector_RMS': float(np.sqrt(np.mean(np.sum(x*x, axis=-1)))),
            'component_MAE': float(np.mean(np.abs(x))),
            'maximum_absolute': float(np.abs(x).max()),
            'maximum_vector_norm': float(np.linalg.norm(x, axis=-1).max()),
            'per_feature_signed_mean': x.mean(axis=0).tolist(),
            'per_feature_RMSE': np.sqrt(np.mean(x*x, axis=0)).tolist(),
            'per_feature_MAE': np.mean(np.abs(x), axis=0).tolist(),
            'per_feature_maximum_absolute': np.abs(x).max(axis=0).tolist()}


def sigmoid(x):
    return np.exp(-np.logaddexp(0., -np.asarray(x, dtype=np.float64)))


def cohorts(ids):
    result = [('ALL', np.ones(152, bool))]
    result += [(p, np.asarray([r['path'] == p for r in ids])) for p in ('A0', 'A2')]
    result += [(f'{p}/k{k}', np.asarray([r['path'] == p and r['k'] == k for r in ids]))
               for p in ('A0', 'A2') for k in (0, 49, 99, 149)]
    result += [(m, np.asarray([r['mode'] == m for r in ids])) for m in ('baseline', 'direction')]
    assert len(ids) == 152 and len(result) == 13
    assert [int(mask.sum()) for _, mask in result] == [152, 76, 76] + [19]*8 + [24, 128]
    assert {r['path'] for r in ids} == {'A0', 'A2'} and {r['mode'] for r in ids} == {'baseline', 'direction'}
    assert {r['k'] for r in ids} == {0, 49, 99, 149}
    return result


def summarize(forward, loss, data, initial_logits, mask):
    assert mask.shape == (152,) and mask.dtype == bool and mask.any()
    pred = forward['prediction58_FP32'].astype('f8')
    delta = loss['native_point_delta52_FP32'].astype('f8')
    scale = data['effective_s52_FP32'].astype('f8')
    labels = data['boundary_constraint_witness']
    assert delta.shape == (152, 52) and scale.shape == (52,) and (scale > 0).all()
    assert labels.shape == (152, 2) and labels.dtype == bool
    z = forward['z12_FP32'].astype('f8')
    logits = forward['mode_logits2_FP32'].astype('f8')
    p = sigmoid(logits)
    guess = logits > 0  # Exact ties deliberately classified negative; count them.
    bce_elements = np.logaddexp(0., logits) - logits*labels
    cons = np.concatenate(((loss['r_v_hat_F64']-data['r_v_saved_m_s_F64'])/scale[46:49],
                           (loss['r_w_hat_F64']-data['r_w_saved_rad_s_F64'])/scale[49:52]), axis=-1)
    result = {
        'groups': int(mask.sum()),
        'native52_scaled_MSE_F64_of_saved_FP32_delta': float(np.mean((delta[mask]/scale)**2)),
        'mode_BCE_F64_of_saved_FP32_logits': float(bce_elements[mask].mean()),
        'mode_Brier_F64': float(np.mean((p[mask]-labels[mask])**2)),
        'mode_Brier_from_captured_FP32_probability_F64_reduction': float(np.mean((loss['mode_probability2_FP32'][mask].astype('f8')-labels[mask])**2)),
        'impulse_z_MSE_F64': float(np.mean((z[mask]-loss['z_target_F64'][mask])**2)),
        'saved_defect_consistency_MSE_F64': float(np.mean(cons[mask]**2)),
        'state_native_delta': {n: vector_stats(delta[mask, lo:hi], unit) for n, (lo, hi, unit) in BLOCKS.items()},
        'raw_R9_difference': vector_stats((pred-data['target58_FP32'])[:, 43:52][mask], 'raw matrix entry; vector_RMS is Frobenius_RMS'),
        'mode': {}, 'boundary_impulse_error': {}, 'dimensionless_impulse_error': {}, 'observed_defect_match_error': {},
        'raw58_difference': {},
    }
    for name, lo, hi, unit in [('q', 0, 20, 'rad'), ('qd', 20, 40, 'rad/s'), ('p', 40, 43, 'm'),
                                ('R_raw9', 43, 52, 'raw matrix entry'), ('v', 52, 55, 'm/s'), ('omega', 55, 58, 'rad/s')]:
        result['raw58_difference'][name] = vector_stats((pred-data['target58_FP32'])[mask, lo:hi], unit)
    result['native52_per_feature'] = vector_stats(delta[mask], 'Mixed units: per-feature only; do not interpret this aggregate component_RMSE/vector_RMS physically')
    # Only named per-feature fields are meaningful across the 52 mixed-unit entries.
    result['native52_per_feature'] = {k: v for k, v in result['native52_per_feature'].items() if k.startswith('per_feature_') or k == 'unit'}
    result['mode']['both_bits_correct_groups'] = int(np.all(guess[mask] == labels[mask], axis=1).sum())
    for j, name in enumerate(('HO', 'nonhand')):
        y, g = labels[mask, j], guess[mask, j]
        result['mode'][name] = {'positive_witness': int(y.sum()), 'negative_witness': int((~y).sum()),
                                'TP': int((g & y).sum()), 'FP': int((g & ~y).sum()),
                                'TN': int((~g & ~y).sum()), 'FN': int((~g & y).sum()),
                                'exact_zero_logit_ties': int((logits[mask, j] == 0).sum()),
                                'BCE_F64': float(bce_elements[mask, j].mean()),
                                'Brier_F64': float(np.mean((p[mask, j]-y)**2))}
        for kind, sl in [('zJ', slice(0, 3)), ('zK', slice(3, 6))]:
            result['dimensionless_impulse_error'][kind+'_'+name] = vector_stats((z-loss['z_target_F64'])[mask, j, sl], 'dimensionless')
        for kind, key, target_key, unit in [('J', 'J_hat_palm_F64', 'J_palm_F64', 'N s'),
                                           ('K', 'K_hat_palm_F64', 'K_palm_F64', 'N m s')]:
            result['boundary_impulse_error'][kind+'_'+name] = vector_stats((loss[key]-data[target_key])[mask, j], unit)
    for kind, key, target_key, unit in [('J', 'J_hat_palm_F64', 'J_palm_F64', 'N s'), ('K', 'K_hat_palm_F64', 'K_palm_F64', 'N m s')]:
        result['boundary_impulse_error'][kind+'_two_source_total'] = vector_stats((loss[key].sum(axis=1)-data[target_key].sum(axis=1))[mask], unit)
    for key, target_key, label, unit in [('r_v_hat_F64', 'r_v_saved_m_s_F64', 'linear', 'm/s'),
                                        ('r_w_hat_F64', 'r_w_saved_rad_s_F64', 'angular', 'rad/s')]:
        result['observed_defect_match_error'][label] = vector_stats((loss[key]-data[target_key])[mask], unit)
    current = forward['self15_logits_FP32'].astype('f8')
    probs, initial_probs = sigmoid(current), sigmoid(initial_logits)
    result['self15_initial_drift'] = {
        'logit': vector_stats((current-initial_logits)[mask], 'logit; no true self15 labels'),
        'probability': vector_stats((probs-initial_probs)[mask], 'probability; no accuracy claim'),
        'mean_sum_sigmoid': float(probs[mask].sum(axis=1).mean()),
        'initial_mean_sum_sigmoid': float(initial_probs[mask].sum(axis=1).mean()),
        'mean_change_sum_sigmoid': float((probs-initial_probs)[mask].sum(axis=1).mean()),
        'strict_positive_logit_bit_flips': int(((current > 0) != (initial_logits > 0))[mask].sum()),
        'per_feature_strict_positive_logit_bit_flips': ((current > 0) != (initial_logits > 0))[mask].sum(axis=0).tolist(),
        'current_exact_zero_logit_ties': int((current[mask] == 0).sum()),
        'initial_exact_zero_logit_ties': int((initial_logits[mask] == 0).sum()),
        'captured_FP32_mean_penalty_rate': float(loss['self15_penalty_rate_FP32'][mask].astype('f8').mean()),
        'captured_FP32_mean_penalty_component_dt002': float(loss['self15_penalty_component_FP32'][mask].astype('f8').mean()),
        'F64_rescore_mean_penalty_rate': float(-probs[mask].sum(axis=1).mean()),
        'F64_rescore_mean_penalty_component_dt002': float(-probs[mask].sum(axis=1).mean()*.02),
        'probability_precision_scope': 'Unprefixed sigmoid/drift/Brier statistics use stable F64 rescore from saved FP32 logits; captured FP32 arrays/penalty summaries are separately named, not claimed bit-identical.',
        'scope': 'Initial function drift only. No self15 label, task reward, physical clearance or penalty accuracy qualification.',
    }
    return result


def row_table(role, forward, loss, data, ids, initial_logits):
    d = loss['native_point_delta52_FP32'].astype('f8')
    s = data['effective_s52_FP32'].astype('f8')
    z = forward['z12_FP32'].astype('f8')
    logits = forward['mode_logits2_FP32'].astype('f8')
    labels = data['boundary_constraint_witness']
    cons = np.concatenate(((loss['r_v_hat_F64']-data['r_v_saved_m_s_F64'])/s[46:49],
                           (loss['r_w_hat_F64']-data['r_w_saved_rad_s_F64'])/s[49:52]), axis=-1)
    bce = np.logaddexp(0., logits)-logits*labels
    self_logits = forward['self15_logits_FP32'].astype('f8')
    for j, identity in enumerate(ids):
        row = {'role': role, 'row_id': j, 'path': identity['path'], 'k': identity['k'], 'mode': identity['mode'], 'group_id': identity['group_id'],
               'native52_scaled_MSE_F64': float(np.mean((d[j]/s)**2)), 'mode_BCE_F64': float(bce[j].mean()),
               'impulse_z_MSE_F64': float(np.mean((z[j]-loss['z_target_F64'][j])**2)), 'consistency_MSE_F64': float(np.mean(cons[j]**2)),
               'self15_logit_vector_drift': float(np.linalg.norm(self_logits[j]-initial_logits[j])),
               'captured_FP32_self15_penalty_component': float(loss['self15_penalty_component_FP32'][j])}
        for name, (lo, hi, unit) in BLOCKS.items():
            row[name+'_native_delta_vector_norm'] = float(np.linalg.norm(d[j, lo:hi]))
        for c, boundary in enumerate(('HO', 'nonhand')):
            row[boundary+'_mode_logit'] = float(logits[j, c])
            row[boundary+'_mode_witness'] = int(labels[j, c])
            row[boundary+'_predicted_bit_strict_positive'] = int(logits[j, c] > 0)
            for kind, key, truth in [('J', 'J_hat_palm_F64', 'J_palm_F64'), ('K', 'K_hat_palm_F64', 'K_palm_F64')]:
                row[boundary+'_'+kind+'_error_vector_norm'] = float(np.linalg.norm(loss[key][j, c]-data[truth][j, c]))
        yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    run = args.run_dir.resolve(strict=True)
    producer = read(run/'report.json')
    assert producer['status'] == 'COMPLETED_FIXED_INITIAL_AND_TWO_AFTER1000_EVAL_NOT_CONFIRMATION'
    assert producer['checks'] and all(v is True for v in producer['checks'].values())
    for item in producer['outputs'].values():
        assert Path(item['path']).parent.samefile(run) and pin(item['path'])['sha256'] == item['sha256']
    data = arrays(run/'prepared_inputs_and_context.npz')
    ids = read(run/'group152_identities.json')
    assert [r['row_id'] for r in ids] == list(range(152))
    assert len({r['group_id'] for r in ids}) == 152
    assert np.array_equal(np.asarray([r['group_id'] for r in ids]), data['source_group_id152'])
    fs = {r: arrays(run/(r+'_forward152.npz')) for r in ROLES}
    ls = {r: arrays(run/(r+'_descriptive_losses.npz')) for r in ROLES}
    for f in fs.values():
        for k, shape in [('prediction58_FP32', (152, 58)), ('self15_logits_FP32', (152, 15)),
                         ('z12_FP32', (152, 2, 6)), ('mode_logits2_FP32', (152, 2))]:
            assert f[k].shape == shape and f[k].dtype == np.float32 and np.isfinite(f[k]).all()
    for loss in ls.values():
        for k, shape in [('native_point_squared_scaled52_FP32', (152, 52)), ('mode_probability2_FP32', (152, 2)),
                         ('self15_probability15_FP32', (152, 15)), ('self15_penalty_rate_FP32', (152,)), ('self15_penalty_component_FP32', (152,))]:
            assert loss[k].shape == shape and loss[k].dtype == np.float32 and np.isfinite(loss[k]).all()
    out = args.output.absolute();out.mkdir(exist_ok=False)
    all_stats = {r: {name: summarize(fs[r], ls[r], data, fs[ROLES[0]]['self15_logits_FP32'].astype('f8'), mask)
                     for name, mask in cohorts(ids)} for r in ROLES}
    flat = []
    for role, groups in all_stats.items():
        for name, s in groups.items():
            row = {'role': role, 'cohort': name, 'groups': s['groups']}
            for key in ('native52_scaled_MSE_F64_of_saved_FP32_delta', 'mode_BCE_F64_of_saved_FP32_logits',
                        'impulse_z_MSE_F64', 'saved_defect_consistency_MSE_F64'):
                row[key] = s[key]
            for block in BLOCKS:
                row[block+'_vector_RMS'] = s['state_native_delta'][block]['vector_RMS']
            row['self15_logit_component_RMSE_vs_initial'] = s['self15_initial_drift']['logit']['component_RMSE']
            row['self15_mean_sum_sigmoid'] = s['self15_initial_drift']['mean_sum_sigmoid']
            flat.append(row)
    with (out/'all39_model_cohort_summary.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    rows = [row for role in ROLES for row in row_table(role, fs[role], ls[role], data, ids, fs[ROLES[0]]['self15_logits_FP32'].astype('f8'))]
    assert len(rows) == 456
    with (out/'all456_model_row_metrics.csv').open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    comparisons = {}
    for a, b in [(ROLES[1], ROLES[0]), (ROLES[2], ROLES[0]), (ROLES[2], ROLES[1])]:
        comparisons[a+'_minus_'+b] = {}
        for group, x in all_stats[a].items():
            y = all_stats[b][group]
            comparisons[a+'_minus_'+b][group] = {k: x[k]-y[k] for k in ('native52_scaled_MSE_F64_of_saved_FP32_delta', 'mode_BCE_F64_of_saved_FP32_logits', 'impulse_z_MSE_F64', 'saved_defect_consistency_MSE_F64')}
            comparisons[a+'_minus_'+b][group]['captured_FP32_mean_self15_penalty_component_difference'] = x['self15_initial_drift']['captured_FP32_mean_penalty_component_dt002']-y['self15_initial_drift']['captured_FP32_mean_penalty_component_dt002']
    report = {
        'schema': 'full151105_boundary_component_saved_prediction_description:1',
        'status': 'COMPLETED_DESCRIPTIVE_ALL152_FIXED_FINAL1000_NOT_GENERALIZATION',
        'producer': pin(run/'report.json'), 'source': pin(__file__),
        'cohorts': [n for n, _ in cohorts(ids)], 'roles': list(ROLES), 'statistics': all_stats,
        'signed_metric_differences_not_a_composite_winner': comparisons,
        'captured_global_losses': {r: {key: float(ls[r][key]) for key in ('L_state_FP32', 'L_mode_FP32', 'L_imp_F64', 'L_cons_F64', 'L_public')} for r in ROLES},
        'scope': 'Dependent training rows from one cached robot lineage. Scalar F64 recomputation from saved FP32 delta/logits may differ from captured Torch FP32 mean; neither is a replay of native Log or original optimizer iteration. All13 cohorts retained; no accuracy threshold, significance test, model selection or composite winner.',
        'rotation_scope': 'native LogR residual on original rawR9 under producer chart guard; raw matrix Frobenius distance is descriptive, not a projected SO3 geodesic certificate.',
        'self15_scope': 'Drift relative to initial only; no true15 target labels in this experiment. New2 boundary modes have different semantics.',
        'mode_threshold': 'Strict logit>0; exactzero counted and classifiednegative. All constraint witnesses retained including zero-force records.',
        'new_model_weight_load_FW_AD_MJB_Env_physics_GPU': 0,
        'accuracy_task_CHORD_physicalgradient_ego_qualified': False,
        'outputs': {'cohort_CSV': pin(out/'all39_model_cohort_summary.csv'), 'row_CSV': pin(out/'all456_model_row_metrics.csv')},
    }
    write(out/'report.json', report)
    preview = {}
    for r in ROLES:
        s = all_stats[r]['ALL']
        preview[r] = {k: s[k] for k in ('native52_scaled_MSE_F64_of_saved_FP32_delta', 'mode_BCE_F64_of_saved_FP32_logits', 'impulse_z_MSE_F64', 'saved_defect_consistency_MSE_F64')}
        preview[r].update(p_vector_RMS_m=s['state_native_delta']['p']['vector_RMS'], native_LogR_vector_RMS_rad=s['state_native_delta']['native_LogR']['vector_RMS'], omega_vector_RMS_rad_s=s['state_native_delta']['omega']['vector_RMS'], self15_logit_component_RMSE_vs_initial=s['self15_initial_drift']['logit']['component_RMSE'], self15_mean_sum_sigmoid=s['self15_initial_drift']['mean_sum_sigmoid'])
    print(json.dumps({'report': pin(out/'report.json'), 'ALL': preview}, ensure_ascii=False))


if __name__ == '__main__':
    main()
