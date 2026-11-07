"""Independent small saved-row fixtures; no production/dense data imported."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT/'src/full151_gradient/dense_saved_rows_v1.py'
MATH = ROOT/'src/full151_gradient/observability_saved_math_v1.py'
OUT = ROOT/'outputs/full151105_dense_saved_rows_preparation_v1'


def pin(p):
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main():
    OUT.mkdir(exist_ok=False)
    pins = {'rows': pin(ROWS), 'math': pin(MATH), 'checker': pin(Path(__file__))}
    for key, p in [('rows', ROWS), ('math', MATH), ('checker', Path(__file__))]:
        ast.parse(p.read_bytes()); (OUT/(key+'_source_before_execution.py')).write_bytes(p.read_bytes())
    (OUT/'predata_freeze.json').write_text(json.dumps({'source_pins': pins, 'actual_dense_read': False}, indent=2))
    rows, math = module('dense_row_math', ROWS), module('independent_proxy_math', MATH)
    pairsets = [[([1, 9], -.004)], [([1, 9], -.004), ([9, 2], -.004), ([1, 2], -.1), ([8, 9], -.2)], [([1, 9], -.002)], []]
    samples = []
    for i, pairs in enumerate(pairsets):
        contacts = [{'geom': g, 'dist': d, 'force_normal': 0.} for g, d in pairs]
        samples.append({'path': 'A0', 'mode': 'donor', 'phase': 0, 'branch_id': None, 'substep': i+1,
            'canonical_pre_tick': i, 'source_frame': 200, 'nacon': len(pairs), 'nefc': 0, 'contacts': contacts,
            'raw_wp_contact': {'geom': [g for g, d in pairs], 'worldid': [0]*len(pairs), 'type': [0]*len(pairs), 'efc_address': [-1]*len(pairs)}})
    ids = np.array([1, 1], dtype='i8')
    sdf = np.array([[-.002, .01], [-.002, .01], [-.004, .01], [-.003, .01]], dtype='f4')
    p = math.proxy_from_saved_sdf(sdf, ids)
    score = {'sampled_sdf_m': sdf, 'P': p['P'], 'BP': p['BP'], 'minimum_sdf_m': p['min_sdf_m'], 'geom_ids': p['geom_ids'], 'per_geom_min_sdf_m': p['pergeom_min_sdf_m']}
    result = rows.reconstruct_rows(samples, score, ids, {1, 2}, {9}, 10, math)
    r = result['rows']; stats = result['statistics']['all1808']
    checks = {'source_exact_captured_P_BP_and_min_pergeom': result['captured_proxy_arrays_match'],
        'HO_not_self_or_other_deepest_no_force_filter': r[1]['native_generated_HO_depth_m'] == .004 and r[0]['native_HO_depth_gt3mm'],
        'primary_all_exact_ties_preserved': len(r[1]['all_exact_tied_deepest_pairs']) == 2,
        'secondary_allviolating_pairs_overlap_primary': len(r[1]['all_violating_pairs_SECONDARY']) == 2,
        'coverage_gap_and_undersensitive_nonexclusive': r[1]['primary_deepest_tie_coverage']['geom_set_gap_ids'] == [2] and r[1]['primary_deepest_tie_coverage']['covered_proxy_undersensitivity_ids'] == [1],
        'empty_generated_HO_not_missing_or_positive_gap_certificate': r[3]['native_generated_HO_count'] == 0 and r[3]['native_generated_HO_depth_m'] == 0,
        'FP32_threshold_equality_retained': r[3]['proxy_min_sdf_exactly_minus_FP32_allowance'] and not r[3]['proxy_P_positive'],
        'wholecohort_confusion_no_dropped_rows': stats['rows'] == 4 and stats['native_positive_proxy_zero'] == 2 and stats['proxy_alert_without_nativeHO3mm'] == 1 and stats['both_nonpositive'] == 1}
    changed = {k: v.copy() for k, v in score.items()}; changed['P'][0] = 1
    checks['mutated_P_array_rejected_numeric_not_bool_trusted'] = not rows.reconstruct_rows(samples, changed, ids, {1, 2}, {9}, 10, math)['captured_proxy_arrays_match']
    changed_samples = copy.deepcopy(samples); changed_samples[0]['raw_wp_contact']['geom'][0][0] = 100
    try:
        rows.reconstruct_rows(changed_samples, score, ids, {1, 2}, {9}, 10, math)
        checks['unknown_native_geom_rejected'] = False
    except ValueError:
        checks['unknown_native_geom_rejected'] = True
    flat = np.broadcast_to(np.eye(3, dtype='f4').reshape(1, 9), (4, 9)).copy()
    checks['strict_saved_flat9_lossless_logical3x3'] = rows.logical_world_rotation_from_saved_flat9(flat, 4).shape == (4, 3, 3) and rows.logical_world_rotation_from_saved_flat9(flat, 4).tobytes() == flat.tobytes()
    try:
        rows.logical_world_rotation_from_saved_flat9(flat.astype('f8'), 4)
        checks['wrong_stored_dtype_rejected_not_repaired'] = False
    except ValueError:
        checks['wrong_stored_dtype_rejected_not_repaired'] = True
    checks['no_backend_model_or_native_imports'] = not any(k in sys.modules for k in ('torch', 'warp', 'mujoco', 'scipy', 'dex_supermimic'))
    report = {'schema': 'full151105_dense_saved_rows_preparation:1', 'source_pins': pins, 'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'synthetic_only': True, 'actual_dense_reads': 0, 'factory_MJB_FK_WM_GPU_calls': 0, 'file_adapter': 'PENDING_REV4_AND_FROZEN_PRODUCER_ABI'}
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    ready = {'source_pins': pins, 'cpu_report': pin(OUT/'report.json'), 'cpu_checks': {'passed': report['passed'], 'total': report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready, indent=2))
    print(json.dumps({'READY': pin(OUT/'READY.json'), **ready}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
