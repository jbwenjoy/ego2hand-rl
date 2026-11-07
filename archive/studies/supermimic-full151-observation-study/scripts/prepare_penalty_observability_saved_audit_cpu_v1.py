"""Small synthetic-only algebra preparation; never opens dense result arrays."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT/'src/full151_gradient/observability_saved_math_v1.py'
PROTOCOL = ROOT/'protocols/full151105_solver_phase_penalty_observability_CPU_prereg_v2.json'
OUT = ROOT/'outputs/full151105_observability_saved_auditor_preparation_v1'


def pin(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    OUT.mkdir(exist_ok=False)
    pins = {'math': pin(MATH), 'checker': pin(Path(__file__)), 'preregistration': pin(PROTOCOL)}
    for name, path in [('math', MATH), ('checker', Path(__file__))]:
        ast.parse(path.read_bytes())
        (OUT/(name+'_source_before_execution.py')).write_bytes(path.read_bytes())
    checklist = {
        'converter_precedes_scoring': ['1808 PRE+452 endpoint rows=2260;160firstanchor comparisons reuse pre rows',
            'q20/qd20 exact FP32 bits; other18 arithmetic8eps32; all1808 solverpose same-preframe',
            'all452 same-reader palm positions/matrices authoritative, constant perpath, native fixed-root mapping bound',
            'native root body ID != mocap ID; proxy sample_body_ids local FK IDs != native body IDs',
            'body angular velocity -> world -> reader palm, original quat normalization, qvel and post/pre timeline preserved',
            'all mapping/shape/finite/clock/actualu/source/capacity gates pass before 1factory or any proxy batch'],
        'saved_scoring': ['1808rows original order;15batches14x128+16;1factory;0WM/Actor/AD',
            'all1808x1344 saved SDF -> FP32 P, -.1P, rowmin/pergeommin, samplegeomcoverage',
            'native HO only exact enabled geom masks/WORLD0; include zero-force contacts; no all-category depth substitution',
            'allnative>3mm pairs saved; PRIMARY exact deepest ties, SECONDARY all violating pairs overlapping evidence',
            'P>0 and sourceFP32 min<-.003 separately, exact threshold equality retained; no new noise/tolerance threshold',
            'allrows and path/mode confusion/continuous values; emptygeneratedHO observed0 != geometric clearance'],
        'scope': ['no exclusive miss cause; no falsepositive claim from missing generated contacts',
            'no physicalgradient/wholehidden/contactforce/independentN or methodbenefit qualification',
            'buffers/weights/RNG unchanged are source-bound unless actual saved raw states independently available']}
    write(OUT/'equations_frozen_before_synthetic.json', {'source_pins': pins,
        'actual_dense_outputs_read': False, 'checklist': checklist})
    spec = importlib.util.spec_from_file_location('independent_saved_math', MATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    checks = {}
    checks['count_schedule'] = m.COUNTS['solver_pre']+m.COUNTS['post_endpoints'] == 2260 and 14*128+16 == 1808
    sdf = np.array([[-.002, -.004, .01], [-.003, .1, .2]], dtype='f4')
    proxy = m.proxy_from_saved_sdf(sdf, np.array([1, 2, 2]))
    checks['FP32_penalty_boundary_and_geommin'] = bool(np.array_equal(proxy['P_positive'], [True, False]) and proxy['source_F32_min_equal_allowance'][1] and proxy['pergeom_min_sdf_m'].shape == (2, 2))
    native = m.native_ho_witnesses(np.array([[1, 9], [9, 2], [1, 2], [8, 9]]), np.array([-.004, -.004, -.02, -.03]), np.zeros(4, dtype='i8'), {1, 2}, {9}, range(10))
    checks['native_exact_HO_excludes_self_environment_and_retains_ties'] = native['observed_generated_HO_depth_m'] == .004 and len(native['deepest_exact_tie_contact_indices']) == 2 and len(native['all_violating_pairs_secondary']) == 2
    labels = m.miss_labels([1, 2], np.array([1]), np.array([-.002], dtype='f4'))
    checks['mixed_tie_coverage_labels_overlap_not_exclusive'] = {r['label'] for r in labels} == {'GEOM_SET_COVERAGE_GAP', 'SAME_POSE_PROXY_UNDERSENSITIVITY_CANDIDATE'}
    empty = m.native_ho_witnesses(np.empty((0, 2), dtype='i8'), np.array([]), np.array([], dtype='i8'), {1}, {9}, range(10))
    checks['valid_empty_generated_contact_scope'] = empty['observed_generated_HO_depth_m'] == 0 and empty['generated_HO_contact_count'] == 0 and 'not' in empty['empty_means']
    try:
        m.native_ho_witnesses(np.array([[1, 99]]), np.array([0.]), np.array([0]), {1}, {9}, range(10))
        checks['unknown_geom_rejected_not_zero'] = False
    except ValueError:
        checks['unknown_geom_rejected_not_zero'] = True
    x = np.zeros((1, 58), dtype='f4'); y = x.copy(); y[0, 40] = np.finfo('f4').eps
    checks['arithmetic_pose_bound'] = m.state58_comparison(x, y)['other18_within_arithmetic']
    y = x.copy(); y[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    checks['q_ulp_changes_reject_exact_bits'] = not m.state58_comparison(x, y)['q_qd_exact_bits']
    checks['support_absent_not_high_score'] = bool(m.support_match_from_saved(np.zeros((1, 1, 1, 4)), np.ones((1, 1, 1, 4)))[0] == 0)
    checks['allrow_confusion_partition'] = sum(m.confusion(np.array([1, 0, 1, 0], bool), np.array([1, 1, 0, 0], bool)).values()) == 4
    checks['no_framework_geometry_or_model_modules'] = not bool({'torch', 'warp', 'mujoco', 'scipy'} & set(sys.modules))
    report = {'schema': 'full151105_observability_saved_algebra_preparation:1',
        'source_pins': pins, 'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'actual_dense_outputs_read': False, 'synthetic_only': True,
        'actual_converter_calls': 0, 'model_loads': 0, 'geometry_queries': 0, 'physics_steps': 0,
        'checklist': checklist, 'actual_saved_audit_status': 'PENDING_ROOT_TERMINAL_AND_FROZEN_OUTPUT_ABI'}
    write(OUT/'report.json', report)
    write(OUT/'READY.json', {'source_pins': pins, 'cpu_report': pin(OUT/'report.json'),
        'cpu_checks': {'passed': report['passed'], 'total': report['total']}, 'actual_audit': False})
    print(json.dumps({'report': pin(OUT/'report.json'), 'checks': checks, 'READY': pin(OUT/'READY.json')}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
