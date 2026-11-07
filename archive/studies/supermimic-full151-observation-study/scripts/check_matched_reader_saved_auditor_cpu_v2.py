"""Small synthetic saved-ABI fixtures only; no actual replay output reads."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT/'scripts/audit_matched_native_GPU_reader_saved_cpu_v2.py'
MATH = ROOT/'src/full151_gradient/matched_reader_saved_math_v1.py'
OUT = ROOT/'outputs/full151105_matched_reader_saved_auditor_preparation_v2'


def pin(p):
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False)
    pins = {'auditor': pin(AUDITOR), 'math': pin(MATH), 'checker': pin(Path(__file__))}
    for name, p in [('auditor', AUDITOR), ('math', MATH), ('checker', Path(__file__))]:
        ast.parse(p.read_bytes()); (OUT/(name+'_source_before_execution.py')).write_bytes(p.read_bytes())
    (OUT/'source_freeze_before_fixture.json').write_text(json.dumps({'source_pins': pins, 'actual_replay_arrays_read': False}, indent=2))
    spec = importlib.util.spec_from_file_location('independent_saved_auditor_fixture', AUDITOR)
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a); m = a.math_module()
    x = np.zeros((3, 58), dtype='f4'); y = x.copy()
    def saved(u, v):
        d = m.old_rule(u, v, True)
        return {'actual': u, 'reference': v, **{k: d[k] for k in ('difference_F64', 'old_bound_F64', 'old_flags')}}
    good = saved(x, y)
    checks = {'positive_exact_full58_saved_fixture': a.rule_evidence(m, good, x, y, True)[0]}
    bad = {k: v.copy() for k, v in good.items()}; bad['old_flags'][0, 0] = False
    checks['reject_mutated_saved_flag'] = not a.rule_evidence(m, bad, x, y, True)[0]
    bad = {k: v.copy() for k, v in good.items()}; bad['difference_F64'][0, 0] = 1
    checks['reject_mutated_saved_difference'] = not a.rule_evidence(m, bad, x, y, True)[0]
    y = x.copy(); y[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    checks['q_ulp_requires_exact_bits'] = not m.old_rule(x, y, True)['all_passed']
    y = x.copy(); y[1, 57] = np.float32(2e-6)
    checks['angular_original8eps_failure_preserved'] = not m.old_rule(x, y, True)['all_passed']
    rng = {k: np.array([1, 2], dtype='u1') for k in ('Python_pickle', 'NumPy_pickle', 'Torch_CPU', 'Torch_CUDA_device0')}
    checks['raw_rng_equal_positive'] = a.raw_rng_same(rng, rng)
    altered = {k: v.copy() for k, v in rng.items()}; altered['Torch_CUDA_device0'][0] = 3
    checks['raw_rng_mutation_rejected'] = not a.raw_rng_same(rng, altered)
    u = np.zeros((2260, 58), dtype='f4'); v = u.copy(); u[2259, 40] = np.float32(.01)
    delta, description = m.all2260_gpu_cpu_description(u, v)
    checks['all2260_delta_preserved_per_unit_descriptive'] = delta.shape == (2260, 58) and delta[2259, 40] == float(np.float32(.01)) and description['p_m']['FP32_byte_different_elements'] == 1
    mapping = np.arange(20, dtype='i4')
    checks['MJB_int32_to_native_long_preserves_values'] = a.native_long_index(mapping, 27).dtype == np.int64 and np.array_equal(a.native_long_index(mapping, 27), mapping)
    for label, badmap in [('bool', np.ones(20, dtype=bool)), ('outofrange', np.arange(20, 40, dtype='i4'))]:
        try:
            a.native_long_index(badmap, 27)
            checks['reject_native_mapping_'+label] = False
        except ValueError:
            checks['reject_native_mapping_'+label] = True
    checks['no_backend_model_or_converter_imports'] = not any(k in sys.modules for k in ('torch', 'warp', 'mujoco', 'scipy', 'dex_supermimic'))
    checks['frozen_sources_current_after_fixture'] = all(pin(p)['sha256'] == pins[k]['sha256'] for k, p in [('auditor', AUDITOR), ('math', MATH), ('checker', Path(__file__))])
    report = {'schema': 'full151105_matched_reader_saved_auditor_preparation:1', 'source_pins': pins,
        'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'synthetic_only': True, 'actual_replay_outputs_read': False, 'native_Reader_calls': 0,
        'models': 0, 'Env': False, 'physics_steps': 0, 'GPU': False}
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    ready = {'source_pins': pins, 'cpu_report': pin(OUT/'report.json'), 'cpu_checks': {'passed': report['passed'], 'total': report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready, indent=2))
    print(json.dumps({'READY': pin(OUT/'READY.json'), **ready}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
