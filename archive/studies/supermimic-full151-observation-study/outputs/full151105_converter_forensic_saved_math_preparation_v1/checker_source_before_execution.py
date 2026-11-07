"""Freeze small independent forensic algebra; synthetic inputs only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'src/full151_gradient/converter_forensic_saved_math_v1.py'
OUT = ROOT/'outputs/full151105_converter_forensic_saved_math_preparation_v1'


def pin(p):
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False)
    sources = {'math': pin(PATH), 'checker': pin(Path(__file__))}
    for label, p in [('math', PATH), ('checker', Path(__file__))]:
        ast.parse(p.read_bytes()); (OUT/(label+'_source_before_execution.py')).write_bytes(p.read_bytes())
    spec = importlib.util.spec_from_file_location('forensic_saved_only', PATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    a = np.zeros((2, 58), dtype='f4'); b = a.copy()
    _, zero = m.state58_errors(a, b, 2)
    b[0, 55] = np.float32(2e-6)
    arrays, changed = m.state58_errors(a, b, 2)
    q = a.copy(); q[0, 0] = np.nextafter(np.float32(0), np.float32(1))
    _, qchange = m.state58_errors(a, q, 2)
    A = np.broadcast_to(np.eye(3), (2, 3, 3)); x = np.array([[1., 2., 3.], [-1., 2., 1.]])
    cancelled = m.transform_cancellation_inventory(A, x)
    checks = {'all_rows_zero_exact': zero['original_gate_qualified'],
        'retain_arithmetic_failure_no_relaxation': not changed['original_gate_qualified'] and arrays['original8eps32_pass_mask'].shape == (2, 58) and changed['blocks']['omega_reader_from_BODY']['failed_arithmetic_rows'] == 1,
        'retain_q_ulp_failure_even_within_arithmetic_bound': not qchange['q_qd_exact_original_rule'],
        'cancellation_is_diagnostic_not_new_bound': np.array_equal(cancelled['cancellation_amount_not_error_bound'], np.zeros((2, 3))),
        'expected_all_counts_no_proxy': m.EXPECTED['converted'] == m.EXPECTED['pre']+m.EXPECTED['endpoint'] == 2260 and m.EXPECTED['factory'] == m.EXPECTED['proxy_forwards'] == 0,
        'no_framework_model_or_native_imports': not bool({'torch', 'warp', 'mujoco', 'scipy'} & set(sys.modules))}
    report = {'schema': 'full151105_converter_forensic_saved_math_preparation:1', 'source_pins': sources,
        'passed': sum(checks.values()), 'total': len(checks), 'checks': checks, 'synthetic_only': True,
        'actual_new_forensic_outputs_read': False, 'actual_converter_calls': 0, 'models': 0, 'GPU': False,
        'old_case_gate': 'UNCHANGED_FAIL_RETAINED', 'actual_saved_adapter': 'PENDING_NEW_PRODUCER_ABI_AND_ROOT_TERMINAL'}
    with (OUT/'report.json').open('x') as f:json.dump(report, f, indent=2)
    ready = {'source_pins': sources, 'cpu_report': pin(OUT/'report.json'), 'cpu_checks': {'passed': report['passed'], 'total': report['total']}}
    with (OUT/'READY.json').open('x') as f:json.dump(ready, f, indent=2)
    print(json.dumps({'READY': pin(OUT/'READY.json'), **ready}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
