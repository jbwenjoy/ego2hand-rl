"""Small synthetic NumPy alignment fixtures; no actual dense output access."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT/'src/full151_gradient/dense_packet_alignment_saved_math_v1.py'
OUT = ROOT/'outputs/full151105_dense_packet_alignment_saved_math_preparation_v1'


def pin(p):
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False)
    pins = {'math': pin(MATH), 'checker': pin(Path(__file__))}
    for name, p in [('math', MATH), ('checker', Path(__file__))]:
        ast.parse(p.read_bytes()); (OUT/(name+'_before_execution.py')).write_bytes(p.read_bytes())
    spec = importlib.util.spec_from_file_location('dense_saved_math', MATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    x = np.zeros((2, 58), dtype='f4'); pp = np.array([[1, 2, 3], [2, 3, 4]], dtype='f4')
    R = np.broadcast_to(np.eye(3, dtype='f4'), (2, 3, 3)).copy()
    pos = pp+np.array([.1, .2, .3], dtype='f4'); obj = R.copy()
    x[:, 40:43] = pos-pp; x[:, 43:52] = obj.reshape(2, 9)
    d = m.alignment(x, pp, R, pos, obj, 2)
    checks = {'identity_frame_allrows_pass': d['alignment_all_passed'],
        'original8eps_component_arrays_preserved': d['old_flags'].shape == (2, 12) and d['reference_pose12_F64'].dtype == np.float64}
    wrong = pp.copy(); wrong[1, 0] += .1
    checks['wrong_palm_frame_rejected'] = not m.alignment(x, wrong, R, pos, obj, 2)['alignment_all_passed']
    wrongR = obj.copy(); wrongR[0] = np.diag([-1, -1, 1])
    checks['wrong_orientation_rejected'] = not m.alignment(x, pp, R, pos, wrongR, 2)['alignment_all_passed']
    try:
        m.alignment(x[:1], pp, R, pos, obj, 2)
        checks['missing_row_rejected_not_dropped'] = False
    except ValueError:
        checks['missing_row_rejected_not_dropped'] = True
    checks['no_backend_native_model_imports'] = not any(k in sys.modules for k in ('torch', 'warp', 'mujoco', 'scipy', 'dex_supermimic'))
    r = {'schema': 'full151105_dense_packet_alignment_saved_math_preparation:1', 'source_pins': pins,
        'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'synthetic_only': True, 'actual_dense_reads': 0, 'native_state_FK_MJB_model_GPU_calls': 0,
        'actual_dense_adapter': 'PENDING_PRODUCER_V2_FROZEN_ABI'}
    (OUT/'report.json').write_text(json.dumps(r, indent=2))
    ready = {'source_pins': pins, 'cpu_report': pin(OUT/'report.json'), 'cpu_checks': {'passed': r['passed'], 'total': r['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready, indent=2))
    print(json.dumps({'READY': pin(OUT/'READY.json'), **ready}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
