"""Small independent saved-adapter metadata/algebra fixture, no dense reads."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT/'scripts/audit_qualified_solver_phase_penalty_saved_cpu_v1.py'
OUT = ROOT/'outputs/full151105_qualified_dense_saved_auditor_preparation_v2'


def pin(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False)
    source = pin(AUDITOR)
    for label, path in [('auditor', AUDITOR), ('checker', Path(__file__))]:
        ast.parse(path.read_bytes()); (OUT/(label+'_source_before_execution.py')).write_bytes(path.read_bytes())
    (OUT/'pre_fixture_freeze.json').write_text(json.dumps({'auditor': source, 'actual_dense_arrays_read': False}, indent=2))
    spec = importlib.util.spec_from_file_location('qualified_dense_saved_adapter', AUDITOR)
    a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
    ready = {'schema': 'full151105_qualified_dense_penalty_CPU_contract_READY:1',
        'status': 'PASS_SYNTHETIC_QUALIFIED_PACKET_ALIGNMENT_AND_HOOK_ONLY', 'cpu_checks': {'passed': 72, 'total': 72}}
    proof = {'schema': 'full151105_qualified_dense_penalty_CPU_contract_proof:1', 'status': ready['status'],
        'passed': 72, 'total': 72, 'checks': {'synthetic_'+str(i): True for i in range(72)}}
    checks = {'synthetic_metadata72_shape_positive_not_actualproof': a.CPU_ready_metadata_passed(ready, proof)}
    bad = copy.deepcopy(proof); bad['passed'] = True
    checks['native_integer_counter_not_bool'] = not a.CPU_ready_metadata_passed(ready, bad)
    bad = copy.deepcopy(proof); bad['checks']['synthetic_0'] = False
    checks['one_false_ready_check_rejected'] = not a.CPU_ready_metadata_passed(ready, bad)
    bad = copy.deepcopy(proof); bad['schema'] = 'old_wrong_schema'
    checks['wrong_proof_schema_rejected'] = not a.CPU_ready_metadata_passed(ready, bad)
    m = a.independent_modules()
    checks['all_four_frozen_independent_math_dependencies_bound'] = set(m) == {'proxy_math', 'rows', 'alignment', 'old_rule'}
    p = m['proxy_math'].proxy_from_saved_sdf(np.array([[-.004, .01]], dtype='f4'), np.array([1, 1], dtype='i8'))
    checks['proxy_P_BP_use_source_FP32_math'] = p['P'][0] > 0 and p['BP'][0] < 0 and p['min_sdf_m'][0] == np.float32(-.004)
    x = np.zeros((2, 58), dtype='f4'); R = np.broadcast_to(np.eye(3, dtype='f4'), (2, 3, 3)).copy(); x[:, 43:52] = R.reshape(2, 9)
    zeros = np.zeros((2, 3), dtype='f4')
    checks['alignment_adapter_exact_key_shapes'] = m['alignment'].alignment(x, zeros, R, zeros, R, 2)['old_flags'].shape == (2, 12)
    checks['no_backend_native_model_or_dense_loader_executed'] = not any(k in sys.modules for k in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'dex_supermimic', 'mjlab', 'scipy'))
    checks = {k: bool(v) for k, v in checks.items()}
    report = {'schema': 'full151105_qualified_dense_saved_auditor_preparation:1', 'source_pins': {'auditor': source, 'checker': pin(Path(__file__)),
            **{k: pin(path) for k, (path, _) in a.DEPENDENCIES.items()}},
        'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'actual_dense_reads': 0, 'native_State_MJB_FK_objective_WM_GPU_calls': 0,
        'synthetic_only': True, 'actual_scientific_saved_review': 'WAIT_ROOT_TERMINAL'}
    (OUT/'report.json').write_text(json.dumps(report, indent=2))
    ready = {'source_pins': report['source_pins'], 'cpu_report': pin(OUT/'report.json'), 'cpu_checks': {'passed': report['passed'], 'total': report['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready, indent=2))
    print(json.dumps({'READY': pin(OUT/'READY.json'), **ready}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
