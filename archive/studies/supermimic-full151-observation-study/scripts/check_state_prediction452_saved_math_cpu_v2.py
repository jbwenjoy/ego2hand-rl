"""Small synthetic-only error/ordering fixtures. No actual model predictions."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT/'src/full151_gradient/state_prediction452_saved_math_v1.py'
OUT = ROOT/'outputs/full151105_state_prediction452_saved_math_preparation_v2'


def pin(p):
    return {'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False)
    pins = {'math': pin(PATH), 'checker': pin(Path(__file__))}
    for name, p in [('math', PATH), ('checker', Path(__file__))]:
        ast.parse(p.read_bytes()); (OUT/(name+'_source_before_execution.py')).write_bytes(p.read_bytes())
    (OUT/'predata_freeze.json').write_text(json.dumps({'source_pins': pins, 'future_predictions_read': False}, indent=2))
    spec = importlib.util.spec_from_file_location('saved452math', PATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    identities = m.expected_identities(); x = np.zeros((452, 58), dtype='f4'); y = x.copy()
    z = m.all_statistics(x, y, identities)
    checks = {'exact452_300donor_24baseline_128direction_order': m.validate_identity_order(identities) and len([r for r in identities if r['mode'] == 'donor']) == 300,
        'zero_error_all_units': all(v['component_RMSE'] == 0 for v in z['statistics']['all452']['blocks'].values()),
        'terminal149_and_no_u150_retained': sum(r['k'] == 149 for r in identities) == 40 and all(r['terminal_observation'] == 150 and not r['valid_next_policy_decision'] for r in identities if r['k'] == 149)}
    x[:, :20] = np.float32(7); x[:, 40:43] = np.float32(2); x[:, 43:52] = np.float32(3)
    z = m.all_statistics(x, y, identities)
    blocks = z['statistics']['all452']['blocks']
    checks['q_not_wrapped_or_mixed_with_rotation'] = blocks['q']['component_RMSE'] == 7 and blocks['R']['component_RMSE'] == 3
    checks['p_vector_vs_component_RMS_explicit_units'] = blocks['p']['component_RMSE'] == 2 and blocks['p']['vector_RMS'] == np.sqrt(12)
    checks['R_only9_raw_elements_no_angle_projection'] = blocks['R']['Frobenius_RMS'] == 9 and set(blocks['R']) == {'start','stop_exclusive','component_count','units','component_RMSE','mean_absolute','maximum_absolute','Frobenius_RMS'}
    checks['signed_subtraction_promoted_before_operation'] = z['signed_error452x58_F64'].dtype == np.float64 and z['signed_error452x58_F64'][0,0] == 7
    changed = copy.deepcopy(identities); changed[0], changed[1] = changed[1], changed[0]
    try:
        m.validate_identity_order(changed); checks['reordered_transition_rejected'] = False
    except ValueError:
        checks['reordered_transition_rejected'] = True
    u = np.zeros((452, 20), dtype='f4'); raw = np.broadcast_to(u[:, None, :], (452, 4, 20)).copy()
    checks['all_four_absolute_actualu_held'] = m.held_controls_exact(raw, u)
    raw[400,3,19] = np.nextafter(np.float32(0),np.float32(1))
    checks['one_substep_u_ulp_difference_rejected'] = not m.held_controls_exact(raw, u)
    checks['no_model_native_geometry_or_backend_import'] = not any(k in sys.modules for k in ('torch','warp','mujoco','scipy','dex_supermimic'))
    checks = {k:bool(v) for k,v in checks.items()}
    r = {'schema':'full151105_state_prediction452_saved_math_preparation:1','source_pins':pins,
        'passed':sum(checks.values()),'total':len(checks),'checks':checks,'synthetic_only':True,
        'future_prediction_reads':0,'model_Torch_loader_FK_MJB_GPU_calls':0,'taskscore_goal_AD':0}
    (OUT/'report.json').write_text(json.dumps(r,indent=2))
    ready = {'source_pins':pins,'cpu_report':pin(OUT/'report.json'),'cpu_checks':{'passed':r['passed'],'total':r['total']}}
    (OUT/'READY.json').write_text(json.dumps(ready,indent=2))
    print(json.dumps({'READY':pin(OUT/'READY.json'),**ready},indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
