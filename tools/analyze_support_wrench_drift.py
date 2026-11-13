"""Saved actual contact wrench versus original nominal dynamic demand clocks."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin


def main(args):
    if args.output.exists():
        raise ValueError('Fresh saved wrench diagnostic')
    r = json.loads(args.report.read_text())
    review = json.loads(args.review.read_text())
    if verify_file_pin(review['run']).resolve() != args.report.resolve() or review['passed'] != review['total']:
        raise ValueError('Reviewed execution required')
    p = json.loads(verify_file_pin(r['proposal']).read_text())
    original = json.loads(verify_file_pin(p['run']).read_text())
    target_report = json.loads(verify_file_pin(original['dynamic_wrench']).read_text())
    targets = np.load(verify_file_pin(target_report['targets']))
    qref = np.load(verify_file_pin(p['clip']))['reference_qpos']
    selected = np.load(verify_file_pin(p['artifacts']['selected']))
    efc = json.loads(verify_file_pin(review['EFC_review']).read_text())
    args.output.mkdir()
    rows = []
    for c in r['cases']:
        z = np.load(verify_file_pin(c['trace']))
        e = next(x for x in efc['cases'] if x['dt'] == c['dt'])
        w = np.load(verify_file_pin(e['wrenches']))
        t = np.arange(len(z['pre_qpos'])) * c['dt']
        demand_world = np.stack([np.interp(t, targets['time_s'], targets[key][:, j])
                                  for key in ['force_world', 'torque_COM_world'] for j in range(3)], axis=1)
        qa = c['model_layout']['object_qposadr']
        R = Rotation.from_quat(z['pre_qpos'][:, qa + 3:qa + 7][:, [1, 2, 3, 0]]).as_matrix()
        demand_BODY = np.c_[np.einsum('nji,nj->ni', R, demand_world[:, :3]),
                            np.einsum('nji,nj->ni', R, demand_world[:, 3:])]
        actual = w['object_BODY_COM_wrench']
        delta = actual - demand_BODY
        force = np.linalg.norm(delta[:, :3], axis=1)
        torque = np.linalg.norm(delta[:, 3:], axis=1)
        bad = np.flatnonzero((force > .03) | (torque > .0015))
        low = np.flatnonzero(z['pre_hand_normal_N'] < .1)
        path = args.output / f'wrench_drift_{c["dt"]:g}.npz'
        np.savez_compressed(path, pre_time_s=t, actual_wrench_BODY=actual, nominal_demand_actual_BODY=demand_BODY,
                            force_error_N=force, torque_error_Nm=torque, hand_normal_N=z['pre_hand_normal_N'])
        rows.append(dict(dt=c['dt'], initial_execution_force_error_N=float(force[0]),
            initial_execution_torque_error_Nm=float(torque[0]),
            first_outside_proposal_tolerance_s=None if not len(bad) else float(t[bad[0]]),
            first_hand_normal_below_0_1N_s=None if not len(low) else float(t[low[0]]),
            first_control_shift_from_search_rad=float(abs(z['post_ctrl'][0, selected['aids']] - selected['ctrl20']).max()),
            terminal_duration_s=c['summary']['metrics']['actual_duration_s'], artifact=file_pin(path)))
    result = dict(schema='support_pen_saved_wrench_drift:1', source=file_pin(__file__), run=file_pin(args.report),
        review=file_pin(args.review), nominal_targets=file_pin(verify_file_pin(target_report['targets'])),
        rows=rows, neural_updates=0, simulation_integration_steps=0,
        scope='Actual PRE-step EFC wrench compared with original nominal WORLD force/moment demand rotated into the ACTUAL object BODY, interpolated on original pre-step clock. Conditional inertia/derivatives, no feedback correction, measured force supervision or causal attribution. First action uses original reference phase1 rather than search ctrl phase0.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
