"""Independently recompute saved model/clone integrity and failed100ms gates."""
from pathlib import Path
import argparse
import json
import sys
import tomllib
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin


def metric_errors(q, v, targetq, targetv, qids, dofs, qa, da):
    R = Rotation.from_quat(q[:, :, qa + 3:qa + 7].reshape(-1, 4)[:, [1, 2, 3, 0]])
    Rt = Rotation.from_quat(np.broadcast_to(targetq[:, :, qa + 3:qa + 7], (*q.shape[:2], 4)).reshape(-1, 4)[:, [1, 2, 3, 0]])
    omega = R.apply(v[:, :, da + 3:da + 6].reshape(-1, 3))
    omegat = Rt.apply(np.broadcast_to(targetv[:, :, da + 3:da + 6], (*v.shape[:2], 3)).reshape(-1, 3).copy())
    return dict(hand_position_rad=float(abs(q[:, :, qids] - targetq[:, :, qids]).max()),
        object_position_m=float(np.linalg.norm(q[:, :, qa:qa + 3] - targetq[:, :, qa:qa + 3], axis=-1).max()),
        object_rotation_rad=float((R * Rt.inv()).magnitude().max()),
        hand_velocity_rad_s=float(abs(v[:, :, dofs] - targetv[:, :, dofs]).max()),
        object_velocity_m_s=float(np.linalg.norm(v[:, :, da:da + 3] - targetv[:, :, da:da + 3], axis=-1).max()),
        object_omega_rad_s=float(np.linalg.norm(omega - omegat, axis=1).max()))


def main(args):
    if args.output.exists():
        raise ValueError('Fresh saved repeatability review')
    r = json.loads(args.report.read_text())
    for pin in r['sources']:
        verify_file_pin(pin)
    cfg = tomllib.loads(verify_file_pin(r['config']).read_text())
    gates = tomllib.loads(verify_file_pin(r['gates']).read_text())
    proposal = json.loads(verify_file_pin(r['proposal']).read_text())
    selected = np.load(verify_file_pin(proposal['artifacts']['selected']))
    expected = {(dt, binding, repeat) for dt in cfg['dt'] for binding in cfg['bindings'] for repeat in range(cfg['repeats'])}
    if {(p['dt'], p['binding'], p['repeat']) for p in r['probes']} != expected or len(r['probes']) != len(expected):
        raise ValueError('Exact preregistered repeatability cohort')
    items = []
    for p in r['probes']:
        z = np.load(verify_file_pin(p['trace']))
        case = next(c for c in r['cases'] if c['dt'] == p['dt'])
        actual = np.load(verify_file_pin(case['trace']))
        T = round(.1 / p['dt'])
        qa, da = case['model_layout']['object_qposadr'], case['model_layout']['object_dofadr']
        qids, dofs = selected['qids'], selected['dofs']
        errors = metric_errors(z['qpos'], z['qvel'], actual['post_qpos'][:T, None], actual['post_qvel'][:T, None], qids, dofs, qa, da)
        spread = metric_errors(z['qpos'], z['qvel'], z['qpos'][:, :1], z['qvel'][:, :1], qids, dofs, qa, da)
        first = next(x for x in r['probes'] if x['dt'] == p['dt'] and x['binding'] == p['binding'] and x['repeat'] == 0)
        firstz = np.load(verify_file_pin(first['trace']))
        repeated = metric_errors(z['qpos'][:, :1], z['qvel'][:, :1], firstz['qpos'][:, :1], firstz['qvel'][:, :1], qids, dofs, qa, da)
        merror = max(abs(errors[k] - p['actual_errors'][k]) for k in errors)
        serror = max(abs(spread[k] - p['identical_world_spread'][k]) for k in spread)
        rerror = max(abs(repeated[k] - p['first_world_repeat_error'][k]) for k in repeated)
        reported_gates = {key: value <= gates[key] for key, value in errors.items()}
        model_arrays = np.load(verify_file_pin(p['inventory']['snapshot']))
        scalars = json.loads(verify_file_pin(p['inventory']['scalars']).read_text())
        diagnostic = z['capacity_aid_mocap']
        checks = dict(exact_population_shape=z['qpos'].shape == (T, 128, 27),
            original_initial_pose=np.array_equal(z['initial_qpos'], np.tile(actual['initial_qpos'], (128, 1))),
            original_initial_velocity=np.array_equal(z['initial_qvel'], np.tile(actual['initial_qvel'], (128, 1))),
            initial_memory=np.array_equal(z['initial_ema'], np.tile(selected['delta20'].astype(np.float32), (128, 1))),
            initial_warm_time=not np.any(z['initial_qacc_warmstart']) and not np.any(z['initial_time']),
            original_control=float(abs(z['ctrl'] - actual['post_ctrl'][:T, None]).max()) <= gates['control_rad'],
            original_clock=float(abs(z['time'][:, :, 0] - actual['post_GPU_time'][:T, None]).max()) < 1e-6,
            no_capacity_overflow=np.max(diagnostic[:, :, 0]) <= 128 * 128 and not np.any(diagnostic[:, :, 1]) and np.max(z['nefc']) <= 512,
            no_aid=not np.any(diagnostic[:, :, 2:4]), fixed_mocap=not np.any(diagnostic[:, :, 4:]),
            model_snapshot_layout=len(model_arrays.files) == p['inventory']['array_fields'] and len(scalars) == p['inventory']['scalar_fields'],
            finite_snapshot=all(np.isfinite(model_arrays[k]).all() for k in model_arrays.files),
            errors_reconstructed=merror < 1e-7, spread_reconstructed=serror < 1e-7, repeats_reconstructed=rerror < 1e-7,
            unchanged_failed_gates=reported_gates == p['fidelity'] and not all(reported_gates.values()))
        items.append(dict(dt=p['dt'], binding=p['binding'], repeat=p['repeat'], checks={k: bool(v) for k, v in checks.items()},
                          max_metric_readback_error=max(merror, serror, rerror), fidelity=reported_gates))
    args.output.mkdir()
    result = dict(schema='pen_repeatability_saved_review:1', source=file_pin(__file__), run=file_pin(args.report),
        cases=items, passed=sum(sum(p['checks'].values()) for p in items), total=sum(len(p['checks']) for p in items),
        neural_updates=0, simulation_integration_steps=0,
        scope='Independent saved state/clock/control/capacity/no-aid/mocap and rotation/frame/spread/fidelity metric readback. Full-model device restoration hash is a runtime check, not independently rerun here. Passing review confirms the failed100ms fidelity experiment, not successful planning.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], total=result['total'])))
    if result['passed'] != result['total']:
        raise ValueError('Preserve failed saved repeatability readback')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
