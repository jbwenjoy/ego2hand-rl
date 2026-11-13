"""Independent intervention/control/rotation/envelope and EFC readback."""
from pathlib import Path
import argparse
import json
import sys
import tomllib
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.warp_saved_review import review_saved_warp_contacts


def main(args):
    if args.output.exists():
        raise ValueError('Fresh prospective envelope review')
    r = json.loads(args.report.read_text())
    for pin in r['sources'] + r['task_sources']:
        verify_file_pin(pin)
    cfg = tomllib.loads(verify_file_pin(r['config']).read_text())
    p = json.loads(verify_file_pin(r['proposal']).read_text())
    selected = np.load(verify_file_pin(p['artifacts']['selected']))
    qref = np.load(verify_file_pin(p['clip']))['reference_qpos']
    profile = np.load(verify_file_pin(r['profiles']))
    directions = np.sin(np.pi * (np.arange(20) + .5)[None] * np.arange(1, 5)[:, None] / 20)
    directions /= np.max(abs(directions), axis=1, keepdims=True)
    names = ['BASELINE'] + [f'D{i}_{sign}' for i in range(1, 5) for sign in ['PLUS', 'MINUS']]
    expected = {(name, dt) for name in names for dt in cfg['dt']}
    args.output.mkdir()
    if args.efc_report is None:
        efc = review_saved_warp_contacts(args.report, args.output / 'EFC', expected)
        efc_path = args.output / 'EFC/report.json'
    else:
        efc_path = args.efc_report.resolve()
        efc = json.loads(efc_path.read_text())
        verify_file_pin(efc['source'])
        if verify_file_pin(efc['run']).resolve() != args.report.resolve() or {(c['name'],c['dt']) for c in efc['cases']} != expected or len(efc['cases']) != len(expected):
            raise ValueError('Exact existing EFC receipt cohort required')
    cases = []
    for c in r['cases']:
        index = names.index(c['name'])
        offset = np.zeros(20) if index == 0 else directions[(index - 1) // 2] * .04 * (1 if (index - 1) % 2 == 0 else -1)
        z = np.load(verify_file_pin(c['trace']))
        f = np.load(verify_file_pin(c['initialization']['prospective_forecast']))
        band = np.load(verify_file_pin(c['forecast_validation']['envelope']))
        raw = (2 * selected['delta20']).astype(np.float32)
        perturb = offset.astype(np.float32)
        expected_raw = np.tile(raw, (len(z['actions20']), 1))
        expected_raw[:5] += perturb
        target = []
        memory = selected['delta20'].astype(np.float32)
        for phase, action in zip(z['pre_action_phase'], expected_raw):
            memory = .5 * np.clip(.5 * action, -.6, .6) + .5 * memory
            target.append(qref[int(phase)] + memory)
        n = round(.1 / c['dt'])
        ticks = np.arange(1, n + 1) * c['dt']
        qa = c['model_layout']['object_qposadr']
        reference = Slerp(z['time_s'], Rotation.from_quat(z['quaternion_wxyz'][:, [1, 2, 3, 0]]))(ticks)
        pred = Rotation.from_quat(f['qpos'][:, :, qa + 3:qa + 7].reshape(-1, 4)[:, [1, 2, 3, 0]])
        tiled_reference = Rotation.from_quat(np.repeat(reference.as_quat(), 128, axis=0))
        plog = (pred * tiled_reference.inv()).as_rotvec().reshape(n, 128, 3)
        actual = Rotation.from_quat(z['post_qpos'][:n, qa + 3:qa + 7][:, [1, 2, 3, 0]])
        alog = (actual * reference.inv()).as_rotvec()
        predicted = np.concatenate([f['qpos'][:, :, qa:qa + 3], plog], axis=2)
        observed = np.concatenate([z['post_qpos'][:n, qa:qa + 3], alog], axis=1)
        margin = np.array([.0005] * 3 + [.01] * 3)
        lo, hi = predicted.min(axis=1), predicted.max(axis=1)
        error = max(float(abs(lo - band['minimum']).max()), float(abs(hi - band['maximum']).max()),
                    float(abs(observed - band['actual_position_rotation_log']).max()))
        covered = ((observed >= lo - margin) & (observed <= hi + margin)).all(axis=1)
        # Exact saved-value raw boundaries avoid claiming significance for
        # floating zero-width SO3-envelope endpoint roundoff.
        literal_raw = ((band['actual_position_rotation_log'] >= band['minimum']) &
                       (band['actual_position_rotation_log'] <= band['maximum'])).all(axis=1)
        checks = dict(profile_directions=np.array_equal(profile['directions'], directions),
            profile_offset=np.array_equal(profile['raw_offsets'][index], offset),
            role=c['role'] == ('CALIBRATION' if index <= 2 else 'CHECK') and c['supervision'] == 'NONEXPERT_NOT_FIT',
            original_raw_actions=np.array_equal(expected_raw, z['actions20']),
            original_EMA_PD=float(abs(np.repeat(target, round(.02 / c['dt']), axis=0) - z['post_ctrl'][:, selected['aids']]).max()) < 2e-7,
            original_phase=np.array_equal(z['pre_action_phase'], np.arange(1, len(z['actions20']) + 1)),
            selected_initial_pose=np.array_equal(z['initial_qpos'], selected['qpos'].astype(np.float32).astype(float)),
            complete_forecast=f['qpos'].shape == (n, 128, 27),
            forecast_actual_control=float(abs(f['ctrl'] - z['post_ctrl'][:n, None]).max()) < 2e-7,
            independent_rotation_and_bounds=error < 1e-10,
            fixed_margins=np.array_equal(margin, band['margin']),
            literal_raw_coverage=np.array_equal(literal_raw, band['raw_covered']),
            margin_coverage=np.array_equal(covered, band['margin_covered']) and bool(np.all(covered)),
            original_first_terminal=not np.any(z['terminated_truncated'][:-1]) and bool(z['terminated_truncated'][-1, 0]),
            no_aid=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
            failed_task_preserved=not c['summary']['qualified'],
            capacity=not np.any(f['overflow']) and np.max(f['nacon']) <= 16384 and np.max(f['nefc']) <= 512)
        cases.append(dict(name=c['name'], dt=c['dt'], checks={k: bool(v) for k, v in checks.items()},
                          maximum_frame_readback_error=error))
    result = dict(schema='pen_action_envelope_saved_review:1', run=file_pin(args.report), source=file_pin(__file__),
        cases=cases, passed=sum(sum(c['checks'].values()) for c in cases), total=sum(len(c['checks']) for c in cases),
        EFC_review=file_pin(efc_path), EFC_checks=[efc['passed'], efc['total']],
        combined_review_passed=all(all(c['checks'].values()) for c in cases) and efc['passed']==efc['total'],
        neural_updates=0, simulation_integration_steps=0,
        scope='Independent fixed actions/roles, EMA/PD, world SO3-log envelopes, margins, capacity/no-aid and actual EFC saved-data checks. Passing envelope coverage is not deterministic equivalence, a calibrated probability, planning readiness or successful manipulation.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], total=result['total'], EFC=result['EFC_checks'])))
    if not result['combined_review_passed']:
        raise ValueError('Preserve failed action/envelope readback')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--efc-report', type=Path)
    main(p.parse_args())
