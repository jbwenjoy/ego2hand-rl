"""Replay frozen clip_003 prefixes at W1 before handing control back to the parent Actor."""
import argparse
from pathlib import Path
import tomllib

import numpy as np

from evaluate_pen_dev_clip import b, config, evaluate_cases, file_pin, raw_prefix_array, read, verify_file_pin, write_json


ARMS = ('base', 'nominal', 'action_only', 'joint')
DEFAULT_ACTOR = b.ROOT/'runs/pen_scaled_train_v1/actor_02781.pt'


def load_selection(pin, projected):
    selection = read(pin)
    order = (*ARMS, 'joint_init_only') if 'joint_init_only' in selection['arms'] else ARMS
    if selection['clip'] != 'clip_003' or set(selection['arms']) != set(order):
        raise ValueError('Expected the frozen clip_003 prefix arms')
    plans = {}
    for name in order:
        arm = selection['arms'][name]
        q = np.asarray(arm['initial_hand_q'], dtype=np.float64)
        raw = raw_prefix_array(arm['raw_prefix'])
        pd = np.asarray(arm['pd_sequence'], dtype=np.float64)
        if pd.shape == (0,):
            pd = pd.reshape(0, 20)
        if q.shape != (20,) or not np.isfinite(q).all() or pd.shape != raw.shape or not np.isfinite(pd).all():
            raise ValueError(f'Invalid projected hand or planned PD shape for {name}')
        if name in ('base', 'nominal', 'action_only'):
            np.testing.assert_allclose(q, projected, rtol=0, atol=1e-7, err_msg=f'{name} changed the start')
        plans[name] = (q, raw, pd)
    if len(plans['base'][1]) or not len(plans['nominal'][1]):
        raise ValueError('Expected a prefix-free parent and nonempty nominal prefix')
    if len({len(plans[name][1]) for name in ARMS[1:]}) != 1:
        raise ValueError('The three prefix arms must have the same planned length')
    if 'joint_init_only' in plans:
        if len(plans['joint_init_only'][1]):
            raise ValueError('The joint initialization-only arm requires the frozen Actor from phase one')
        np.testing.assert_array_equal(plans['joint_init_only'][0], plans['joint'][0])
    return plans


def quaternion_gap_deg(a, b):
    cosine = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.rad2deg(2 * np.arccos(np.clip(cosine, 0, 1))))


def summarize_case(row, plan, preload, ref):
    q, raw, planned_pd = plan
    with np.load(verify_file_pin(row['prefix_execution']), allow_pickle=False) as execution:
        actual = execution['post_ctrl']
        count = len(actual)
        if row['prefix_controls'] != count or count != min(row['controls'], len(raw)):
            raise ValueError('Inconsistent prefix execution length')
        entry_pose = execution['entry_object_qpos']
        entry_error = dict(hand_max_rad=float(abs(execution['entry_hand_q'] - q).max()),
            preload_ctrl_max_rad=float(abs(execution['entry_ctrl'] - preload).max()),
            object_xyz_m=float(np.linalg.norm(entry_pose[:3] - ref['COM'][0])),
            object_angle_deg=quaternion_gap_deg(entry_pose[3:], ref['quat_wxyz'][0]))
        pd_error = (abs(actual - planned_pd[:count, None, :]).max(axis=(1, 2)) if count else np.empty(0))
        handoff_error = None
        if row['handoff_phase'] is not None:
            expected = raw[-1] if len(raw) else execution['entry_raw']
            handoff_error = float(abs(execution['handoff_previous_raw'] - expected).max())
            np.testing.assert_array_equal(execution['handoff_obs'][-20:], execution['handoff_previous_raw'])
            np.testing.assert_array_equal(execution['handoff_previous_raw'], expected)
    case = read(row['case'])
    with np.load(verify_file_pin(case['trace']), allow_pickle=False) as trace:
        prefix_end = None
        if count:
            i = 8*count-1
            prefix_end = dict(phase=count, COM_error_m=float(np.linalg.norm(trace['post_COM'][i] - ref['COM'][count])),
                angle_error_deg=quaternion_gap_deg(trace['post_quaternion_wxyz'][i], ref['quat_wxyz'][count]))
    return dict(rep=row['rep'], controls=row['controls'], valid_s=row['valid_s'], stop=row['stop'],
        reasons=row['reasons'], reference_completion=row['stop']=='REFERENCE_COMPLETION',
        strict_success=row['summary']['qualified'], failed_gates=[k for k,v in row['summary']['gates'].items() if not v],
        first_failed_physical_phase=next((i for i,c in enumerate(case['checks'],1) if not c['ok']), None),
        first_phase_audit_pass=case['checks'][0]['ok'], entry_error=entry_error,
        prefix_controls_executed=count, prefix_end=prefix_end, handoff_phase=row['handoff_phase'],
        handoff_previous_raw_error=handoff_error, prefix_pd_max_error_rad=float(pd_error.max()) if count else None,
        prefix_pd_max_error_by_phase_rad=pd_error.tolist(), metrics=row['summary']['metrics'], case=row['case'])


def run(selection, out, repetitions, actor_path=DEFAULT_ACTOR):
    if repetitions < 1:
        raise ValueError('Specify at least one repetition')
    root = b.ROOT / 'runs'
    selection_pin = file_pin(selection)
    preload_pin = file_pin(root / 'pen_refit_dev_preload_v1/report.json')
    projected_report = read(read(read(preload_pin)['intent'])['source'])
    pose_pin = next(r['variants']['projection']['trace'] for r in projected_report['rows'] if r['clip']=='clip_003')
    with np.load(verify_file_pin(pose_pin), allow_pickle=False) as projected:
        plans = load_selection(selection_pin, projected['initial_hand_q'])
    actor = file_pin(actor_path)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'intent.json', dict(selection=selection_pin, projected_pose=pose_pin, preload=preload_pin,
        frozen_actor=actor, arm_order=list(plans), repetitions=repetitions, worlds=1, environment_support_weight=0,
        scope='Frozen development candidates. Original phase-one W1 reference, physical audit, native termination and strict success gates. Deterministic repeats measure repeatability.'))
    results = {}
    for arm, plan in plans.items():
        folder = out / arm
        folder.mkdir()
        q, raw, _ = plan
        objective, item = config(folder, 'clip_003', Path(preload_pin['path']), initial_hand_q=q)
        adapter = Path(tomllib.loads(objective.read_text())['adapter_config'])
        cfg = tomllib.loads(adapter.read_text())
        write_json(folder/'intent.json', dict(master=file_pin(out/'intent.json'), arm=arm,
            projected=cfg['projected_pin'], preload=cfg['control_pin'], reference=item['series'],
            frozen_actor=actor, raw_prefix_controls=len(raw), objective=file_pin(objective), adapter=file_pin(adapter)))
        rows = evaluate_cases(folder, objective, item, {arm: actor}, {arm: 0.}, repetitions,
            refit_export=True, raw_prefixes={arm: raw})
        with np.load(verify_file_pin(cfg['control_pin'])) as ctrl, np.load(verify_file_pin(item['series'])) as ref:
            cases = [summarize_case(row, plan, ctrl['ctrl'], ref) for row in rows]
        results[arm] = dict(native_report=file_pin(folder/'report.json'), planned_prefix_controls=len(raw),
            full_count=sum(case['reference_completion'] for case in cases),
            strict_count=sum(case['strict_success'] for case in cases), cases=cases)
        write_json(out/'progress.json', dict(arms=results))
    write_json(out/'report.json', dict(intent=file_pin(out/'intent.json'), clip='clip_003', arms=results))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--actor', type=Path, default=DEFAULT_ACTOR, help='Frozen Actor checkpoint for every arm')
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    run(args.selection.resolve(), args.out.resolve(), args.repetitions, args.actor.resolve())
