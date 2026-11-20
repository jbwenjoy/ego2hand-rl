"""Select paired startup interventions on the corrected CPU pen model before native evaluation."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.pen_corrective_lookahead import STATE, score_candidates
from supermimic_research.strict_json import write_json


ROOT = Path(__file__).resolve().parents[1]
HORIZON = 8
SUPPORT_WEIGHT = 0.2


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def row_for(report, name):
    return next(row for row in report['rows'] if row['clip'] == name)


def transmitted_targets(raws, refq, ema0, limits):
    ema = ema0.astype(np.float32)
    targets = []
    for raw, ref in zip(raws, refq, strict=True):
        ema = 0.5 * ema + 0.25 * np.clip(raw, -1, 1)
        targets.append(np.clip(ref.astype(np.float32) + ema, *limits))
    return np.asarray(targets)


def select(out, clip, power):
    if clip not in {f'clip_{i:03}' for i in range(2, 7)}:
        raise ValueError('Select a development clip')
    if not 1 <= power <= 10:
        raise ValueError('Expected Sobol power between 1 and 10')
    out.mkdir(exist_ok=False, parents=True)
    runs = ROOT / 'runs'
    pre_pin = file_pin(runs / 'pen_refit_dev_preload_v1/report.json')
    pre = read(pre_pin)
    projection = read(read(pre['intent'])['source'])
    catalog = read(read(projection['intent'])['screen'])
    source = read(catalog['intent'])
    case = read(source['case'])
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(source['cpu_model'])))
    qids = case['qids']
    aids = case['aids']
    np.testing.assert_array_equal(aids, np.arange(model.nu))
    joint_ids = model.actuator_trnid[aids, 0]
    np.testing.assert_array_equal(qids, model.jnt_qposadr[joint_ids])
    lower, upper = model.jnt_range[joint_ids].T
    limits = np.array([np.maximum(lower, model.actuator_ctrlrange[aids, 0]),
                       np.minimum(upper, model.actuator_ctrlrange[aids, 1])], dtype=np.float32)
    assert np.all(model.actuator_ctrllimited) and model.neq == 0 and not np.any(model.body_gravcomp)
    layout = case['model_layout']
    body, qa, da = (layout[k] for k in ('object_body', 'object_qposadr', 'object_dofadr'))
    np.testing.assert_array_equal(model.body_ipos[body], np.zeros(3))
    with np.load(verify_file_pin(case['trace'])) as f:
        mount_pos, mount_quat = f['initial_mocap_pos'][0], f['initial_mocap_quat'][0]
    item = catalog['results'][clip]
    with np.load(verify_file_pin(row_for(projection, clip)['variants']['projection']['trace'])) as f:
        q0, object_pose = f['initial_hand_q'], np.r_[f['object_pos'], f['object_quat']]
    with np.load(verify_file_pin(row_for(pre, clip)['holds']['preloaded']['trace'])) as f:
        ctrl0 = f['ctrl']
    with np.load(verify_file_pin(item['source'])) as f:
        refq = f['reference_qpos'][:HORIZON+1].astype(np.float32)
    with np.load(verify_file_pin(item['series'])) as f:
        pos, quat = f['COM'][:HORIZON+1], f['quat_wxyz'][:HORIZON+1]
    np.testing.assert_allclose(object_pose[:3], pos[0], atol=1e-6)
    assert abs(object_pose[3:] @ quat[0]) > 1-1e-6
    rot = Rotation.from_quat(quat[:, [1, 2, 3, 0]])
    ref_v = np.diff(pos, axis=0) / 0.02
    ref_omega = (rot[1:] * rot[:-1].inv()).as_rotvec() / 0.02
    ema0 = (ctrl0 - refq[0]).astype(np.float32)
    if np.abs(2*ema0).max() > 1:
        raise ValueError('Preload exceeds RAW range')
    # The five fingers share one flexion direction within a finger. The second
    # knot allows a different sign at handoff. Both arms see identical samples.
    samples = qmc.Sobol(20, scramble=True, seed=1120).random_base2(power)
    knots = np.zeros((len(samples)+1, 2, 5, 4))
    draws = (2*samples-1).reshape(-1, 2, 5, 2)
    knots[1:, :, :, 0] = draws[..., 0]
    knots[1:, :, :, 1] = draws[..., 1]
    knots[1:, :, :, 2] = draws[..., 0]
    knots[1:, :, :, 3] = .65 * draws[..., 0]
    knots = knots.reshape(-1, 2, 20)
    mix = np.linspace(0, 1, HORIZON)[:, None]
    sources = [Path(__file__), ROOT/'src/supermimic_research/mechanics/pen_corrective_lookahead.py']
    write_json(out/'intent.json', dict(clip=clip, preload=pre_pin, source=item['source'],
               reference=item['series'], cpu_model=source['cpu_model'], code=[file_pin(p) for p in sources],
               horizon=HORIZON, seed=1120, candidates_per_arm=len(knots), raw_amplitude=.18,
               initial_hand_rad=.035, support_weight=SUPPORT_WEIGHT,
               rank='Finite candidates, longest geometrically and task-valid prefix; then lowest full-horizon mean pose_cost + 0.2 support_cost; then smallest index.',
               state='Unchanged free object, clock, mount, zero velocities, original preload, initial EMA and RAW. Joint arm couples initial hand change to first knot.',
               controls='Native residual actor after the prefix. Base starts with residual actor. Nominal follows the zero-amplitude prefix. Joint-init-only uses joint winner q0 with actor from the beginning.'))
    arms = {'base': dict(initial_hand_q=q0.tolist(), raw_prefix=[], pd_sequence=[])}
    ledger = []
    for arm in ('action_only', 'joint'):
        pool = []
        data = mujoco.MjData(model)
        for index, (first, last) in enumerate(knots):
            q = np.clip(q0 + (.035 * first if arm == 'joint' else 0), lower, upper).astype(np.float32)
            raws = np.clip(2*ema0 + .18*((1-mix)*first + mix*last), -1, 1).astype(np.float32)
            targets = transmitted_targets(raws, refq[1:], ema0, limits)
            mujoco.mj_resetData(model, data)
            data.mocap_pos[:], data.mocap_quat[:] = mount_pos, mount_quat
            data.qpos[qids], data.qpos[qa:qa+7], data.ctrl[aids] = q, object_pose, ctrl0
            state = np.empty(mujoco.mj_stateSize(model, STATE))
            mujoco.mj_getState(model, data, state, STATE)
            score, = score_candidates(model, state, targets[None], pos[1:], quat[1:], body=body,
                                      qa=qa, da=da, ref_v=ref_v, ref_omega=ref_omega)
            if score['finite']:
                valid = score['valid_controls']
                for step, interval in enumerate(score['intervals']):
                    if interval['shaft_deg'] > 60 or interval['mm'] > 50:
                        valid = min(valid, step)
                objective = score['pose_cost'] + SUPPORT_WEIGHT * score['support_cost']
                rank = (0, -valid, objective, index)
            else:
                valid, objective, rank = 0, None, (1, 0, 0, index)
            row = dict(arm=arm, candidate=index, valid_controls=valid, objective=objective,
                       native_steps=score['native_steps'], finite=score['finite'],
                       endpoint_mm=score['endpoint_mm'], endpoint_deg=score['endpoint_deg'])
            ledger.append(row)
            pool.append((rank, row, q, raws, targets, score))
        best = min(pool, key=lambda entry: entry[0])
        if not best[1]['finite']:
            raise ValueError(f'No finite candidate for {arm}')
        def proposal(entry):
            _, row, q, raw, targets, score = entry
            return dict(initial_hand_q=q.tolist(), raw_prefix=raw.tolist(), pd_sequence=targets.tolist(),
                        cpu=row, intervals=score['intervals'])
        if arm == 'action_only':
            arms['nominal'] = proposal(pool[0])
        arms[arm] = proposal(best)
        print(arm, best[1], flush=True)
    arms['joint_init_only'] = dict(initial_hand_q=arms['joint']['initial_hand_q'], raw_prefix=[], pd_sequence=[])
    write_json(out/'candidates.json', ledger)
    write_json(out/'selection.json', dict(clip=clip, arms=arms, intent=file_pin(out/'intent.json'),
                                        candidates=file_pin(out/'candidates.json')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--clip', default='clip_003')
    parser.add_argument('--power', type=int, default=6)
    args = parser.parse_args()
    select(args.out.resolve(), args.clip, args.power)
