"""Fixed bounded interventions and prospective100ms ensemble envelopes."""
from pathlib import Path
import argparse
import copy
import json
import sys
import tomllib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import ArtifactRegistry, file_pin, verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case, write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.runtime.gpu_pen_rollout import GPUPenRollout
from supermimic_research.runtime.gpu_model_snapshot import capture_model


def envelope(predicted, actual, times, ref_q, cfg, qa):
    target = Slerp(actual['time_s'], Rotation.from_quat(ref_q[:, [1, 2, 3, 0]]))(times)
    Rref = target.as_matrix()
    pq = predicted['qpos'][:, :, qa + 3:qa + 7]
    aq = actual['post_qpos'][:len(times), qa + 3:qa + 7]
    Rp = Rotation.from_quat(pq.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*pq.shape[:2], 3, 3)
    Ra = Rotation.from_quat(aq[:, [1, 2, 3, 0]]).as_matrix()
    plog = Rotation.from_matrix((Rp @ Rref[:, None].transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).as_rotvec().reshape(*pq.shape[:2], 3)
    alog = Rotation.from_matrix(Ra @ Rref.transpose(0, 2, 1)).as_rotvec()
    p = np.concatenate([predicted['qpos'][:, :, qa:qa + 3], plog], axis=2)
    a = np.concatenate([actual['post_qpos'][:len(times), qa:qa + 3], alog], axis=1)
    lo, hi = p.min(axis=1), p.max(axis=1)
    margin = np.array([cfg['position_component_margin_m']] * 3 + [cfg['rotation_log_component_margin_rad']] * 3)
    raw = ((a >= lo) & (a <= hi)).all(axis=1)
    covered = ((a >= lo - margin) & (a <= hi + margin)).all(axis=1)
    return dict(time_s=times, minimum=lo, maximum=hi, actual_position_rotation_log=a,
                raw_covered=raw, margin_covered=covered, margin=margin,
                prediction_rotation_log_max=np.max(np.linalg.norm(plog, axis=-1)),
                actual_rotation_log_max=np.max(np.linalg.norm(alog, axis=-1)))


def main(args):
    import torch
    import warp as wp
    import mujoco_warp as mw
    if args.output.exists():
        raise ValueError('Fresh prospective intervention cohort')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    if (cfg['direction_count'], cfg['nworld'], cfg['horizon_controls'], cfg['raw_amplitude']) != (4, 128, 5, .04):
        raise ValueError('Fixed prospective profiles and100ms horizon')
    bp = (cp.parent / cfg['base_config']).resolve()
    base = tomllib.loads(bp.read_text())
    pp = (cp.parent / cfg['proposal']).resolve()
    proposal = json.loads(pp.read_text())
    selected = np.load(verify_file_pin(proposal['artifacts']['selected']))
    qref = np.load(verify_file_pin(proposal['clip']))['reference_qpos']
    directions = np.sin(np.pi * (np.arange(20) + .5)[None] * np.arange(1, 5)[:, None] / 20)
    directions /= np.max(abs(directions), axis=1, keepdims=True)
    profiles = [('BASELINE', 'CALIBRATION', np.zeros(20))]
    for direction in range(4):
        for side in [1, -1]:
            profiles.append((f'D{direction + 1}_{"PLUS" if side == 1 else "MINUS"}',
                             'CALIBRATION' if direction == 0 else 'CHECK', side * .04 * directions[direction]))
    torch.set_num_threads(4)
    wp.set_device('cuda:0')
    registry = ArtifactRegistry(bp.parent / base['artifact_manifest'])
    args.output.mkdir()
    np.savez_compressed(args.output / 'profiles.npz', directions=directions,
                        raw_offsets=np.stack([p[2] for p in profiles]))
    sources = [file_pin(__file__)] + [file_pin(ROOT / p) for p in [
        'src/supermimic_research/runtime/controlled_evaluation.py',
        'src/supermimic_research/runtime/pen_gpu_initialization.py',
        'src/supermimic_research/runtime/gpu_pen_rollout.py',
        'src/supermimic_research/runtime/gpu_contact_oracle.py',
        'src/supermimic_research/runtime/gpu_model_snapshot.py',
        'src/supermimic_research/mechanics/warp_contact_labels.py']]
    intent = dict(schema='pen_bounded_action_envelope:1', config=file_pin(cp), base_config=file_pin(bp),
                  proposal=file_pin(pp), profiles=file_pin(args.output / 'profiles.npz'), sources=sources,
                  roles=cfg['roles'], neural_updates=0)
    write(args.output / 'intent.json', intent)
    cases = []
    for dt in cfg['dt']:
        for name, role, offset in profiles:
            folder = args.output / f'{name}_{dt}'
            raw_base = torch.as_tensor(2 * selected['delta20'], device='cuda', dtype=torch.float32)[None]
            perturb = torch.as_tensor(offset, device='cuda', dtype=torch.float32)[None]
            forecast = {}

            def initialize(env):
                receipt = initialize_and_query_pen(env, folder, selected['source_qpos'], selected['qpos'],
                    selected['delta20'], selected['target_BODY'], support_all_robot_contacts=True)
                model = copy.copy(env.sim.mj_model)
                for key in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'geom_size', 'geom_friction', 'geom_solref', 'geom_solimp',
                            'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'actuator_ctrlrange', 'jnt_range', 'dof_damping', 'dof_armature']:
                    getattr(model, key)[:] = getattr(env.sim.wp_model, key).numpy()[0]
                data = mujoco.MjData(model)
                mujoco.mj_setConst(model, data)
                mw.get_data_into(data, model, env.sim.wp_data)
                _, scalars, before = capture_model(env.sim.wp_model)
                clone = GPUPenRollout(model, data, cfg['nworld'], selected['delta20'], device_model=env.sim.wp_model, nconmax=128, njmax=512)
                rows = []
                reference = torch.as_tensor(qref, device='cuda', dtype=torch.float32)
                for phase in range(1, cfg['horizon_controls'] + 1):
                    clone.set_action((raw_base + perturb).expand(cfg['nworld'], -1), reference[phase])
                    for _ in range(clone.decimation):
                        clone.physics_step()
                        row = clone.readback(clone.output())
                        row['overflow'] = clone.views['overflow'].cpu().numpy().copy()
                        row['nacon'] = np.full(cfg['nworld'], int(clone.data.nacon.numpy()[0]))
                        rows.append(row)
                    clone.refresh_boundary()
                forecast.update({key: np.stack([row[key] for row in rows]) for key in rows[0]})
                path = folder / 'prospective_forecast.npz'
                np.savez_compressed(path, **forecast)
                _, scalar_after, after = capture_model(env.sim.wp_model)
                return {**receipt, 'prospective_forecast': file_pin(path),
                        'forecast_model_unchanged': before == after and scalars == scalar_after}

            def provider(env, observation, k):
                return raw_base + perturb if k < cfg['horizon_controls'] else raw_base

            case = evaluate_case('hand2-pen-spin', 'INTERVENTION', None, {**base, 'physics_dt': dt},
                bp.parent, registry, folder, initializer=initialize, action_provider=provider)
            case.update(name=name, role=role, supervision='NONEXPERT_NOT_FIT')
            actual = np.load(verify_file_pin(case['trace']))
            n = min(len(actual['post_qpos']), len(forecast['qpos']))
            common = {k: v[:n] for k, v in forecast.items()}
            band = envelope(common, actual, np.arange(1, n + 1) * dt, actual['quaternion_wxyz'], cfg, case['model_layout']['object_qposadr'])
            envelope_path = folder / 'envelope.npz'
            np.savez_compressed(envelope_path, **band)
            complete = n == round(.1 / dt)
            no_wrap = max(band['prediction_rotation_log_max'], band['actual_rotation_log_max']) < np.pi - .001
            coverage = dict(complete_100ms_actual=complete, rotation_log_unambiguous=bool(no_wrap),
                model_unchanged=case['initialization']['forecast_model_unchanged'],
                original_PD=float(abs(common['ctrl'] - actual['post_ctrl'][:n, None]).max()) < 2e-7,
                prediction_capacity=bool(np.max(common['nacon']) <= 128 * 128 and not np.any(common['overflow']) and np.max(common['nefc']) <= 512),
                all_margin_ticks=bool(complete and no_wrap and np.all(band['margin_covered'])))
            case['forecast_validation'] = dict(checks=coverage, raw_coverage=float(np.mean(band['raw_covered'])),
                margin_coverage=float(np.mean(band['margin_covered'])),
                maximum_position_envelope_width_m=float(np.linalg.norm(band['maximum'][:, :3] - band['minimum'][:, :3], axis=1).max()),
                maximum_rotation_log_envelope_width_rad=float(np.linalg.norm(band['maximum'][:, 3:] - band['minimum'][:, 3:], axis=1).max()),
                envelope=file_pin(envelope_path), passed=bool(all(coverage.values())))
            cases.append(case)
            write(folder / 'report.json', case)
            write(args.output / 'progress.json', dict(cases=cases))
            print(json.dumps(dict(name=name, dt=dt, role=role, envelope_passed=case['forecast_validation']['passed'],
                coverage=case['forecast_validation']['margin_coverage'], duration=case['summary']['metrics']['actual_duration_s'])), flush=True)
    for pin in sources:
        verify_file_pin(pin)
    task_sources = []
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('dex_supermimic') and path:
            if not Path(path).resolve().is_relative_to(ROOT / 'src'):
                raise ValueError('Unexpected historical task source')
            task_sources.append(file_pin(path))
    write(args.output / 'report.json', dict(**intent, cases=cases, task_sources=task_sources,
        roles_passed={role: [sum(c['forecast_validation']['passed'] for c in cases if c['role'] == role), sum(c['role'] == role for c in cases)] for role in ['CALIBRATION', 'CHECK']},
        scope='Prospective128-world100ms sample-envelope coverage under fixed independent bounded action directions. Not calibrated probability, deterministic equivalence, robust planning readiness, neural training, expert demonstrations or task success.'))


if __name__ == '__main__':
    import torch
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    with torch.jit.optimized_execution(False):
        main(p.parse_args())
