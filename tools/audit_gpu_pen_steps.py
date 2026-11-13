"""Saved physical step/EMA/phase fidelity for single and resident batch clones."""
from pathlib import Path
import argparse
import json
import sys
import time
import tomllib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.runtime.gpu_pen_rollout import GPUPenRollout
from tools.audit_gpu_contact_oracle import harvest


def discrepancies(actual, saved, oracle, ticks):
    q = actual['qpos'][:ticks]
    v = actual['qvel'][:ticks]
    a, d = oracle.qa, oracle.da
    targetq = saved['post_qpos'][:ticks, None, :]
    targetv = saved['post_qvel'][:ticks, None, :]
    q1, q2 = q[:, :, a + 3:a + 7], targetq[:, :, a + 3:a + 7]
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=-1, keepdims=True)
    angle = 2 * np.arccos(np.clip(abs(np.sum(q1 * q2, axis=-1)), 0, 1))
    R = Rotation.from_quat(q1.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*q1.shape[:-1], 3, 3)
    R0 = Rotation.from_quat(q2.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*q2.shape[:-1], 3, 3)
    omega = np.einsum('...ij,...j->...i', R, v[:, :, d + 3:d + 6])
    omega0 = np.einsum('...ij,...j->...i', R0, targetv[:, :, d + 3:d + 6])
    return dict(hand_position_rad=float(abs(q[:, :, oracle.qids] - targetq[:, :, oracle.qids]).max()),
        object_position_m=float(np.linalg.norm(q[:, :, a:a + 3] - targetq[:, :, a:a + 3], axis=-1).max()),
        object_rotation_rad=float(angle.max()),
        hand_velocity_rad_s=float(abs(v[:, :, oracle.dofs] - targetv[:, :, oracle.dofs]).max()),
        object_velocity_m_s=float(np.linalg.norm(v[:, :, d:d + 3] - targetv[:, :, d:d + 3], axis=-1).max()),
        object_omega_rad_s=float(np.linalg.norm(omega - omega0, axis=-1).max()))


def main(args):
    import torch
    if args.output.exists():
        raise ValueError('Fresh physical step audit')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    rp = (cp.parent / cfg['run']).resolve()
    r = json.loads(rp.read_text())
    pp = (cp.parent / cfg['proposal']).resolve()
    p = json.loads(pp.read_text())
    selected = np.load(verify_file_pin(p['artifacts']['selected']))
    qref = np.load(verify_file_pin(p['clip']))['reference_qpos']
    out = args.output.resolve()
    out.mkdir()
    sources = [file_pin(__file__), file_pin(ROOT / 'src/supermimic_research/runtime/gpu_pen_rollout.py'),
               file_pin(ROOT / 'src/supermimic_research/runtime/gpu_contact_oracle.py'),
               file_pin(ROOT / 'tools/audit_gpu_contact_oracle.py')]
    intent = dict(schema='GPU_pen_steps_audit:1', config=file_pin(cp), run=file_pin(rp), proposal=file_pin(pp), sources=sources,
                  neural_updates=0, role='PHYSICAL_CLONE_FIDELITY_NOT_EXPERT_NOT_FIT')
    (out / 'intent.json').write_text(json.dumps(intent, indent=2) + '\n')
    cases = []
    for c in r['cases']:
        z = np.load(verify_file_pin(c['trace']))
        for nw in cfg['nworld']:
            start = time.perf_counter()
            m = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
            for key, value in dict(np.load(verify_file_pin(c['GPU_physical_context']))).items():
                getattr(m, key)[:] = value[0]
            data = mujoco.MjData(m)
            mujoco.mj_setConst(m, data)
            mujoco.mj_setState(m, data, z['initial_native_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
            oracle = GPUPenRollout(m, data, nw, selected['delta20'], nconmax=cfg['nconmax'], njmax=cfg['njmax'])
            initial = {k: v.cpu().numpy() for k, v in oracle.state().items()}
            initial_mocap = {k: oracle.views[k].cpu().numpy().copy() for k in ['mocap_pos', 'mocap_quat']}
            reference = torch.as_tensor(qref, device='cuda', dtype=torch.float32)
            rows, ema_rows = [], []
            for tick in range(len(z['post_qpos'])):
                control = tick // oracle.decimation
                if tick % oracle.decimation == 0:
                    raw = torch.as_tensor(z['actions20'][control], device='cuda', dtype=torch.float32)[None].expand(nw, -1)
                    oracle.set_action(raw, reference[int(z['pre_action_phase'][control])])
                    ema_rows.append(oracle.ema.cpu().numpy().copy())
                pre = oracle.views['qpos'].clone()
                oracle.physics_step()
                rows.append(oracle.readback(oracle.output()))
                if tick == len(z['post_qpos']) - 1:
                    contacts = harvest(oracle)
                    last_R = oracle.views['xmat'][:, oracle.body].reshape(nw, 3, 3).cpu().numpy().copy()
                    last_COM = pre[:, oracle.qa:oracle.qa + 3].cpu().numpy().copy()
                if (tick + 1) % oracle.decimation == 0:
                    oracle.refresh_boundary()
            actual = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
            horizon = discrepancies(actual, z, oracle, round(cfg['horizon_s'] / c['dt']))
            complete = discrepancies(actual, z, oracle, len(rows))
            memory = selected['delta20'].astype(np.float32)
            expected_ema = []
            for raw in z['actions20']:
                memory = .5 * np.clip(.5 * raw, -.6, .6) + .5 * memory
                expected_ema.append(memory.copy())
            ema_error = float(abs(np.stack(ema_rows) - np.array(expected_ema)[:, None]).max())
            ctrl_error = float(abs(actual['ctrl'] - z['post_ctrl'][:, None]).max())
            # Independent contact-point wrench at the last PRE-step geometry.
            a, b = contacts['geom'].T
            oa, ob = m.geom_bodyid[a] == oracle.body, m.geom_bodyid[b] == oracle.body
            ids = np.flatnonzero(oa ^ ob)
            wid = contacts['worldid'][ids]
            sign = np.where(ob[ids], 1., -1.)
            frame = contacts['frame'][ids].reshape(-1, 3, 3)
            local = contacts['local_force6'][ids]
            F = sign[:, None] * np.einsum('nji,nj->ni', frame, local[:, :3])
            T = sign[:, None] * np.einsum('nji,nj->ni', frame, local[:, 3:]) + np.cross(contacts['pos'][ids] - last_COM[wid], F)
            wrench = np.zeros((nw, 6))
            np.add.at(wrench, wid, np.c_[np.einsum('nji,nj->ni', last_R[wid], F), np.einsum('nji,nj->ni', last_R[wid], T)])
            residual = wrench - actual['wrench_BODY_COM'][-1]
            force_error = float(np.linalg.norm(residual[:, :3], axis=1).max())
            torque_error = float(np.linalg.norm(residual[:, 3:], axis=1).max())
            checks = {key: value <= cfg[key] for key, value in horizon.items()}
            checks.update(initial_pose=bool(np.array_equal(initial['qpos'], np.tile(z['initial_qpos'], (nw, 1)))),
                initial_velocity=bool(np.array_equal(initial['qvel'], np.tile(z['initial_qvel'], (nw, 1)))),
                initial_warmstart=bool(np.array_equal(initial['qacc_warmstart'], np.tile(data.qacc_warmstart.astype(np.float32), (nw, 1)))),
                fixed_mocap=bool(all(np.array_equal(v, oracle.views[k].cpu().numpy()) for k, v in initial_mocap.items())),
                original_EMA=ema_error <= cfg['control_rad'], original_PD=ctrl_error <= cfg['control_rad'],
                EFC_force=force_error <= cfg['EFC_force_N'], EFC_torque=torque_error <= cfg['EFC_torque_Nm'],
                no_aid=not bool(oracle.views['qfrc_applied'].count_nonzero()) and not bool(oracle.views['xfrc_applied'].count_nonzero()),
                capacity=bool(contacts['capacity_ok']) and bool((actual['nefc'] <= oracle.data.njmax).all()),
                original_clock=bool(np.max(abs(actual['time'][:, :, 0] - z['post_GPU_time'][:, None])) < 1e-6),
                finite=bool(all(np.isfinite(v).all() for v in actual.values())))
            name = f'dt{c["dt"]:g}_worlds{nw}'
            path = out / (name + '.npz')
            np.savez_compressed(path, **actual, ema=np.stack(ema_rows), last_EFC_wrench=wrench)
            contact_path = out / (name + '_contacts.npz')
            np.savez_compressed(contact_path, **contacts)
            entry = dict(dt=c['dt'], nworld=nw, checks=checks, horizon_errors=horizon, complete_prefix_errors=complete,
                         EMA_error_rad=ema_error, PD_error_rad=ctrl_error,
                         force_identity_error_N=force_error, torque_identity_error_Nm=torque_error,
                         physical_world_ticks=oracle.physics_ticks, setup_integration_world_ticks=nw,
                         wall_including_setup_and_readback_s=time.perf_counter() - start,
                         trace=file_pin(path), last_contacts=file_pin(contact_path))
            cases.append(entry)
            print(json.dumps({k: entry[k] for k in ['dt', 'nworld', 'checks', 'horizon_errors']}), flush=True)
    for pin in sources:
        verify_file_pin(pin)
    result = dict(**intent, cases=cases, passed=sum(sum(c['checks'].values()) for c in cases),
                  total=sum(len(c['checks']) for c in cases), ready_for_100ms_planning=all(all(c['checks'].values()) for c in cases),
                  scope='Physical clone integration and original control/EMA/phase fidelity over preregistered100ms, with whole failed prefixes reported separately. No new physical task success, neural training or WM-gradient evidence.')
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], total=result['total'], ready=result['ready_for_100ms_planning'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
