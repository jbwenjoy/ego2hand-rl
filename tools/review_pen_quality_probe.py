"""Independent source-state, impulse and full contact-wrench probe readback."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin


def main(args):
    if args.output.exists():
        raise ValueError('Fresh quality-probe readback')
    r = json.loads(args.report.read_text())
    for pin in r['sources']:
        verify_file_pin(pin)
    actual = json.loads(verify_file_pin(r['run']).read_text())
    rows = []
    for c in r['cases']:
        source = next(s for s in actual['cases'] if s['name'] == c['source_name'] and s['dt'] == c['dt'])
        original = np.load(verify_file_pin(source['trace']))
        physical = np.load(verify_file_pin(source['GPU_physical_context']))
        m = mujoco.MjModel.from_binary_path(str(verify_file_pin(source['host_topology'])))
        z = np.load(verify_file_pin(c['state']))
        contacts = np.load(verify_file_pin(c['contacts']))
        tick, body = c['source_tick'], source['model_layout']['object_body']
        qa, da = source['model_layout']['object_qposadr'], source['model_layout']['object_dofadr']
        R = Rotation.from_quat(original['pre_qpos'][tick, qa + 3:qa + 7][[1, 2, 3, 0]]).as_matrix()
        COM = original['pre_qpos'][tick, qa:qa + 3]
        F, T = np.zeros((128, 3)), np.zeros((128, 3))
        local_error = 0.
        for k, (a, b) in enumerate(contacts['geom']):
            wid = int(contacts['worldid'][k])
            dim = int(contacts['dim'][k])
            force = np.zeros(6)
            if contacts['efc_address'][k, 0] >= 0:
                edges = contacts['efc_force'][wid, contacts['efc_address'][k, :2 * (dim - 1)]].astype(float)
                force[0] = edges.sum()
                for j in range(dim - 1):
                    force[j + 1] = (edges[2 * j] - edges[2 * j + 1]) * contacts['friction'][k, j]
            local_error = max(local_error, float(abs(force - contacts['local_force6'][k]).max()))
            oa, ob = m.geom_bodyid[a] == body, m.geom_bodyid[b] == body
            if oa == ob:
                continue
            sign = 1. if ob else -1.
            frame = contacts['frame'][k].reshape(3, 3)
            f, t = sign * frame.T @ force[:3], sign * frame.T @ force[3:]
            F[wid] += f
            T[wid] += t + np.cross(contacts['pos'][k] - COM, f)
        mass = float(physical['body_mass'][0, body])
        dv = z['qvel'][:, da:da + 3] - original['pre_qvel'][tick, da:da + 3]
        impulse = mass * dv - c['dt'] * (F + mass * m.opt.gravity)
        passed = int(np.sum(np.linalg.norm(impulse, axis=1) <= 5e-7))
        force_error = float(np.linalg.norm(F - z['qfrc_constraint'][:, da:da + 3], axis=1).max())
        torque_error = float(np.linalg.norm(T @ R - z['qfrc_constraint'][:, da + 3:da + 6], axis=1).max())
        checks = dict(source_qpos=np.array_equal(z['initial_qpos'], np.tile(original['pre_qpos'][tick].astype(np.float32), (128, 1))),
            source_qvel=np.array_equal(z['initial_qvel'], np.tile(original['pre_qvel'][tick].astype(np.float32), (128, 1))),
            source_ctrl=np.array_equal(z['initial_ctrl'], np.tile(original['post_ctrl'][tick].astype(np.float32), (128, 1))),
            source_warmstart=np.array_equal(z['initial_qacc_warmstart'], np.tile(original['post_qacc'][tick - 1].astype(np.float32), (128, 1))),
            local_force_labels=local_error < 1e-12, full_force_identity=force_error < 1e-4,
            full_torque_identity=torque_error < 1e-5,
            impulse_readback=float(abs(impulse - z['impulse']).max()) < 1e-12,
            original_gate_count=passed == c['passed_worlds'],
            iteration_counts=[int(z['solver_niter'].min()), int(z['solver_niter'].max())] == c['solver_iterations'])
        rows.append(dict(dt=c['dt'], variant=c['variant'], repeat=c['repeat'], checks=checks,
                         passed_worlds=passed, total_worlds=128, force_identity_error_N=force_error,
                         torque_identity_error_Nm=torque_error))
    args.output.mkdir()
    result = dict(schema='pen_quality_probe_saved_review:1', source=file_pin(__file__), run=file_pin(args.report),
        cases=rows, passed=sum(sum(c['checks'].values()) for c in rows), total=sum(len(c['checks']) for c in rows),
        neural_updates=0, simulation_integration_steps=0,
        scope='Saved source warmstart/controls, direct EFC decode, full COM force/torque and unchanged impulse-gate count readback. Passing review confirms mixed numerical-probe outcomes; not all-world or trajectory qualification.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=result['passed'], total=result['total'])))
    if result['passed'] != result['total']:
        raise ValueError('Preserve failed numerical-probe readback')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
