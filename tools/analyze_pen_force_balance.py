"""Separate final free-body force balance from integration roundoff, saved only."""
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
        raise ValueError('Fresh force-balance decomposition')
    r = json.loads(args.report.read_text())
    efc = json.loads(args.efc_report.read_text())
    if verify_file_pin(efc['run']).resolve() != args.report.resolve():
        raise ValueError('Same actual EFC run required')
    args.output.mkdir()
    rows = []
    for c in r['cases']:
        z = np.load(verify_file_pin(c['trace']))
        p = np.load(verify_file_pin(c['GPU_physical_context']))
        e = next(e for e in efc['cases'] if e['name'] == c['name'] and e['dt'] == c['dt'])
        wrench = np.load(verify_file_pin(e['wrenches']))
        m = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
        b, da = c['model_layout']['object_body'], c['model_layout']['object_dofadr']
        if m.body_parentid[b] != 0 or np.any(p['body_ipos'][0, b]) or m.neq or np.any(m.body_gravcomp):
            raise ValueError('Unassisted world-parent free body with zero COM offset')
        mass = float(p['body_mass'][0, b])
        R = Rotation.from_quat(p['body_iquat'][0, b][[1, 2, 3, 0]]).as_matrix()
        inertia = R @ np.diag(p['body_inertia'][0, b]) @ R.T
        acceleration = z['post_qacc'][:, da:da + 6]
        inertia_acceleration = np.c_[mass * acceleration[:, :3], acceleration[:, 3:] @ inertia.T]
        bias = z['post_qfrc_bias'][:, da:da + 6]
        other = z['post_qfrc_passive'][:, da:da + 6] + z['post_qfrc_actuator'][:, da:da + 6] + z['post_qfrc_applied'][:, da:da + 6]
        constraint = z['post_qfrc_constraint'][:, da:da + 6]
        efc_generalized = np.c_[wrench['world_COM_wrench'][:, :3], wrench['object_BODY_COM_wrench'][:, 3:]]
        balance = inertia_acceleration + bias - other - efc_generalized
        direct_balance = inertia_acceleration + bias - other - constraint
        dv = z['post_qvel'][:, da:da + 3] - z['pre_qvel'][:, da:da + 3]
        integration_roundoff = mass * (dv - c['dt'] * acceleration[:, :3])
        gravity_bias_correction = -c['dt'] * (mass * m.opt.gravity + bias[:, :3] - other[:, :3])
        predicted_impulse_residual = c['dt'] * balance[:, :3] + integration_roundoff + gravity_bias_correction
        impulse = wrench['linear_impulse_residual_N_s']
        reconstruction_error = float(abs(predicted_impulse_residual - impulse).max())
        bad = np.flatnonzero(np.linalg.norm(impulse, axis=1) > 5e-7)
        path = args.output / f'{c["name"]}_{c["dt"]}.npz'
        np.savez_compressed(path, free_generalized_balance=balance, direct_generalized_balance=direct_balance,
            linear_impulse_residual_N_s=impulse, integration_roundoff_N_s=integration_roundoff,
            gravity_bias_correction_N_s=gravity_bias_correction, predicted_impulse_residual_N_s=predicted_impulse_residual)
        rows.append(dict(name=c['name'], dt=c['dt'], bad_ticks=bad.tolist(), bad_POST_times_s=((bad + 1) * c['dt']).tolist(),
            maximum_impulse_residual_N_s=float(np.linalg.norm(impulse, axis=1).max()),
            maximum_final_linear_balance_N=float(np.linalg.norm(balance[:, :3], axis=1).max()),
            maximum_final_angular_balance_Nm=float(np.linalg.norm(balance[:, 3:], axis=1).max()),
            maximum_integration_roundoff_N_s=float(np.linalg.norm(integration_roundoff, axis=1).max()),
            algebraic_reconstruction_error_N_s=reconstruction_error,
            checks=dict(reconstructed_impulse=reconstruction_error < 1e-12,
                        no_external_force=not np.any(z['post_qfrc_applied']) and not np.any(z['post_xfrc_applied']),
                        finite=bool(np.isfinite(balance).all())), artifact=file_pin(path)))
    result = dict(schema='pen_saved_force_balance:1', source=file_pin(__file__), run=file_pin(args.report),
        EFC_review=file_pin(args.efc_report), cases=rows, passed=sum(sum(c['checks'].values()) for c in rows),
        total=sum(len(c['checks']) for c in rows), failed_original_impulse_cases=sum(bool(c['bad_ticks']) for c in rows),
        neural_updates=0, simulation_integration_steps=0,
        scope='Algebraic saved free-body equation/impulse decomposition with actual EFC forces and source acceleration/bias/passive/actuation. A remaining final equation residual is not attributed to a unique solver/kernel or fixed by this analysis; original two impulse gates remain failed.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['passed', 'total', 'failed_original_impulse_cases']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--efc-report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
