"""Independent saved-candidate, full geometry, EFC and support-primal review."""
from pathlib import Path
import argparse
import json
import sys
import tomllib
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.compound_collision import eligible_pairs
from supermimic_research.mechanics.support import static_feasibility


def decode(z):
    forces = np.zeros((len(z['geom']), 6))
    for k, dim in enumerate(z['dim']):
        dim = int(dim)
        if z['efc_address'][k, 0] < 0:
            continue
        ids = z['efc_address'][k, :2 * (dim - 1)]
        world = int(z['worldid'][k])
        if np.any((ids < 0) | (ids >= z['nefc'][world])):
            raise ValueError('Saved EFC indices outside original world')
        edges = z['efc_force'][world, ids].astype(float)
        forces[k, 0] = sum(edges)
        for j in range(dim - 1):
            forces[k, j + 1] = (edges[2 * j] - edges[2 * j + 1]) * z['friction'][k, j]
    return forces


def basis_and_actual(model, data, body, hand, dofs, z):
    R = data.xmat[body].reshape(3, 3)
    COM = data.xipos[body]
    W, L = [], []
    wrench, reaction = np.zeros(6), np.zeros(20)
    local = decode(z)
    for k, (a, b) in enumerate(z['geom']):
        oa, ob = model.geom_bodyid[a] == body, model.geom_bodyid[b] == body
        if oa == ob or z['efc_address'][k, 0] < 0:
            continue
        other = int(a if ob else b)
        if other not in hand:
            raise ValueError('Nonhand support forbidden')
        sign = 1. if ob else -1.
        frame = z['frame'][k].reshape(3, 3)
        F, T = sign * frame.T @ local[k, :3], sign * frame.T @ local[k, 3:]
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, z['pos'][k], int(model.geom_bodyid[other]))
        wrench += np.r_[R.T @ F, R.T @ (T + np.cross(z['pos'][k] - COM, F))]
        reaction -= jp[:, dofs].T @ F + jr[:, dofs].T @ T
        dim = int(z['dim'][k])
        for j in range(dim - 1):
            for side in [1., -1.]:
                ray = np.zeros(6)
                ray[0] = 1
                ray[j + 1] = side * z['friction'][k, j]
                F, T = sign * frame.T @ ray[:3], sign * frame.T @ ray[3:]
                W.append(np.r_[R.T @ F, R.T @ (T + np.cross(z['pos'][k] - COM, F))])
                L.append(jp[:, dofs].T @ F + jr[:, dofs].T @ T)
    return np.array(W).T, np.array(L).T, wrench, reaction, local


def main(args):
    if args.output.exists():
        raise ValueError('Fresh saved support review')
    r = json.loads(args.report.read_text())
    for pin in r['sources']:
        verify_file_pin(pin)
    cfg = tomllib.loads(verify_file_pin(r['config']).read_text())
    base = json.loads(verify_file_pin(r['run']).read_text())
    case = next(c for c in base['cases'] if c['name'] == cfg['case_name'] and c['dt'] == cfg['dt'])
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(case['host_topology'])))
    for k, v in dict(np.load(verify_file_pin(case['GPU_physical_context']))).items():
        getattr(model, k)[:] = v[0]
    if int(model.opt.cone) != 0:
        raise ValueError('Recorded pyramidal contact solver only')
    selected = np.load(verify_file_pin(r['artifacts']['selected']))
    qids, dofs, aids = [selected[k] for k in ['qids', 'dofs', 'aids']]
    jids = model.actuator_trnid[aids, 0]
    body = model.body('object/cube').id
    trace = np.load(verify_file_pin(case['trace']))
    hand = {g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('robot/') and (model.geom_contype[g] or model.geom_conaffinity[g])}
    parts = {g for g in range(model.ngeom) if model.geom_bodyid[g] == body and (model.geom_contype[g] or model.geom_conaffinity[g])}
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_setState(model, data, trace['initial_native_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
    data.qpos[:] = selected['qpos']
    data.qvel[:] = 0
    data.ctrl[aids] = selected['ctrl20']
    mujoco.mj_forward(model, data)
    pairs = [(a, b) for a, b in eligible_pairs(model, sorted(parts | hand)) if (a in parts) != (b in parts)] + eligible_pairs(model, sorted(hand))
    gaps = np.array([mujoco.mj_geomDistance(model, data, a, b, .02, np.zeros(6)) for a, b in pairs])
    checks = dict(complete_geometry_pairs=np.array_equal(pairs, selected['pairs']) and len(pairs) == 252,
                  native_geometry_readback=bool(np.max(abs(gaps - selected['gaps'])) < 1e-12))
    records = {}
    for name in ['moving', 'static']:
        row = np.load(verify_file_pin(r['artifacts'][name]))
        contacts = np.load(verify_file_pin(r['artifacts'][name + '_contacts']))
        W, L, wrench, reaction, local = basis_and_actual(model, data, body, hand, dofs, contacts)
        checks[name + '_EFC_labels'] = bool(np.max(abs(local - contacts['local_force6'])) < 1e-12)
        err = wrench - row['wrench_BODY_COM'][0]
        checks[name + '_wrench_identity'] = bool(np.linalg.norm(err[:3]) < 1e-4 and np.linalg.norm(err[3:]) < 1e-5)
        checks[name + '_pose'] = bool(np.array_equal(row['qpos'][0], selected['qpos']))
        if name == 'moving':
            dynamic = static_feasibility(W, selected['target_BODY'], radius=.125)
            checks['full_dynamic_cone_primal'] = bool(dynamic['success'])
            records.update(dynamic_W=W, dynamic_coefficients=dynamic['witness']['ray_coefficients_N'])
            force_err = np.linalg.norm(wrench[:3] - selected['target_BODY'][:3])
            torque_err = np.linalg.norm(wrench[3:] - selected['target_BODY'][3:])
            checks['moving_full_target'] = bool(force_err <= cfg['force_scale_N'] and torque_err <= cfg['torque_scale_Nm'])
            checks['source_velocity'] = bool(np.array_equal(row['qvel'][0], trace['initial_qvel']))
        else:
            other = row['qfrc_constraint'][0, dofs] - reaction
            offset = data.qfrc_bias[dofs] - data.qfrc_passive[dofs] - other
            q = data.qpos[qids]
            qref0 = np.load(verify_file_pin(r['clip']))['reference_qpos'][0].astype(float)
            lo = np.maximum.reduce([q - cfg['preload_bound_rad'], qref0 - cfg['joint_bound_rad'], model.jnt_range[jids, 0], model.actuator_ctrlrange[aids, 0]])
            hi = np.minimum.reduce([q + cfg['preload_bound_rad'], qref0 + cfg['joint_bound_rad'], model.jnt_range[jids, 1], model.actuator_ctrlrange[aids, 1]])
            gain = model.actuator_gainprm[aids, 0]
            bias = model.actuator_biasprm[aids, 0] + model.actuator_biasprm[aids, 1] * data.actuator_length[aids]
            cap_lo = np.maximum(model.actuator_forcerange[aids, 0], model.jnt_actfrcrange[jids, 0])
            cap_hi = np.minimum(model.actuator_forcerange[aids, 1], model.jnt_actfrcrange[jids, 1])
            lower, upper = [np.clip(gain * u + bias, cap_lo, cap_hi) for u in [lo, hi]]
            target_static = np.r_[data.xmat[body].reshape(3, 3).T @ (-model.body_mass[body] * model.opt.gravity), np.zeros(3)]
            static = static_feasibility(W, target_static, loads=L, offset=offset, lower=lower, upper=upper, radius=.125)
            checks['static_bounded_primal'] = bool(static['success'])
            records.update(static_W=W, static_L=L, static_offset=offset, static_lower=lower, static_upper=upper,
                           static_target=target_static, static_coefficients=static['witness']['ray_coefficients_N'])
    target = selected['target_BODY']
    acc = selected['target_joint_acceleration']
    rounds, score_errors = [], []
    for i in range(cfg['iterations']):
        rows = np.load(verify_file_pin(r['artifacts'][f'round_{i:02}']))['candidates']
        if rows.shape != (512, 112):
            raise ValueError('Exact saved population')
        score = (((rows[:, 80:86] - target) / np.array([cfg['force_scale_N']] * 3 + [cfg['torque_scale_Nm']] * 3))**2).sum(axis=1)
        score += cfg['hand_accel_weight'] * (((rows[:, 86:106] - acc) / cfg['hand_accel_scale_rad_s2'])**2).mean(axis=1)
        score += cfg['joint_regularization'] * ((rows[:, :20] / cfg['joint_bound_rad'])**2).sum(axis=1)
        score += (np.maximum(rows[:, 106] - cfg['object_penetration_m'], 0) / cfg['collision_scale_m'])**2
        score += (np.maximum(rows[:, 107] - cfg['self_penetration_m'], 0) / cfg['self_collision_scale_m'])**2
        score += cfg['nonhand_contact_penalty'] * rows[:, 108]
        score_errors.append(bool(np.allclose(score, rows[:, -1], rtol=1e-5, atol=1e-4)))
        rounds.append(rows)
    allrows = np.concatenate(rounds)
    chosen = allrows[np.argmin(allrows[:, -1])]
    checks['all_scores_reconstructed'] = all(score_errors)
    checks['fixed_budget_global_selection'] = bool(np.array_equal(chosen, selected['candidate_row']) and len(allrows) == 10240)
    checks['selected_pose_and_control'] = bool(np.array_equal(chosen[40:60], selected['qpos'][qids]) and np.array_equal(chosen[60:80], selected['ctrl20']))
    checks['unchanged_nonjoint'] = bool(np.array_equal(np.delete(selected['qpos'], qids), np.delete(selected['source_qpos'], qids)))
    checks['object_and_self_geometry'] = bool(gaps[:63].min() >= -cfg['object_penetration_m'] and gaps[63:].min() >= -cfg['self_penetration_m'])
    args.output.mkdir()
    path = args.output / 'primal_readback.npz'
    np.savez_compressed(path, **records, gaps=gaps)
    result = dict(schema='GPU_pen_support_saved_review:1', run=file_pin(args.report), source=file_pin(__file__),
                  checks=checks, passed=sum(checks.values()), total=len(checks), primals=file_pin(path),
                  moving_force_error_N=float(force_err), moving_torque_error_Nm=float(torque_err),
                  neural_updates=0, simulation_integration_steps=0,
                  scope='Saved candidate score/selection, EFC, full geometry and numerical LP primal review only; static smooth/Jacobian readback uses native double parameters, not a physical rerun or trajectory success.')
    (args.output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not all(checks.values()):
        raise ValueError('Preserve failed independent review')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
