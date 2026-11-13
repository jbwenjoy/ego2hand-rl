"""Saved-snapshot fidelity, actual EFC wrench and equal-query GPU batching audit."""
from pathlib import Path
import argparse
import json
import sys
import time
import tomllib
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.runtime.gpu_contact_oracle import GPUContactOracle
from supermimic_research.mechanics.warp_contact_labels import decode_local_forces


def harvest(oracle):
    """Audit boundary only: each source array is transferred once, not per world."""
    d = oracle.data
    count = int(d.nacon.numpy()[0])
    limit = min(count, d.naconmax)
    z = {k: getattr(d.contact, k).numpy()[:limit].copy() for k in
         ['worldid', 'dim', 'geom', 'dist', 'pos', 'frame', 'friction', 'efc_address']}
    z['efc_force'] = d.efc.force.numpy()
    z['nefc'] = d.nefc.numpy()
    local = np.zeros((limit, 6))
    invalid = np.zeros(limit, bool)
    for world in range(oracle.nworld):
        ids = np.flatnonzero(z['worldid'] == world)
        local[ids], invalid[ids] = decode_local_forces(
            int(oracle.model.opt.cone), z['dim'][ids], z['friction'][ids],
            z['efc_address'][ids], z['efc_force'][world], int(z['nefc'][world]))
    z['local_force6'] = local
    z['invalid'] = invalid
    z['nacon'] = np.array(count)
    z['capacity_ok'] = np.array(count <= d.naconmax and np.all(z['nefc'] <= d.njmax)
                                and not invalid.any()
                                and np.all((z['worldid'] >= 0) & (z['worldid'] < oracle.nworld)))
    return z


def contact_wrenches(model, body, R, COM, z, nworld):
    """Independent EFC -> local -> world moment arm -> BODY COM aggregation."""
    a, b = z['geom'].T
    oa, ob = model.geom_bodyid[a] == body, model.geom_bodyid[b] == body
    ids = np.flatnonzero(oa ^ ob)
    sign = np.where(ob[ids], 1., -1.)
    frame = z['frame'][ids].reshape(-1, 3, 3)
    F = sign[:, None] * np.einsum('nji,nj->ni', frame, z['local_force6'][ids, :3])
    T = sign[:, None] * np.einsum('nji,nj->ni', frame, z['local_force6'][ids, 3:])
    T += np.cross(z['pos'][ids] - COM, F)
    result = np.zeros((nworld, 6))
    np.add.at(result, z['worldid'][ids], np.c_[F @ R, T @ R])
    return result


def save(out, name, values):
    path = out / (name + '.npz')
    np.savez_compressed(path, **values)
    return file_pin(path)


def main(args):
    import torch
    if args.output.exists():
        raise ValueError('Fresh oracle audit output')
    config_path = args.config.resolve()
    cfg = tomllib.loads(config_path.read_text())
    if cfg['nworld'] != 81 or cfg['repetitions'] != 3 or cfg['fd_step_rad'] != .001:
        raise ValueError('Fixed preregistered query cohort')
    run_path = (config_path.parent / cfg['run']).resolve()
    report = json.loads(run_path.read_text())
    case = next(c for c in report['cases'] if c['name'] == cfg['case_name'] and c['dt'] == cfg['dt'])
    trace = np.load(verify_file_pin(case['trace']))
    snap = np.load(verify_file_pin(case['initialization']['snapshot']))
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(case['host_topology'])))
    physical = np.load(verify_file_pin(case['GPU_physical_context']))
    for key in physical.files:
        getattr(model, key)[:] = physical[key][0]
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_setState(model, data, snap['static_native_integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
    out = args.output.resolve()
    out.mkdir()
    sources = [file_pin(__file__), file_pin(ROOT / 'src/supermimic_research/runtime/gpu_contact_oracle.py'),
               file_pin(ROOT / 'src/supermimic_research/mechanics/warp_contact_labels.py')]
    intent = dict(schema='GPU_contact_oracle_audit:1', config=file_pin(config_path), run=file_pin(run_path),
                  sources=sources, role='PHYSICS_DIAGNOSTIC_NOT_EXPERT_NOT_FIT', neural_updates=0,
                  simulation_integration_steps=0)
    (out / 'intent.json').write_text(json.dumps(intent, indent=2) + '\n')
    setup_start = time.perf_counter()
    oracle = GPUContactOracle(model, data, 1, nconmax=cfg['nconmax'], njmax=cfg['njmax'])
    q0 = oracle.base_qpos[oracle.qindex].clone()
    u0 = oracle.base_ctrl[oracle.aindex].clone()
    static = oracle.readback(oracle.query(q0[None], u0[None]))
    static_contacts = harvest(oracle)
    R, COM = snap['BODY_rotation_world'], snap['COM_world']
    actual_static = contact_wrenches(model, oracle.body, R, COM, static_contacts, 1)
    old = dict(geom=snap['GPU_geom'], frame=snap['GPU_frame'], pos=snap['GPU_position'],
               local_force6=snap['GPU_local_force6'], worldid=np.zeros(len(snap['GPU_geom']), dtype=int))
    saved_static = contact_wrenches(model, oracle.body, R, COM, old, 1)
    static_error = static['wrench_BODY_COM'] - saved_static
    static_identity = static['wrench_BODY_COM'] - actual_static
    pins = dict(static=save(out, 'static', static), static_contacts=save(out, 'static_contacts', static_contacts))
    # Moving queries retain the task's original velocity; the static snapshot
    # is a separate replay test, never silently substituted for source motion.
    data.qvel[:] = trace['initial_qvel']
    single = GPUContactOracle(model, data, 1, nconmax=cfg['nconmax'], njmax=cfg['njmax'])
    batch = GPUContactOracle(model, data, 81, nconmax=cfg['nconmax'], njmax=cfg['njmax'])
    x0 = torch.cat([q0, u0])
    h = cfg['fd_step_rad']
    offsets = torch.cat([torch.zeros((1, 40), device='cuda'), h * torch.eye(40, device='cuda'),
                         -h * torch.eye(40, device='cuda')])
    x = x0[None] + offsets
    # Warm each path with the exact workload before timing.
    batch.readback(batch.query(x[:, :20], x[:, 20:]))
    single.readback(single.query(x[:1, :20], x[:1, 20:]))
    setup_s = time.perf_counter() - setup_start
    batched_times, individual_times = [], []
    all_differences = []
    batched = individual = None
    for rep in range(cfg['repetitions']):
        torch.cuda.synchronize()
        start = time.perf_counter()
        b = batch.readback(batch.query(x[:, :20], x[:, 20:]))
        batched_times.append(time.perf_counter() - start)
        torch.cuda.synchronize()
        start = time.perf_counter()
        rows = [single.readback(single.query(x[i:i + 1, :20], x[i:i + 1, 20:])) for i in range(81)]
        s = {k: np.concatenate([row[k] for row in rows]) for k in rows[0]}
        individual_times.append(time.perf_counter() - start)
        diff = b['wrench_BODY_COM'] - s['wrench_BODY_COM']
        all_differences.append(diff)
        pins[f'batch_{rep}'] = save(out, f'batch_{rep}', b)
        pins[f'individual_{rep}'] = save(out, f'individual_{rep}', s)
        batched, individual = b, s
    contacts = harvest(batch)
    pins['moving_batch_contacts'] = save(out, 'moving_batch_contacts', contacts)
    direct_wrench = contact_wrenches(model, batch.body, R, COM, contacts, 81)
    identity = batched['wrench_BODY_COM'] - direct_wrench
    difference = np.concatenate(all_differences)
    center = batched['wrench_BODY_COM'][0]
    J = ((batched['wrench_BODY_COM'][1:41] - batched['wrench_BODY_COM'][41:]) / (2 * h)).T
    direction = np.sin(np.arange(1, 41, dtype=float))
    direction /= np.linalg.norm(direction)
    direction_gpu = torch.as_tensor(direction, device='cuda', dtype=torch.float32)
    directional = []
    directional_modes = []
    for step in [h, h / 2]:
        values = []
        for side in [1, -1]:
            point = x0 + side * step * direction_gpu
            row = single.readback(single.query(point[None, :20], point[None, 20:]))
            directional.append(row['wrench_BODY_COM'][0])
            values.append(row['wrench_BODY_COM'][0])
            c = harvest(single)
            directional_modes.append(sorted([tuple(map(int, g)) for g, a in zip(c['geom'], c['efc_address']) if a[0] >= 0]))
        # Derivatives are saved below; no autograd through simulator is claimed.
    d_h = (directional[0] - directional[1]) / (2 * h)
    d_half = (directional[2] - directional[3]) / h
    abs_floor = np.r_[np.full(3, cfg['derivative_force_absolute_N_per_rad']),
                      np.full(3, cfg['derivative_torque_absolute_Nm_per_rad'])]
    deriv_limits = abs_floor + cfg['derivative_relative_tolerance'] * np.maximum(abs(d_h), abs(d_half))
    allfinite = all(np.isfinite(v).all() for rows in [static, batched, individual] for v in rows.values())
    joint_ids = model.actuator_trnid[batch.aids, 0]
    lower = np.maximum(model.jnt_range[joint_ids, 0], model.actuator_ctrlrange[batch.aids, 0])
    upper = np.minimum(model.jnt_range[joint_ids, 1], model.actuator_ctrlrange[batch.aids, 1])
    state_ok = (np.array_equal(batched['qvel'], np.tile(trace['initial_qvel'], (81, 1)))
                and np.array_equal(batched['qpos'][:, np.setdiff1d(np.arange(model.nq), batch.qids)],
                                   np.tile(trace['initial_qpos'][np.setdiff1d(np.arange(model.nq), batch.qids)], (81, 1))))
    def errors(value):
        return dict(force_N=float(np.linalg.norm(value[:, :3], axis=1).max()),
                    torque_Nm=float(np.linalg.norm(value[:, 3:], axis=1).max()))
    def force_gate(value):
        e = errors(value)
        return e['force_N'] <= cfg['force_tolerance_N'] and e['torque_Nm'] <= cfg['torque_tolerance_Nm']
    checks = dict(static_saved_replay=force_gate(static_error), static_EFC_identity=force_gate(static_identity),
                  moving_EFC_identity=force_gate(identity), batch_matches_individual=force_gate(difference),
                  finite=allfinite, original_velocity_and_nonjoint_state=state_ok,
                  legal_joint_and_control_bounds=bool(np.all(batched['qpos'][:, batch.qids] >= lower)
                      and np.all(batched['qpos'][:, batch.qids] <= upper)
                      and np.all(batched['ctrl'][:, batch.aids] >= lower)
                      and np.all(batched['ctrl'][:, batch.aids] <= upper)),
                  capacity=bool(static_contacts['capacity_ok'] and contacts['capacity_ok']),
                  no_integration=bool(np.all(batched['time'] == static['time'][0])),
                  directional_step_consistency=bool(np.all(abs(d_h - d_half) <= deriv_limits)),
                  directional_axis_consistency=bool(np.all(abs(J @ direction - d_half) <= deriv_limits)))
    pins['derivatives'] = save(out, 'derivatives', dict(J=J, direction=direction, d_h=d_h, d_half=d_half,
                                                     J_direction=J @ direction, limits=deriv_limits, center_wrench=center))
    for pin in sources:
        verify_file_pin(pin)
    result = dict(**intent, checks=checks, passed=sum(checks.values()), total=len(checks), artifacts=pins,
                  static_saved_error=errors(static_error), static_EFC_error=errors(static_identity),
                  moving_EFC_error=errors(identity), batch_individual_error=errors(difference),
                  derivative_mode_sets=directional_modes, setup_and_warmup_s=setup_s,
                  timing=dict(queries_per_sweep=81, repetitions=3, batched_s=batched_times,
                              individual_s=individual_times,
                              median_query_speed_ratio=float(np.median(individual_times) / np.median(batched_times)),
                              includes='State writes, captured GPU forward, output conversion and aggregate CPU readback; excludes setup, compilation, contact audit and file IO.'),
                  query_counts=dict(static=oracle.query_count, individual=single.query_count, batched=batch.query_count),
                  scope='Actual GPU forward queries, local finite differences and equal-query batching only; not training, WM gradients, full-cost retargeting speed or successful manipulation.')
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['checks', 'passed', 'total', 'static_saved_error', 'batch_individual_error', 'timing']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
