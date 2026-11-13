"""Fixed-budget actual-GPU dynamic-wrench proposals, without neural training."""
from pathlib import Path
import argparse
import json
import sys
import time
import tomllib
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.runtime.gpu_contact_oracle import GPUContactOracle
from supermimic_research.runtime.gpu_contact_proposals import GPUContactMetrics, proposal_score
from supermimic_research.mechanics.compound_collision import eligible_pairs
from supermimic_research.mechanics.indexed_contact_support import indexed_static_support, friction_rays
from supermimic_research.mechanics.support import static_feasibility
from tools.audit_gpu_contact_oracle import harvest, contact_wrenches, save


def wrench_basis(model, body, hand, R, COM, contacts):
    rays = []
    for k, (a, b) in enumerate(contacts['geom']):
        oa, ob = model.geom_bodyid[a] == body, model.geom_bodyid[b] == body
        if oa == ob or (a if ob else b) not in hand or contacts['efc_address'][k, 0] < 0:
            continue
        frame = contacts['frame'][k].reshape(3, 3)
        sign = 1. if ob else -1.
        for ray in friction_rays(int(contacts['dim'][k]), contacts['friction'][k]):
            F = sign * frame.T @ ray[:3]
            T = sign * frame.T @ ray[3:] + np.cross(contacts['pos'][k] - COM, F)
            rays.append(np.r_[R.T @ F, R.T @ T])
    return np.array(rays).reshape(-1, 6).T


def main(args):
    import torch
    import mujoco_warp as mw
    if args.output.exists():
        raise ValueError('Fresh fixed-budget support search')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    if (cfg['population'], cfg['iterations'], cfg['elites'], cfg['seed']) != (512, 20, 32, 6201):
        raise ValueError('Fixed proposal budget and seed')
    rp = (cp.parent / cfg['run']).resolve()
    r = json.loads(rp.read_text())
    ap = (cp.parent / cfg['oracle_audit']).resolve()
    audit = json.loads(ap.read_text())
    needed = ['static_saved_replay', 'static_EFC_identity', 'moving_EFC_identity',
              'batch_matches_individual', 'capacity', 'no_integration']
    if not all(audit['checks'][k] for k in needed):
        raise ValueError('Oracle fidelity prerequisite')
    for pin in audit['sources']:
        verify_file_pin(pin)
    case = next(c for c in r['cases'] if c['name'] == cfg['case_name'] and c['dt'] == cfg['dt'])
    trace = np.load(verify_file_pin(case['trace']))
    snap = np.load(verify_file_pin(case['initialization']['snapshot']))
    model = mujoco.MjModel.from_binary_path(str(verify_file_pin(case['host_topology'])))
    for key, value in dict(np.load(verify_file_pin(case['GPU_physical_context']))).items():
        getattr(model, key)[:] = value[0]
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    mujoco.mj_setState(model, data, trace['initial_native_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
    p = json.loads(verify_file_pin(r['projection']).read_text())
    pc = verify_file_pin(p['config'])
    projection_cfg = tomllib.loads(pc.read_text())
    clip_path = (pc.parent / projection_cfg['source_clip']).resolve()
    qref = np.load(clip_path)['reference_qpos'].astype(np.float64)
    target_accel = (2 * qref[0] - 5 * qref[1] + 4 * qref[2] - qref[3]) / .02**2
    wr = json.loads(verify_file_pin(r['dynamic_wrench']).read_text())
    target = np.load(verify_file_pin(wr['targets']))['wrench_BODY_COM'][0]
    out = args.output.resolve()
    out.mkdir()
    sources = [file_pin(__file__)] + [file_pin(ROOT / f) for f in [
        'src/supermimic_research/runtime/gpu_contact_oracle.py',
        'src/supermimic_research/runtime/gpu_contact_proposals.py',
        'src/supermimic_research/mechanics/indexed_contact_support.py',
        'src/supermimic_research/mechanics/warp_contact_labels.py',
        'src/supermimic_research/mechanics/compound_collision.py',
        'src/supermimic_research/mechanics/support.py', 'tools/audit_gpu_contact_oracle.py']]
    intent = dict(schema='GPU_pen_dynamic_support_search:1', config=file_pin(cp), run=file_pin(rp),
                  audit=file_pin(ap), clip=file_pin(clip_path), sources=sources,
                  role='DIAGNOSTIC_ONLY_NOT_EXPERT_NOT_FIT', neural_updates=0, simulation_integration_steps=0)
    (out / 'intent.json').write_text(json.dumps(intent, indent=2) + '\n')
    start = time.perf_counter()
    batch = GPUContactOracle(model, data, cfg['population'], nconmax=cfg['nconmax'], njmax=cfg['njmax'])
    metrics = GPUContactMetrics(batch)
    source_q = torch.as_tensor(snap['source_initial_qpos'][batch.qids], device='cuda', dtype=torch.float32)
    ref0 = torch.as_tensor(qref[0], device='cuda', dtype=torch.float32)
    jids = model.actuator_trnid[batch.aids, 0]
    lower = torch.as_tensor(np.maximum(model.jnt_range[jids, 0], model.actuator_ctrlrange[batch.aids, 0]), device='cuda', dtype=torch.float32)
    upper = torch.as_tensor(np.minimum(model.jnt_range[jids, 1], model.actuator_ctrlrange[batch.aids, 1]), device='cuda', dtype=torch.float32)
    qlo, qhi = torch.maximum(lower, source_q - cfg['joint_bound_rad']), torch.minimum(upper, source_q + cfg['joint_bound_rad'])
    ulo, uhi = torch.maximum(lower, ref0 - cfg['joint_bound_rad']), torch.minimum(upper, ref0 + cfg['joint_bound_rad'])
    target_gpu = torch.as_tensor(target, device='cuda', dtype=torch.float32)
    accel_gpu = torch.as_tensor(target_accel, device='cuda', dtype=torch.float32)
    center = torch.cat([batch.base_qpos[batch.qindex] - source_q,
                        batch.base_ctrl[batch.aindex] - batch.base_qpos[batch.qindex]])
    std = torch.as_tensor([cfg['initial_joint_std_rad']] * 20 + [cfg['initial_preload_std_rad']] * 20, device='cuda')
    initial_std = std.clone()
    generator = torch.Generator(device='cuda').manual_seed(cfg['seed'])
    best_param = center.clone()
    best_score = float('inf')
    best_row = None
    pins = {}
    history = []

    def evaluate(params):
        q = (source_q + params[:, :20]).clamp(qlo, qhi)
        lo = torch.maximum(ulo, q - cfg['preload_bound_rad'])
        hi = torch.minimum(uhi, q + cfg['preload_bound_rad'])
        u = (q + params[:, 20:]).clamp(lo, hi)
        bounded = torch.cat([q - source_q, u - q], dim=1)
        result = batch.query(q, u)
        contact = metrics.read()
        score = proposal_score(result['wrench_BODY_COM'], result['hand_qacc'], contact,
                               target_gpu, accel_gpu, q - source_q, cfg)
        packed = torch.cat([bounded, q, u, result['wrench_BODY_COM'], result['hand_qacc'], contact, score[:, None]], dim=1).cpu().numpy()
        return bounded, score, packed

    # Declared preflight: same corrected center repeated in all 512 worlds.
    _, _, baseline = evaluate(center[None].expand(cfg['population'], -1))
    pins['preflight'] = save(out, 'preflight', dict(candidates=baseline))
    setup_s = time.perf_counter() - start
    search_start = time.perf_counter()
    for iteration in range(cfg['iterations']):
        params = center + torch.randn((cfg['population'], 40), generator=generator, device='cuda') * std
        params[0] = torch.cat([batch.base_qpos[batch.qindex] - source_q,
                               batch.base_ctrl[batch.aindex] - batch.base_qpos[batch.qindex]])
        params[1] = best_param
        params[2].zero_()  # Original reference with no preload, not a new seed.
        bounded, scores, rows = evaluate(params)
        index = int(np.argmin(rows[:, -1]))
        if rows[index, -1] < best_score:
            best_score = float(rows[index, -1])
            best_param = bounded[index].clone()
            best_row = rows[index].copy()
        elites = bounded[torch.topk(scores, cfg['elites'], largest=False).indices]
        alpha = cfg['mean_update']
        center = alpha * elites.mean(dim=0) + (1 - alpha) * center
        std = (alpha * elites.std(dim=0, unbiased=False) + (1 - alpha) * std).clamp(min=cfg['minimum_std_rad'])
        std = torch.minimum(std, initial_std)
        path = out / f'round_{iteration:02}.npz'
        np.savez_compressed(path, candidates=rows, center=center.cpu().numpy(), std=std.cpu().numpy())
        pins[f'round_{iteration:02}'] = file_pin(path)
        history.append(dict(iteration=iteration, best_score=best_score,
                            selected_force_error_N=float(np.linalg.norm(best_row[80:83] - target[:3])),
                            selected_torque_error_Nm=float(np.linalg.norm(best_row[83:86] - target[3:]))))
        print(json.dumps(history[-1]), flush=True)
    search_s = time.perf_counter() - search_start
    # Independently query the selected candidate in one world and preserve raw EFC.
    single = GPUContactOracle(model, data, 1, nconmax=cfg['nconmax'], njmax=cfg['njmax'])
    single_metrics = GPUContactMetrics(single)
    q = torch.as_tensor(best_row[40:60], device='cuda')[None]
    u = torch.as_tensor(best_row[60:80], device='cuda')[None]
    moving = single.readback(single.query(q, u))
    moving_metrics = single_metrics.read().cpu().numpy()
    moving_contacts = harvest(single)
    pins['moving'] = save(out, 'selected_moving', moving)
    pins['moving_contacts'] = save(out, 'selected_moving_contacts', moving_contacts)
    host = mujoco.MjData(model)
    mw.get_data_into(host, model, single.data)
    host.contact.adhesion[:] = 0
    R = host.xmat[single.body].reshape(3, 3).copy()
    COM = host.xipos[single.body].copy()
    hand = {g for g in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('robot/')
            and (model.geom_contype[g] or model.geom_conaffinity[g])}
    parts = {g for g in range(model.ngeom) if model.geom_bodyid[g] == single.body and (model.geom_contype[g] or model.geom_conaffinity[g])}
    W = wrench_basis(model, single.body, hand, R, COM, moving_contacts)
    dynamic_cone = static_feasibility(W, target, radius=.125)
    actual = contact_wrenches(model, single.body, R, COM, moving_contacts, 1)
    identity = actual - moving['wrench_BODY_COM']
    allpairs = eligible_pairs(model, sorted(parts | hand))
    pairs = [(a, b) for a, b in allpairs if (a in parts) != (b in parts)] + eligible_pairs(model, sorted(hand))
    mujoco.mj_kinematics(model, host)
    mujoco.mj_comPos(model, host)
    gaps = np.array([mujoco.mj_geomDistance(model, host, a, b, .02, np.zeros(6)) for a, b in pairs])
    cross_count = sum((a in parts) != (b in parts) for a, b in pairs)
    # A separate actual static query keeps this diagnostic distinct from the moving target.
    single.base_qvel.zero_()
    static = single.readback(single.query(q, u))
    static_contacts = harvest(single)
    mw.get_data_into(host, model, single.data)
    host.contact.adhesion[:] = 0
    q_np, u_np = q.cpu().numpy()[0].astype(float), u.cpu().numpy()[0].astype(float)
    lo = np.maximum(np.maximum(q_np - cfg['preload_bound_rad'], lower.cpu().numpy()), ref0.cpu().numpy() - cfg['joint_bound_rad'])
    hi = np.minimum(np.minimum(q_np + cfg['preload_bound_rad'], upper.cpu().numpy()), ref0.cpu().numpy() + cfg['joint_bound_rad'])
    support = indexed_static_support(model, host, single.body, parts, hand, single.dofs, single.aids, lo, hi)
    pins['static'] = save(out, 'selected_static', static)
    pins['static_contacts'] = save(out, 'selected_static_contacts', static_contacts)
    force_error = float(np.linalg.norm(moving['wrench_BODY_COM'][0, :3] - target[:3]))
    torque_error = float(np.linalg.norm(moving['wrench_BODY_COM'][0, 3:] - target[3:]))
    q_full = moving['qpos'][0]
    source = snap['source_initial_qpos']
    bounds = bool(np.all(abs(q_np - source[single.qids]) <= cfg['joint_bound_rad'] + 1e-7)
                  and np.all(abs(u_np - qref[0]) <= cfg['joint_bound_rad'] + 1e-7)
                  and np.all(abs(u_np - q_np) <= cfg['preload_bound_rad'] + 1e-7)
                  and np.all(q_np >= lower.cpu().numpy()) and np.all(q_np <= upper.cpu().numpy())
                  and np.all(u_np >= lower.cpu().numpy()) and np.all(u_np <= upper.cpu().numpy()))
    checks = dict(full_force=force_error <= cfg['force_scale_N'], full_moment=torque_error <= cfg['torque_scale_Nm'],
                  dynamic_cone=bool(dynamic_cone['success']), bounds=bounds,
                  all_pair_object_geometry=bool(gaps[:cross_count].min() >= -cfg['object_penetration_m']),
                  all_pair_self_geometry=bool(gaps[cross_count:].min() >= -cfg['self_penetration_m']),
                  no_nonhand_contact=bool(moving_metrics[0, 2] == 0),
                  capacity=bool(moving_contacts['capacity_ok'] and static_contacts['capacity_ok']),
                  EFC_force_identity=bool(np.linalg.norm(identity[0, :3]) <= .0001 and np.linalg.norm(identity[0, 3:]) <= .00001),
                  source_velocity=bool(np.array_equal(moving['qvel'][0], trace['initial_qvel'])),
                  object_wrist_unchanged=bool(np.array_equal(np.delete(q_full, single.qids), np.delete(source, single.qids))),
                  finite=bool(all(np.isfinite(v).all() for v in moving.values())))
    pins['selected'] = save(out, 'selected', dict(qpos=q_full, delta20=u_np - qref[0], ctrl20=u_np,
        source_qpos=source, qids=single.qids, dofs=single.dofs, aids=single.aids,
        target_BODY=target, target_joint_acceleration=target_accel, candidate_row=best_row,
        moving_wrench_basis=W, pairs=np.array(pairs), gaps=gaps,
        static_wrench_basis=support['wrench_BODY'], static_target_BODY=support['target_BODY']))
    for pin in sources:
        verify_file_pin(pin)
    result = dict(**intent, artifacts=pins, history=history, checks=checks, passed=sum(checks.values()), total=len(checks),
                  proposal_qualified=all(checks.values()), selected_score=best_score,
                  force_error_N=force_error, torque_error_Nm=torque_error,
                  max_object_penetration_m=float(max(0, -gaps[:cross_count].min())),
                  max_self_penetration_m=float(max(0, -gaps[cross_count:].min())),
                  geometry_pair_counts=dict(object_hand=cross_count, hand_self=len(pairs) - cross_count),
                  static_bounded_PD_feasible=bool(support['lp']['success']),
                  dynamic_cone_feasible=bool(dynamic_cone['success']),
                  static_unbounded_cone_feasible=bool(static_feasibility(support['wrench_BODY'], support['target_BODY'], radius=.125)['success']),
                  nominal_reference_uncertainty='Original 4-point derivatives and nominal simulator inertia; not measured source contact force.',
                  setup_and_preflight_s=setup_s, search_loop_including_candidate_files_s=search_s,
                  query_counts=dict(search_candidates=cfg['population'] * cfg['iterations'],
                                    preflight=cfg['population'], explicit_batch_total=batch.query_count,
                                    selected_single=single.query_count,
                                    setup_forward_API_calls_per_oracle=2),
                  candidate_columns='joint_delta20,preload20,q20,ctrl20,wrench6,hand_qacc20,contact_metrics5,score1',
                  scope='One fixed-seed bounded physical proposal search. Selection is not test/generalization evidence. No integration, neural training, source time stretch, object pose/force assistance or complete-task success.')
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['checks', 'proposal_qualified', 'force_error_N', 'torque_error_Nm', 'static_bounded_PD_feasible', 'query_counts']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
