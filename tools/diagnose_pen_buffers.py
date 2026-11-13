"""Fresh/reused data and source-row aligned contact/solver diagnosis."""
from pathlib import Path
import argparse
import json
import sys
import tomllib
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.runtime.gpu_model_snapshot import restore_model_arrays, capture_model
from supermimic_research.runtime.gpu_pen_rollout import GPUPenRollout
from supermimic_research.mechanics.constraint_alignment import canonical_constraint_rows
from tools.audit_gpu_contact_oracle import harvest
from tools.audit_gpu_pen_steps import discrepancies


def detailed_receipt(clone):
    c = harvest(clone)
    count = len(c['geom'])
    c['geomcollisionid'] = clone.data.contact.geomcollisionid.numpy()[:count].copy()
    n = int(c['nefc'].max())
    if clone.wp_model.is_sparse:
        raise ValueError('This controlled dense Jacobian diagnosis does not decode sparse layouts')
    f = {key: getattr(clone.data.efc, key).numpy()[:, :n].copy()
         for key in ['type', 'id', 'D', 'aref', 'force']}
    f['J'] = clone.data.efc.J.numpy()[:, :n, :clone.model.nv].copy()
    result = {'contact_' + k: v for k, v in c.items()}
    result.update({'constraint_' + k: v for k, v in f.items()})
    for key in ['qfrc_smooth', 'qacc_smooth', 'qfrc_bias', 'qfrc_passive', 'solver_niter']:
        result[key] = getattr(clone.data, key).numpy().copy()
    base = canonical_constraint_rows(c, f, 0)
    errors = {key: 0. for key in ['J', 'D', 'aref', 'force']}
    same_keys = True
    row_orders = set()
    contact_keys = []
    for world in range(clone.nworld):
        aligned = canonical_constraint_rows(c, f, world)
        row_orders.add(tuple(aligned['source_addresses']))
        if not np.array_equal(base['keys'], aligned['keys']):
            same_keys = False
            continue
        for key in errors:
            errors[key] = max(errors[key], float(np.max(abs(aligned['fields'][key] - base['fields'][key]), initial=0)))
        ids = np.flatnonzero(c['worldid'] == world)
        contact_keys.append(tuple(sorted((int(c['geom'][k, 0]), int(c['geom'][k, 1]), int(c['geomcollisionid'][k])) for k in ids)))
    summary = dict(all_physical_constraint_keys_equal=same_keys, aligned_max_errors=errors,
                   source_row_orders=len(row_orders), contact_modes=len(set(contact_keys)),
                   solver_iterations_min=int(result['solver_niter'].min()), solver_iterations_max=int(result['solver_niter'].max()),
                   solver_at_cap_worlds=int(np.sum(result['solver_niter'] >= clone.wp_model.opt.iterations)))
    return result, summary


def main(args):
    import torch
    import warp as wp
    import mujoco_warp as mw
    if args.output.exists():
        raise ValueError('Fresh causal buffer diagnosis')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    if cfg['anchors_s'] != [0., .02, .04] or cfg['nworld'] != 128 or cfg['repeats'] != 3:
        raise ValueError('Fixed source anchors, worlds and repetitions')
    sp = (cp.parent / cfg['source']).resolve()
    source = json.loads(sp.read_text())
    pp = (cp.parent / cfg['proposal']).resolve()
    proposal = json.loads(pp.read_text())
    selected = np.load(verify_file_pin(proposal['artifacts']['selected']))
    qref = np.load(verify_file_pin(proposal['clip']))['reference_qpos']
    gates_path = (cp.parent / cfg['gates']).resolve()
    gates = tomllib.loads(gates_path.read_text())
    out = args.output.resolve()
    out.mkdir()
    sources = [file_pin(__file__)] + [file_pin(ROOT / p) for p in [
        'src/supermimic_research/runtime/gpu_model_snapshot.py', 'src/supermimic_research/runtime/gpu_pen_rollout.py',
        'src/supermimic_research/runtime/gpu_contact_oracle.py', 'src/supermimic_research/mechanics/constraint_alignment.py',
        'tools/audit_gpu_contact_oracle.py', 'tools/audit_gpu_pen_steps.py']]
    intent = dict(schema='pen_GPU_buffer_diagnosis:1', config=file_pin(cp), source_run=file_pin(sp),
                  proposal=file_pin(pp), gates=file_pin(gates_path), sources=sources, neural_updates=0,
                  role='CAUSAL_PHYSICS_DIAGNOSTIC_NOT_EXPERT_NOT_FIT')
    (out / 'intent.json').write_text(json.dumps(intent, indent=2) + '\n')
    torch.set_num_threads(4)
    wp.set_device('cuda:0')
    cases = []
    for dt in cfg['dt']:
        src = next(c for c in source['cases'] if c['dt'] == dt)
        z = np.load(verify_file_pin(src['trace']))
        inventory = next(p['inventory'] for p in source['probes'] if p['dt'] == dt)
        values = dict(np.load(verify_file_pin(inventory['snapshot'])))
        scalars = json.loads(verify_file_pin(inventory['scalars']).read_text())
        m = mujoco.MjModel.from_binary_path(str(verify_file_pin(src['host_topology'])))
        for key, value in dict(np.load(verify_file_pin(src['GPU_physical_context']))).items():
            getattr(m, key)[:] = value[0]
        device_model = mw.put_model(m)
        restore_model_arrays(device_model, values, scalars)
        _, _, model_before = capture_model(device_model)
        for anchor in cfg['anchors_s']:
            tick0 = round(anchor / dt)
            phase0 = round(anchor / .02) + 1
            data = mujoco.MjData(m)
            mujoco.mj_setConst(m, data)
            mujoco.mj_setState(m, data, z['initial_native_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
            if tick0:
                for key, trace_key in [('qpos', 'post_qpos'), ('qvel', 'post_qvel'), ('ctrl', 'post_ctrl'), ('qacc_warmstart', 'post_qacc')]:
                    getattr(data, key)[:] = z[trace_key][tick0 - 1]
                data.time = float(z['post_GPU_time'][tick0 - 1])
            expected_initial = {key: getattr(data, key).copy().astype(np.float32) for key in ['qpos', 'qvel', 'ctrl', 'qacc_warmstart']}
            for mode in cfg['modes']:
                clone = None
                for rep in range(cfg['repeats']):
                    if clone is None or mode == 'FRESH_DATA':
                        clone = GPUPenRollout(m, data, cfg['nworld'], selected['delta20'], device_model=device_model, nconmax=128, njmax=512)
                    clone.reset_clone()
                    initial = {k: v.cpu().numpy().copy() for k, v in clone.state().items()}
                    folder = out / f'{dt:g}_{anchor:g}_{mode}_{rep}'
                    folder.mkdir()
                    reference = torch.as_tensor(qref, device='cuda', dtype=torch.float32)
                    raw = torch.as_tensor(2 * selected['delta20'], device='cuda', dtype=torch.float32)[None].expand(cfg['nworld'], -1)
                    steps = round(.1 / dt) if anchor == 0 else 1
                    rows, details, iterations = [], [], []
                    first_pre = None
                    for tick in range(steps):
                        if tick % clone.decimation == 0:
                            clone.set_action(raw, reference[phase0 + tick // clone.decimation])
                        pre = clone.views['qpos'].cpu().numpy().copy()
                        clone.physics_step()
                        row = clone.readback(clone.output())
                        row['pre_qpos'] = pre
                        row['warmstart'] = clone.views['qacc_warmstart'].cpu().numpy().copy()
                        row['solver_niter'] = clone.data.solver_niter.numpy().copy()
                        row['overflow'] = clone.views['overflow'].cpu().numpy().copy()
                        iterations.append(row['solver_niter'])
                        rows.append(row)
                        if tick == 0 or (anchor == 0 and tick * dt < cfg['detailed_duration_s']):
                            receipt, summary = detailed_receipt(clone)
                            path = folder / f'constraints_{tick:03}.npz'
                            np.savez_compressed(path, **receipt)
                            details.append(dict(tick=tick, pre_time_s=anchor + tick * dt, receipt=file_pin(path), **summary))
                        if (tick + 1) % clone.decimation == 0:
                            clone.refresh_boundary()
                    actual = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
                    path = folder / 'trace.npz'
                    np.savez_compressed(path, **actual, **{'initial_' + k: v for k, v in initial.items()})
                    initial_exact = all(np.array_equal(initial[k], np.tile(v, (128, 1))) for k, v in expected_initial.items())
                    world0 = dict(post_qpos=actual['qpos'][:, 0], post_qvel=actual['qvel'][:, 0])
                    spread = discrepancies(actual, world0, clone, len(rows))
                    errors = discrepancies(actual, z, clone, len(rows)) if anchor == 0 else None
                    fidelity = None if errors is None else {key: value <= gates[key] for key, value in errors.items()}
                    checks = dict(exact_reset_inputs=initial_exact,
                                  equal_inputs_across_worlds=all(np.array_equal(v, np.broadcast_to(v[:1], v.shape)) for v in initial.values()),
                                  original_control=bool(abs(actual['ctrl'] - z['post_ctrl'][tick0:tick0 + steps, None]).max() <= gates['control_rad']),
                                  capacity=not np.any(actual['overflow']) and np.max(actual['nefc']) <= 512,
                                  no_aid=not bool(clone.views['qfrc_applied'].count_nonzero()) and not bool(clone.views['xfrc_applied'].count_nonzero()),
                                  finite=all(np.isfinite(v).all() for v in actual.values()))
                    checks = {key: bool(value) for key, value in checks.items()}
                    entry = dict(dt=dt, anchor_s=anchor, mode=mode, repeat=rep, checks=checks,
                        world_spread=spread, actual_errors=errors, fidelity=fidelity, trace=file_pin(path), details=details,
                        solver_at_cap_ticks=int(np.sum(np.stack(iterations) >= device_model.opt.iterations)),
                        physical_world_ticks=steps * 128, constructor_setup_world_ticks=128 if mode == 'FRESH_DATA' or rep == 0 else 0)
                    cases.append(entry)
                    (out / 'progress.json').write_text(json.dumps(dict(cases=cases), indent=2) + '\n')
                    print(json.dumps(dict(dt=dt, anchor=anchor, mode=mode, repeat=rep, checks=checks,
                        spread_object_m=spread['object_position_m'], solver_cap_world_ticks=entry['solver_at_cap_ticks'], first_detail=details[0]['aligned_max_errors'])), flush=True)
        _, _, after = capture_model(device_model)
        if model_before != after:
            raise ValueError('Readonly full device model changed')
    for pin in sources:
        verify_file_pin(pin)
    result = dict(**intent, cases=cases, integrity_passed=sum(sum(c['checks'].values()) for c in cases),
        integrity_total=sum(len(c['checks']) for c in cases),
        fidelity_passed=sum(sum(c['fidelity'].values()) for c in cases if c['fidelity'] is not None),
        fidelity_total=sum(len(c['fidelity']) for c in cases if c['fidelity'] is not None),
        ready_for_100ms_planning=all(all(c['fidelity'].values()) for c in cases if c['fidelity'] is not None),
        scope='Fresh versus same-buffer reset probes with exact full-model snapshots and actual saved warmstart/control phase. Offline canonical row alignment only; no engine reordering, numerical/physical option edits, policy planning, neural training or success claim.')
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
