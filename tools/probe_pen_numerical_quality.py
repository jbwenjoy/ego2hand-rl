"""Declared numerical-quality interventions at every failed actual impulse state."""
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
from supermimic_research.runtime.gpu_pen_rollout import GPUPenRollout
from supermimic_research.runtime.gpu_model_snapshot import restore_model_arrays
from tools.audit_gpu_contact_oracle import harvest
from supermimic_research.runtime.numerical_options import set_model_numerical_options


def main(args):
    import warp as wp
    import mujoco_warp as mw
    if args.output.exists():
        raise ValueError('Fresh fixed numerical-quality probe')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    rp, ep, mp = [(cp.parent / cfg[k]).resolve() for k in ['run', 'efc_review', 'model_source']]
    run, efc, models = [json.loads(path.read_text()) for path in [rp, ep, mp]]
    if verify_file_pin(efc['run']).resolve() != rp:
        raise ValueError('Exact original impulse review')
    out = args.output.resolve()
    out.mkdir()
    sources = [file_pin(__file__)] + [file_pin(ROOT / p) for p in [
        'src/supermimic_research/runtime/gpu_pen_rollout.py', 'src/supermimic_research/runtime/gpu_contact_oracle.py',
        'src/supermimic_research/runtime/gpu_model_snapshot.py', 'src/supermimic_research/runtime/numerical_options.py', 'tools/audit_gpu_contact_oracle.py']]
    intent = dict(schema='pen_numerical_quality_probe:1', config=file_pin(cp), run=file_pin(rp),
                  EFC_review=file_pin(ep), model_source=file_pin(mp), sources=sources, neural_updates=0)
    (out / 'intent.json').write_text(json.dumps(intent, indent=2) + '\n')
    wp.set_device('cuda:0')
    results = []
    for e in efc['cases']:
        if e['checks']['linear_impulse']:
            continue
        c = next(c for c in run['cases'] if c['name'] == e['name'] and c['dt'] == e['dt'])
        z = np.load(verify_file_pin(c['trace']))
        w = np.load(verify_file_pin(e['wrenches']))
        bad = np.flatnonzero(np.linalg.norm(w['linear_impulse_residual_N_s'], axis=1) > cfg['impulse_tolerance_N_s'])
        model_case = next(k for k in models['cases'] if k['dt'] == c['dt'])
        if c['host_topology']['sha256'] != model_case['host_topology']['sha256']:
            raise ValueError('Same compiled host topology prerequisite')
        physical = dict(np.load(verify_file_pin(c['GPU_physical_context'])))
        old_physical = dict(np.load(verify_file_pin(model_case['GPU_physical_context'])))
        if any(not np.array_equal(v, old_physical[k]) for k, v in physical.items()):
            raise ValueError('All source physical fields must match exact snapshot')
        inventory = next(k['inventory'] for k in models['probes'] if k['dt'] == c['dt'])
        values = dict(np.load(verify_file_pin(inventory['snapshot'])))
        scalars = json.loads(verify_file_pin(inventory['scalars']).read_text())
        for tick in bad:
            if tick < 1:
                raise ValueError('This failure-triggered protocol requires a captured preceding warmstart')
            for variant in cfg['variants']:
                m = mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])))
                for key, value in physical.items():
                    getattr(m, key)[:] = value[0]
                device = mw.put_model(m)
                restore_model_arrays(device, values, scalars)
                if 'numerical_options' in cfg:
                    set_model_numerical_options(m,device,cfg['numerical_options'][variant])
                elif variant in ['ITER50', 'TIGHT']:
                    m.opt.iterations = device.opt.iterations = cfg['iterations']
                if 'numerical_options' not in cfg and variant == 'TIGHT':
                    m.opt.ls_iterations = device.opt.ls_iterations = cfg['ls_iterations']
                    m.opt.tolerance = cfg['tolerance']
                    m.opt.ls_tolerance = cfg['ls_tolerance']
                    device.opt.tolerance.fill_(cfg['tolerance'])
                    device.opt.ls_tolerance.fill_(cfg['ls_tolerance'])
                data = mujoco.MjData(m)
                mujoco.mj_setConst(m, data)
                mujoco.mj_setState(m, data, z['initial_native_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
                data.qpos[:] = z['pre_qpos'][tick]
                data.qvel[:] = z['pre_qvel'][tick]
                data.ctrl[:] = z['post_ctrl'][tick]
                data.qacc_warmstart[:] = z['post_qacc'][tick - 1]
                data.time = float(z['post_GPU_time'][tick - 1])
                clone = GPUPenRollout(m, data, 128, np.zeros(20), device_model=device, nconmax=128, njmax=512)
                body, da = c['model_layout']['object_body'], c['model_layout']['object_dofadr']
                mass = float(physical['body_mass'][0, body])
                for rep in range(3):
                    clone.reset_clone()
                    before = clone.state()
                    clone.physics_step()
                    state = clone.readback(clone.output())
                    contacts = harvest(clone)
                    ids = np.flatnonzero((m.geom_bodyid[contacts['geom'][:, 0]] == body) ^ (m.geom_bodyid[contacts['geom'][:, 1]] == body))
                    sign = np.where(m.geom_bodyid[contacts['geom'][ids, 1]] == body, 1., -1.)
                    frame = contacts['frame'][ids].reshape(-1, 3, 3)
                    force = sign[:, None] * np.einsum('nji,nj->ni', frame, contacts['local_force6'][ids, :3])
                    world = np.zeros((128, 3))
                    np.add.at(world, contacts['worldid'][ids], force)
                    dv = state['qvel'][:, da:da + 3] - data.qvel[da:da + 3]
                    impulse = mass * dv - c['dt'] * (world + mass * m.opt.gravity)
                    qacc = state['qacc'][:, da:da + 3]
                    roundoff = mass * (dv - c['dt'] * qacc)
                    balance = mass * qacc - mass * m.opt.gravity - world
                    niter = clone.data.solver_niter.numpy()
                    folder = out / f'{c["dt"]}_{tick}_{variant}_{rep}'
                    folder.mkdir()
                    np.savez_compressed(folder / 'state.npz', **state, impulse=impulse, roundoff=roundoff,
                        linear_balance=balance, solver_niter=niter,
                        **{'initial_' + k: v.cpu().numpy() for k, v in before.items()})
                    np.savez_compressed(folder / 'contacts.npz', **contacts)
                    force_error = np.linalg.norm(world - state['qfrc_constraint'][:, da:da + 3], axis=1)
                    passed_worlds = np.linalg.norm(impulse, axis=1) <= 5e-7
                    entry = dict(dt=c['dt'], source_name=c['name'], source_tick=int(tick), variant=variant, repeat=rep,
                        maximum_impulse_residual_N_s=float(np.linalg.norm(impulse, axis=1).max()),
                        passed_worlds=int(passed_worlds.sum()), total_worlds=128,
                        maximum_integration_roundoff_N_s=float(np.linalg.norm(roundoff, axis=1).max()),
                        maximum_force_balance_N=float(np.linalg.norm(balance, axis=1).max()),
                        maximum_EFC_identity_error_N=float(force_error.max()),
                        solver_iterations=[int(niter.min()), int(niter.max())],
                        checks=dict(EFC_force_identity=bool(np.all(force_error <= 1e-4)), capacity=bool(contacts['capacity_ok']),
                                    finite=bool(all(np.isfinite(v).all() for v in state.values()))),
                        state=file_pin(folder / 'state.npz'), contacts=file_pin(folder / 'contacts.npz'))
                    results.append(entry)
                    print(json.dumps({k: entry[k] for k in ['dt', 'source_tick', 'variant', 'repeat', 'passed_worlds', 'maximum_impulse_residual_N_s', 'solver_iterations']}), flush=True)
    for pin in sources:
        verify_file_pin(pin)
    result = dict(**intent, cases=results, scope='Explicit local numerical-option interventions at every original failed source state,128 worlds/3 one-step repeats. No physical-parameter edits, original-gate relaxation, full task/100ms qualification, installed kernel change or neural training.')
    (out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
