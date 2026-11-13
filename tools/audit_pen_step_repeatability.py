"""Exact GPU model reconstruction and repeated identical-input clone evidence."""
from pathlib import Path
import argparse
import copy
import json
import sys
import tomllib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
from supermimic_research.artifacts import ArtifactRegistry, file_pin, verify_file_pin
from supermimic_research.runtime.controlled_evaluation import evaluate_case, write
from supermimic_research.runtime.pen_gpu_initialization import initialize_and_query_pen
from supermimic_research.runtime.gpu_pen_rollout import GPUPenRollout
from supermimic_research.runtime.gpu_model_snapshot import capture_model, restore_model_arrays
from tools.audit_gpu_pen_steps import discrepancies


def main(args):
    import torch
    import warp as wp
    import mujoco
    import mujoco_warp as mw
    if args.output.exists():
        raise ValueError('Fresh exact-model repeatability experiment')
    cp = args.config.resolve()
    cfg = tomllib.loads(cp.read_text())
    if (cfg['nworld'], cfg['repeats'], cfg['horizon_controls']) != (128, 3, 5):
        raise ValueError('Fixed identical-input repeatability cohort')
    bp = (cp.parent / cfg['base_config']).resolve()
    base = tomllib.loads(bp.read_text())
    gp = (cp.parent / cfg['gates']).resolve()
    gates = tomllib.loads(gp.read_text())
    pp = (cp.parent / cfg['proposal']).resolve()
    proposal = json.loads(pp.read_text())
    selected = np.load(verify_file_pin(proposal['artifacts']['selected']))
    qref = np.load(verify_file_pin(proposal['clip']))['reference_qpos']
    torch.set_num_threads(4)
    wp.set_device('cuda:0')
    registry = ArtifactRegistry(bp.parent / base['artifact_manifest'])
    args.output.mkdir()
    sources = [file_pin(__file__)] + [file_pin(ROOT / p) for p in [
        'src/supermimic_research/runtime/gpu_pen_rollout.py',
        'src/supermimic_research/runtime/gpu_contact_oracle.py',
        'src/supermimic_research/runtime/gpu_model_snapshot.py',
        'src/supermimic_research/runtime/controlled_evaluation.py',
        'src/supermimic_research/runtime/pen_gpu_initialization.py', 'tools/audit_gpu_pen_steps.py']]
    intent = dict(schema='GPU_pen_model_repeatability:1', config=file_pin(cp), base_config=file_pin(bp),
                  gates=file_pin(gp), proposal=file_pin(pp), sources=sources, neural_updates=0)
    write(args.output / 'intent.json', intent)
    cases, probes = [], []
    for dt in cfg['dt']:
        folder = args.output / f'actual_{dt}'
        captures = []
        inventory = {}

        def initialize(env):
            receipt = initialize_and_query_pen(env, folder, selected['source_qpos'], selected['qpos'],
                selected['delta20'], selected['target_BODY'], support_all_robot_contacts=True)
            native_model = copy.copy(env.sim.mj_model)
            for key in ['body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'geom_size', 'geom_friction', 'geom_solref', 'geom_solimp',
                        'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange', 'actuator_ctrlrange', 'jnt_range', 'dof_damping', 'dof_armature']:
                getattr(native_model, key)[:] = getattr(env.sim.wp_model, key).numpy()[0]
            native_data = mujoco.MjData(native_model)
            mujoco.mj_setConst(native_model, native_data)
            mw.get_data_into(native_data, native_model, env.sim.wp_data)
            values, scalars, before = capture_model(env.sim.wp_model)
            snapshot = folder / 'full_device_model_arrays.npz'
            np.savez_compressed(snapshot, **values)
            write(folder / 'full_device_model_scalars.json', scalars)
            reconstructed = mw.put_model(native_model)
            restore_model_arrays(reconstructed, values, scalars)
            _, reconstructed_scalars, reconstructed_hash = capture_model(reconstructed)
            inventory.update(array_fields=len(values), scalar_fields=len(scalars),
                full_reconstruction_exact=before == reconstructed_hash and scalars == reconstructed_scalars,
                snapshot=file_pin(snapshot), scalars=file_pin(folder / 'full_device_model_scalars.json'))
            for binding in cfg['bindings']:
                device_model = env.sim.wp_model if binding == 'LIVE_READONLY' else reconstructed
                clone = GPUPenRollout(native_model, native_data, cfg['nworld'], selected['delta20'], device_model=device_model, nconmax=128, njmax=512)
                reference = torch.as_tensor(qref, device='cuda', dtype=torch.float32)
                raw = torch.as_tensor(2 * selected['delta20'], device='cuda', dtype=torch.float32)[None].expand(cfg['nworld'], -1)
                for rep in range(cfg['repeats']):
                    clone.reset_clone()
                    initial = {k: v.cpu().numpy().copy() for k, v in clone.state().items()}
                    mocap0 = {k: clone.views[k].clone() for k in ['mocap_pos', 'mocap_quat']}
                    rows = []
                    for phase in range(1, cfg['horizon_controls'] + 1):
                        clone.set_action(raw, reference[phase])
                        for _ in range(clone.decimation):
                            clone.physics_step()
                            row = clone.readback(clone.output())
                            diagnostics = torch.stack([
                                clone.views['nacon'][0].expand(cfg['nworld']), clone.views['overflow'],
                                clone.views['qfrc_applied'].abs().amax(dim=1),
                                clone.views['xfrc_applied'].reshape(cfg['nworld'], -1).abs().amax(dim=1),
                                (clone.views['mocap_pos'] - mocap0['mocap_pos']).reshape(cfg['nworld'], -1).abs().amax(dim=1),
                                (clone.views['mocap_quat'] - mocap0['mocap_quat']).reshape(cfg['nworld'], -1).abs().amax(dim=1)
                            ], dim=1).cpu().numpy()
                            row['capacity_aid_mocap'] = diagnostics
                            rows.append(row)
                        clone.refresh_boundary()
                    actual = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
                    path = folder / f'{binding}_{rep}.npz'
                    np.savez_compressed(path, **actual, **{'initial_' + k: v for k, v in initial.items()})
                    captures.append(dict(binding=binding, repeat=rep, clone=clone, trace=actual, initial=initial, pin=file_pin(path)))
            _, after_scalars, after = capture_model(env.sim.wp_model)
            inventory['live_model_unchanged'] = before == after and scalars == after_scalars
            return {**receipt, 'repeatability_inventory': inventory}

        case = evaluate_case('hand2-pen-spin', 'CONSTANT_RESIDUAL', None, {**base, 'physics_dt': dt},
            bp.parent, registry, folder, initializer=initialize, constant_action=2 * selected['delta20'])
        case['name'] = 'REPEATABILITY_CONSTANT_RESIDUAL'
        cases.append(case)
        saved = np.load(verify_file_pin(case['trace']))
        for capture in captures:
            actual, clone, initial = capture['trace'], capture['clone'], capture['initial']
            errors = discrepancies(actual, saved, clone, len(actual['qpos']))
            world0 = {key: actual[value][:, 0] for key, value in [('post_qpos', 'qpos'), ('post_qvel', 'qvel')]}
            spread = discrepancies(actual, world0, clone, len(actual['qpos']))
            ref = next(c for c in captures if c['binding'] == capture['binding'] and c['repeat'] == 0)
            first_repeat = dict(post_qpos=ref['trace']['qpos'][:, 0], post_qvel=ref['trace']['qvel'][:, 0])
            repeat_error = discrepancies({k: v[:, :1] for k, v in actual.items()}, first_repeat, clone, len(actual['qpos']))
            fidelity = {key: value <= gates[key] for key, value in errors.items()}
            checks = dict(full_model_reconstruction=inventory['full_reconstruction_exact'],
                live_model_unchanged=inventory['live_model_unchanged'],
                initial_pose=np.array_equal(initial['qpos'], np.tile(saved['initial_qpos'], (cfg['nworld'], 1))),
                initial_velocity=np.array_equal(initial['qvel'], np.tile(saved['initial_qvel'], (cfg['nworld'], 1))),
                initial_ema=np.array_equal(initial['ema'], np.tile(selected['delta20'].astype(np.float32), (cfg['nworld'], 1))),
                initial_warmstart=not bool(np.any(initial['qacc_warmstart'])),
                initial_time=not bool(np.any(initial['time'])),
                controls=bool(abs(actual['ctrl'] - saved['post_ctrl'][:len(actual['ctrl']), None]).max() <= gates['control_rad']),
                original_clock=bool(abs(actual['time'][:, :, 0] - saved['post_GPU_time'][:len(actual['time']), None]).max() < 1e-6),
                capacity=bool(np.all(actual['capacity_aid_mocap'][:, :, 0] <= cfg['nworld'] * 128)
                              and not np.any(actual['capacity_aid_mocap'][:, :, 1]) and np.all(actual['nefc'] <= 512)),
                no_aid=not bool(np.any(actual['capacity_aid_mocap'][:, :, 2:4])),
                fixed_mocap=not bool(np.any(actual['capacity_aid_mocap'][:, :, 4:])),
                finite=all(np.isfinite(v).all() for v in actual.values()))
            probes.append(dict(dt=dt, binding=capture['binding'], repeat=capture['repeat'], trace=capture['pin'],
                checks=checks, fidelity=fidelity, actual_errors=errors, identical_world_spread=spread,
                first_world_repeat_error=repeat_error, inventory=inventory))
        print(json.dumps(dict(dt=dt, probes=[{k: p[k] for k in ['binding', 'repeat', 'checks', 'fidelity', 'identical_world_spread']} for p in probes if p['dt'] == dt])), flush=True)
    for pin in sources:
        verify_file_pin(pin)
    write(args.output / 'report.json', dict(**intent, cases=cases, probes=probes,
        integrity_passed=sum(sum(p['checks'].values()) for p in probes), integrity_total=sum(len(p['checks']) for p in probes),
        fidelity_passed=sum(sum(p['fidelity'].values()) for p in probes), fidelity_total=sum(len(p['fidelity']) for p in probes),
        ready_for_100ms_planning=all(all(p['checks'].values()) and all(p['fidelity'].values()) for p in probes),
        scope='Exact nested device-model reconstruction and100ms identical-input/reset-repeatability versus a fresh unchanged task. No injected randomness, shorter-horizon substitution, changed acceptance, physical planning, neural training or success claim.'))


if __name__ == '__main__':
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    with torch.jit.optimized_execution(False):
        main(parser.parse_args())
