"""Bounded, zero-update W32 integration and saved raw-contact oracle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import mujoco
import numpy as np
import torch
from mjlab.sim.sim import Simulation
from mujoco_warp._src import solver

from qualify_resident_contact_native import source_packet
from qualify_resident_contact_replay import canonical_joints, empty_window, fold, oracle
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import WorkCounter, saved_bytes_equal
from supermimic_research.mechanics.portable_contact_records import contacts
from supermimic_research.runtime.explicit_collision_backend import backend_scope
from supermimic_research.runtime.owned_pen_state import capture, restore
from supermimic_research.runtime.pen_frozen_cost_rollout_v2 import require_structure
from supermimic_research.runtime.pen_guarded_w32 import GuardedW32PenRuntime, guarded_w32
from supermimic_research.runtime.pen_stable_friction_binding_v2 import code_digest
from supermimic_research.strict_json import write_json


def packet(runtime):
    sim = runtime.env.sim
    model, data = sim.mj_model, sim.wp_data
    body = runtime.features.body
    contact = contacts(sim.wp_model, data, model, body)
    dof = model.jnt_dofadr[model.body_jntadr[body]]
    native = data.qfrc_constraint.numpy()[:, dof:dof + 6]
    torque = np.einsum('wij,wj->wi', contact['R_efc_phase_world'], native[:, 3:])
    return dict(qpos=data.qpos.numpy(), time=data.time.numpy(), cone=int(sim.wp_model.opt.cone),
                contact=contact, constraint_world=np.c_[native[:, :3], torque],
                no_applied_force=not (data.qfrc_applied.numpy().any() or data.xfrc_applied.numpy().any()))


def audit_window(model, body, bounds, saved):
    start, packets, actual = (saved[k] for k in ('start', 'packets', 'actual'))
    joints = canonical_joints(model)
    expected = empty_window(start)
    coverage = np.zeros(32, dtype=bool)
    maximum = dict(force_N=0., torque_Nm=0.)
    checks = dict(eight_native_ticks=len(packets) == 8)
    for index, row in enumerate(packets, 1):
        c = row['contact']
        checks[f'tick_{index}_raw'] = row['no_applied_force'] and all(c['flags'].values())
        checks[f'tick_{index}_clock'] = bool(np.allclose(row['time'], start['time'] + index * .0025, atol=3e-8, rtol=0))
        instant, _, _ = oracle(model, body, joints, bounds, row)
        fold(expected, instant, row)
        coverage |= instant['Fn6'].sum(axis=1) > .1
        residual = np.abs(row['constraint_world'] - instant['wrench'][:, 0])
        maximum['force_N'] = max(maximum['force_N'], float(residual[:, :3].max()))
        maximum['torque_Nm'] = max(maximum['torque_Nm'], float(residual[:, 3:].max()))
    checks.update(all_worlds_have_hand_contact=bool(coverage.all()),
                  constraint_force=maximum['force_N'] <= 5e-4, constraint_torque=maximum['torque_Nm'] <= 5e-5,
                  all_worlds_eight_samples=bool(np.all(actual['counts'][:, 0] == 8)))
    for key, value in expected.items():
        checks[f'window_{key}'] = bool(np.allclose(actual[key], value, atol=1e-8, rtol=1e-10))
        maximum[f'window_{key}'] = float(np.max(np.abs(actual[key].astype(np.float64) - value)))
    return dict(checks=checks, maximum=maximum, passed=sum(checks.values()), total=len(checks))


def partial_reset(runtime, selected):
    data = runtime.env.sim.wp_data
    fields = ('qpos', 'qvel', 'ctrl', 'time')
    before = {key: getattr(data, key).numpy().copy() for key in fields}
    phase_before = runtime.motion.time_steps.cpu().numpy().copy()
    runtime.reset(torch.tensor(selected, device=runtime.device))
    after = {key: getattr(data, key).numpy().copy() for key in fields}
    phase_after = runtime.motion.time_steps.cpu().numpy().copy()
    keep = np.setdiff1d(np.arange(32), selected)
    checks = {f'unselected_{key}': before[key][keep].tobytes() == after[key][keep].tobytes() for key in fields}
    checks.update(selected_time=bool(np.all(after['time'][selected] == 0)),
                  selected_phase=bool(np.all(phase_after[selected] == 1)),
                  unselected_phase=np.array_equal(phase_before[keep], phase_after[keep]))
    return dict(checks=checks, before=before, after=after, phase_before=phase_before, phase_after=phase_after)


def require(checks):
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise ValueError(f'W32 checks failed: {failed}')


def main(config):
    config = config.resolve()
    cfg = tomllib.loads(config.read_text(encoding='utf-8'))
    paths = {key: (config.parent / cfg[key]).resolve() for key in ('objective', 'collision', 'arithmetic', 'w1_evidence', 'protocol', 'output')}
    out = paths.pop('output')
    if (cfg['worlds'], cfg['controls_before_restore'], cfg['controls_after_restore'], cfg['ticks_per_control'], cfg['native_budget'], cfg['reset_ids']) != (32, 4, 2, 8, 1536, [0, 7, 31]):
        raise ValueError('Use the declared fixed W32 protocol and budget')
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh W32 output required')
    inputs = {key: file_pin(path) for key, path in paths.items()}
    arithmetic = json.loads(verify_file_pin(inputs['arithmetic']).read_text(encoding='utf-8'))
    earlier = json.loads(verify_file_pin(inputs['w1_evidence']).read_text(encoding='utf-8'))
    if arithmetic['passed'] != arithmetic['total'] or not all(arithmetic['checks'].values()) or earlier['native_integrations_in_producers'] != 11616:
        raise ValueError('Completed arithmetic and W1 evidence required')
    factory_manifest = json.loads(verify_file_pin(arithmetic['factory_manifest']).read_text(encoding='utf-8'))
    out.mkdir()
    sources = sorted((ROOT / 'src').rglob('*.py')) + [config, Path(__file__), paths['protocol'], paths['objective'], paths['collision'],
        ROOT / 'tools/qualify_resident_contact_native.py', ROOT / 'tools/qualify_resident_contact_replay.py']
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    receipt = freeze_sources(sources, out / 'source_snapshot', ROOT, revision)
    write_json(out / 'source_receipt.json', receipt)
    write_json(out / 'intent.json', dict(inputs=inputs, sources=file_pin(out / 'source_receipt.json'), config=file_pin(config), budget=1536))

    runtime, packets = None, None
    counter, layouts, rows, checks = WorkCounter(), [], [], {}
    original_step, original_context = Simulation.step, solver._create_solver_context
    original_done, original_factory = solver._solve_done, solver._linesearch_iterative_kernel
    done_code = code_digest(original_done)
    stage = 'construct'

    def observe_step(sim):
        if packets is None or runtime is None or sim is not runtime.env.sim:
            raise ValueError('Native integration outside the scheduled controls')
        if counter.native_attempted + sim.num_envs > cfg['native_budget']:
            raise ValueError('Native budget exceeded')
        result = counter.native_call(original_step, sim)
        packets.append(packet(runtime))
        return result

    def observe_context(model, data):
        ctx = original_context(model, data)
        layout = dict(worlds=data.nworld, nv=model.nv, nv_pad=model.nv_pad,
                      grad=list(ctx.grad.shape), search=list(ctx.search.shape), Mgrad=list(ctx.Mgrad.shape))
        layouts.append(layout)
        require(dict(actual_layout=layout == dict(worlds=32, nv=26, nv_pad=28, grad=[32, 28], search=[32, 26], Mgrad=[32, 0])))
        return ctx

    def construct(binding, name):
        calls, contexts = len(binding.audit['calls']), len(layouts)
        instance = GuardedW32PenRuntime(paths['objective'], 32)
        try:
            sim = instance.env.sim
            require(dict(constructor_graphs=sim.step_graph is not None and sim.forward_graph is not None,
                         guarded_factory=len(binding.audit['calls']) > calls,
                         context_observed=len(layouts) > contexts,
                         fixed_wrist=require_structure(sim.mj_model),
                         cached=instance.metadata()['fixed_frame_features'] == 'CACHED'))
        except BaseException:
            instance.close()
            raise
        checks[f'{name}_captured_guarded_graphs'] = True
        return instance

    def control(index):
        nonlocal packets
        sim = runtime.env.sim
        phase = runtime.motion.time_steps.cpu().numpy().copy()
        start = source_packet(runtime)
        world = torch.arange(32, device=runtime.device)[:, None]
        joint = torch.arange(20, device=runtime.device)[None, :]
        action = runtime.initial_raw + .005 * torch.sin(.2 * index + .13 * world + .03 * joint)
        packets = []
        try:
            obs, reward, term, trunc, info = runtime.step(action)
            saved = dict(start=start, packets=packets, actual=runtime.contact_metrics.numpy()['window'],
                         raw=action.cpu(), phase=phase, phase_used=info['reference_phase'].cpu().numpy(),
                         phase_after=runtime.motion.time_steps.cpu().numpy(), terminated=term.cpu(), truncated=trunc.cpu())
        finally:
            packets = None
        path = out / f'control_{index:02}.pt'
        torch.save(saved, path)
        results = audit_window(sim.mj_model, runtime.features.body, sim.wp_model.jnt_range.numpy(),
                               torch.load(path, map_location='cpu', weights_only=False))
        results['checks'].update(phase=np.array_equal(saved['phase_used'], phase) and np.array_equal(saved['phase_after'], phase + 1),
            no_terminal_worlds=not bool((term | trunc).any()), finite=all(bool(torch.isfinite(value).all()) for value in [*obs.values(), reward]))
        results.update(passed=sum(results['checks'].values()), total=len(results['checks']), saved=file_pin(path))
        write_json(out / f'control_{index:02}.json', results)
        require(results['checks'])
        rows.append(file_pin(out / f'control_{index:02}.json'))

    Simulation.step, solver._create_solver_context = observe_step, observe_context
    try:
        with backend_scope(paths['collision']) as collision, guarded_w32(factory_manifest) as binding, torch.no_grad(), torch.jit.optimized_execution(False):
            try:
                runtime = construct(binding, 'first')
                contract = runtime.metadata()['termination_backend']
                model_path = out / 'model.mjb'
                mujoco.mj_saveModel(runtime.env.sim.mj_model, str(model_path))
                for index in range(4):
                    stage = f'control_{index}'
                    if index == 2:
                        reset = partial_reset(runtime, cfg['reset_ids'])
                        torch.save(reset, out / 'reset.pt')
                        require(reset['checks'])
                        checks['selected_reset'] = True
                    control(index)
                stage = 'checkpoint'
                boundary = counter.native
                saved = capture(runtime, receipt)
                checkpoint = out / 'state.pt'
                torch.save(saved, checkpoint)
                loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
                checks['serialized_exact'] = saved_bytes_equal(saved, loaded)
                del saved
                runtime.close()
                runtime = None
                stage = 'fresh_restore'
                runtime = construct(binding, 'resume')
                incompatible = dict(loaded, metadata={key: value for key, value in loaded['metadata'].items() if key != 'termination_backend'})
                fresh = capture(runtime, receipt)
                rejected = False
                try:
                    restore(runtime, incompatible)
                except ValueError as error:
                    rejected = 'runtime/schema contract mismatch' in str(error)
                checks['wrong_identity_rejected_without_write'] = rejected and saved_bytes_equal(fresh, capture(runtime, receipt))
                del fresh
                audit = restore(runtime, loaded)
                write_json(out / 'restore.json', audit)
                checks['fresh_restore_exact'] = audit['all_array_bytes_equal'] and audit['all_scalar_values_equal'] and audit['pointers_preserved']
                checks['zero_setup_and_restore_integrations'] = counter.native == boundary == 1024
                require(checks)
                del loaded, incompatible
                for index in range(4, 6):
                    stage = f'control_{index}'
                    control(index)
                checks['runtime_accounting'] = runtime.owned_native_integrations == runtime.physics_integrations == 1536
            finally:
                if runtime is not None:
                    runtime.close()
            checks['collision_runtimes_closed'] = not bool(collision.runtime_set)
        checks['solver_restored'] = binding.audit['bindings_restored'] and solver._linesearch_iterative_kernel is original_factory and solver._solve_done is original_done and code_digest(original_done) == done_code
        checks['coverage_and_budget'] = len(rows) == 6 and counter.native == counter.native_attempted == 1536 and counter.optimizer == 0
        verify_source_snapshot(receipt, live=True)
        require(checks)
        report = out / 'report.json'
        write_json(report, dict(schema='pen_guarded_w32_native_qualification:1', intent=file_pin(out / 'intent.json'), contract=contract,
            layouts=layouts, binding=binding.audit, rows=rows, model=file_pin(model_path), checkpoint=file_pin(checkpoint),
            reset=file_pin(out / 'reset.pt'), restore=file_pin(out / 'restore.json'), checks=checks,
            passed=sum(checks.values()), total=len(checks), normalizer_fits=0, **counter.metadata()))
        print(json.dumps(dict(report=file_pin(report), checks=[sum(checks.values()), len(checks)])), flush=True)
    except BaseException as error:
        write_json(out / 'failure.json', dict(stage=stage, type=type(error).__name__, message=str(error), rows=rows,
            layouts=layouts, checks=checks, normalizer_fits=0, **counter.metadata()))
        raise
    finally:
        Simulation.step, solver._create_solver_context = original_step, original_context


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    main(parser.parse_args().config)
