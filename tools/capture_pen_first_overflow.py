"""Bounded diagnostic replay with external first-overflow buffers."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import torch
import warp as wp
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import WorkCounter, persist_failed_learner, write_metadata as write
from supermimic_research.learning.pen_support_ppo import SupportPenPPO, pilot_settings
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.runtime.first_overflow_observer import FirstOverflowObserver
from supermimic_research.runtime.gpu_model_snapshot import model_leaves
from train_native_pen_ppo import finite_training_state


def freeze(cp, paths, out):
    files = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), cp, paths['objective_config'], paths['protocol'],
        ROOT / 'tools/train_native_pen_ppo.py', ROOT / 'tools/qualify_first_overflow_observer.py']
    receipt = freeze_sources(files, out / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    return receipt


def main(a):
    started = time.perf_counter()
    cp = a.config.resolve(); cfg = tomllib.loads(cp.read_text())
    if (cfg['schema'], cfg['study'], cfg['first_iteration'], cfg['max_collections'], cfg['rollout_controls'],
            cfg['num_envs'], cfg['arm']) != (1, 'FIRST_NATIVE_OVERFLOW_DIAGNOSTIC', 202, 99, 64, 32, 'RESIDUAL'):
        raise ValueError('Fixed registered diagnostic scope required')
    paths = {k: (cp.parent / cfg[k]).resolve() for k in ['objective_config', 'parent', 'protocol', 'prior_ledger', 'registration', 'qualification']}
    parent = dict(path=str(paths['parent']), bytes=cfg['parent_bytes'], sha256=cfg['parent_sha256'])
    state = torch.load(verify_file_pin(parent), map_location='cpu', weights_only=False)
    if (state['iteration'], state['optimizer_steps'], state['training_transitions'], state['metadata']['arm'],
            state['native']['worlds']) != (201, 6432, 411648, 'RESIDUAL', 32):
        raise ValueError('Valid recovery201 parent required')
    verify_source_snapshot(state['native']['sources'], live=True)
    qa = json.loads(paths['qualification'].read_text())
    if (qa['schema'], qa['passed'], qa['total'], qa['actual_native_integrations'], qa['optimizer_steps']) != ('first_overflow_observer_GPU_qualification:1', 10, 10, 0, 0):
        raise ValueError('GPU observer qualification required')
    verify_source_snapshot(json.loads(verify_file_pin(qa['source_receipt']).read_text()), live=True)
    inputs = {k: file_pin(paths[k]) for k in ['objective_config', 'protocol', 'prior_ledger', 'qualification']}
    if state['metadata']['objective_config'] != inputs['objective_config']:
        raise ValueError('Unchanged physical/objective contract required')
    budget = dict(max_collections=99, max_native_integrations=1622016, max_Adam_calls=3136)
    if a.register:
        out = paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh registration required')
        out.mkdir(); freeze(cp, paths, out)
        write(paths['registration'], dict(schema='first_native_overflow_diagnostic_registration:1', config=file_pin(cp),
            inputs=inputs, parent=parent, budget=budget, source_receipt=file_pin(out / 'source_receipt.json')))
        print(json.dumps(dict(registration=file_pin(paths['registration']), budget=budget)), flush=True)
        return
    reg = json.loads(paths['registration'].read_text())
    if reg['config'] != file_pin(cp) or reg['inputs'] != inputs or reg['parent'] != parent or reg['budget'] != budget:
        raise ValueError('Registration inputs changed')
    verify_source_snapshot(json.loads(verify_file_pin(reg['source_receipt']).read_text()), live=True)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'): raise ValueError('Fresh diagnostic output required')
    out.mkdir(); receipt = freeze(cp, paths, out)
    from mjlab.sim.sim import Simulation
    original_step, original_forward = Simulation.step, Simulation.forward
    counter = WorkCounter(); observer = runtime = learner = None
    step_calls = forward_calls = 0
    iteration = 0

    def observe(sim, kind, operation):
        nonlocal forward_calls
        if observer is None or sim is not runtime.env.sim:
            return operation()
        # Both readings and original Warp launch share the owned stream. No
        # added per-native host scalar check or simulator writes.
        with torch.cuda.stream(runtime.contact_metrics.torch_stream), wp.ScopedStream(runtime.contact_metrics.stream):
            if kind == 2: forward_calls += 1
            observer.before()
            result = operation()
            observer.after(iteration=iteration, kind=kind, step_call=step_calls, forward_call=forward_calls)
            return result

    def step(sim):
        nonlocal step_calls
        if observer is not None: step_calls += 1
        return observe(sim, 1, lambda: counter.native_call(original_step, sim))

    def forward(sim):
        return observe(sim, 2, lambda: original_forward(sim))

    Simulation.step, Simulation.forward = step, forward
    phase = 'construction'; rows = []
    try:
        torch.set_num_threads(4)
        _, ppo = pilot_settings(paths['objective_config'])
        with torch.jit.optimized_execution(False):
            runtime = SupportPenRuntime(paths['objective_config'], 32)
            learner = SupportPenPPO(runtime, ppo, 'RESIDUAL')
            phase = 'ordinary_parent201_reload'
            load = learner.load(state); write(out / 'reload.json', load)
            if not all(v for v in load.values() if type(v) is bool) or not all(load['native'][k] for k in ['all_array_bytes_equal', 'all_scalar_values_equal', 'pointers_preserved']):
                raise ValueError('Ordinary parent201 reload failed')
            counter.bind_optimizer(learner.optimizer)
            d = runtime.env.sim.wp_data
            views = {k: wp.to_torch(getattr(d, k)) for k in ['overflow', 'qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'time', 'geom_xpos', 'geom_xmat']}
            views['phase'] = runtime.motion.time_steps
            observer = FirstOverflowObserver(views)
            # Pin complete physical Model inputs once; no dynamic state restore.
            phase = 'model_evidence'
            arrays, scalars = model_leaves(runtime.env.sim.wp_model)
            torch.save(dict(arrays={k: v.numpy().copy() for k, v in arrays.items()}, scalars=scalars), out / 'native_model_inputs.pt')
            import mujoco
            mujoco.mj_saveModel(runtime.env.sim.mj_model, str(out / 'model.mjb'))
            for iteration in range(202, 301):
                phase = f'collect_{iteration}'
                learner.collect(64)
                phase = f'health_{iteration}'
                health = learner.consistency()
                packet = observer.packet()
                worlds = torch.nonzero(packet['raw_mask'], as_tuple=False).flatten().tolist()
                invalid = health['passed'] != health['total']
                last = iteration == 300
                row = dict(iteration=iteration, health=health, captured_worlds=worlds, optimized=False,
                           **counter.metadata())
                if worlds or invalid or last:
                    phase = f'terminal_evidence_{iteration}'
                    torch.save(packet, out / 'first_overflow_NOT_RESUMABLE.pt')
                    evidence = persist_failed_learner(learner, out, receipt)
                    rows.append(row); write(out / f'health_{iteration:04d}.json', row)
                    break
                phase = f'update_{iteration}'
                before = counter.optimizer
                stats = learner.update()
                if counter.optimizer - before != 32 or stats['optimizer_steps'] != 32 or not finite_training_state(learner):
                    raise ValueError('Diagnostic update finite/accounting failure')
                row.update(optimized=True, **counter.metadata())
                rows.append(row); write(out / f'health_{iteration:04d}.json', row)
                print(json.dumps(dict(iteration=iteration, checks=[health['passed'], health['total']], captured_worlds=worlds)), flush=True)
            verify_source_snapshot(receipt, live=True)
            if counter.native != len(rows) * 16384 or counter.optimizer != sum(r['optimized'] for r in rows) * 32:
                raise ValueError('Actual diagnostic budget mismatch')
            result = dict(schema='first_native_overflow_diagnostic:1', config=file_pin(cp), registration=file_pin(paths['registration']),
                parent=parent, source_receipt=file_pin(out / 'source_receipt.json'), reload=file_pin(out / 'reload.json'),
                model_inputs=file_pin(out / 'native_model_inputs.pt'), host_model=file_pin(out / 'model.mjb'),
                observer_packet=file_pin(out / 'first_overflow_NOT_RESUMABLE.pt'), terminal_evidence=evidence,
                rows=rows, captured_worlds=worlds, raw_mask=packet['raw_mask'].tolist(), events=packet['event'].tolist(),
                stopped_on_observed_event=bool(worlds), stopped_on_failed_gate=invalid, collection_budget_exhausted=last,
                diagnostic_native_step_calls=step_calls, diagnostic_forward_calls=forward_calls, **counter.metadata(),
                total_runner_wall_s=time.perf_counter() - started,
                scope='Bounded separate numerical diagnostic, not complete recovery or final task comparison. First per-world operation-return flag and field bytes retained before later reset. Collision pair/reproduction not proven; no package/driver/CUDA/physics/gate changes.')
            write(out / 'report.json', result)
            print(json.dumps(dict(report=file_pin(out / 'report.json'), raw_mask=result['raw_mask'],
                captured_worlds=worlds, **counter.metadata())), flush=True)
    except Exception as error:
        base = dict(type=type(error).__name__, message=str(error), phase=phase, **counter.metadata(),
                    registration=file_pin(paths['registration']), source_receipt=file_pin(out / 'source_receipt.json'))
        write(out / 'failure_exception.json', base)
        if observer is not None and not observer.active:
            torch.save(observer.packet(), out / 'partial_first_overflow_NOT_RESUMABLE.pt')
        evidence = persist_failed_learner(learner, out, receipt)
        write(out / 'failure.json', dict(base, evidence=evidence, total_runner_wall_s=time.perf_counter() - started))
        raise
    finally:
        if runtime is not None: runtime.close()
        Simulation.step, Simulation.forward = original_step, original_forward


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path); p.add_argument('--register', action='store_true')
    a = p.parse_args()
    if not a.register and a.output is None: p.error('--output required')
    main(a)
