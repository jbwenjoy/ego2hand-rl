"""Matched ordinary parent200 restores with persistent failure evidence."""
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
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.learning.pen_matched_recovery import (
    ARMS, SCHEMA, WorkCounter, budget, checked_health, persist_failed_learner,
    validate_parent, validate_initial_pair, saved_bytes_equal, write_metadata as write)
from supermimic_research.learning.pen_support_ppo import SupportPenPPO, pilot_settings
from supermimic_research.learning.native_pen_ppo import cpu_tree, tensor_tree_equal
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from train_native_pen_ppo import finite_training_state


def settings(cp):
    cfg = tomllib.loads(cp.read_text())
    work = budget(cfg)
    paths = {k: (cp.parent / cfg[k]).resolve() for k in ['objective_config', 'qualification', 'original_run', 'prior_ledger', 'protocol', 'registration']}
    for key in ['load_qualification', 'startup_failure']:
        if key in cfg:
            paths[key] = (cp.parent / cfg[key]).resolve()
    parents = {arm: dict(cfg['parents'][arm], path=str((cp.parent / cfg['parents'][arm]['path']).resolve())) for arm in ARMS}
    return cfg, paths, parents, work


def freeze(cp, paths, out):
    files = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), cp, paths['objective_config'], paths['protocol'], ROOT / 'tools/train_native_pen_ppo.py']
    receipt = freeze_sources(files, out / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    return receipt


def validate_inputs(paths, parents, *, require_load_qualification=True):
    failure = json.loads((paths['original_run'] / 'failure.json').read_text())
    if (failure['phase'], failure['actual_native_integrations'], failure['optimizer_steps']) != ('SUPPORT_206', 36143104, 70560):
        raise ValueError('Original stopped-cohort accounting differs')
    qualification = json.loads(paths['qualification'].read_text())
    objective = file_pin(paths['objective_config'])
    if qualification['config'] != objective or (qualification['passed'], qualification['total'], qualification['actual_native_integrations'], qualification['optimizer_steps']) != (75, 75, 4096, 0):
        raise ValueError('Original unchanged qualification required')
    verify_source_snapshot(json.loads(verify_file_pin(qualification['source_receipt']).read_text()), live=True)
    for arm in ARMS:
        state = torch.load(verify_file_pin(parents[arm]), map_location='cpu', weights_only=False)
        validate_parent(state, arm, objective)
        if not all(bool(torch.isfinite(v).all()) for module in state['modules'].values() for v in module.values()):
            raise ValueError('Parent modules are nonfinite')
    initial = [torch.load(paths['original_run'] / arm / 'initial_state.pt', map_location='cpu', weights_only=False) for arm in ARMS]
    validate_initial_pair(*initial)
    if 'startup_failure' in paths:
        stopped = json.loads(paths['startup_failure'].read_text())
        if stopped['actual_native_integrations'] or stopped['optimizer_steps'] or stopped['phase'] != 'RESIDUAL_parent200_load':
            raise ValueError('Preserve zero-step startup failure separately')
    if require_load_qualification and 'load_qualification' in paths:
        qa = json.loads(paths['load_qualification'].read_text())
        if (qa['schema'] != 'pen_matched_recovery_parent_load_qualification:1' or qa['parents'] != parents
                or qa['passed'] != 21 or qa['total'] != 21 or qa['actual_native_integrations'] or qa['optimizer_steps']):
            raise ValueError('Both ordinary parent load qualifications required')
        verify_source_snapshot(json.loads(verify_file_pin(qa['source_receipt']).read_text()), live=True)
    return qualification


def save(learner, receipt, path):
    if path.exists():
        raise ValueError('No checkpoint replacement')
    temporary = path.with_suffix('.tmp')
    state = learner.checkpoint(receipt)
    torch.save(state, temporary)
    temporary.replace(path)
    return file_pin(path), state


def main(a):
    runner_started = time.perf_counter()
    cp = a.config.resolve()
    cfg, paths, parents, work = settings(cp)
    qualification = validate_inputs(paths, parents)
    inputs = {k: file_pin(paths[k]) for k in ['objective_config', 'qualification', 'prior_ledger', 'protocol']}
    inputs['original_failure'] = file_pin(paths['original_run'] / 'failure.json')
    for key in ['load_qualification', 'startup_failure']:
        if key in paths:
            inputs[key] = file_pin(paths[key])
    if a.register:
        out = paths['registration'].parent
        if out.exists() or not out.is_relative_to(ROOT / 'runs'):
            raise ValueError('Fresh registration required')
        out.mkdir()
        freeze(cp, paths, out)
        write(paths['registration'], dict(schema='pen_matched_recovery_registration:1', config=file_pin(cp),
            inputs=inputs, parents=parents, budgets=work, arm_order=ARMS, source_receipt=file_pin(out / 'source_receipt.json'),
            original_initial={arm: file_pin(paths['original_run'] / arm / 'initial_state.pt') for arm in ARMS},
            accounting=dict(original_failed_native=36143104, original_failed_optimizer=70560,
                bounded_replay_native=98304, bounded_replay_optimizer=192, preparation_native=4928,
                prior_zero_step_startup_native=0, prior_zero_step_startup_optimizer=0, prior_zero_step_startup_wall_s=None,
                reused_prefix_already_in_original_native=6553600, reused_prefix_already_in_original_optimizer=12800),
            scope=cfg['scope']))
        print(json.dumps(dict(registration=file_pin(paths['registration']), budgets=work)), flush=True)
        return
    registration = json.loads(paths['registration'].read_text())
    if registration['config'] != file_pin(cp) or registration['inputs'] != inputs or registration['parents'] != parents:
        raise ValueError('Registered inputs changed')
    verify_source_snapshot(json.loads(verify_file_pin(registration['source_receipt']).read_text()), live=True)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh recovery run required')
    out.mkdir()
    receipt = freeze(cp, paths, out)
    _, ppo = pilot_settings(paths['objective_config'])
    from mjlab.sim.sim import Simulation
    original = Simulation.step
    counter = WorkCounter()
    # The ordinary native closure contract records this bound callback's
    # qualified name. Preserve the declared main.<locals>.step interface.
    def step(sim):
        return counter.native_call(original, sim)
    Simulation.step = step
    runtime = learner = None
    phase = 'construction'
    folder = out
    completed_arms = []
    try:
        torch.set_num_threads(4)
        with torch.jit.optimized_execution(False):
            for arm_index, arm in enumerate(ARMS):
                folder = out / arm
                folder.mkdir()
                phase = arm + '_parent200_load'
                arm_started = time.perf_counter()
                native_start, update_start = counter.native, counter.optimizer
                state = torch.load(verify_file_pin(parents[arm]), map_location='cpu', weights_only=False)
                runtime = SupportPenRuntime(paths['objective_config'], 32)
                learner = SupportPenPPO(runtime, ppo, arm)
                load = learner.load(state)
                write(folder / 'parent_reload.json', load)
                if not all(v for v in load.values() if type(v) is bool) or not load['native']['all_array_bytes_equal'] or not load['native']['all_scalar_values_equal']:
                    raise ValueError('Ordinary parent reload differs')
                counter.bind_optimizer(learner.optimizer)
                last_checkpoint = parents[arm]
                loop_times = []
                for iteration in range(201, 2001):
                    phase = f'{arm}_collect_{iteration}'
                    started = time.perf_counter()
                    learner.collect(64)
                    phase = f'{arm}_consistency_{iteration}'
                    health_path = folder / f'collector_health_{iteration:04d}.json'
                    health = checked_health(learner, health_path)
                    due = iteration == 201 or iteration % 100 == 0 or iteration == 2000
                    memory = None
                    if due:
                        path = folder / f'memory_{iteration:04d}.pt'
                        torch.save(cpu_tree(learner.memory), path)
                        memory = file_pin(path)
                    phase = f'{arm}_update_{iteration}'
                    before = counter.optimizer
                    stats = learner.update()
                    if counter.optimizer - before != 32 or stats['optimizer_steps'] != 32 or not finite_training_state(learner):
                        raise ValueError('Independent Adam count or finite state failed')
                    checkpoint, captured_checkpoint = save(learner, receipt, folder / f'checkpoint_{iteration:04d}.pt') if due else (None, None)
                    loop_s = time.perf_counter() - started
                    readback = None
                    if iteration == 201:
                        saved = torch.load(verify_file_pin(checkpoint), map_location='cpu', weights_only=False)
                        readback = dict(modules=tensor_tree_equal(saved['modules'], dict(actor=cpu_tree(learner.actor.state_dict()), critic=cpu_tree(learner.critic.state_dict()))),
                                        optimizer=tensor_tree_equal(saved['optimizer'], cpu_tree(learner.optimizer.state_dict())),
                                        full_saved_state_bytes=saved_bytes_equal(saved, captured_checkpoint),
                                        counters=(saved['iteration'], saved['optimizer_steps'], saved['training_transitions']) == (201, 6432, 411648))
                        if not all(readback.values()):
                            raise ValueError('First CPU serialization readback differs')
                    captured_checkpoint = None
                    loop_times.append(loop_s)
                    typical = sorted(loop_times[-50:])[len(loop_times[-50:]) // 2]
                    row = dict(arm=arm, iteration=iteration, collector=file_pin(health_path), checkpoint=checkpoint,
                        previous_checkpoint=last_checkpoint, memory=memory, optimizer_steps=learner.optimizer_steps,
                        training_transitions=learner.training_transitions, loop_s=loop_s, first_CPU_readback=readback,
                        added_native_integrations=counter.native - native_start, added_optimizer_steps=counter.optimizer - update_start,
                        estimated_total_remaining_loop_s=((2000 - iteration) + (1 - arm_index) * 1800) * typical,
                        ETA_scope='Median up to50 loops;startup,parent load,first CPU readback,next-arm construction and final evaluation excluded.')
                    write(folder / f'health_{iteration:04d}.json', row)
                    if checkpoint is not None:
                        last_checkpoint = checkpoint
                    print(json.dumps(dict(arm=arm, iteration=iteration, checks=[health['passed'], health['total']], checkpoint=checkpoint,
                                          estimated_total_remaining_loop_s=row['estimated_total_remaining_loop_s'])), flush=True)
                if (counter.native - native_start, counter.optimizer - update_start, learner.optimizer_steps, learner.training_transitions) != (29491200, 57600, 64000, 4096000):
                    raise ValueError('Registered additional/cumulative arm work differs')
                completed_arms.append(dict(arm=arm, parent=parents[arm], initial=registration['original_initial'][arm],
                    checkpoint=last_checkpoint, start_iteration=200, iterations=2000, added_iterations=1800,
                    actual_native_integrations=counter.native - native_start, added_optimizer_steps=counter.optimizer - update_start,
                    optimizer_steps=learner.optimizer_steps, training_transitions=learner.training_transitions,
                    final_health=file_pin(folder / 'health_2000.json'), parent_reload=file_pin(folder / 'parent_reload.json')))
                completed_arms[-1]['arm_wall_s_excluding_close'] = time.perf_counter() - arm_started
                runtime.close()
                runtime = learner = None
            verify_source_snapshot(receipt, live=True)
            for pin in list(inputs.values()) + list(parents.values()):
                verify_file_pin(pin)
            checks = dict(two_final_arms=len(completed_arms) == 2, added_native_total=counter.native == 58982400,
                          added_optimizer_total=counter.optimizer == 115200, no_partial_calls=counter.metadata()['completed_call_accounting_exact'],
                          equal_cumulative_policy_budget=all(a['optimizer_steps'] == 64000 and a['training_transitions'] == 4096000 for a in completed_arms))
            if not all(checks.values()):
                raise ValueError('Matched recovery budget failed')
            result = dict(schema=SCHEMA, config=inputs['objective_config'], recovery_config=file_pin(cp), registration=file_pin(paths['registration']),
                source_receipt=file_pin(out / 'source_receipt.json'), arms=completed_arms, checks=checks,
                passed=sum(checks.values()), total=len(checks), **counter.metadata(), causal_policy_budget=work,
                unique_named_lineage_total=dict(native_integrations=36143104 + 98304 + 4928 + counter.native,
                                               optimizer_steps=70560 + 192 + counter.optimizer),
                total_runner_wall_s=time.perf_counter() - runner_started,
                wall_scope='Main entry through source/budget checks;includes parent loads/checkpoint QA,excludes Python imports. Earlier lineage wall total remains incomplete.',
                lineage_wall_total_complete=False, scope=cfg['scope'])
            write(out / 'report.json', result)
            print(json.dumps(dict(report=file_pin(out / 'report.json'), checks=[result['passed'], result['total']], **counter.metadata())), flush=True)
    except Exception as error:
        base = dict(type=type(error).__name__, message=str(error), phase=phase, completed_arms=completed_arms,
                    **counter.metadata(), total_runner_wall_s=time.perf_counter() - runner_started,
                    registration=file_pin(paths['registration']), source_receipt=file_pin(out / 'source_receipt.json'))
        write(out / 'failure_exception.json', base)
        try:
            evidence = persist_failed_learner(learner, folder, receipt)
        except Exception as capture_error:
            evidence = dict(failure_evidence_capture_error=dict(type=type(capture_error).__name__, message=str(capture_error)))
        write(out / 'failure.json', dict(base, evidence=evidence, total_runner_wall_s=time.perf_counter() - runner_started,
                                       immediate_exception=file_pin(out / 'failure_exception.json')))
        raise
    finally:
        if runtime is not None:
            runtime.close()
        Simulation.step = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--register', action='store_true')
    args = parser.parse_args()
    if not args.register and args.output is None:
        parser.error('--output required for training')
    main(args)
