"""Bounded diagnostic replay;never silently continue or replace a failed trial."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.learning.pen_support_ppo import SupportPenPPO, pilot_settings
from supermimic_research.learning.native_pen_ppo import cpu_tree
from train_native_pen_ppo import finite_training_state


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(a):
    run = ROOT / 'runs/pen_long_control_training_v1'
    failure_path = run / 'failure.json'
    original_failure = json.loads(failure_path.read_text())
    if (original_failure['phase'], original_failure['actual_native_integrations'], original_failure['optimizer_steps']) != ('SUPPORT_206', 36143104, 70560):
        raise ValueError('Explicit observed original failure required')
    health200 = json.loads((run / 'SUPPORT/health_0200.json').read_text())
    checkpoint = health200['checkpoint']
    state = torch.load(verify_file_pin(checkpoint), map_location='cpu', weights_only=False)
    if (state['iteration'], state['optimizer_steps'], state['training_transitions']) != (200, 6400, 409600):
        raise ValueError('Fixed SUPPORT200 diagnostic source required')
    verify_source_snapshot(state['native']['sources'], live=True)
    cp = verify_file_pin(state['metadata']['objective_config'])
    cfg, ppo = pilot_settings(cp)
    if cfg['study'] != 'LONG_CONTROL_BASELINE':
        raise ValueError('Original long-study configuration required')
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh separate diagnostic output required')
    out.mkdir()
    protocol = ROOT / 'docs/pen-support206-replay-protocol-v1.md'
    paths = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), protocol, cp, ROOT / 'tools/train_native_pen_ppo.py']
    receipt = freeze_sources(paths, out / 'source_snapshot', ROOT,
                             subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    from mjlab.sim.sim import Simulation
    original = Simulation.step
    observed = updates = 0
    runtime = None
    phase = 'construction'

    def step(sim):
        nonlocal observed
        original(sim)
        observed += sim.num_envs

    Simulation.step = step
    rows = []
    try:
        torch.set_num_threads(4)
        with torch.jit.optimized_execution(False):
            runtime = SupportPenRuntime(cp, 32)
            learner = SupportPenPPO(runtime, ppo, 'SUPPORT')
            phase = 'ordinary_checkpoint200_reload'
            load = learner.load(state)
            if not all(v for v in load.values() if type(v) is bool) or not load['native']['all_array_bytes_equal'] or not load['native']['all_scalar_values_equal']:
                raise ValueError('Ordinary SUPPORT200 load failed')
            write(out / 'reload.json', load)
            for index in range(201, 207):
                phase = f'collect_{index}'
                learner.collect(64)
                memory_path = out / f'memory_{index:04d}.pt'
                torch.save(cpu_tree(learner.memory), memory_path)
                health = learner.consistency()
                row = dict(iteration=index, health=health, memory=file_pin(memory_path),
                           failed_checks=[k for k, v in health['checks'].items() if not v], optimized=False)
                if health['passed'] != health['total']:
                    # A retained diagnostic bundle is explicitly NOT an ordinary
                    # loadable physics checkpoint or permission to use this batch.
                    torch.save(dict(schema='failed_pen_collector_diagnostic_NOT_RESUMABLE:1',
                        modules=dict(actor=cpu_tree(learner.actor.state_dict()), critic=cpu_tree(learner.critic.state_dict())),
                        optimizer=cpu_tree(learner.optimizer.state_dict()),
                        iteration=learner.iteration, optimizer_steps=learner.optimizer_steps,
                        training_transitions=learner.training_transitions, observations=cpu_tree(learner.observations)),
                        out / 'failed_policy_diagnostic.pt')
                    row['failed_policy_diagnostic'] = file_pin(out / 'failed_policy_diagnostic.pt')
                    rows.append(row)
                    write(out / f'health_{index:04d}.json', row)
                    print(json.dumps(dict(iteration=index, failed_checks=row['failed_checks'], optimized=False)), flush=True)
                    break
                phase = f'update_{index}'
                stats = learner.update()
                updates += stats['optimizer_steps']
                if not finite_training_state(learner):
                    raise ValueError('Nonfinite diagnostic learner after update')
                path = out / f'checkpoint_{index:04d}.pt'
                torch.save(learner.checkpoint(receipt), path)
                row.update(optimized=True, checkpoint=file_pin(path))
                rows.append(row)
                write(out / f'health_{index:04d}.json', row)
                print(json.dumps(dict(iteration=index, failed_checks=[], optimized=True)), flush=True)
            verify_source_snapshot(receipt, live=True)
            result = dict(schema='pen_support206_failure_diagnostic_replay:1', original_failure=file_pin(failure_path),
                parent_checkpoint=checkpoint, config=file_pin(cp), protocol=file_pin(protocol),
                source_receipt=file_pin(out / 'source_receipt.json'), reload=file_pin(out / 'reload.json'), rows=rows,
                actual_native_integrations=observed, optimizer_steps=updates,
                stopped_on_failed_gate=bool(rows[-1]['failed_checks']),
                original_failure_cause_proven=False,
                scope='Separately budgeted six-round maximum diagnostic. Replayed W32 trajectories need not reproduce the original. Failed batches never optimized;all raw memories retained. Not original trial continuation,equal-budget final comparison,task success or ego results.')
            if observed != len(rows) * 16384 or updates != sum(row['optimized'] for row in rows) * 32:
                raise ValueError('Diagnostic actual counters differ')
            write(out / 'report.json', result)
            print(json.dumps(dict(report=file_pin(out / 'report.json'), native=observed, optimizer=updates,
                                  failed_checks=rows[-1]['failed_checks'])), flush=True)
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), phase=phase,
              actual_native_integrations=observed, optimizer_steps=updates))
        raise
    finally:
        if runtime is not None:
            runtime.close()
        Simulation.step = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
