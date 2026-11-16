"""Unstepped ordinary R/S parent200 loads and exact startup-difference evidence."""
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
from supermimic_research.learning.pen_matched_recovery import ARMS, WorkCounter, write_metadata as write
from supermimic_research.learning.pen_support_ppo import SupportPenPPO, pilot_settings
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from train_pen_matched_recovery import settings, validate_inputs


def main(a):
    cp = a.config.resolve()
    cfg, paths, parents, work = settings(cp)
    validate_inputs(paths, parents, require_load_qualification=False)
    out = a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT / 'runs'):
        raise ValueError('Fresh unstepped load qualification required')
    old = torch.load(verify_file_pin(parents['RESIDUAL']), map_location='cpu', weights_only=False)
    failed_path = ROOT / 'runs/pen_matched_recovery_training_v1/RESIDUAL/failed_native_NOT_RESUMABLE.pt'
    previous = torch.load(failed_path, map_location='cpu', weights_only=False)
    saved = old['native']['python']['description']
    actual = previous['native']['python']['description']
    differences = {key: dict(saved=saved[key], failed_startup=actual[key]) for key in saved if saved[key] != actual[key]}
    opaque_changes = [dict(saved=x, failed_startup=y) for x, y in zip(saved['opaque'], actual['opaque']) if x != y]
    expected = (set(differences) == {'opaque'} and len(opaque_changes) == 1
                and opaque_changes[0]['saved']['qualified_name'] == '__main__.main.<locals>.step'
                and opaque_changes[0]['failed_startup']['qualified_name'] == '__main__.main.<locals>.<lambda>')
    if not expected:
        raise ValueError('Previous startup difference is not the independently observed callback contract')
    out.mkdir()
    sources = sorted((ROOT / 'src').rglob('*.py')) + [Path(__file__), ROOT / 'tools/train_pen_matched_recovery.py', cp, paths['protocol']]
    receipt = freeze_sources(sources, out / 'source_snapshot', ROOT,
        subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    write(out / 'source_receipt.json', receipt)
    write(out / 'previous_startup_closure_diff.json', dict(parent=parents['RESIDUAL'], failed_native=file_pin(failed_path),
          changed_description_groups=list(differences), opaque_changes=opaque_changes,
          scope='Observed prior mismatch only;current loader still checks the entire unchanged saved description,no metadata editing or relaxed guard.'))
    from mjlab.sim.sim import Simulation
    original = Simulation.step
    counter = WorkCounter()
    def step(sim):
        return counter.native_call(original, sim)
    Simulation.step = step
    runtime = None
    checks = {'previous_difference_exactly_callback_name': expected}
    audits = {}
    _, ppo = pilot_settings(paths['objective_config'])
    try:
        torch.set_num_threads(4)
        with torch.jit.optimized_execution(False):
            for arm in ARMS:
                state = torch.load(verify_file_pin(parents[arm]), map_location='cpu', weights_only=False)
                runtime = SupportPenRuntime(paths['objective_config'], 32)
                learner = SupportPenPPO(runtime, ppo, arm)
                audit = learner.load(state)
                audits[arm] = audit
                checks.update({arm + '_' + key: value for key, value in audit.items() if type(value) is bool})
                for key in ['all_array_bytes_equal', 'all_scalar_values_equal', 'pointers_preserved']:
                    checks[arm + '_' + key] = audit['native'][key]
                runtime.close()
                runtime = None
            checks['no_native_integrations'] = counter.native == counter.native_attempted == 0
            checks['no_optimizer_steps'] = counter.optimizer == counter.optimizer_attempted == 0
            verify_source_snapshot(receipt, live=True)
            result = dict(schema='pen_matched_recovery_parent_load_qualification:1', config=file_pin(cp), parents=parents,
                source_receipt=file_pin(out / 'source_receipt.json'), previous_closure_diff=file_pin(out / 'previous_startup_closure_diff.json'),
                audits=audits, checks=checks, passed=sum(checks.values()), total=len(checks), **counter.metadata(),
                scope='Both original200 parents restored ordinarily with complete strict closure/source/state checks and immediate byte readback. No physics step,optimizer update,source rebinding or future W32 identity claim. Separate from the actual recovery reloads.')
            write(out / 'report.json', result)
            print(json.dumps(dict(report=file_pin(out / 'report.json'), checks=[result['passed'], result['total']], **counter.metadata())), flush=True)
            if result['passed'] != result['total']:
                raise ValueError('Preserve failed ordinary load qualification')
    except Exception as error:
        write(out / 'failure.json', dict(type=type(error).__name__, message=str(error), **counter.metadata()))
        raise
    finally:
        if runtime is not None:
            runtime.close()
        Simulation.step = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
