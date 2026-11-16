"""Exercise the actual pending full54 prerequisite before any GPU imports."""
import argparse
import builtins
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_solve_exit_observer import SolveExitGraphObserver
from supermimic_research.strict_json import write_json


def main(a):
    started=time.perf_counter();cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh pending-prerequisite check required')
    paths={k:(cp.parent/cfg[k]).resolve() for k in ['required_closed_producer','required_closed_review','protocol']}
    if all(paths[k].exists() for k in ['required_closed_producer','required_closed_review']):raise ValueError('This check targets the pending stage;do not relabel a closed stage')
    preparation=ROOT/'runs/pen_solver_tick_preparation_v1/report.json';prep=json.loads(preparation.read_text())
    verify_source_snapshot(json.loads(verify_file_pin(prep['source_receipt']).read_text()),live=True)
    attempts=[];original=builtins.__import__;guard_error=None
    def guarded_import(name,*args,**kwargs):
        if name.split('.')[0] in ['torch','warp','mujoco','mujoco_warp','mjlab']:
            attempts.append(name)
            raise AssertionError('GPU/simulator import attempted before prerequisite refusal')
        return original(name,*args,**kwargs)
    with patch('builtins.__import__',side_effect=guarded_import):
        try:
            # Deliberately no simulator object: the real constructor must refuse
            # the current missing reports before inspecting/allocating anything.
            SolveExitGraphObserver(object(),config_path=cp,installed_solver_pin={})
        except ValueError as error:
            guard_error=str(error)
    focused=ROOT/'runs/pen_solve_exit_observer_tests_v1.log';tests=ROOT/'runs/pen_solve_exit_observer_tests580_v1.log'
    checks=dict(actual_pending_constructor_refused=guard_error=='Wait for both complete full54 producer and final V3 review',
        no_simulator_import_attempts=not attempts,original_preparation103=prep['passed']==prep['total']==103,
        focused12='Ran 12 tests' in focused.read_text() and '\nOK\n' in focused.read_text(),
        full580='Ran 580 tests' in tests.read_text() and '\nOK\n' in tests.read_text())
    out.mkdir();sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),cp,paths['protocol'],ROOT/'tests/test_pen_solve_exit_observer.py']
    receipt=freeze_sources(sources,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    result=dict(schema='pen_solve_exit_pending_guard:1',config=file_pin(cp),preparation=file_pin(preparation),
        prior_evidence=file_pin(ROOT/'provenance/pen_solver_tick_preparation_v1/ledger.json'),
        source_receipt=file_pin(out/'source_receipt.json'),tests=[file_pin(focused),file_pin(tests)],
        pending_paths={k:str(paths[k]) for k in ['required_closed_producer','required_closed_review']},
        guard_error=guard_error,blocked_simulator_import_attempts=attempts,checks=checks,
        passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        CUDA_graph_qualified=False,private_failing_tick_context_captured=False,wall_s=time.perf_counter()-started,
        scope='Actual pending-config guard and CPU binding/copy tests only. No observer graph built,native replay,physical-label fix or Actor/ego admission. Full54 producer remains separate.')
    verify_source_snapshot(receipt,live=True);write_json(out/'report.json',result)
    print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native=0)),flush=True)
    if not all(checks.values()):raise ValueError('Preserve pending guard discrepancy')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
