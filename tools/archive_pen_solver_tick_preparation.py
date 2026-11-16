"""Immutable CPU readiness evidence; no live producer progress is pinned."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())


def main():
    out=ROOT/'runs/pen_solver_tick_preparation_archive_v1'
    dest=ROOT/'provenance/pen_solver_tick_preparation_v1'
    if out.exists() or dest.exists():raise ValueError('Fresh readiness archive required')
    root=ROOT/'runs/pen_solver_tick_preparation_v1/report.json';r=read(root)
    tests=ROOT/'runs/pen_solver_tick_tests568_v1.log';focused=ROOT/'runs/pen_solver_tick_balance_tests_v1.log'
    checks=dict(preparation103=r['passed']==r['total']==103 and all(r['checks'].values()),
        all10=len(r['rows'])==10,all5_failures=sum(not c['original_metrics']['checks']['linear_impulse'] for c in r['rows'])==5,
        all_source_epochs=all(c['source_epoch_scope']=='SOURCE_SNAPSHOT_ONLY_NOT_ORIGINAL_FAILING_TICK' for c in r['rows']),
        packed_layouts=all(c['M_shape']==[1,71] and c['M_dtype']=='float32' and c['nv']==26 and c['solver_factorization_is_sparse'] is False for c in r['rows']),
        future_cap=r['future_native_cap']==960,
        no_native_updates=r['actual_native_integrations']==r['optimizer_steps']==r['normalizer_fits']==0,
        tests568='Ran 568 tests' in tests.read_text() and '\nOK\n' in tests.read_text(),
        focused6='Ran 6 tests' in focused.read_text() and '\nOK\n' in focused.read_text())
    if not all(checks.values()):raise ValueError('Readiness checkpoint discrepancy '+repr(checks))
    docs=[ROOT/'docs'/p for p in ['pen-solver-tick-preparation-results-2026-11-17.md','research-plan-v1.46.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__),ROOT/'tests/test_pen_solver_tick_balance.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],preparation_checks=[103,103],tests=568,
        failure_conditions=5,comparison_conditions=5,future_native_cap=960,
        actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        full54_complete=False,private_failing_tick_context_captured=False,
        observer_implemented_and_GPU_qualified=False,
        scope='CPU source/layout/arithmetic readiness only. Full54 producer separate and live at preparation. '
              'No native reproduction,physical-label fix,Actor/ego/whole-goal or acceleration claim.')
    write_json(out/'review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    parent=ROOT/'provenance/pen_selected_recursive_results_v1/ledger.json';previous=read(parent)
    inherited=previous['inherited_verified_manifests']+[previous['additional_verified_manifest']];old=set()
    for pin in inherited:
        for record in read(verify_file_pin(pin))['records']:old.add(json.dumps(record,sort_keys=True))
    todo=[file_pin(p) for p in [root,tests,focused,*docs,out/'review.json',out/'source_receipt.json']];verified={}
    forbidden={str((ROOT/'runs/pen_native_screened_rollout_v1/work.json').resolve()),str((ROOT/'runs/pen_native_screened_rollout_v1.log').resolve())}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if pin['path'] in forbidden:raise ValueError('No mutable producer progress pins')
        if key in old or key in verified:continue
        p=verify_file_pin(pin);verified[key]=pin
        if p.suffix=='.json' and p.is_relative_to(ROOT/'runs'):todo.extend(pins(read(p)))
    verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True);verify_source_snapshot(receipt,live=True)
    write_json(out/'additional_verified_pins.json',dict(records=list(verified.values()),inherited_manifests=inherited,scope='New files fully hashed. Parent verified manifests inherited;not all parent files cold-rehashed. Mutable producer-root progress excluded.'))
    dest.mkdir();write_json(dest/'summary.json',summary)
    write_json(dest/'ledger.json',dict(schema='pen_solver_tick_readiness_ledger:1',prior_evidence=file_pin(parent),inherited_verified_manifests=inherited,preparation=file_pin(root),review=file_pin(out/'review.json'),summary=file_pin(dest/'summary.json'),documents=[file_pin(p) for p in docs],tests=[file_pin(tests),file_pin(focused)],source_receipt=file_pin(out/'source_receipt.json'),additional_verified_manifest=file_pin(out/'additional_verified_pins.json'),additional_verified_pins=len(verified),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0,full54_complete=False))
    print(json.dumps(dict(ledger=file_pin(dest/'ledger.json'),checks=summary['checks'],new_pins=len(verified),native=0)),flush=True)


if __name__=='__main__':main()
