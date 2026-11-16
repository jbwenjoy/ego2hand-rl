"""Immutable partial checkpoint;never claims completion of the live full54 study."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())


def main():
    out=ROOT/'runs/pen_screened_running_archive_v1';archive=ROOT/'provenance/pen_screened_running_v1';producer=ROOT/'runs/pen_native_screened_rollout_v1'
    if out.exists() or archive.exists() or (producer/'report.json').exists():raise ValueError('Fresh incomplete-study checkpoint required')
    roots=dict(index=ROOT/'runs/pen_native_screened_review_v3/indices/completed_10.json',prior_index=ROOT/'runs/pen_native_screened_review_v2/indices/completed_04.json',failed_diagnostic=ROOT/'runs/pen_impulse_balance_v1/report.json',diagnostic=ROOT/'runs/pen_impulse_balance_v2/report.json',figure=ROOT/'runs/pen_impulse_balance_figure_v1/report.json',preview=ROOT/'runs/pen_native_screened_repeat0_preview_RESIDUAL_v1/report.json')
    r={k:read(p) for k,p in roots.items()};index=r['index'];cases=[read(verify_file_pin(x['report'])) for x in index['reviewed_cases']];first9=[x['summary'] for x in cases[:9]];test=ROOT/'runs/pen_native_screened_review_tests562_v1.log'
    checks=dict(partial10=len(cases)==10 and not index['complete'] and index['declared_total']==54,all16_case_reviews=all(c['passed']==c['total']==16 and all(c['checks'].values()) for c in cases),all_input_capsules=all(c['input_file_capsule']['all_bytes_verified_after'] and c['input_file_capsule']['same_file_inventory'] and c['input_file_capsule']['records_before_read'] for c in cases),all_first9_fixed=all(s['teacher']=='RESIDUAL' and s['repetition']==0 for s in first9),first9_zero_success=sum(s['full_task_success'] for s in first9)==0,
        first9_native=sum(s['native_queries'] for s in first9)*8==14104,original_failed_diagnostic=r['failed_diagnostic']['passed']==3 and r['failed_diagnostic']['total']==6,corrected6=r['diagnostic']['passed']==r['diagnostic']['total']==6 and all(r['diagnostic']['checks'].values()),preview5=r['preview']['passed']==r['preview']['total']==5 and all(r['preview']['checks'].values()) and r['preview']['review']==file_pin(roots['index']),no_offline_native=all(c['actual_native_integrations']==c['optimizer_steps']==0 for c in cases) and all(r[k]['actual_native_integrations']==r[k]['optimizer_steps']==0 for k in ['failed_diagnostic','diagnostic','figure','preview']),tests562='Ran 562 tests' in test.read_text() and '\nOK\n' in test.read_text())
    if not all(checks.values()):raise ValueError('Running checkpoint discrepancy '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['pen-screened-running-results-2026-11-17.md','research-plan-v1.44.md']];out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__)],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write_json(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],full54_complete=False,reviewed_cases=10,first9=first9,first9_native_integrations=14104,offline_native_integrations=0,optimizer_steps=0,normalizer_fits=0,diagnostic_rows=676,impulse_failures_preserved=5,failed_diagnostic_checks=[3,6],corrected_diagnostic_checks=[6,6],preview_checks=[5,5],tests=562,
        scope='Sealed partial cases/diagnostic/preview only. Running producer native total/outcomes remain pending;no model ranking,isolated speed,Actor/ego/hardware or full goal success claim.')
    write_json(out/'review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    parent=ROOT/'provenance/pen_live_lifecycle_results_v1/ledger.json';previous=read(parent);verify_file_pin(previous['verified_manifest']);old={json.dumps(p,sort_keys=True) for p in read(verify_file_pin(previous['verified_manifest']))['records']};verified={};todo=[file_pin(p) for p in list(roots.values())+docs+[test,out/'review.json',out/'source_receipt.json']];allowed={p.parent for p in roots.values()}|{out,producer,ROOT/'runs/pen_native_screened_review_v3',ROOT/'runs/pen_native_screened_review_v2',ROOT/'runs/pen_native_screened_review_v1',ROOT/'runs/pen_native_screened_rollout_registration_v1'}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified or key in old:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    for c in cases:verify_source_snapshot(read(verify_file_pin(c['source_receipt'])),live=True)
    for k in ['failed_diagnostic','diagnostic','figure','preview']:verify_source_snapshot(read(verify_file_pin(r[k]['source_receipt'])),live=True)
    verify_source_snapshot(receipt,live=True);write_json(out/'additional_verified_pins.json',dict(records=list(verified.values()),parent_manifest=previous['verified_manifest'],scope='New checkpoint dependencies hashed here. Parent manifest inherited from its prior verified archive,not all cold-rehashed in this partial checkpoint. Mutable producer-root work/log files excluded.'))
    archive.mkdir();write_json(archive/'summary.json',summary);write_json(archive/'ledger.json',dict(schema='pen_screened_running_checkpoint_ledger:1',prior_evidence=file_pin(parent),parent_verified_manifest=previous['verified_manifest'],roots={k:file_pin(p) for k,p in roots.items()},review=file_pin(out/'review.json'),summary=file_pin(archive/'summary.json'),documents=[file_pin(p) for p in docs],source_receipt=file_pin(out/'source_receipt.json'),additional_verified_manifest=file_pin(out/'additional_verified_pins.json'),additional_verified_pins=len(verified),full54_complete=False,offline_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],new_pins=len(verified),full54=False)),flush=True)


if __name__=='__main__':main()
