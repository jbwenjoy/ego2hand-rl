"""Seal the first18 diagnostic without pinning mutable full54 producer files."""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'), str(ROOT/'tools')]
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.frozen_sources import freeze_sources, verify_source_snapshot
from supermimic_research.strict_json import write_json
from archive_pen_source_ccd_evidence import pins


def read(p):
    return json.loads(Path(p).read_text())


def main():
    out=ROOT/'runs/pen_selected_recursive_archive_v1'
    archive=ROOT/'provenance/pen_selected_recursive_results_v1'
    if out.exists() or archive.exists():
        raise ValueError('Fresh immutable checkpoint required')
    roots={k:ROOT/p for k,p in dict(
        index='runs/pen_native_screened_review_v3/indices/completed_18.json',
        summary='runs/pen_native_screened_summary_18_v1/report.json',
        support_video='runs/pen_native_screened_repeat0_preview_SUPPORT_v1/report.json',
        diagnostic_registration='runs/pen_selected_rollout_diagnostic_registration_v1/registration.json',
        data='runs/pen_selected_rollout_diagnostic_data_v1/report.json',
        diagnostic='runs/pen_selected_rollout_diagnostic_v1/report.json',
        review='runs/pen_selected_rollout_diagnostic_review_v1/report.json').items()}
    r={k:read(p) for k,p in roots.items()}
    case_reviews=[read(verify_file_pin(x['report'])) for x in r['index']['reviewed_cases']]
    checks=dict(first18=len(case_reviews)==18 and not r['index']['complete'],
        all_cases16=all(c['passed']==c['total']==16 and all(c['checks'].values()) for c in case_reviews),
        summary7=r['summary']['passed']==r['summary']['total']==7,
        support_video5=r['support_video']['passed']==r['support_video']['total']==5,
        data5=r['data']['passed']==r['data']['total']==5,
        diagnostic21=r['diagnostic']['passed']==r['diagnostic']['total']==21,
        independent170=r['review']['passed']==r['review']['total']==170,
        all_boolean_checks=all(all(r[k]['checks'].values()) for k in ['summary','support_video','data','diagnostic','review']),
        original8_masks=r['data']['window_counts']['1']['physical_masked_windows']==8,
        first18_native=r['summary']['summary']['native_integrations']==27144,
        first18_no_success=r['summary']['summary']['full_task_successes']==0,
        all270_groups=len(list(csv.DictReader(verify_file_pin(r['review']['per_case']).open())))==270
                      and sum(len(v) for v in r['review']['groups'].values())==15,
        no_new_native=all(r[k].get('actual_native_integrations',r[k].get('native_integrations'))==0
                          and r[k]['optimizer_steps']==0 for k in ['summary','support_video','data','diagnostic','review']))
    if not all(checks.values()):
        raise ValueError('Checkpoint discrepancy '+repr(checks))
    docs=[ROOT/'docs'/n for n in ['pen-selected-recursive-results-2026-11-17.md','research-plan-v1.45.md']]
    out.mkdir()
    sources=sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__),ROOT/'tools/review_pen_selected_model_rollouts.py']
    receipt=freeze_sources(sources,out/'source_snapshot',ROOT,
        subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    summary=dict(checks=[sum(checks.values()),len(checks)],reviewed_cases=18,
        declared_cases=54,full54_complete=False,original_selected_controls=1765,
        original_native_queries=3393,original_native_integrations=27144,
        full_task_successes=0,physical_stops=8,task_failures=10,
        qualified_windows=[1757,1739,1703,1631,1487],
        data_checks=[5,5],diagnostic_checks=[21,21],independent_checks=[170,170],
        offline_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='First18 sealed development records only;full54 producer may continue. '
              'Large recursive tails;no Actor/ego/independent-seed/whole-goal or speed claim.')
    write_json(out/'review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),
                                   summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    parent=ROOT/'provenance/pen_screened_running_v1/ledger.json'
    previous=read(parent)
    old=set()
    inherited=[]
    for key in ['parent_verified_manifest','additional_verified_manifest']:
        pin=previous[key]
        inherited.append(pin)
        for record in read(verify_file_pin(pin))['records']:
            old.add(json.dumps(record,sort_keys=True))
    forbidden={str((ROOT/'runs/pen_native_screened_rollout_v1/work.json').resolve()),
               str((ROOT/'runs/pen_native_screened_rollout_v1.log').resolve())}
    verified={}
    todo=[file_pin(p) for p in list(roots.values())+docs+[out/'review.json',out/'source_receipt.json']]
    while todo:
        pin=todo.pop()
        if pin['path'] in forbidden:
            raise ValueError('Mutable producer progress is outside this checkpoint')
        key=json.dumps(pin,sort_keys=True)
        if key in old or key in verified:
            continue
        path=verify_file_pin(pin)
        verified[key]=pin
        if path.suffix=='.json' and path.is_relative_to(ROOT/'runs'):
            todo.extend(pins(read(path)))
    for item in r.values():
        if 'source_receipt' in item:
            verify_source_snapshot(read(verify_file_pin(item['source_receipt'])),live=True)
    verify_source_snapshot(receipt,live=True)
    write_json(out/'additional_verified_pins.json',dict(records=list(verified.values()),
        inherited_manifests=inherited,
        scope='New dependencies fully hashed here. Prior verified manifests inherited, '
              'not all parent files cold-rehashed. No mutable producer-root work/log pins.'))
    archive.mkdir()
    write_json(archive/'summary.json',summary)
    write_json(archive/'ledger.json',dict(schema='pen_selected_recursive_checkpoint:1',
        prior_evidence=file_pin(parent),inherited_verified_manifests=inherited,
        roots={k:file_pin(p) for k,p in roots.items()},review=file_pin(out/'review.json'),
        summary=file_pin(archive/'summary.json'),documents=[file_pin(p) for p in docs],
        source_receipt=file_pin(out/'source_receipt.json'),
        additional_verified_manifest=file_pin(out/'additional_verified_pins.json'),
        additional_verified_pins=len(verified),full54_complete=False,
        offline_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],
                          new_pins=len(verified),full54=False)),flush=True)


if __name__=='__main__':
    main()
