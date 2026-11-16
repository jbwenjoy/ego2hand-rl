"""Close the bounded native full-rehash comparison without changing defaults."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=ROOT/'runs/native_full_rehash_archive_v1';archive=ROOT/'provenance/native_full_rehash_results_v1'
    if out.exists() or archive.exists():raise ValueError('Fresh integration archive required')
    roots=dict(integration=ROOT/'runs/native_full_rehash_integration_v1/report.json',review=ROOT/'runs/native_full_rehash_review_v1/report.json',registration=ROOT/'runs/native_full_rehash_integration_registration_v1/registration.json')
    r={k:read(p) for k,p in roots.items()};run,review,reg=[r[k] for k in ['integration','review','registration']]
    checks=dict(integration7=run['passed']==run['total']==7 and all(run['checks'].values()),
        review14=review['passed']==review['total']==14 and all(review['checks'].values()) and review['integration']==file_pin(roots['integration']),
        original_physics_all180=review['valid_rows']==review['total_rows']==180 and review['row_flags']==[3600,3600],
        exact1440_native=run['actual_native_integrations']==run['attempted_native_integrations']==reg['native_budget']==1440,
        exact_restore_counts=run['restore_completed']==202 and run['restore_attempts']==210 and reg['expected_positive_restores']==198 and reg['negative_warmup_restores']==4 and reg['expected_guard_denials']==8,
        zero_optimizer=all(x['optimizer_steps']==0 for x in [run,review]),
        no_default_or_epoch=not run['source_epoch_used'] and not run['default_changed'],
        all120_matched_comparisons=len(review['matched_inputs'])==120 and all(all(c['identical_inputs'].values()) for c in review['matched_inputs']),
        all8_failure_snapshots=len(review['negative_snapshot_reviews'])==8 and all(all(n['checks'].values()) for n in review['negative_snapshot_reviews']))
    for t in review['timing_rounds']:
        name=t['source_name']+'_R'+str(t['round']);phases={p['mode']:p for p in run['phases'] if p['source_name']==t['source_name'] and p['round']==t['round']}
        for field in ['phase_wall_s','branch_wall_sum_s']:
            actual={k:p[field] for k,p in phases.items()};mean=(actual['ORIGINAL_A']+actual['ORIGINAL_B'])/2
            checks[name+'_'+field]=t[field]['times']==actual and t[field]['original_bracket_mean_s']==mean and t[field]['candidate_reduction_vs_bracket_mean']==1-actual['SINGLE_RESOLVE']/mean
        for mode in phases:
            cases=[c for c in run['cases'] if c['source_name']==t['source_name'] and c['round']==t['round'] and c['mode']==mode]
            sums={k:sum(c['stages'][k] for c in cases) for k in cases[0]['stages']}
            checks[name+'_'+mode+'_stages']=len(cases)==10 and phases[mode]['branch_wall_sum_s']==sum(c['branch_wall_s'] for c in cases) and t['stages_s'][mode]==sums
    for source,summary in review['timing_summary'].items():
        entries=[t for t in review['timing_rounds'] if t['source_name']==source]
        for field,value in summary.items():
            medians={m:float(np.median([t[field]['times'][m] for t in entries])) for m in ['ORIGINAL_A','SINGLE_RESOLVE','ORIGINAL_B']}
            checks[source+'_'+field+'_medians']=len(entries)==3 and medians==value['median_times_s'] and value['median_candidate_reduction_vs_bracket_mean']==float(np.median([t[field]['candidate_reduction_vs_bracket_mean'] for t in entries]))
    tests=ROOT/'runs/full_rehash_scope_tests523_v1.log';checks['full523_tests']='Ran 523 tests' in tests.read_text() and '\nOK\n' in tests.read_text()
    receipts=[x['source_receipt'] for x in r.values()]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    if not all(checks.values()):raise ValueError('Integration archive discrepancy: '+repr([k for k,v in checks.items() if not v]))
    docs=[ROOT/'docs'/n for n in ['native-full-rehash-results-2026-11-16.md','research-plan-v1.38.md']]
    out.mkdir();receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+docs+[Path(__file__),ROOT/'tools/review_native_full_rehash.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(out/'source_receipt.json',receipt)
    differences={k:max(c['observed_max_absolute_differences'][k] for c in review['matched_inputs']) for k in review['matched_inputs'][0]['observed_max_absolute_differences']}
    summary=dict(checks=[sum(checks.values()),len(checks)],integration_checks=[7,7],review_checks=[14,14],physical_rows=[180,180],row_flags=[3600,3600],
        native_integrations=1440,attempted_native_integrations=1440,optimizer_steps=0,added_review_native_integrations=0,
        restore_completed=202,restore_attempts=210,guard_denials=8,matched_input_comparisons=120,
        observed_max_absolute_differences=differences,timing_summary=review['timing_summary'],integration_wall_s=run['wall_s'],
        default_changed=False,source_epoch_used=False,
        scope='Bounded W1 full-rehash implementation diagnostic only;all180 saved packets and8 denied state pairs reviewed. Both modes have same explicit scope instrumentation. No default/W32 training/task/Actor/ego speedup claim;old epoch anomaly remains unresolved.')
    write(out/'independent_review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),summary=summary,source_receipt=file_pin(out/'source_receipt.json')))
    prior_path=ROOT/'provenance/pen_backend_probe_profile_results_v1/ledger.json';prior=read(prior_path);verified={}
    for pin in read(verify_file_pin(prior['verified_manifest']))['records']:verify_file_pin(pin);verified[json.dumps(pin,sort_keys=True)]=pin
    todo=[file_pin(p) for p in list(roots.values())+docs+[out/'independent_review.json',out/'source_receipt.json',prior_path,tests]]+receipts
    allowed={p.parent for p in roots.values()}|{out}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(root) for root in allowed):todo.extend(pins(read(path)))
    write(out/'verified_pins.json',dict(records=list(verified.values())));archive.mkdir();write(archive/'summary.json',summary)
    write(archive/'ledger.json',dict(schema='native_full_rehash_results_ledger:1',prior_evidence=file_pin(prior_path),roots={k:file_pin(p) for k,p in roots.items()},summary=file_pin(archive/'summary.json'),
        review=file_pin(out/'independent_review.json'),source_receipts=receipts+[file_pin(out/'source_receipt.json')],documents=[file_pin(p) for p in docs],
        verified_manifest=file_pin(out/'verified_pins.json'),verified_pins=len(verified),native_integrations=1440,added_review_native_integrations=0,optimizer_steps=0,default_changed=False))
    print(json.dumps(dict(ledger=file_pin(archive/'ledger.json'),checks=summary['checks'],pins=len(verified),physical_rows=summary['physical_rows'])),flush=True)


if __name__=='__main__':main()
