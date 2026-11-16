"""Seal both native processes, corrected reviews and the retained reader failure."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from archive_pen_source_ccd_evidence import pins
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.strict_json import write_json


def main():
    out=ROOT/'runs/pen_termination_comparator_checkpoint_v1'; dest=ROOT/'provenance/pen_termination_comparator_v1'
    if out.exists() or dest.exists(): raise ValueError('Fresh final comparator checkpoint required')
    prior=file_pin(ROOT/'provenance/pen_termination_rule_cpu_v1/ledger.json'); parent=json.loads(verify_file_pin(prior).read_text())
    manifests=parent['inherited_verified_manifests']+[parent['additional_verified_manifest']]
    identity=lambda p:json.dumps(p,sort_keys=True)
    known=set()
    for p in manifests: known.update(identity(q) for q in pins(json.loads(verify_file_pin(p).read_text())))
    names=dict(worker_registration='pen_termination_comparator_registration_v1/registration.json',
        reader_registration_v1='pen_termination_comparator_reader_registration_v1/registration.json',
        reader_registration_v2='pen_termination_comparator_reader_registration_v2/registration.json',
        nativeA='pen_termination_comparator_v1/A/report.json',nativeB='pen_termination_comparator_v1/B/report.json',
        failed_reader_v1='pen_termination_comparator_review_v1/A/report.json',
        readerA='pen_termination_comparator_review_v2/A/report.json',readerB='pen_termination_comparator_review_v2/B/report.json',
        correction_review='pen_termination_reader_correction_review_v1/report.json',summary='pen_termination_comparator_summary_v1/report.json')
    roots={k:file_pin(ROOT/'runs'/v) for k,v in names.items()}; reports={k:json.loads(verify_file_pin(p).read_text()) for k,p in roots.items()}
    checks={k:d['passed']==d['total'] and all(d['checks'].values()) for k,d in reports.items() if 'checks' in d and k!='failed_reader_v1'}
    checks['native_exact1920']=sum(reports[k]['actual_native_integrations'] for k in ['nativeA','nativeB'])==1920 and all(reports[k]['actual_native_integrations']==reports[k]['attempted_native_integrations']==960 and reports[k]['setup_extra_native']==0 and not reports[k]['setup_native_unknown'] for k in ['nativeA','nativeB'])
    checks['zero_updates']=all(d.get('optimizer_steps',0)==0 and d.get('normalizer_fits',0)==0 for d in reports.values())
    failed=reports['failed_reader_v1']; bad=[k for k,v in failed['checks'].items() if not v]
    checks['retained_expected_failed_review']=failed['passed']==3373 and failed['total']==3493 and len(bad)==120 and all(k.endswith('_metadata_audit') for k in bad)
    checks['all20_cpu_imports']=len(reports['worker_registration']['source_import_cpu_preflight'])==20 and all(r['audit']['rollback_exact'] and r['audit']['native_python_rng_unchanged'] and all(r['audit']['unchanged_top_level_fields'].values()) for r in reports['worker_registration']['source_import_cpu_preflight'])
    checks['both_corrected_reviews']=all(reports[k]['reader_version']==2 and reports[k]['passed']==reports[k]['total']==3493 for k in ['readerA','readerB'])
    if not all(checks.values()): raise ValueError('Final comparator contract mismatch')
    documents=[ROOT/'docs'/n for n in ['pen-termination-comparator-results-2026-11-17.md','research-plan-v1.52.md','pen-termination-reader-v2-serialization-note.md','pen-termination-rule-comparator-protocol-v1.md']]
    tools=[ROOT/'tools'/n for n in ['run_pen_termination_comparator_v1.py','review_pen_termination_comparator_v1.py','review_pen_termination_comparator_v2.py','summarize_pen_termination_comparator_v1.py','review_pen_termination_reader_correction_v1.py']]
    logs=[ROOT/'runs'/n for n in ['pen_termination_comparator_tests601_v1.log','pen_termination_audit_json_v2_tests.log','pen_termination_comparator_A_v1.log','pen_termination_comparator_B_v1.log','pen_termination_comparator_review_A_v1.log','pen_termination_comparator_review_A_v2.log','pen_termination_comparator_review_B_v2.log']]
    out.mkdir(); dest.mkdir(); receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+tools+documents+[Path(__file__),ROOT/'tests/test_pen_termination_import_design.py',ROOT/'tests/test_pen_termination_audit_json_v2.py'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()); write_json(out/'source_receipt.json',receipt)
    todo=list(roots.values())+[file_pin(out/'source_receipt.json')]+[file_pin(p) for p in logs]
    verified={}; reached=set()
    while todo:
        p=todo.pop(); key=identity(p); path=Path(p['path']).resolve()
        if path==ROOT/'runs/pen_native_screened_rollout_v1/work.json': raise ValueError('Mutable full54 progress cannot enter checkpoint')
        if key in known: reached.add(key); continue
        if key in verified: continue
        verify_file_pin(p); verified[key]=p
        if path.suffix=='.json' and path.is_relative_to(ROOT/'runs'): todo.extend(pins(json.loads(path.read_text())))
        if len(verified)%2000==0: print(json.dumps(dict(new_verified=len(verified),inherited_reached=len(reached))),flush=True)
    for p in list(roots.values())+manifests+[prior]+[file_pin(p) for p in logs]: verify_file_pin(p)
    verify_source_snapshot(receipt,live=True)
    write_json(out/'additional_verified_pins.json',dict(records=list(verified.values()),inherited_reached=len(reached),scope='New pins cold-verified here;prior evidence inherited through exact manifest identities.'))
    write_json(out/'review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
    write_json(dest/'ledger.json',dict(schema='pen_termination_comparator_checkpoint:1',prior_evidence=prior,inherited_verified_manifests=manifests,
        roots=roots,documents=[file_pin(p) for p in documents],retained_logs=[file_pin(p) for p in logs],source_receipt=file_pin(out/'source_receipt.json'),
        review=file_pin(out/'review.json'),additional_verified_manifest=file_pin(out/'additional_verified_pins.json'),additional_verified_pins=len(verified),inherited_pins_reached=len(reached),
        native_integrations_in_producers=1920,archival_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        scope='All240 targeted queries,two actual processes and corrected saved reviews. Failed V1 review/cost,nonrescues,regression,scope gate crossings and cap50 retained. '
              'New pins cold-verified;prior manifests inherited,not all historical bytes rehashed. Candidate unqualified;no Actor/task/ego/speed or whole-goal completion claim.'))
    print(json.dumps(dict(ledger=file_pin(dest/'ledger.json'),checks=[sum(checks.values()),len(checks)],new_pins=len(verified),inherited_reached=len(reached))),flush=True)


if __name__=='__main__': main()
