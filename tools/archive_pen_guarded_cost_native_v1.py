"""Seal guarded arithmetic evidence,all native queries and independent reviews."""
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
    out=ROOT/'runs/pen_guarded_cost_native_checkpoint_v1'; dest=ROOT/'provenance/pen_guarded_cost_native_v1'
    if out.exists() or dest.exists(): raise ValueError('Fresh guarded cost checkpoint required')
    prior=file_pin(ROOT/'provenance/pen_stable_friction_qualification_v1/ledger.json'); parent=json.loads(verify_file_pin(prior).read_text())
    manifests=parent['inherited_verified_manifests']+[parent['additional_verified_manifest']]
    identity=lambda p:json.dumps(p,sort_keys=True); known=set()
    for p in manifests: known.update(identity(q) for q in pins(json.loads(verify_file_pin(p).read_text())))
    names=dict(arithmetic='pen_stable_friction_backend_qualification_v4/report.json',arithmetic_review='pen_guarded_cost_arithmetic_saved_review_v1/report.json',
        rounding='pen_friction_rounding_probe_v2/report.json',retained_import_failure='pen_friction_rounding_probe_v1/failure_closure.json',
        worker_registration='pen_stable_friction_comparator_registration_v1/registration.json',reader_registration='pen_stable_friction_comparator_reader_registration_v1/registration.json',
        nativeA='pen_stable_friction_comparator_v1/A/report.json',nativeB='pen_stable_friction_comparator_v1/B/report.json',
        readerA='pen_stable_friction_comparator_review_v1/A/report.json',readerB='pen_stable_friction_comparator_review_v1/B/report.json',
        summary='pen_stable_friction_comparator_summary_v1/report.json')
    roots={k:file_pin(ROOT/'runs'/v) for k,v in names.items()}; reports={k:json.loads(verify_file_pin(p).read_text()) for k,p in roots.items()}
    checks={k:r['passed']==r['total'] and all(r['checks'].values()) for k,r in reports.items() if 'checks' in r}
    checks['all1920_exact']=all(reports[k]['actual_native_integrations']==reports[k]['attempted_native_integrations']==960 and reports[k]['setup_extra_native']==0 and not reports[k]['setup_native_unknown'] for k in ['nativeA','nativeB'])
    checks['both3493_reviews']=all(reports[k]['passed']==reports[k]['total']==3493 and reports[k]['recomputed_array_elements']==70560 for k in ['readerA','readerB'])
    checks['B_after_reviewA']=reports['nativeB']['prior_replica_review']==roots['readerA'] and reports['nativeA']['prior_replica_review'] is None
    checks['all20_imports']=len(reports['worker_registration']['source_import_cpu_preflight'])==20 and all(row['audit']['rollback_exact'] and row['audit']['native_python_rng_unchanged'] and all(row['audit']['unchanged_top_level_fields'].values()) for row in reports['worker_registration']['source_import_cpu_preflight'])
    checks['no_updates_or_normalizers']=all(r.get('optimizer_steps',0)==r.get('normalizer_fits',0)==0 for r in reports.values())
    if not all(checks.values()): raise ValueError('Complete guarded native evidence required')
    documents=[ROOT/'docs'/n for n in ['pen-guarded-friction-native-results-2026-11-17.md','research-plan-v1.56.md','pen-stable-friction-comparator-protocol-v1.md']]
    tools=[ROOT/'tools'/n for n in ['probe_pen_friction_rounding_v1.py','probe_pen_friction_rounding_v2.py','qualify_pen_stable_friction_backend_v4.py',
        'review_pen_guarded_cost_qualification_v1.py','close_pen_rounding_probe_import_failure_v1.py','run_pen_stable_friction_comparator_v1.py','review_pen_stable_friction_comparator_v1.py','summarize_pen_stable_friction_comparator_v1.py']]
    logs=[ROOT/'runs'/n for n in ['pen_friction_rounding_probe_v1.log','pen_friction_rounding_probe_v2.log','pen_guarded_cost_tests638_v1.log','pen_guarded_cost_design_v1_tests.log',
        'pen_stable_friction_backend_qualification_v4.log','pen_guarded_cost_arithmetic_saved_review_v1.log','pen_stable_friction_comparator_register_v1.log',
        'pen_stable_friction_comparator_reader_register_v1.log','pen_stable_friction_comparator_A_v1.log','pen_stable_friction_comparator_B_v1.log',
        'pen_stable_friction_comparator_review_A_v1.log','pen_stable_friction_comparator_review_B_v1.log','pen_stable_friction_comparator_summary_v1.log']]
    out.mkdir(); dest.mkdir()
    tests=sorted((ROOT/'tests').glob('test_pen_stable_friction*.py'))+[ROOT/'tests/test_pen_friction_rounding_probe_v2.py']
    receipt=freeze_sources(sorted((ROOT/'src').rglob('*.py'))+tools+tests+documents+[Path(__file__),ROOT/'configs/pen_stable_friction_comparator_v1.toml'],out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    todo=list(roots.values())+[file_pin(out/'source_receipt.json')]+[file_pin(p) for p in logs]; verified={}; reached=set()
    while todo:
        p=todo.pop(); key=identity(p); path=Path(p['path']).resolve()
        if path==ROOT/'runs/pen_native_screened_rollout_v1/work.json': raise ValueError('Mutable historical progress cannot enter this checkpoint')
        if key in known: reached.add(key); continue
        if key in verified: continue
        verify_file_pin(p); verified[key]=p
        if path.suffix=='.json' and path.is_relative_to(ROOT/'runs'): todo.extend(pins(json.loads(path.read_text())))
    for p in list(roots.values())+manifests+[prior]+[file_pin(p) for p in logs]: verify_file_pin(p)
    verify_source_snapshot(receipt,live=True)
    write_json(out/'additional_verified_pins.json',dict(records=list(verified.values()),inherited_reached=len(reached),scope='New pins cold-verified;prior evidence inherited through exact manifest identities.'))
    write_json(out/'review.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=0,optimizer_steps=0,normalizer_fits=0))
    write_json(dest/'ledger.json',dict(schema='pen_guarded_cost_native_checkpoint:1',prior_evidence=prior,inherited_verified_manifests=manifests,
        roots=roots,documents=[file_pin(p) for p in documents],retained_logs=[file_pin(p) for p in logs],source_receipt=file_pin(out/'source_receipt.json'),
        review=file_pin(out/'review.json'),additional_verified_manifest=file_pin(out/'additional_verified_pins.json'),additional_verified_pins=len(verified),inherited_pins_reached=len(reached),
        native_integrations_in_producers=1920,archival_native_integrations=0,optimizer_steps=0,normalizer_fits=0,
        explicit_saved_input_GPU_probe_launches=9,explicit_GPU_qualification_launches=6,retained_import_failure_GPU_launches=0,
        additional_bounded_GPU_unit_test_launches=1,additional_test_evidence='Observed terminal test output;formal retained qualification independently covers the production helpers.',
        scope='All240 prospective cost queries,two fresh processes and independent reviews. Earlier arithmetic failures/import failure retained. '
              'W1 inspected-source evidence only;no automatic W32,Actor,full-task,ego,hardware,speed or whole-goal completion admission.'))
    print(json.dumps(dict(ledger=file_pin(dest/'ledger.json'),checks=[sum(checks.values()),len(checks)],new_pins=len(verified),inherited_reached=len(reached))),flush=True)


if __name__=='__main__': main()
