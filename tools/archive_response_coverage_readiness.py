"""Seal old-fixture coverage regression and frozen future review protocol."""
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    out=ROOT/'provenance/pen_backend_coverage_readiness_v1';analysis=ROOT/'runs/pen_backend_coverage_readiness_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh coverage archive required')
    prior=ROOT/'provenance/pen_backend_arm_transition_v1/ledger.json';rp=ROOT/'runs/pen_backend_coverage_regression_v1/report.json';r=read(rp);reg_path=verify_file_pin(r['registration']);reg=read(reg_path)
    checks=dict(regression11=r['passed']==r['total']==11 and all(r['checks'].values()),old_rows=sum(c['rows'] for c in r['cohorts'].values())==5832,
        masks_preserved=(r['cohorts']['TRAIN']['valid_rows'],r['cohorts']['VALIDATE']['valid_rows'])==(3866,1935),
        eligibility_preserved=r['cohorts']['TRAIN']['eligible_group_counts']=={'rank4':189,'rank20':949},
        no_new_work=(r['actual_native_integrations'],r['optimizer_steps'],r['normalizer_fits'],r['new_backend_rows'])==(0,0,0,0))
    log=ROOT/'runs/pen_backend_coverage_tests489_v1.log';txt=log.read_text();checks['tests489']='Ran 489 tests in 10.199s' in txt and '\nOK\n' in txt
    train=ROOT/'runs/pen_backend_long_training_v1/source_receipt.json';response=ROOT/'runs/pen_backend_response_registration_v1/registration.json'
    receipts=[r['source_receipt'],reg['source_receipt'],file_pin(train),read(response)['source_receipt']]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    checks['all_source_receipts_live']=True
    if not all(checks.values()):raise ValueError('Coverage readiness archive failed: '+repr(checks))
    docs=[ROOT/'docs'/n for n in ['pen-backend-data-review-protocol-v1.md','pen-backend-coverage-readiness-results-2026-11-16.md','research-plan-v1.31.md']]
    scripts=[Path(__file__),ROOT/'tools/review_backend_pen_responses.py',ROOT/'tools/verify_response_coverage_regression.py'];analysis.mkdir()
    receipt=freeze_sources(docs+scripts,analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(analysis/'source_receipt.json',receipt)
    todo=[file_pin(p) for p in [prior,rp,reg_path,log,response,analysis/'source_receipt.json']+docs+scripts]+receipts;verified={};allowed={rp.parent,reg_path.parent,analysis}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(p) for p in allowed):todo.extend(pins(read(path)))
    out.mkdir();write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'summary.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),regression=file_pin(rp),registration=file_pin(reg_path),
        old_fixture_rows=5832,old_invalid_rows=31,TRAIN_eligible_groups={'rank4':189,'rank20':949},added_actual_native_integrations=0,added_optimizer_steps=0,normalizer_fits=0,new_backend_rows=0,
        scope='Pure old-fixture numerical preservation and future new-backend review readiness only. New collection/physics/gradient/Actor/task/ego evidence pending.'))
    write(out/'ledger.json',dict(schema='backend_coverage_readiness_ledger:1',prior_evidence=file_pin(prior),summary=file_pin(out/'summary.json'),regression=file_pin(rp),registration=file_pin(reg_path),
        source_receipts=receipts+[file_pin(analysis/'source_receipt.json')],documents=[file_pin(p) for p in docs],verified_manifest=file_pin(analysis/'verified_pins.json'),verified_pins=len(verified),
        added_actual_native_integrations=0,added_optimizer_steps=0,normalizer_fits=0,new_backend_rows=0))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],verified_pins=len(verified))),flush=True)


if __name__=='__main__':main()
