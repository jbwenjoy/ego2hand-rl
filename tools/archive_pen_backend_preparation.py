"""Seal numerical regression and preparation readiness, not native admission."""
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
    out=ROOT/'provenance/pen_backend_preparation_v1';analysis=ROOT/'runs/pen_backend_preparation_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh preparation archive required')
    rp=ROOT/'runs/pen_backend_preparation_numeric_regression_v1/report.json';r=read(rp);prior=ROOT/'provenance/pen_backend_response_design_v1/ledger.json'
    checks=dict(regression62=r['passed']==r['total']==62 and all(r['checks'].values()),eight_records=len(r['records'])==8,
        four_invalid=sum(not row['original_valid'] for row in r['records'])==4,no_new_work=(r['actual_native_integrations'],r['optimizer_steps'],r['new_backend_rows'])==(0,0,0),
        no_prior_rebinding=r['prior_evidence']==file_pin(prior))
    verify_source_snapshot(read(verify_file_pin(r['source_receipt'])),live=True)
    log=ROOT/'runs/pen_backend_preparation_tests478_v1.log';txt=log.read_text();checks['tests478']='Ran 478 tests in 10.380s' in txt and '\nOK\n' in txt
    training=ROOT/'runs/pen_backend_long_training_v1/source_receipt.json';verify_source_snapshot(read(training),live=True);checks['training_sources_unchanged']=True
    registration=ROOT/'runs/pen_backend_response_registration_v1/registration.json';reg=read(registration);verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True);checks['response_sources_unchanged']=True
    if not all(checks.values()):raise ValueError('Independent preparation review failed: '+repr(checks))
    docs=[ROOT/'docs'/n for n in ['pen-backend-preparation-results-2026-11-16.md','research-plan-v1.28.md','pen-backend-response-preparation-protocol-v1.md']]
    scripts=[Path(__file__),ROOT/'tools/prepare_backend_pen_responses.py',ROOT/'tools/verify_backend_probe_preparation.py']
    analysis.mkdir();receipt=freeze_sources(docs+scripts,analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(analysis/'source_receipt.json',receipt)
    todo=[file_pin(p) for p in [rp,prior,log,registration,training,analysis/'source_receipt.json']+docs+scripts]
    verified={};allowed={rp.parent,analysis}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(p) for p in allowed):todo.extend(pins(read(path)))
    out.mkdir();write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'summary.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),regression=file_pin(rp),test_log=file_pin(log),old_fixture_rows=8,old_invalid_rows=4,
        actual_native_integrations=0,optimizer_steps=0,new_backend_rows=0,scope='Offline numerical/source/denominator readiness only;actual new-backend collection/preparation/WM/gradient/task/ego admission pending.'))
    write(out/'ledger.json',dict(schema='backend_preparation_readiness_ledger:1',prior_evidence=file_pin(prior),summary=file_pin(out/'summary.json'),regression=file_pin(rp),
        source_receipts=[r['source_receipt'],file_pin(analysis/'source_receipt.json')],documents=[file_pin(p) for p in docs],verified_manifest=file_pin(analysis/'verified_pins.json'),verified_pins=len(verified),
        training_source_receipt=file_pin(training),response_registration=file_pin(registration),actual_native_integrations=0,optimizer_steps=0,new_backend_rows=0))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],verified_pins=len(verified))),flush=True)


if __name__=='__main__':main()
