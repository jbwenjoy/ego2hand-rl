"""Review frozen response registration and deny premature collection, CPU only."""
import argparse
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
import collect_backend_pen_responses as collector


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    out=ROOT/'provenance/pen_backend_response_design_v1';analysis=ROOT/'runs/pen_backend_response_design_review_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh response design review required')
    rp=ROOT/'runs/pen_backend_response_registration_v1/registration.json';reg=read(rp)
    prior=ROOT/'provenance/pen_backend_launch_v1/ledger.json';verify_file_pin(reg['inputs']['prior_ledger'])
    checks=dict(schema=reg['schema']=='backend_physical_response_registration:1',two_roles=set(reg['specs'])=={'TRAIN_CONTROLLED_PHYSICAL_RESPONSE','VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT'},
        budget_max=sum(b['max_total_native'] for b in reg['budgets'].values())==51008,Adam_zero=all(b['optimizer_steps']==0 for b in reg['budgets'].values()))
    for pin in list(reg['inputs'].values())+list(reg['specs'].values())+list(reg['contexts'].values()):verify_file_pin(pin)
    verify_source_snapshot(read(verify_file_pin(reg['source_receipt'])),live=True)
    # Exercise the actual entry point. Any native scope/runtime construction is
    # forbidden here; missing future final training must fail before either.
    cp=ROOT/'configs/pen_backend_response_train_v1.toml';future_training=Path(reg['future_training_report']);future_evaluation=Path(reg['future_evaluation_report'])
    outputs=[Path(p) for p in reg['outputs'].values()]
    checks['prerequisites_actually_pending']=not future_training.exists() and not future_evaluation.exists()
    if not checks['prerequisites_actually_pending'] or any(p.exists() for p in outputs):raise ValueError('This review is only for the pre-collection pending state')
    rejected=None
    with patch.object(collector,'backend_scope',side_effect=AssertionError('No backend scope before prerequisites')) as scope,patch.object(collector,'BackendSupportPenRuntime',side_effect=AssertionError('No runtime before prerequisites')) as runtime:
        try:collector.main(SimpleNamespace(config=cp,mode='collect'))
        except FileNotFoundError as error:rejected=dict(type=type(error).__name__,message=str(error),filename=str(error.filename))
        checks['future_training_denied']=rejected is not None and Path(rejected['filename']).resolve()==future_training.resolve()
        checks['zero_scope_constructions']=scope.call_count==0;checks['zero_runtime_constructions']=runtime.call_count==0
    checks['production_outputs_absent']=not any(p.exists() for p in outputs)
    testlog=ROOT/'runs/pen_backend_response_contract_tests474_v1.log';log=testlog.read_text();checks['tests474']='Ran 474 tests in 10.147s' in log and '\nOK\n' in log
    training_source=ROOT/'runs/pen_backend_long_training_v1/source_receipt.json';verify_source_snapshot(read(training_source),live=True);checks['running_training_sources_unchanged']=True
    command=f"Get-CimInstance Win32_Process -Filter 'ProcessId={a.pid}' | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    process=json.loads(raw) if raw.strip() else None
    checks['training_process_observed']=bool(process and process['ProcessId']==a.pid and '--mode train' in process['CommandLine'] and 'pen_backend_long_training_v1' in process['CommandLine'])
    health_paths=sorted((ROOT/'runs/pen_backend_long_training_v1/RESIDUAL').glob('health_*.json'));health_rows=[read(p) for p in health_paths]
    checks['complete_prefix']=bool(health_rows) and [r['iteration'] for r in health_rows]==list(range(1,len(health_rows)+1))
    checks['all_prefix_health25']=all((h:=read(verify_file_pin(row['collector'])))['passed']==h['total']==25 and all(h['checks'].values()) for row in health_rows)
    if not all(checks.values()):raise ValueError('Design/readiness review differs: '+repr(checks))
    docs=[ROOT/'docs/pen-backend-response-design-results-2026-11-16.md',ROOT/'docs/research-plan-v1.27.md']
    analysis.mkdir();receipt=freeze_sources([Path(__file__)]+docs,analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(analysis/'source_receipt.json',receipt)
    out.mkdir();write(out/'summary.json',dict(schema='backend_response_design_review:1',observed_at_UTC=datetime.now(timezone.utc).isoformat(),registration=file_pin(rp),
        checks=checks,passed=sum(checks.values()),total=len(checks),prerequisite_denial=rejected,test_log=file_pin(testlog),
        process_observation=process,completed_training_prefix=len(health_rows),last_health=file_pin(health_paths[-1]),prefix_health=[file_pin(p) for p in health_paths],
        added_actual_native_integrations=0,added_optimizer_steps=0,new_response_rows=0,collector_physical_execution_qualified=False,
        scope='CPU design/precondition/hidden-identity contracts only. Planned new-backend native probe/decoder/WM/gradient admission remains unproven. Process is a timestamped observation.'))
    write(out/'ledger.json',dict(schema='backend_response_design_ledger:1',prior_evidence=file_pin(prior),registration=file_pin(rp),summary=file_pin(out/'summary.json'),
        source_receipts=[reg['source_receipt'],file_pin(analysis/'source_receipt.json')],documents=[file_pin(p) for p in docs],running_training_source_receipt=file_pin(training_source),
        added_actual_native_integrations=0,added_optimizer_steps=0,scope='Registration/precondition readiness only,not response collection or task/WM admission.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],iteration=len(health_rows),new_response_rows=0)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);main(p.parse_args())
