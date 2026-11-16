"""Seal a bounded launch observation; never infer live work from files alone."""
import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_pin,verify_source_snapshot


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    out=ROOT/'provenance/pen_backend_launch_v1';analysis=ROOT/'runs/pen_backend_launch_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh launch archive required')
    prior_path=ROOT/'provenance/pen_backend_integration_v1/ledger.json';prior=read(prior_path)
    aliases=read(verify_file_pin(prior['source_resolutions']));records=read(verify_file_pin(prior['verified_manifest']))['records']
    for row in records:
        if file_pin(verify_source_pin(row['observed'],aliases))!=row['resolved']:raise ValueError('Prior resolution changed')
    for pin in prior['source_receipts']:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    qs=read(verify_file_pin(prior['summary']));checks=dict(integration_independent=qs['passed']==qs['total']==27,prior_pins=len(records)==994)
    roots={k:ROOT/'runs'/p for k,p in dict(registration='pen_backend_long_registration_v1/registration.json',
        preparation='pen_backend_long_preparation_v1/report.json',initial_W1='pen_backend_long_W1_initial_admission_v1/report.json',
        first_W1='pen_backend_long_W1_first_admission_v1/report.json',first_health='pen_backend_long_training_v1/RESIDUAL/health_0001.json',
        initialization='pen_backend_long_training_v1/RESIDUAL/initialization.json').items()}
    data={k:read(p) for k,p in roots.items()};reg=data['registration'];backend=reg['backend_id']
    for key in ['registration','preparation','initial_W1','first_W1']:
        item=data[key];verify_source_snapshot(read(verify_file_pin(item['source_receipt'])),live=True)
        checks[key+'_backend']=item['backend_id']==backend
    for pin in reg['inputs'].values():verify_file_pin(pin)
    for key in ['preparation','initial_W1','first_W1']:
        item=data[key];checks[key+'_checks']=item['passed']==item['total']==5 and all(item['checks'].values())
        checks[key+'_registration']=item['registration']==file_pin(roots['registration'])
    checks['preparation_budget']=(data['preparation']['actual_native_integrations'],data['preparation']['optimizer_steps'])==(2048,0)
    for key in ['initial_W1','first_W1']:
        item=data[key];audit=item['audit']
        checks[key+'_unstepped']=(item['actual_native_integrations'],item['optimizer_steps'])==(0,0)
        checks[key+'_model_and_actor']=audit['full_W1_model_arrays']==404 and audit['full_W1_model_scalars']==195 and audit['full_W1_model_bytes_equal'] and audit['actor_normalizer_bytes_equal'] and not audit['old_native_backend_imported']
    checks['first_checkpoint_same']=data['first_W1']['audit']['actor_checkpoint']==data['first_health']['checkpoint']
    verify_file_pin(data['first_health']['checkpoint']);checks['first_readback']=all(data['first_health']['full_first_readback'].values())
    init=data['initialization'];checks['fresh_initial_load']=init['patches']==[] and init['modules_identical'] and init['observations_identical'] and init['load']['native']['all_array_bytes_equal'] and init['load']['native']['all_scalar_values_equal']
    # Capture a bounded completed prefix. A newer file appearing is deliberately ignored.
    paths=sorted((ROOT/'runs/pen_backend_long_training_v1/RESIDUAL').glob('health_*.json'))
    rows=[read(p) for p in paths]
    checks['contiguous_prefix']=[r['iteration'] for r in rows]==list(range(1,len(rows)+1))
    checks['exact_completed_counts']=all((r['actual_native_integrations'],r['optimizer_steps'],r['training_transitions'])==(r['iteration']*16384,r['iteration']*32,r['iteration']*2048) for r in rows)
    collectors=[read(verify_file_pin(r['collector'])) for r in rows]
    checks['all_completed_health']=all(h['passed']==h['total']==25 and all(h['checks'].values()) for h in collectors)
    testlog=ROOT/'runs/pen_backend_study_tests464_v1.log';text=testlog.read_text();checks['tests464']='Ran 464 tests in 9.653s' in text and '\nOK\n' in text
    command=f"Get-CimInstance Win32_Process -Filter 'ProcessId={a.pid}' | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    process=json.loads(raw) if raw.strip() else None
    checks['live_process_observed']=bool(process and process['ProcessId']==a.pid and 'run_pen_backend_control_study.py' in process['CommandLine'] and '--mode train' in process['CommandLine'] and 'pen_backend_long_training_v1' in process['CommandLine'])
    if not all(checks.values()):raise ValueError('Launch observation failed: '+repr(checks))
    now=datetime.now(timezone.utc);median=statistics.median([r['loop_wall_s'] for r in rows[-50:]])
    remaining=(4000-len(rows))*median
    gpu=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)).strip()
    docs=[ROOT/'docs/pen-backend-launch-results-2026-11-16.md',ROOT/'docs/research-plan-v1.26.md']
    analysis.mkdir();receipt=freeze_sources([Path(__file__)]+docs,analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(analysis/'source_receipt.json',receipt)
    out.mkdir();write(out/'summary.json',dict(schema='bounded_backend_launch_observation:1',observed_at_UTC=now.isoformat(),
        checks=checks,passed=sum(checks.values()),total=len(checks),process=process,last_completed_iteration=len(rows),
        completed_training_native=rows[-1]['actual_native_integrations'],completed_training_Adam=rows[-1]['optimizer_steps'],
        preparation_native=2048,additional_W1_admission_native=0,additional_W1_admission_Adam=0,
        median_last_up_to50_loop_s=median,estimated_remaining_loop_s=remaining,estimated_loop_end_UTC=(now+timedelta(seconds=remaining)).isoformat(),
        ETA_scope='Bounded completed R prefix projected to remaining R/S loops;excludes next-arm construction/final evaluation/render;future contacts may change cost. W1 admission overlapped briefly;not a throughput comparison.',
        GPU_devicewide_instantaneous_csv=gpu,completed_health=[file_pin(p) for p in paths],roots={k:file_pin(p) for k,p in roots.items()},test_log=file_pin(testlog),
        admitted_scope='Launch/readback/unstepped W1 contracts only. SUPPORT not started,complete trial and strict physical policy performance unproven. Process is a timestamped observation,not a future liveness guarantee.'))
    write(out/'ledger.json',dict(schema='fresh_backend_launch_ledger:1',prior_evidence=file_pin(prior_path),roots={k:file_pin(p) for k,p in roots.items()},
        summary=file_pin(out/'summary.json'),source_receipt=file_pin(analysis/'source_receipt.json'),documents=[file_pin(p) for p in docs],
        training_source_receipt=file_pin(ROOT/'runs/pen_backend_long_training_v1/source_receipt.json'),
        scope='Timestamped prefix,not terminal training accounting;read actual process/report on continuation.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],iteration=len(rows),remaining_loop_s=remaining,GPU=gpu)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);main(p.parse_args())
