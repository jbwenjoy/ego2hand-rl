"""Seal actual R2000/S1 saved-state equality and entire completed R prefix."""
import argparse
import json
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from archive_pen_source_ccd_evidence import pins


def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    out=ROOT/'provenance/pen_backend_arm_transition_v1';analysis=ROOT/'runs/pen_backend_arm_transition_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh transition archive required')
    prior=ROOT/'provenance/mug_rgb_projection_v1/ledger.json';rp=ROOT/'runs/pen_backend_second_arm_review_v1/report.json';wp=ROOT/'runs/pen_backend_long_W1_support_first_admission_v1/report.json'
    r,w=read(rp),read(wp);checks=dict(saved_review29=r['passed']==r['total']==29 and all(r['checks'].values()) and not r['full_initial_root_differences'],
        W1_admission5=w['passed']==w['total']==5 and all(w['checks'].values()),same_first_S=w['audit']['actor_checkpoint']==r['inputs']['SUPPORT_checkpoint'],
        no_added_work=(r['added_actual_native_integrations'],r['added_optimizer_steps'],w['actual_native_integrations'],w['optimizer_steps'])==(0,0,0,0),
        prefix_costs=(r['training_prefix_native'],r['training_prefix_Adam'],r['shared_preparation_native'],r['cohort_prefix_native'])==(32784384,64032,2048,32786432))
    raw=ROOT/'runs/pen_backend_long_training_v1';health_paths=[raw/'RESIDUAL'/f'health_{i:04d}.json' for i in range(1,2001)]
    rows=[read(p) for p in health_paths];collectors=[read(verify_file_pin(row['collector'])) for row in rows]
    checks['all2000_R_identities']=[h['iteration'] for h in rows]==list(range(1,2001)) and all(h['arm']=='RESIDUAL' for h in rows)
    checks['all2000_R_exact_counts']=all((h['actual_native_integrations'],h['optimizer_steps'],h['training_transitions'])==(h['iteration']*16384,h['iteration']*32,h['iteration']*2048) for h in rows)
    checks['all2000_R_health25']=all(h['passed']==h['total']==25 and all(h['checks'].values()) for h in collectors)
    due=[h for h in rows if h['checkpoint'] is not None];checks['all_due_R_saves']=[h['iteration'] for h in due]==[1]+list(range(100,2001,100))
    for h in due:verify_file_pin(h['checkpoint'])
    training_source=raw/'source_receipt.json';response_registration=ROOT/'runs/pen_backend_response_registration_v1/registration.json'
    receipts=[r['source_receipt'],w['source_receipt'],file_pin(training_source),read(response_registration)['source_receipt']]
    for pin in receipts:verify_source_snapshot(read(verify_file_pin(pin)),live=True)
    checks['source_receipts_live']=True
    command=f"Get-CimInstance Win32_Process -Filter 'ProcessId={a.pid}' | Select-Object ProcessId,CreationDate,CommandLine | ConvertTo-Json -Compress"
    text=subprocess.check_output(['powershell','-NoProfile','-Command',command],text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));process=json.loads(text) if text.strip() else None
    checks['training_process_observed']=bool(process and process['ProcessId']==a.pid and '--mode train' in process['CommandLine'] and 'pen_backend_long_training_v1' in process['CommandLine'])
    sp=sorted((raw/'SUPPORT').glob('health_*.json'))[-1];support=read(sp)
    checks['observed_S_counts']=(support['actual_native_integrations'],support['optimizer_steps'],support['training_transitions'])==(support['iteration']*16384,support['iteration']*32,support['iteration']*2048)
    sh=read(verify_file_pin(support['collector']));checks['observed_S_health25']=sh['passed']==sh['total']==25 and all(sh['checks'].values())
    if not all(checks.values()):raise ValueError('Independent arm transition review failed: '+repr(checks))
    docs=[ROOT/'docs'/n for n in ['pen-backend-arm-transition-results-2026-11-16.md','research-plan-v1.30.md']]
    scripts=[Path(__file__),ROOT/'tools/review_backend_arm_transition.py'];analysis.mkdir();receipt=freeze_sources(docs+scripts,analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(analysis/'source_receipt.json',receipt)
    todo=[file_pin(p) for p in [rp,wp,prior,sp,response_registration,analysis/'source_receipt.json']+docs+scripts+health_paths]+receipts
    allowed={rp.parent,wp.parent,raw,analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        path=verify_file_pin(pin);verified[key]=pin
        if path.suffix=='.json' and any(path.is_relative_to(p) for p in allowed):todo.extend(pins(read(path)))
    out.mkdir();write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'summary.json',dict(checks=checks,passed=sum(checks.values()),total=len(checks),saved_review=file_pin(rp),W1_admission=file_pin(wp),
        complete_R_rounds=2000,R_due_saves=21,R_collector_flags_checked=50000,training_prefix_native=32784384,training_prefix_Adam=64032,shared_preparation_native=2048,cohort_prefix_native=32786432,
        observed_at_UTC=datetime.now(timezone.utc).isoformat(),process_observation=process,observed_SUPPORT_iteration=support['iteration'],observed_SUPPORT_health=file_pin(sp),
        observed_remaining_loop_s=support['estimated_total_remaining_loop_s'],ETA_scope=support['ETA_scope'],added_actual_native_integrations=0,added_optimizer_steps=0,
        scope='Completed R/fresh-common S1 state/counter evidence and timestamped live S observation,not complete matched trial or strict task/WM/ego/speed result.'))
    write(out/'ledger.json',dict(schema='new_backend_arm_transition_ledger:1',prior_evidence=file_pin(prior),summary=file_pin(out/'summary.json'),roots=dict(saved_review=file_pin(rp),W1_admission=file_pin(wp)),
        source_receipts=receipts+[file_pin(analysis/'source_receipt.json')],documents=[file_pin(p) for p in docs],verified_manifest=file_pin(analysis/'verified_pins.json'),verified_pins=len(verified),
        added_actual_native_integrations=0,added_optimizer_steps=0))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],verified_pins=len(verified),observed_S=support['iteration'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);main(p.parse_args())
