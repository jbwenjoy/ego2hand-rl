"""Seal integrated backend checks and ordinary state reloads with actual costs."""
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_pin,verify_source_snapshot
from supermimic_research.learning.backend_pen_ppo import require_saved_backend


def read(p):return json.loads(Path(p).read_text())


def main():
    out=ROOT/'provenance/pen_backend_integration_v1';analysis=ROOT/'runs/pen_backend_integration_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh integration archive required')
    pp=ROOT/'provenance/pen_ccd_recovery_candidate_v1/ledger.json';prior=read(pp);aliases=read(verify_file_pin(prior['source_resolutions']))
    for row in read(verify_file_pin(prior['verified_manifest']))['records']:
        if file_pin(verify_source_pin(row['observed'],aliases))!=row['resolved']:raise ValueError('Prior source resolution changed')
    names=dict(integration='pen_backend_integration_v1/report.json',registration='pen_backend_integration_registration_v1/registration.json')
    roots={k:file_pin(ROOT/'runs'/v) for k,v in names.items()};report=read(verify_file_pin(roots['integration']));run=Path(roots['integration']['path']).parent
    checks=dict(root_all6=report['passed']==report['total']==6 and all(report['checks'].values()),
        native_budget=report['actual_native_integrations']==report['attempted_native_integrations']==42240,
        optimizer_budget=report['optimizer_steps']==report['attempted_optimizer_steps']==128,
        case_order=[c['label'] for c in report['cases']]==['W1_RESIDUAL','W1_SUPPORT','W32_RESIDUAL','W32_SUPPORT'])
    local=[]
    for case in report['cases']:
        health=read(verify_file_pin(case['checks']));audit=read(verify_file_pin(case['reload']));state=torch.load(verify_file_pin(case['checkpoint']),map_location='cpu',weights_only=False)
        require_saved_backend(state,state['metadata']['collision_backend'])
        W=state['native']['worlds'];expected=109 if case['label']=='W1_RESIDUAL' else 105
        checks[case['label']+'_all_checks']=health['passed']==health['total']==expected and all(health['checks'].values())
        checks[case['label']+'_actual_budget']=case['native_integrations']==640*W and case['optimizer_steps']==32
        checks[case['label']+'_ordinary_load']=all(v for v in audit.values() if type(v)is bool) and all(audit['native'][k] for k in ['all_array_bytes_equal','all_scalar_values_equal','pointers_preserved'])
        checks[case['label']+'_trained_state']=(state['iteration'],state['optimizer_steps'],state['training_transitions'],state['calibration_transitions'])==(1,32,68*W,8*W) and bool(state['optimizer']['state'])
        raw=torch.load(verify_file_pin(case['raw_oracle']),map_location='cpu',weights_only=False)
        checks[case['label']+'_raw_coverage']=len(raw['packets'])==32 and len(raw['starts'])==4 and all(all(p['contact']['flags'].values()) and p['no_applied_force'] for p in raw['packets'])
        local.append(dict(label=case['label'],maximum=health['maximum'],native_array_fields=audit['native']['array_fields'],native_scalar_fields=audit['native']['scalar_fields']))
    rejection=read(run/'W1_RESIDUAL/rejections.json');checks['four_rejections']=len(rejection)==4 and all(v.get('no_partial_write',v.get('no_native',False)) for v in rejection)
    backend=read(verify_file_pin(report['backend_audit']));checks['actual_graph_source_observed']=backend['entry_calls']==32 and backend['kernel_factory_calls']==160 and backend['live_runtimes']==0
    log=ROOT/'runs/pen_backend_contract_tests461_v1.log';checks['tests461']='Ran 461 tests in 9.641s' in log.read_text() and '\nOK\n' in log.read_text()
    if not all(checks.values()):raise ValueError('Independent integration review differs: '+repr(checks))
    docs=[ROOT/'docs/pen-backend-integration-results-2026-11-16.md'];scripts=[Path(__file__),ROOT/'tools/qualify_pen_backend_integration.py',ROOT/'tools/register_face_normal_backend.py']
    analysis.mkdir();receipt=freeze_sources(docs+scripts,analysis/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip());write(analysis/'source_receipt.json',receipt)
    receipts=[run/'source_receipt.json',Path(roots['registration']['path']).parent/'source_receipt.json',analysis/'source_receipt.json']
    merged={json.dumps(v['original'],sort_keys=True):v for v in aliases['records']}
    for p in receipts:
        source=read(p);verify_source_snapshot(source,live=True)
        for v in source['records']:merged.setdefault(json.dumps(v['observed'],sort_keys=True),dict(original=v['observed'],archived=v['snapshot'],reason='Exact integrated backend full-path/bytes/SHA source.'))
    aliases=dict(schema='explicit_source_relocations:1',records=list(merged.values()))
    todo=list(roots.values())+[file_pin(p) for p in receipts+docs+scripts+[pp,log,ROOT/'runs/pen_backend_integration_console_v1.log']]
    todo += [file_pin(p) for p in run.rglob('*.json') if 'source_snapshot' not in p.parts]
    allowed={run,Path(roots['registration']['path']).parent,analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,aliases);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(p) for p in allowed):todo.extend(pins(read(resolved)))
    out.mkdir();write(out/'source_resolutions.json',aliases);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'summary.json',dict(roots=roots,checks=checks,passed=sum(checks.values()),total=len(checks),cases=local,
        added_actual_work=dict(native_integrations=42240,optimizer_steps=128),old_native_migrated=False,
        admitted_scope='Fresh controlled research runtime/state integration only;not final task,WM gradient,ego,hardware or performance admission.',
        successful_run=True,current_live_training=False,failure_handler_scope='Successful CLI path verified;long runner must capture failure and close runtime inside active backend scope.'))
    write(out/'ledger.json',dict(schema='explicit_backend_integration_ledger:1',prior_evidence=file_pin(pp),roots=roots,summary=file_pin(out/'summary.json'),documents=[file_pin(p) for p in docs],
        source_receipts=[file_pin(p) for p in receipts],source_resolutions=file_pin(out/'source_resolutions.json'),verified_manifest=file_pin(analysis/'verified_pins.json'),verified_pins=len(verified),
        actual_native_integrations=42240,optimizer_steps=128,current_live_training=False))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),checks=[sum(checks.values()),len(checks)],verified_pins=len(verified))),flush=True)


if __name__=='__main__':main()
