"""Archive independent TRAIN coverage without changing historical CHECK data."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_train_coverage_results';analysis=ROOT/'runs/pen_train_coverage_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence destination')
    prior_path=ROOT/'provenance/source_epoch_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in old:verify_source_pin(entry['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    reports=dict(collection=ROOT/'runs/pen_response_train_coverage_v1/report.json',preparation=ROOT/'runs/pen_response_train_preparation_v1/report.json',diagnosis_initial=ROOT/'runs/pen_response_train_momentum_diagnosis_v1/report.json',diagnosis=ROOT/'runs/pen_response_train_momentum_diagnosis_v2/report.json')
    data={key:json.loads(p.read_text()) for key,p in reports.items()};run=data['collection'];prep=data['preparation']
    if any(data[k]['passed']!=data[k]['total'] for k in ['collection','preparation','diagnosis']) or [data['diagnosis_initial']['passed'],data['diagnosis_initial']['total']]!=[9,825] or len(run['sources'])!=24 or len(run['cases'])!=816 or run['actual_native_integrations']!=8800 or prep['total_rows']!=816:raise ValueError('Observed collection/contract differs;preserve evidence')
    if any(d['optimizer_steps'] for d in data.values()) or any(d['CHECK_response_rows_used'] for d in [run,prep]) or run['restore_mode']!='STRICT' or prep['scales_fitted']:raise ValueError('Unregistered optimization or data mixing')
    log=ROOT/'runs/pen_train_coverage_tests352_v1.log';test_text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 352 tests in ' not in test_text or '\nOK\n' not in test_text:raise ValueError('Completed full352-test evidence required')
    docs=[ROOT/'docs'/n for n in ['pen-response-train-coverage-protocol-v1.md','pen-response-train-coverage-results-2026-11-16.md','research-plan-v1.08.md']]
    config=ROOT/'configs/pen_response_train_coverage_v1.toml';scripts=[ROOT/'tools'/n for n in ['collect_trained_pen_responses.py','prepare_pen_train_coverage.py','diagnose_pen_train_momentum.py','archive_pen_train_coverage.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+[config]+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[verify_file_pin(d['source_receipt']) for d in data.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for entry in source['records']:aliases.setdefault(json.dumps(entry['observed'],sort_keys=True),dict(original=entry['observed'],archived=entry['snapshot'],reason='Exact original full path/bytes/SHA;preserve older CHECK and epoch sources across maintained changes.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for entry in resolutions['records']:verify_source_pin(entry['original'],resolutions)
    engines=[]
    for source in run['sources']:
        if not source['available']:continue
        state=torch.load(verify_file_pin(source['source']),map_location='cpu',weights_only=False);verify_source_snapshot(state['native']['sources'],live=True)
        for pin in state['native']['engine_sources']:verify_file_pin(pin)
        engines+=state['native']['engine_sources']
    logs=[log,ROOT/'runs/pen_response_train_coverage_console_v1.log',ROOT/'runs/pen_response_train_preparation_console_v1.log',ROOT/'runs/pen_response_train_momentum_diagnosis_console_v1.log',ROOT/'runs/pen_response_train_momentum_diagnosis_console_v2.log']
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+[config]+logs+[prior_path]]+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='pen_TRAIN_response_coverage_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks={k:[d['passed'],d['total']] for k,d in data.items()},
        source_states=24,episodes=4,branches=816,qualified_sources=run['qualified_sources'],qualified_rows=prep['qualified_rows'],rejected_rows=prep['rejected_rows'],
        physical_qualification=prep['physical_qualification'],row_gates=[prep['row_gate_passed'],prep['row_gate_total']],qualified_response_groups=prep['qualified_response_groups'],
        actual_native_integrations=8800,optimizer_steps=0,CHECK_response_rows_used=0,restore_mode='STRICT',unit_tests=352,
        prior_epoch_physical_qualification=prior['physical_qualification'],strict_impulse_failure_rows=data['diagnosis']['all_impulse_failure_rows'],recorded_repeat_conditions=data['diagnosis']['repeated_conditions'],
        max_velocity_update_residual_m_s=data['diagnosis']['max_velocity_update_residual_m_s'],
        scope='Independent-noise-history TRAIN physical response coverage within one original reference. Existing developmental CHECK NEVER_FIT. No new WM fit,Actor update,task/video improvement,unseen-reference/ego or PPO-speed claim;prior epoch impulse anomaly unresolved.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='pen_TRAIN_response_coverage_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),
        prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),
        test_log=file_pin(out/log.name),actual_native_integrations=8800,optimizer_steps=0,physical_qualification=prep['physical_qualification'],scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
