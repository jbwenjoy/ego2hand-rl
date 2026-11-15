"""Retain faster source epochs AND the failed initial physical gate."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/source_epoch_results';analysis=ROOT/'runs/source_epoch_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable epoch evidence')
    prior_path=ROOT/'provenance/trained_pen_response_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for e in manifest['records']:
            key=json.dumps(e['observed'],sort_keys=True)
            if key not in old:verify_source_pin(e['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(benchmark='source_epoch_benchmark_v1',physical_review='source_epoch_benchmark_review_v1',amortized='source_epoch_amortized_v1',replay='source_epoch_impulse_replay_v1',outcomes='source_epoch_outcome_analysis_v1')
    reports={k:ROOT/'runs'/v/'report.json' for k,v in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()}
    expected=dict(benchmark=[80,80],physical_review=[6157,6159],amortized=[76,76],replay=[93,93],outcomes=[257,257])
    for key,value in expected.items():
        if [data[key]['passed'],data[key]['total']]!=value:raise ValueError('Observed checks differ '+key)
    if sum(v['actual_native_integrations'] for v in data.values())!=4256 or any(v['optimizer_steps'] for v in data.values()):raise ValueError('Cohort cost differs')
    if data['physical_review']['envelope_passed']!=270 or data['outcomes']['physical_qualification']!='INCOMPLETE_ORIGINAL_IMPULSE_FAILURE_RETAINED':raise ValueError('Failed physical status was changed')
    log=ROOT/'runs/source_epoch_tests345_v1.log';text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 345 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Full345-test evidence required')
    docs=[ROOT/'docs'/n for n in ['source-epoch-benchmark-protocol-v1.md','source-epoch-amortized-protocol-v1.md','source-epoch-impulse-replay-protocol-v1.md','source-validation-epoch-results-2026-11-16.md','research-plan-v1.07.md']]
    configs=[ROOT/'configs'/n for n in ['source_epoch_benchmark_v1.toml','source_epoch_benchmark_review_v1.toml']]
    scripts=[ROOT/'tools'/n for n in ['benchmark_source_epoch.py','review_source_epoch_benchmark.py','benchmark_source_epoch_amortized.py','replay_source_epoch_impulse_case.py','analyze_source_epoch_outcomes.py','archive_source_epoch_results.py','archive_pen_source_ccd_evidence.py','review_owned_pen_state.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for e in source['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Exact original path/bytes/SHA,including experimental source revisions and the executed no-op source-list typo before cleanup.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    engines=[]
    for e in data['benchmark']['sources']:
        state=torch.load(verify_file_pin(e['source']),map_location='cpu',weights_only=False);verify_source_snapshot(state['native']['sources'])
        for pin in state['native']['engine_sources']:verify_file_pin(pin)
        engines+=state['native']['engine_sources']
    logs=[log]+[ROOT/'runs'/n for n in ['source_epoch_benchmark_console_v1.log','source_epoch_benchmark_review_console_v1.log','source_epoch_amortized_console_v1.log','source_epoch_impulse_replay_console_v1.log']]
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]]+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='source_epoch_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,
        whole_batch=data['amortized']['total_s'],whole_batch_ratio=data['amortized']['pooled_ratio'],per_call_only_ratio=data['benchmark']['restore_only_ratio'],
        physical_status=data['outcomes']['physical_qualification'],original_anomaly=data['outcomes']['original_anomaly'],replay_summaries=data['replay']['summaries'],
        observed_endpoint_envelopes=[270,270],qualified_finite_response_pairs=[23,24],W1_W32_contexts=data['outcomes']['W1_W32_contexts'],
        actual_native_integrations=4256,optimizer_steps=0,unit_tests=345,default_loading='STRICT',adoption='EXPLICIT_EXPERIMENTAL_OPTION_NO_AUTOMATIC_TRAIN_ENABLEMENT',
        scope='Measured restore-batch speed and exact source/state checks,with one retained original impulse failure not reproduced in90 selected replays. No full physical equivalence,ordinary PPO/end-to-end speedup,WM-gradient/task/ego claim.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='source_epoch_evidence_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),
        prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),
        test_log=file_pin(out/log.name),actual_native_integrations=4256,optimizer_steps=0,physical_qualification=summary['physical_status'],scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
