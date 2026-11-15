"""Freeze legal trained-actor probes,failed WM gradients and restore profile."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/trained_pen_response_results';analysis=ROOT/'runs/pen_trained_response_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable response evidence')
    prior_path=ROOT/'provenance/pen_centered_support_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text());old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in old:verify_source_pin(entry['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(probe='pen_trained_response_probe_v1',initial_review='pen_trained_response_review_v1',review='pen_trained_response_review_v2',screens='pen_trained_response_screen_summary_v1',profile='pen_trained_source_restore_profile_v1')
    reports={k:ROOT/'runs'/v/'report.json' for k,v in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()}
    expected=dict(probe=[5,5],initial_review=[6522,6522],review=[6525,6525],screens=[586,586],profile=[50,50])
    for key,value in expected.items():
        if [data[key]['passed'],data[key]['total']]!=value:raise ValueError('Observed checks differ '+key)
    if sum(v['actual_native_integrations'] for v in data.values())!=3776 or any(v['optimizer_steps'] for v in data.values()):raise ValueError('Actual no-update cohort cost differs')
    if data['review']['qualified_rows']!=324 or data['probe']['available_cases']!=324 or data['probe']['qualified_sources']!=12:raise ValueError('Declared legal physical cohort differs')
    log=ROOT/'runs/pen_trained_probe_tests337_v1.log';text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 337 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Full337 unit suite required')
    docs=[ROOT/'docs'/n for n in ['pen-trained-response-protocol-v1.md','pen-trained-response-review-protocol-v1.md','pen-trained-response-review-protocol-v2.md','trained-actor-physical-response-results-2026-11-16.md','research-plan-v1.06.md']]
    configs=[ROOT/'configs'/n for n in ['pen_trained_response_probe_v1.toml','pen_trained_response_review_v1.toml','pen_trained_response_review_v2.toml']]
    scripts=[ROOT/'tools'/n for n in ['collect_trained_pen_responses.py','review_trained_pen_responses.py','summarize_trained_pen_response_screens.py','profile_owned_source_restore.py','archive_trained_pen_responses.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for entry in source['records']:aliases.setdefault(json.dumps(entry['observed'],sort_keys=True),dict(original=entry['observed'],archived=entry['snapshot'],reason='Full original path/bytes/SHA,including initial2-model and explicit3-model review versions; no implicit source migration.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for entry in resolutions['records']:verify_source_pin(entry['original'],resolutions)
    engines=[];source_audits=[]
    for entry in data['probe']['sources']:
        source=torch.load(verify_file_pin(entry['source']),map_location='cpu',weights_only=False);verify_source_snapshot(source['native']['sources'])
        for pin in source['native']['engine_sources']:verify_file_pin(pin)
        engines+=source['native']['engine_sources'];source_audits.append(dict(name=entry['name'],source=entry['source'],native_schema=source['native']['schema'],phase=source['phase'],role=data['probe']['role'],
            array_fields=len(source['native']['native']['model']['arrays'])+len(source['native']['native']['data']['arrays'])+len(source['native']['python']['tensors'])+len(source['native']['python']['arrays'])))
    logs=[log,ROOT/'runs/pen_trained_response_probe_console_v1.log',ROOT/'runs/pen_trained_response_review_console_v1.log',ROOT/'runs/pen_trained_response_review_console_v2.log',ROOT/'runs/pen_trained_source_restore_profile_console_v1.log']
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]]+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    frames=data['review']['frame_audits'];summary=dict(schema='trained_pen_response_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,source_audits=source_audits,
        qualified_sources=12,qualified_rows=324,response_pairs=48,model_response_aggregate=data['screens']['aggregate'],frozen_forward=data['review']['frozen_model_forward'],
        frame_rotation_gap_rad=[min(v['rotation_gap_rad'] for v in frames),max(v['rotation_gap_rad'] for v in frames)],maximum_COM_gap_m=max(v['COM_gap_m'] for v in frames),
        restore_profile_measured_wall_s=data['profile']['total_measured_wall_s'],restore_profile_top=data['profile']['top_cumulative'][:12],
        actual_native_integrations=3776,optimizer_steps=0,unit_tests=337,role=data['probe']['role'],
        scope='Legal contact action responses above observed repeat noise,independent full-plant/frame/raw checks; old DIRECT/CV/IMPULSE object JVPs remain worse than zero on stable finite-response subset. No fitting,Actor update,full Jacobian,task/ego or performance speedup qualification.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='trained_pen_response_evidence_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),
        prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),
        test_log=file_pin(out/log.name),actual_native_integrations=3776,optimizer_steps=0,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
