"""Freeze registered centered-score trial and every failed numerical repeat."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_centered_support_results';analysis=ROOT/'runs/pen_centered_support_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable centered-trial archive')
    prior_path=ROOT/'provenance/pen_support_pilot_results_v2/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in old:verify_source_pin(entry['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(score_study='pen_support_score_study_v1',qualification='pen_centered_support_qualification_v1',training='pen_centered_support_training_v1',
        state_review='pen_centered_support_state_review_v1',registration='pen_centered_support_evaluation_registration_v1',evaluation='pen_centered_support_evaluation_v1',
        evaluation_review='pen_centered_support_evaluation_review_v1',common_windows='pen_centered_support_common_windows_v1')
    reports={k:ROOT/'runs'/folder/('registration.json' if k=='registration' else 'report.json') for k,folder in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()}
    expected=dict(score_study=[14,14],qualification=[75,75],training=[5,5],state_review=[216,216],evaluation_review=[351,351],common_windows=[15,15])
    for key,value in expected.items():
        if [data[key]['passed'],data[key]['total']]!=value:raise ValueError('Observed checks differ '+key)
    if len(data['evaluation']['cases'])!=9 or any(c['summary']['qualified'] for c in data['evaluation']['cases']):raise ValueError('Recorded failed evaluation matrix changed')
    integrations=sum(v.get('actual_native_integrations',0) for v in data.values());candidate=data['training']['optimizer_steps'];diagnostic=data['state_review']['diagnostic_optimizer_steps']
    if (integrations,candidate,diagnostic)!=(2108424,4096,64):raise ValueError('Observed total cost differs')
    log=ROOT/'runs/pen_centered_support_tests334_v1.log';text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 334 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Full334 suite log required')
    docs=[ROOT/'docs'/n for n in ['support-score-study-protocol-v1.md','pen-centered-support-training-protocol-v1.md','pen-centered-support-evaluation-protocol-v1.md','centered-support-training-results-2026-11-16.md','research-plan-v1.05.md']]
    configs=[ROOT/'configs'/n for n in ['pen_support_score_study_v1.toml','pen_centered_support_training_v1.toml','pen_centered_support_evaluation_v1.toml']]
    scripts=[ROOT/'tools'/n for n in ['compare_pen_support_scores.py','qualify_pen_support_pilot.py','train_pen_support_pilot.py','evaluate_pen_support_repeats.py',
        'preview_pen_support_pilot.py','review_pen_support_pilot.py','review_pen_support_preview.py','compare_pen_support_executions.py','archive_pen_centered_support.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for entry in source['records']:aliases.setdefault(json.dumps(entry['observed'],sort_keys=True),dict(original=entry['observed'],archived=entry['snapshot'],reason='Exact original full path/size/SHA source version, including absolute/centered schemas and pre-execution registration.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for entry in resolutions['records']:verify_source_pin(entry['original'],resolutions)
    engines=[];states=[]
    for arm in data['training']['arms']:
        state=torch.load(verify_file_pin(arm['checkpoint']),map_location='cpu',weights_only=False);verify_source_snapshot(state['native']['sources'])
        for pin in state['native']['engine_sources']:verify_file_pin(pin)
        engines+=state['native']['engine_sources'];states.append(dict(arm=arm['arm'],checkpoint=arm['checkpoint'],schema=state['schema'],native_schema=state['native']['schema'],
            iteration=state['iteration'],optimizer_steps=state['optimizer_steps'],training_transitions=state['training_transitions'],normalizer_count=int(state['modules']['actor']['norm.count']),role=state['metadata']['role']))
    # Explicit TERMINAL producer logs only. Never hash this archive's open stdout.
    logs=[log]+[ROOT/'runs'/f'pen_centered_support_{name}_console_v1.log' for name in ['qualification','training','state_review','evaluation','evaluation_review']]
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]]+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='centered_support_trial_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,selected_score=data['score_study']['selected_variant'],
        state_audits=states,execution_aggregate=data['evaluation']['aggregate'],common_horizon_descriptive=data['common_windows']['aggregate'],
        actual_native_integrations=integrations,retained_trial_optimizer_steps=candidate,diagnostic_optimizer_steps=diagnostic,total_actual_optimizer_steps=candidate+diagnostic,unit_tests=334,
        scope='Registered equal-budget64-iteration centered-score single-seed/clip trial,9 fresh numerical executions,216-state and351-raw readbacks. All original full tasks fail; mixed COM/orientation evidence, not statistical support benefit,ego/physical-WM-gradient/speedup qualification.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name);write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='centered_support_trial_evidence_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),
        prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),
        test_log=file_pin(out/log.name),actual_native_integrations=integrations,retained_trial_optimizer_steps=candidate,diagnostic_optimizer_steps=diagnostic,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),source_aliases=len(aliases),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
