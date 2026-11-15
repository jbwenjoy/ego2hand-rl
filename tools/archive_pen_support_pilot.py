"""Immutable pilot/failed-evaluation/review evidence with exact historical aliases."""
import json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/pen_support_pilot_results_v2';analysis=ROOT/'runs/pen_support_pilot_archive_v2'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable archive')
    prior_path=ROOT/'provenance/owned_ppo_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for entry in manifest['records']:
            key=json.dumps(entry['observed'],sort_keys=True)
            if key not in old:verify_source_pin(entry['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    names=dict(initial_qualification='pen_support_qualification_v1',qualification='pen_support_qualification_v2',training='pen_support_pilot_v1',
        state_review='pen_support_pilot_state_review_v1',preview_shown='pen_support_pilot_preview_v1',preview_raw='pen_support_pilot_preview_v2',
        independent_preview_review='pen_support_pilot_preview_review_v1',signal_audit='pen_support_signal_audit_v1')
    reports={k:ROOT/'runs'/folder/'report.json' for k,folder in names.items()};data={k:json.loads(p.read_text()) for k,p in reports.items()}
    expected=dict(initial_qualification=[72,72],qualification=[74,74],training=[5,5],state_review=[214,214],independent_preview_review=[114,114],signal_audit=[10,10])
    for key,value in expected.items():
        if [data[key]['passed'],data[key]['total']]!=value:raise ValueError('Recorded checks differ '+key)
    integrations=sum(v.get('actual_native_integrations',0) for v in data.values());pilot_updates=data['training']['optimizer_steps'];diagnostic_updates=data['state_review']['diagnostic_optimizer_steps']
    if integrations!=76320 or pilot_updates!=128 or diagnostic_updates!=64:raise ValueError('Observed total budget differs')
    log=ROOT/'runs/pen_support_tests329_v1.log';text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 329 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Full329 suite evidence required')
    docs=[ROOT/'docs'/n for n in ['geometry-support-pilot-protocol-v1.md','geometry-support-pilot-results-2026-11-16.md','research-plan-v1.04.md']]
    configs=[ROOT/'configs/pen_geometry_support_pilot_v1.toml']
    scripts=[ROOT/'tools'/n for n in ['qualify_pen_support_pilot.py','train_pen_support_pilot.py','preview_pen_support_pilot.py','review_pen_support_pilot.py',
        'review_pen_support_preview.py','audit_pen_support_signal.py','archive_pen_support_pilot.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+scripts+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();analysis.mkdir();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    first_archive_path=ROOT/'provenance/pen_support_pilot_results/ledger.json'
    first_archive=json.loads(first_archive_path.read_text());first_manifest=json.loads(verify_file_pin(first_archive['verified_manifest']).read_text())
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[ROOT/'runs/pen_support_pilot_archive_v1/source_receipt.json',analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for entry in source['records']:
            aliases.setdefault(json.dumps(entry['observed'],sort_keys=True),dict(original=entry['observed'],archived=entry['snapshot'],reason='Exact original full path/bytes/hash including qualifier v1 and displayed preview v1; no basename/SHA-only fallback.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    # v1 accidentally pinned this process's still-open console before its final
    # print. Recover ONLY the exact saved byte prefix by size+SHA, never invent
    # a replacement log or overwrite the completed original/first archive.
    console=ROOT/'runs/pen_support_archive_console_v1.log'
    old_pin=next(e['observed'] for e in first_manifest['records'] if e['observed']['path']==str(console))
    prefix=analysis/'archive_v1_console_prefix.log';prefix.write_bytes(console.read_bytes()[:old_pin['bytes']])
    prefix_pin=file_pin(prefix)
    if any(prefix_pin[k]!=old_pin[k] for k in ['bytes','sha256']):raise ValueError('Original in-progress console prefix cannot be reconstructed exactly')
    resolutions['records'].append(dict(original=old_pin,archived=prefix_pin,reason='v1 in-progress self-console exact byte prefix recovered by its original full size/SHA; terminal console remains separate.'))
    for entry in first_manifest['records']:verify_source_pin(entry['observed'],resolutions)
    for entry in resolutions['records']:verify_source_pin(entry['original'],resolutions)
    engines=[];states=[]
    for arm in data['training']['arms']:
        state=torch.load(verify_file_pin(arm['checkpoint']),map_location='cpu',weights_only=False);verify_source_snapshot(state['native']['sources'])
        for pin in state['native']['engine_sources']:verify_file_pin(pin)
        engines+=state['native']['engine_sources'];states.append(dict(arm=arm['arm'],checkpoint=arm['checkpoint'],schema=state['schema'],native_schema=state['native']['schema'],
            iteration=state['iteration'],optimizer_steps=state['optimizer_steps'],training_transitions=state['training_transitions'],normalizer_count=int(state['modules']['actor']['norm.count']),role=state['metadata']['role']))
    logs=sorted({p for pattern in ['pen_support_unit*.log','pen_support_tests*.log','pen_support*_console_v*.log','pen_support_pilot_console_v*.log'] for p in (ROOT/'runs').glob(pattern) if not p.name.startswith('pen_support_archive_console_')})
    todo=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path,first_archive_path,console,prefix]]+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='geometry_support_pilot_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks=expected,
        initial_check_limit='Qualifier v1 overall no-optimizer check used contact-row count and was uninformative; v2 checks saved counters and empty Adam state directly. Both actually0 optimizer steps.',
        archive_v1_limit='First archive pinned its own incomplete console before final print. v2 excludes live archive consoles, retains v1, and resolves its exact historical prefix by full size/SHA. Independent final revalidation required.',
        state_audits=states,strict_outcomes=[dict(name=c['name'],summary=c['summary'],termination_reasons=c['termination_reasons']) for c in data['preview_raw']['cases']],
        independent_failures=[dict(name=c['name'],first_breaches=c['first_breaches'],common_0_20s=c['common_0_20s']) for c in data['independent_preview_review']['cases']],
        support_zero_wrench_audit=data['signal_audit']['reference'],actual_native_integrations=integrations,retained_pilot_optimizer_steps=pilot_updates,
        diagnostic_optimizer_steps=diagnostic_updates,total_actual_optimizer_steps=pilot_updates+diagnostic_updates,unit_tests=329,
        scope='Registered equal-budget single-seed/single-clip actual pilot with failed original full-task evaluation, complete resume and independent raw readback. Weak/zero-contact high absolute support scores identified offline; no statistical support/ego/physical WM-gradient/speedup claim.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='geometry_support_pilot_evidence_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},
        source_resolutions=file_pin(out/'source_resolutions.json'),source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),
        prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),initial_archive=file_pin(first_archive_path),initial_archive_manifest_reverified=len(first_manifest['records']),
        verified_pins=len(verified),exact_source_aliases=len(resolutions['records']),verified_manifest=file_pin(analysis/'verified_pins.json'),
        test_log=file_pin(out/log.name),actual_native_integrations=integrations,retained_pilot_optimizer_steps=pilot_updates,diagnostic_optimizer_steps=diagnostic_updates,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),aliases=len(resolutions['records']),prior_reverified=len(old))),flush=True)


if __name__=='__main__':main()
