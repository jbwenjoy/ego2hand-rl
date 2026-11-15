"""Retain training2 diagnostics, actual continuation and never-fit EMA probe."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    out=ROOT/'provenance/owned_ppo_results';analysis=ROOT/'runs/owned_ppo_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable training2 evidence')
    prior_path=ROOT/'provenance/owned_pen_state_results/ledger.json';prior=json.loads(prior_path.read_text());previous=json.loads(verify_file_pin(prior['source_resolutions']).read_text())
    old=set();cursor=prior;counts=[]
    while True:
        manifest=json.loads(verify_file_pin(cursor['verified_manifest']).read_text());counts.append(len(manifest['records']))
        for e in manifest['records']:
            key=json.dumps(e['observed'],sort_keys=True)
            if key not in old:verify_source_pin(e['observed'],previous);old.add(key)
        if cursor['schema']=='pen_portable_state_evidence_ledger:1' or 'prior_evidence' not in cursor:break
        cursor=json.loads(verify_file_pin(cursor['prior_evidence']).read_text())
    reports={k:ROOT/'runs'/folder/'report.json' for k,folder in dict(initial='owned_ppo_qualification_v1',qualification='owned_ppo_qualification_v2',initial_review='owned_ppo_review_v1',review='owned_ppo_review_v2',probe='owned_pen_ema_probe_v1').items()}
    data={k:json.loads(p.read_text()) for k,p in reports.items()}
    for k,v in dict(initial=[83,83],qualification=[83,83],initial_review=[51,51],review=[84,84],probe=[8,8]).items():
        if [data[k]['passed'],data[k]['total']]!=v:raise ValueError('Observed checks changed '+k)
    physical=sum(v['actual_native_integrations'] for v in data.values());updates=sum(v.get('diagnostic_optimizer_steps',0)+v.get('diagnostic_clone_optimizer_steps',0)+v.get('diagnostic_continuation_optimizer_steps',0) for v in data.values())
    if physical!=7760 or updates!=224:raise ValueError('Diagnostic budget changed')
    log=ROOT/'runs/owned_ppo_tests322_v1.log';text=log.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 322 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Final322-test log required')
    memory=torch.load(verify_file_pin(data['qualification']['memory']),map_location='cpu',weights_only=False);g=memory['contact_geometry_max'];screen={}
    for i,(name,limit) in enumerate([('object_penetration_m',.002),('self_penetration_m',.0002),('joint_excursion_rad',.001)]):
        value=g[...,i];screen[name]=dict(limit=limit,maximum=float(value.max()),violating_rows=int((value>limit).sum()),total_rows=value.numel(),violating_worlds=int((value>limit).any(0).sum()))
    value=memory['contact_nonhand_max_sum'][...,0];screen['nonhand_N']=dict(limit=.1,maximum=float(value.max()),violating_rows=int((value>.1).sum()),total_rows=value.numel())
    probe_tracking=[]
    for c in data['probe']['cases']:
        rows=torch.load(verify_file_pin(c['raw_record']),map_location='cpu',weights_only=False);max_error=0.
        for row in rows:
            expected=.5*row['EMA_before']+.25*row['applied'];max_error=max(max_error,float((row['EMA_after']-expected).abs().max()))
        if max_error>1e-7:raise ValueError('Independent saved EMA recurrence mismatch')
        probe_tracking.append(dict(name=c['name'],noise_seed=c['noise_seed'],controls=c['controls'],COM_control_end_max_m=max(float(x['COM_error_m'][0]) for x in rows),
            full_angle_control_end_max_rad=max(float(x['full_angle_rad'][0]) for x in rows),EMA_recurrence_max_error=max_error,
            geometry_prefix_qualified=c['geometry_prefix_qualified'],full_task_qualified=c['full_task_qualified']))
    analysis.mkdir();docs=[ROOT/'docs'/n for n in ['owned-ppo-integration-results-2026-11-16.md','research-plan-v1.03.md']]
    configs=[ROOT/'configs'/n for n in ['owned_pen_ppo_qualification_v1.toml','owned_pen_ema_probe_v1.toml']]
    tools=[ROOT/'tools'/n for n in ['qualify_owned_pen_ppo.py','review_owned_pen_ppo.py','probe_owned_pen_ema.py','archive_owned_pen_ppo.py','archive_pen_source_ccd_evidence.py']]
    sources=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+sorted((ROOT/'tests').glob('*.py'))+docs+configs+tools+[ROOT/'README.md',ROOT/'AGENTS.md']
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();receipt=freeze_sources(sources,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json'];aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    for p in receipts:
        source=json.loads(p.read_text());verify_source_snapshot(source)
        for e in source['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],reason='Full original source path/size/hash, including initial uninformative check and extended review versions.'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    engines=[];state_audits=[]
    for key,pin in [('qualification',data['qualification']['checkpoint']),('fresh',data['review']['fresh_state']),('continued',data['review']['continuation_state'])]:
        state=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False);verify_source_snapshot(state['native']['sources'])
        for p in state['native']['engine_sources']:verify_file_pin(p)
        engines.extend(state['native']['engine_sources']);state_audits.append(dict(name=key,state=pin,schema=state['schema'],native_schema=state['native']['schema'],iteration=state['iteration'],
            optimizer_steps=state['optimizer_steps'],training_transitions=state['training_transitions'],calibration_transitions=state['calibration_transitions'],normalizer_count=int(state['modules']['actor']['norm.count']),
            role=state['metadata']['role'],memory_iteration=state['memory_iteration']))
    logs=sorted({p for pattern in ['owned_ppo_unit*.log','owned_ppo_tests*.log','owned_ppo_qualification_console_v*.log','owned_ppo_review_console_v*.log','owned_pen_ema_probe_console_v*.log'] for p in (ROOT/'runs').glob(pattern)})
    todos=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[prior_path]]+engines;allowed={p.parent for p in reports.values()}|{analysis};verified={}
    while todos:
        pin=todos.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todos+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='owned_PPO_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},checks={k:[v['passed'],v['total']] for k,v in data.items()},
        initial_check_limit='v1 file immutability check compared the same read to itself; retained as reported, not independent integrity evidence. v2 records pre-test pin and redoes all work.',
        state_audits=state_audits,primary_geometry_screen=screen,probe_control_end_tracking=probe_tracking,raw_window_max_errors=data['qualification']['raw_max_errors'],
        actual_native_integrations=7760,diagnostic_optimizer_steps=224,retained_candidate_optimizer_steps=0,unit_tests=322,
        scope='Training2/native4 collector/source/optimizer continuation qualified; old tracking-only diagnostic weights, early geometry failures and short never-fit EMA probe are not trained-policy/task/support/WM-gradient/ego or speedup evidence.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(log,out/log.name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='owned_PPO_evidence_ledger:1',analysis_revision=revision,summary=file_pin(out/'summary.json'),roots={k:file_pin(p) for k,p in reports.items()},source_resolutions=file_pin(out/'source_resolutions.json'),
        source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],prior_evidence=file_pin(prior_path),prior_manifest_counts=counts,prior_unique_reverified_pins=len(old),
        verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),test_log=file_pin(out/log.name),
        actual_native_integrations=7760,diagnostic_optimizer_steps=224,retained_candidate_optimizer_steps=0,scope=summary['scope']))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified_pins=len(verified),aliases=len(aliases),prior_reverified=len(old))))


if __name__=='__main__':main()
