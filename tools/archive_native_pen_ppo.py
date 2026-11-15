"""Compact evidence for actual PPO learning, restoration and retained task failures."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main():
    import torch
    out=ROOT/'provenance/pen_native_ppo_results';analysis=ROOT/'runs/pen_native_ppo_archive_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh immutable evidence directories')
    prior_path=ROOT/'provenance/pen_portable_state_results/ledger.json';prior=json.loads(prior_path.read_text())
    aliases_path=verify_file_pin(prior['source_resolutions']);previous=json.loads(aliases_path.read_text());pm=json.loads(verify_file_pin(prior['verified_manifest']).read_text())
    for e in pm['records']:verify_source_pin(e['observed'],previous)
    aliases={json.dumps(e['original'],sort_keys=True):e for e in previous['records']}
    reports={key:ROOT/'runs'/folder/file for key,(folder,file) in dict(
        qualification=('pen_native_ppo_collector_v1','report.json'),first2=('pen_native_ppo_first2_v1','health_0002.json'),
        final=('pen_native_ppo_continuation_v1','health_0032.json'),failed_review=('pen_native_ppo_first2_review_v1','report.json'),
        first_review=('pen_native_ppo_first2_review_v2','report.json'),final_review=('pen_native_ppo_final_review_v1','report.json'),
        evaluation_v1=('pen_native_ppo_evaluation_v1','report.json'),evaluation=('pen_native_ppo_evaluation_v2','report.json'),
        independent=('pen_native_ppo_evaluation_review_v1','report.json'),video=('pen_native_ppo_video_v1','report.json')).items()}
    data={k:json.loads(p.read_text()) for k,p in reports.items()}
    for key,value in [('qualification',[40,40]),('first2',[10,10]),('final',[10,10]),('failed_review',[14,18]),('first_review',[20,20]),('final_review',[20,20]),('independent',[99,99])]:
        if [data[key]['passed'],data[key]['total']]!=value:raise ValueError('Declared result changed '+key)
    if data['final']['training_transitions']!=65536 or data['final']['optimizer_steps']!=1024:raise ValueError('Training budget changed')
    total=data['qualification']['actual_native_integrations']+sum(data[k]['stage_actual_native_integrations'] for k in ['first2','final'])+sum(data[k]['actual_native_integrations'] for k in ['evaluation_v1','evaluation'])
    clones=sum(data[k]['diagnostic_clone_optimizer_steps'] for k in ['failed_review','first_review','final_review'])
    if total!=529544 or clones!=192:raise ValueError('Total native/clone cost changed')
    if any(c['summary']['qualified'] for key in ['evaluation_v1','evaluation'] for c in data[key]['cases']):raise ValueError('Failure-specific study scope changed')
    test=ROOT/'runs/pen_native_ppo_tests285_v1.log';text=test.read_text(encoding='utf-8').replace('\r\n','\n')
    if 'Ran 285 tests in ' not in text or '\nOK\n' not in text:raise ValueError('Final tests missing')
    docs=[ROOT/'docs'/n for n in ['research-plan-v0.98.md','research-plan-v0.99.md','pen-native-ppo-results-2026-11-16.md']]
    configs=[ROOT/'configs'/n for n in ['pen_native_ppo_v1.toml','pen_native_ppo_evaluation_v1.toml','pen_full_pose_native_v1.toml','controlled_evaluation_v4.toml']]
    analysis.mkdir();revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [ROOT/'tools'/n for n in ['archive_native_pen_ppo.py','archive_pen_source_ccd_evidence.py','qualify_native_pen_ppo.py',
        'train_native_pen_ppo.py','review_native_pen_ppo.py','evaluate_native_pen_ppo.py','review_native_pen_ppo_evaluation.py',
        'render_native_pen_ppo.py','render_mug_progress_video.py']]+[ROOT/'tests/test_native_pen_ppo.py']+docs+configs
    receipt=freeze_sources(paths,analysis/'source_snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    receipts=[p.parent/'source_receipt.json' for p in reports.values()]+[analysis/'source_receipt.json',ROOT/'archive/pen_pre_native_ppo_v1/source_receipt.json']
    for path in receipts:
        r=json.loads(path.read_text());verify_source_snapshot(r)
        for e in r['records']:aliases.setdefault(json.dumps(e['observed'],sort_keys=True),dict(original=e['observed'],archived=e['snapshot'],
            reason='Exact original path/size/hash source receipt; no basename/hash-only fallback'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for e in resolutions['records']:verify_source_pin(e['original'],resolutions)
    checkpoint_pins=[data['qualification']['checkpoint'],data['first2']['initial_state'],data['first2']['checkpoint'],data['final']['checkpoint']]
    engines=[];checkpoint_audits=[]
    for pin in checkpoint_pins:
        state=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
        verify_source_snapshot(state['native']['sources'])
        for p in state['native']['engine_sources']:verify_file_pin(p)
        engines+=state['native']['engine_sources']
        checkpoint_audits.append(dict(checkpoint=pin,schema=state['schema'],native_schema=state['native']['schema'],
            iteration=state['iteration'],training_transitions=state['training_transitions'],calibration_transitions=state['calibration_transitions'],
            optimizer_steps=state['optimizer_steps'],actor_norm_count=int(state['modules']['actor']['norm.count']),
            last_saved_phase_range=None if state['memory'] is None else [int(state['memory']['phase_source'].min()),int(state['memory']['phase_source'].max())]))
        del state
    logs=[ROOT/'runs'/n for n in ['pen_native_ppo_unit6_v1.log','pen_native_ppo_unit7_v1.log','pen_native_ppo_tests284_v1.log',
        'pen_native_ppo_collector_console_v1.log','pen_native_ppo_first2_console_v1.log','pen_native_ppo_continuation_console_v1.log',
        'pen_native_ppo_first2_review_console_v1.log','pen_native_ppo_first2_review_console_v2.log','pen_native_ppo_final_review_console_v1.log',
        'pen_native_ppo_evaluation_console_v1.log','pen_native_ppo_evaluation_console_v2.log','pen_native_ppo_evaluation_review_console_v1.log','pen_native_ppo_video_console_v1.log']]
    todos=[file_pin(p) for p in list(reports.values())+receipts+docs+configs+logs+[test,prior_path]]+engines
    allowed={p.parent for p in reports.values()}|{p.parent for p in receipts};verified={}
    while todos:
        pin=todos.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(folder) for folder in allowed):todos+=list(pins(json.loads(resolved.read_text())))
    verify_source_snapshot(receipt,live=True)
    summary=dict(schema='native_pen_PPO_evidence_summary:1',roots={k:file_pin(p) for k,p in reports.items()},
        health={key:[data[key]['passed'],data[key]['total']] for key in ['qualification','first2','final','failed_review','first_review','final_review','independent']},
        checkpoint_audits=checkpoint_audits,training_transitions=65536,normalization_transitions=256,live_optimizer_steps=1024,
        discarded_diagnostic_optimizer_steps=192,actual_native_integrations=529544,
        evaluation=[dict(name=c['name'],summary=c['summary']) for c in data['evaluation']['cases']],
        evaluation_v1=[dict(name=c['name'],summary=c['summary']) for c in data['evaluation_v1']['cases']],
        independent_first_breaches=[dict(name=c['name'],first=c['first_breaches'],common_0_14s=c['common_0_14s']) for c in data['independent']['cases']],
        stage_loop_wall_s={key:data[key]['stage_loop_wall_s'] for key in ['first2','final']},
        GPU_telemetry={key:{k:v for k,v in data[key]['GPU_telemetry'].items() if k!='samples'} for key in ['first2','final']},
        final_unit_tests=285,video=data['video']['video'],
        scope='Actual model-free PPO baseline and combined state loading progressed; original strict task/geometry still fail, W32 future physics/WM gradients unqualified. Same controlled training clip/perfect pose, no ego/CHORD/speedup or full Goal success.')
    out.mkdir();write(out/'summary.json',summary);write(out/'source_resolutions.json',resolutions);shutil.copyfile(test,out/test.name)
    write(analysis/'verified_pins.json',dict(records=list(verified.values())))
    write(out/'ledger.json',dict(schema='native_pen_PPO_evidence_ledger:1',analysis_revision=revision,roots={k:file_pin(p) for k,p in reports.items()},
        summary=file_pin(out/'summary.json'),source_resolutions=file_pin(out/'source_resolutions.json'),prior_evidence=file_pin(prior_path),
        prior_verified_pins=len(pm['records']),verified_pins=len(verified),exact_source_aliases=len(aliases),verified_manifest=file_pin(analysis/'verified_pins.json'),
        source_receipts=[file_pin(p) for p in receipts],documents=[file_pin(p) for p in docs],test_log=file_pin(out/test.name),
        actual_native_integrations=529544,live_optimizer_steps=1024,discarded_diagnostic_optimizer_steps=192,
        scope='Exact checkpoint/native/contact/evaluation/video/log/source pins verified, including failed clone review and both evaluation cohorts. Prior evidence independently reverified; snapshot is a source superset, not proof all modules executed. Full Goal active/incomplete.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified=len(verified),aliases=len(aliases),prior_verified=len(pm['records']))))


if __name__=='__main__':main()
