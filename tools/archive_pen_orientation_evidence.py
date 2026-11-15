"""Exact immutable evidence for the full-orientation experiment and diagnostics."""
import json
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from archive_pen_source_ccd_evidence import pins,write
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin


def main(axis='ORIENTATION'):
    if axis not in ['ORIENTATION','AMPLITUDE']:raise ValueError('Declared evidence axis')
    phase=axis.lower()
    out=ROOT/f'provenance/pen_native_{phase}_results';analysis=ROOT/f'runs/pen_{phase}_analysis_sources_v1'
    if out.exists() or analysis.exists():raise ValueError('Fresh evidence only')
    names=dict(shaft='pen_native_orientation_shaft_v1',full='pen_native_orientation_full_so3_v1',
        shaft_review='pen_native_orientation_shaft_review_v1',full_review='pen_native_orientation_full_so3_review_v1',
        offline='pen_offline_full_orientation_v1',matrix='pen_native_orientation_matrix_review_v1',
        coverage='pen_native_orientation_coverage_v1',video='pen_native_orientation_video_v1')
    review_keys=['shaft_review','full_review'];raw_keys=['shaft','full'];test_log='pen_native_orientation_tests211.log';plan='research-plan-v0.83.md'
    if axis=='AMPLITUDE':
        names=dict(x1='pen_native_amplitude_x1_v1',x4='pen_native_amplitude_x4_v1',x1_review='pen_native_amplitude_x1_review_v1',x4_review='pen_native_amplitude_x4_review_v1',
            matrix='pen_native_amplitude_matrix_review_v1',coverage='pen_native_amplitude_coverage_v1',video='pen_native_amplitude_video_v1')
        review_keys=['x1_review','x4_review'];raw_keys=['x1','x4'];test_log='pen_native_amplitude_tests213.log';plan='research-plan-v0.84.md'
    reports={k:ROOT/'runs'/v/'report.json' for k,v in names.items()}
    for k in review_keys:reports[k+'_EFC']=reports[k].parent/'EFC/report.json'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=[Path(__file__)]+[ROOT/'tools'/p for p in ['archive_pen_source_ccd_evidence.py','rescore_pen_full_orientation.py','compare_pen_native_ccd.py',
        'audit_pen_candidate_coverage.py','render_pen_native_ccd.py','render_mug_progress_video.py']]+[ROOT/'tests/test_pen_native_planning_math.py']
    analysis.mkdir();receipt=freeze_sources(files,analysis/'snapshot',ROOT,revision);write(analysis/'source_receipt.json',receipt)
    prior=ROOT/('provenance/pen_native_ccd_results/supplement_source_resolutions.json' if axis=='ORIENTATION' else 'provenance/pen_native_orientation_results/source_resolutions.json')
    aliases={json.dumps(r['original'],sort_keys=True):r for r in json.loads(prior.read_text())['records']}
    receipt_paths=[analysis/'source_receipt.json']
    for folder in names.values():receipt_paths+=sorted((ROOT/'runs'/folder).glob('*source_receipt.json'))
    for path in receipt_paths:
        manifest=json.loads(path.read_text());verify_source_snapshot(manifest)
        for r in manifest['records']:
            aliases.setdefault(json.dumps(r['observed'],sort_keys=True),dict(original=r['observed'],archived=r['snapshot'],reason='Exact observed source bytes; explicit immutable snapshot'))
    resolutions=dict(schema='explicit_source_relocations:1',records=list(aliases.values()))
    for r in resolutions['records']:verify_source_pin(r['original'],resolutions)
    todo=[file_pin(p) for p in list(reports.values())+receipt_paths+[ROOT/'runs'/test_log,ROOT/'docs'/plan]]
    allowed=[ROOT/'runs'/v for v in names.values()]+[analysis];verified={}
    while todo:
        pin=todo.pop();key=json.dumps(pin,sort_keys=True)
        if key in verified:continue
        resolved=verify_source_pin(pin,resolutions);verified[key]=dict(observed=pin,resolved=file_pin(resolved))
        if resolved.suffix=='.json' and any(resolved.is_relative_to(p) for p in allowed):todo+=list(pins(json.loads(resolved.read_text())))
    matrix=json.loads(reports['matrix'].read_text());coverage=json.loads(reports['coverage'].read_text())
    if matrix['passed']!=matrix['total']:raise ValueError('Matrix identity audit failed')
    if coverage['passed']!=coverage['total']:raise ValueError('Coverage interpretation failed')
    if axis=='ORIENTATION':
        offline=json.loads(reports['offline'].read_text())
        if offline['passed']!=offline['total'] or offline['total']!=1006:raise ValueError('Offline interpretation failed')
        if matrix['actual_integrations']!=796 or matrix['candidate_integrations']!=45372:raise ValueError('Budget changed')
    elif matrix['candidate_integrations']!=57*matrix['actual_integrations']:raise ValueError('Actual and candidate budgets inconsistent')
    actual_ticks=matrix['actual_integrations'];candidate_ticks=matrix['candidate_integrations'];total_ticks=actual_ticks+candidate_ticks
    raw=[json.loads(reports[k].read_text()) for k in raw_keys]
    if any(c['summary']['qualified'] for r in raw for c in r['cases']):raise ValueError('Review scientific conclusion before changing evidence')
    out.mkdir();copies={}
    for key in review_keys+['matrix','coverage','video']+([ 'offline'] if axis=='ORIENTATION' else [])+[k+'_EFC' for k in review_keys]:
        target=out/(key+'.json');shutil.copyfile(reports[key],target)
        if file_pin(target)['sha256']!=file_pin(reports[key])['sha256']:raise ValueError('Copied bytes differ')
        copies[key]=file_pin(target)
    for key,path in {('tests211.log' if axis=='ORIENTATION' else 'tests213.log'):ROOT/'runs'/test_log,'preview.png':reports['video'].parent/'0.005_decoded_mid.png'}.items():
        target=out/key;shutil.copyfile(path,target)
        if file_pin(target)['sha256']!=file_pin(path)['sha256']:raise ValueError('Copied bytes differ')
        copies[key]=file_pin(target)
    write(out/'source_resolutions.json',resolutions)
    write(out/'ledger.json',dict(schema='native_pen_matrix_evidence:2',axis=axis,analysis_revision=revision,roots={k:file_pin(p) for k,p in reports.items()},copied=copies,
        source_receipts=[file_pin(p) for p in receipt_paths],previous_resolutions=file_pin(prior),source_resolutions=file_pin(out/'source_resolutions.json'),
        verified_pins=len(verified),exact_source_aliases=len(aliases),verified=list(verified.values()),actual_integrations=actual_ticks,candidate_integrations=candidate_ticks,total_integrations=total_ticks,
        neural_updates=0,scope='Nested pins verified within this cohort; exact historical input reports verified at boundary. Old failed data/force/task reports unchanged. Offline endpoint coverage is not a new executed trajectory or global feasibility proof. All eight tasks fail; prediction failures remain explicit in separate reviews. Source superset snapshots are not proof every module executed.'))
    print(json.dumps(dict(ledger=file_pin(out/'ledger.json'),verified=len(verified),aliases=len(aliases),native_integrations=total_ticks)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--axis',choices=['ORIENTATION','AMPLITUDE'],default='ORIENTATION');main(p.parse_args().axis)
