"""Retain grasp/gravity/native diagnostic evidence without binary datasets."""
from pathlib import Path
import hashlib,importlib.metadata,json,subprocess,sys,urllib.request
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main():
    import mujoco
    out=ROOT/'provenance/mug_grasp'
    if out.exists():raise ValueError('Fresh retained milestone only')
    stages={name:ROOT/'runs'/run for name,run in dict(kinematic='mug_grasp_ik_v1',contact='mug_contact_ik_v1',
        restoration='mug_contact_restoration_v1',holding='mug_static_hold_v1',saved_review='mug_static_saved_review_v1',
        gravity='mug_gravity_review_v1',figures='mug_holding_figures_v1').items()}
    reports={name:json.loads((p/'report.json').read_text()) for name,p in stages.items()}
    for name,report in reports.items():
        for pin in report.get('sources',[]):verify_file_pin(pin)
        if 'source' in report:verify_file_pin(report['source'])
    hold=reports['holding'];saved=reports['saved_review'];restored=reports['restoration']
    if len(hold['cases'])!=10 or hold['integration_steps']!=9343 or hold['qualified_pair_count']!=0:
        raise ValueError('Unexpected executed holding scope')
    if saved['case_checks_passed']!=saved['case_checks_total'] or saved['case_checks_total']!=140:raise ValueError('Saved review missing')
    if restored['geometry_pass_count']!=1:raise ValueError('Static qualification changed')
    test_log=ROOT/'runs/mug_grasp_tests.log';test_text=test_log.read_text(encoding='utf8')
    if 'Ran 88 tests' not in test_text or '\nOK' not in test_text or 'FAILED' in test_text:raise ValueError('Full executed tests missing')
    frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
    diff=subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT,text=True)
    if diff.strip():raise ValueError('Frozen phone source/config/archive changed')
    out.mkdir(parents=True);copies=[]
    def retain(path,name):
        path=Path(path);target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        copies.append(dict(original=file_pin(path),retained=file_pin(target)))
    for name,p in stages.items():
        retain(p/'report.json',name+'.json')
        if (p/'intent.json').exists():retain(p/'intent.json',name+'_intent.json')
    retain(test_log,'unit_tests.log');retain(stages['figures']/'mug_noaid_holding.png','mug_noaid_holding.png')
    qualified=restored['results'][0]
    for kind in ['candidate','geometry','proposals']:verify_file_pin(qualified[kind])
    retain(qualified['geometry']['path'],'qualified_static_geometry.json')
    gravity_manifest=json.loads(verify_file_pin(reports['gravity']['official_sources']).read_text())
    retain(reports['gravity']['official_sources']['path'],'gravity_sources/manifest.json')
    for item in gravity_manifest['files']:
        verify_file_pin(item);retain(item['path'],'gravity_sources/'+Path(item['path']).name+'.source')
    # The viewer/loader source snapshots are unexecuted Apache-2.0 references.
    url=f"https://raw.githubusercontent.com/facebookresearch/hot3d/{gravity_manifest['revision']}/LICENSE"
    license_bytes=urllib.request.urlopen(url,timeout=20).read();(out/'gravity_sources/LICENSE').write_bytes(license_bytes)
    (out/'gravity_sources/license_source.json').write_text(json.dumps(dict(url=url,sha256=hashlib.sha256(license_bytes).hexdigest(),bytes=len(license_bytes)),indent=2)+'\n')
    model=mujoco.MjModel.from_binary_path(str(ROOT/'runs/right_mug_frames_v4/hand2_mug.mjb'))
    audit_summaries=[]
    for stage in ['kinematic','contact','restoration']:
        for index,item in enumerate(reports[stage]['results']):
            verify_file_pin(item['geometry']);verify_file_pin(item['candidate']);geo=json.loads(Path(item['geometry']['path']).read_text())
            audit_summaries.append(dict(stage=stage,seed=index,geometry=item['geometry'],candidate=item['candidate'],
                passed=geo['passed'],unknown=geo['unknown'],rows=len(geo['rows']),
                all_pairs_clear=geo['all_pairs_clear_with_declared_pad_band'],groups=geo['canonical_contact_candidates'],
                failed=[dict(kind=r['kind'],geoms=[model.geom(g).name for g in r['geom_ids']],gap_m=r.get('gap_m'),status=r['status'])
                        for r in geo['rows'] if not r['passed']]))
    (out/'geometry_summaries.json').write_text(json.dumps(audit_summaries,indent=2,allow_nan=False)+'\n',encoding='utf8')
    for case in hold['cases']:verify_file_pin(case['case']);verify_file_pin(case['trace'])
    summary=dict(schema='mug_grasp_feasibility_milestone:1',source=file_pin(__file__),parent_commit='edc87d0',
        static_candidate_count=2,static_qualified=1,geometry_pairs_per_candidate=3026,
        native_cases=10,native_holding_passes=0,paired_native_successes=0,integration_steps=9343,
        saved_case_checks=[140,140],saved_global_checks=[saved['passed'],saved['total']],unit_tests_passed=88,
        conditional_gravity_mismatch_deg=reports['gravity']['angle_to_simulator_gravity_deg_min_max'],
        source_gravity_per_sequence_measured=False,source_trajectory_replayed=False,source_contact_force_transferred=False,
        neural_updates=0,new_checkpoints=0,phone_training_source_closure_changed=False,goal_complete=False,
        retained=copies,geometry_summary=file_pin(out/'geometry_summaries.json'),
        versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','mujoco','torch']},
        next='Support/moment/actuator-aware initialization and gravity-aware task mapping before any new learning. Full cube/pen, CHORD/WM, multi-trajectory, independent RGB and hardware requirements remain.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=file_pin(out/'summary.json'),retained_files=len(copies),tests=88,static_pass=1,native_pass=0)))


if __name__=='__main__':main()
