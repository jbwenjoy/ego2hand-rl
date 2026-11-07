"""Retain source-conditioned proposals and local support limitations."""
from pathlib import Path
import importlib.metadata,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main():
    out=ROOT/'provenance/mug_patch_proposals'
    if out.exists():raise ValueError('Fresh retained milestone')
    roots={k:ROOT/'runs'/v for k,v in dict(proposal='mug_patch_proposals_v1',coupled='mug_patch_coupled_v1',
        engagement='mug_patch_engagement_v1',support='mug_patch_support_v1',review='mug_patch_review_v3',stills='mug_patch_stills_v1').items()}
    reports={k:json.loads((v/'report.json').read_text()) for k,v in roots.items()};r=reports['review']
    if (r['passed'],r['total'],r['pairs_per_full_audit'],r['full_pair_queries'],r['native_EFC_columns'])!=(5,5,3025,33275,44):raise ValueError('Saved review incomplete')
    if (reports['proposal']['reach_solve_count'],reports['proposal']['geometry_pass_count'],reports['coupled']['geometry_pass_count'],reports['engagement']['geometry_pass_count'])!=(36,0,1,3):raise ValueError('Proposal scope changed')
    if reports['support']['verified_separations']!=3 or any(reports[k]['integration_steps'] for k in ['proposal','coupled','engagement','support']):raise ValueError('Support/integration scope changed')
    test=ROOT/'runs/mug_patch_tests_final.log';text=test.read_text(encoding='utf8')
    if 'Ran 104 tests' not in text or '\nOK' not in text or 'FAILED' in text:raise ValueError('Full test evidence missing')
    frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
    if subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT,text=True).strip():raise ValueError('Frozen learning source changed')
    pins={}
    def check(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=value.keys():
                key=(value['path'],value['sha256'])
                if key not in pins:verify_file_pin(value);pins[key]=value
            else:
                for child in value.values():check(child)
        elif isinstance(value,list):
            for child in value:check(child)
    for report in reports.values():check(report)
    for root in roots.values():
        if (root/'intent.json').exists():check(json.loads((root/'intent.json').read_text()))
    check(json.loads((ROOT/'runs/mug_source_patches_v2/report.json').read_text()))
    out.mkdir(parents=True);retained=[]
    def retain(path,name):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes())
        retained.append(dict(original=file_pin(path),retained=file_pin(target)))
    for name,root in roots.items():
        retain(root/'report.json',name+'.json')
        if (root/'intent.json').exists():retain(root/'intent.json',name+'_intent.json')
    for name in ['anchors.json','reach_solutions.json']:retain(roots['proposal']/name,name)
    for i,row in enumerate(reports['proposal']['candidates']):retain(Path(row['candidate']['path']).parent/'result.json',f'proposal/{i:02d}.json')
    for i,row in enumerate(reports['coupled']['results']):
        folder=Path(row['candidate']['path']).parent
        for name in ['result.json','solver.json','proposals.jsonl']:retain(folder/name,f'coupled/{i:02d}_{name}')
    query_counts=[]
    for i,row in enumerate(reports['engagement']['results']):
        folder=Path(row['candidate']['path']).parent
        for name in ['report.json','proposals.json']:retain(folder/name,f'engagement/{i:02d}_{name}')
        count=json.loads((folder/'proposals.json').read_text())['queries'];query_counts.append(dict(version=i,recorded_before_final_readback=count,actual_including_three_final_queries=count+3))
    # Reviewer failures affect neither the proposal operands nor model. Preserve
    # their exact source bytes and distinguish failed operations from experiments.
    for version in [1,2]:
        root=ROOT/f'runs/mug_patch_review_v{version}'
        retain(root/'terminal.json',f'failed_review{version}/terminal.json')
        retain(root/'source_snapshot/review_mug_patch_proposals.py',f'failed_review{version}/reviewer.py.snapshot')
    retain(test,'unit_tests.log');retain(roots['stills']/'mug_patch_poses.png','mug_patch_poses.png')
    closure=sorted(pins.values(),key=lambda p:(p['path'],p['sha256']))
    (out/'verified_artifact_pins.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf8')
    summary=dict(schema='mug_source_patch_proposals_milestone:1',source=file_pin(__file__),parent_commit='92a0f18',retained=retained,
        verified_pins=file_pin(out/'verified_artifact_pins.json'),verified_pin_count=len(closure),reach_solves=36,material_reach_gate_passes=34,
        composed_candidates=6,composed_geometry_passes=0,coupled_candidates=2,coupled_geometry_passes=1,engaged_candidates=3,
        engaged_geometry_and_native_source_candidate_matches=3,engaged_force_only_feasible=3,engaged_full6D_feasible=0,
        checked_frozen_cone_separations=3,native_EFC_columns_reviewed=44,primal_witnesses_reviewed=r['feasible_primals'],
        successful_saved_review=[5,5],geometry_pairs_per_audit=3025,object_pairs=2816,self_pairs=189,hand_table_pairs=20,
        primary_full_geometry_audits=11,review_full_geometry_audits_including_failed_attempts=23,
        pair_count_correction='Some earlier scope prose said3026/21; actual saved pair lists and compiled filters always yield3025/20. No physics/row edits.',
        preload_query_counter_scope=query_counts,tests=104,integration_steps=0,neural_updates=0,new_checkpoints=0,
        source_contact_truth_known=False,source_gravity_measured=False,independent_RGB_reconstruction=False,full_source_trajectory_replayed=False,
        valid_learning_reward_or_physical_gradient_established=False,frozen_phone_closure_changed=False,goal_complete=False,
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','mujoco','torch']},
        next='Compare current/conditional source gravity against the same uncertain candidate envelope, then rank legitimate alternate material patches with6D separating support and explicit source deviations; preserve full controlled cube/pen, WM/A1/A2, multi-trajectory, RGB and hardware objective.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=file_pin(out/'summary.json'),retained_files=len(retained),pins=len(closure),geometry_pass=1,engaged_three_group=3,full6D=0,tests=104)))


if __name__=='__main__':main()
