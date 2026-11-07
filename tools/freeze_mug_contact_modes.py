"""Retain matched negative trials, source candidates and their review scope."""
from pathlib import Path
import importlib.metadata,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main():
    out=ROOT/'provenance/mug_contact_modes'
    if out.exists():raise ValueError('Fresh retained milestone only')
    folders={name:ROOT/'runs'/path for name,path in dict(modes='mug_contact_modes_v1',mode_review='mug_contact_modes_review_v1',
        patches='mug_source_patches_v2',patch_review='mug_source_patches_review_v1',figures='mug_contact_modes_figures_v1').items()}
    reports={k:json.loads((v/'report.json').read_text()) for k,v in folders.items()};mode=reports['modes'];review=reports['mode_review'];patch=reports['patches'];pr=reports['patch_review']
    if len(mode['cases'])!=12 or mode['integration_steps']!=23485 or mode['qualified_pair_count']!=0:raise ValueError('Trial scope changed')
    if (review['passed'],review['total'],review['trace_checks_passed'],review['trace_checks_total'],review['native_EFC_columns'],review['feasible_primals'])!=(7,7,156,156,224,28):raise ValueError('Native mode review not established')
    if (pr['passed'],pr['total'],patch['raw_candidates'],patch['negative_local_face_offset_count'])!=(9,9,30,9):raise ValueError('Source patch review not established')
    tests=ROOT/'runs/mug_modes_tests.log';text=tests.read_text(encoding='utf8')
    if 'Ran 99 tests' not in text or '\nOK' not in text or 'FAILED' in text:raise ValueError('Full suite evidence missing')
    frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
    diff=subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT,text=True)
    if diff.strip():raise ValueError('Frozen phone closure changed')
    # Recursively check recorded pins, deduplicating large repeated MJB/NPZ reads.
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
    for folder in folders.values():
        if (folder/'intent.json').exists():check(json.loads((folder/'intent.json').read_text()))
    for case in mode['cases']:check(json.loads(verify_file_pin(case['static']).read_text()))
    # Preserve v1's exact executed source: v2 only adds per-finger diagnostics.
    old=ROOT/'runs/mug_source_patches_v1';oldreport=json.loads((old/'report.json').read_text())
    oldpin=next(p for p in oldreport['sources'] if Path(p['path']).name=='inspect_mug_source_patches.py')
    snapshot=old/'source_snapshot/inspect_mug_source_patches.py'
    if file_pin(snapshot)['sha256']!=oldpin['sha256']:raise ValueError('Lost v1 executed source')
    for pin in oldreport['sources']:
        if pin!=oldpin:check(pin)
    for value in [oldreport['inputs'],oldreport['output']]:check(value)
    binding=ROOT/'runs/right_mug_frames_v4/binding.json';check(json.loads(binding.read_text()));check(file_pin(binding))
    out.mkdir(parents=True);retained=[]
    def retain(path,name):
        path=Path(path);target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        retained.append(dict(original=file_pin(path),retained=file_pin(target)))
    for name,folder in folders.items():
        retain(folder/'report.json',name+'.json')
        if (folder/'intent.json').exists():retain(folder/'intent.json',name+'_intent.json')
    retain(folders['modes']/'baseline_barrier.json','baseline_barrier.json')
    for spec in mode['models']:retain(folders['modes']/spec['name']/'variant.json','models/'+spec['name']+'.json')
    for i,case in enumerate(mode['cases']):
        retain(verify_file_pin(case['case']),f'cases/{i:02d}_report.json');retain(verify_file_pin(case['static']),f'cases/{i:02d}_static.json')
    retain(old/'report.json','prefinal_patches_report.json');retain(old/'intent.json','prefinal_patches_intent.json');retain(snapshot,'prefinal_patches_tool.py.snapshot')
    retain(tests,'unit_tests.log');retain(folders['figures']/'mug_contact_modes.png','mug_contact_modes.png')
    closure=sorted(pins.values(),key=lambda v:(v['path'],v['sha256']))
    (out/'verified_artifact_pins.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf8')
    summary=dict(schema='mug_contact_modes_milestone:1',source=file_pin(__file__),parent_commit='f2cdf5b',retained=retained,
        checked_pins=file_pin(out/'verified_artifact_pins.json'),checked_pin_count=len(closure),native_cases=12,integration_steps=23485,
        qualified_pairs=0,complete_holding_passes=0,baseline_dt_trace_byte_parity=True,changed_pairs_per_spin_variant=1152,
        material_coefficients_measured=False,LP_attempts=mode['LP_attempts'],linprog_invocations=mode['linprog_invocations'],
        native_EFC_columns=224,feasible_primals_reviewed=28,trace_checks=[156,156],mode_review=[7,7],patch_review=[9,9],tests=99,
        source_candidate_points=30,negative_local_face_offsets=9,sensitivity_candidate_points=21,
        source_contact_truth_available=False,source_gravity_measured=False,independent_RGB_reconstruction=False,
        full_source_trajectory_replayed=False,valid_learning_reward_established=False,neural_updates=0,new_checkpoints=0,
        frozen_phone_closure_changed=False,goal_complete=False,
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','mujoco','torch']},
        next='Finger/normal/source-candidate conditioned bounded proposals, actual finger/handle fit and six-dimensional native support before learning; preserve full cube/pen, trajectory, WM/A1/A2, multi-trajectory, independent RGB and hardware objective.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=file_pin(out/'summary.json'),retained_files=len(retained),checked_pins=len(closure),native_cases=12,passes=0)))


if __name__=='__main__':main()
