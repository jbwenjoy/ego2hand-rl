"""Retain reviewed causal native data contracts before any new neural update."""
from pathlib import Path
import json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main():
    out=ROOT/'provenance/mug_H20_data'
    if out.exists():raise ValueError('Fresh retained data milestone')
    run=ROOT/'runs/mug_H20_collection_v1';packed=ROOT/'runs/mug_H20_packed_v1';r=json.loads((run/'report.json').read_text());p=json.loads((packed/'report.json').read_text())
    if (p['passed'],p['total'],r['TRAIN_complete_rows'],r['CHECK_complete_rows'],r['branch_cases'],r['total_integration_steps'])!=(7,7,378,193,48,54333):raise ValueError('Reviewed data scope required')
    tests=ROOT/'runs/mug_H20_tests119.log';text=tests.read_text(encoding='utf8')
    if 'Ran 119 tests' not in text or '\nOK' not in text or 'FAILED' in text:raise ValueError('Full119-test evidence missing')
    # Added explicitly imported task modules/configs do not alter the bytes of
    # the old declared phone files. Existing modules/__init__ are untouched.
    frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
    changed=subprocess.check_output(['git','diff','--name-only','--diff-filter=DMRT','9b38f97','--',*frozen],cwd=ROOT,text=True)
    if changed.strip():raise ValueError('Existing frozen phone source/config/archive changed')
    pins={}
    def check(v):
        if isinstance(v,dict):
            if {'path','bytes','sha256'}<=v.keys():
                key=(v['path'],v['sha256'])
                if key not in pins:verify_file_pin(v);pins[key]=v
            else:
                for x in v.values():check(x)
        elif isinstance(v,list):
            for x in v:check(x)
    check(r);check(p);check(json.loads((run/'intent.json').read_text()))
    out.mkdir(parents=True);retained=[]
    def retain(path,name):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());retained.append(dict(original=file_pin(path),retained=file_pin(target)))
    for path,name in [(run/'report.json','collection.json'),(run/'intent.json','collection_intent.json'),(run/'barrier.json','baseline_barriers.json'),(run/'resume.json','restore_barriers.json'),(packed/'report.json','packed_review.json'),(tests,'unit_tests.log')]:retain(path,name)
    for i,episode in enumerate(r['episodes']):retain(Path(episode['transitions']['path']).with_name('report.json'),f'episodes/{i:02d}.json')
    for i,episode in enumerate(r['branches']):retain(Path(episode['transitions']['path']).with_name('report.json'),f'response/{i:02d}.json')
    (out/'verified_artifact_pins.json').write_text(json.dumps(list(pins.values()),indent=2)+'\n',encoding='utf8')
    summary=dict(schema='mug_H20_causal_data_milestone:1',source=file_pin(__file__),parent_commit='36bc80e',retained=retained,
        verified_pins=file_pin(out/'verified_artifact_pins.json'),verified_pin_count=len(pins),native_integration_steps=54333,
        training_episodes=16,check_episodes=8,TRAIN_fullH_rows=378,CHECK_fullH_rows=193,CHECK_response_rows=48,partial_H_excluded=24,
        baseline_trace_byte_parity=[True,True],full_state_controller_restore_checks=[6,6],raw_label_review=[7,7],tests=119,
        current_control_schema='physical_residual_EMA20:1',input_schema=p['input_schema'],target_schema=p['target_schema'],
        existing_phone_source_bytes_changed=False,new_explicit_learning_module='learning/compound_response.py',new_explicit_config='configs/mug_response_first10.toml',
        source_human_contact_truth_known=False,independent_RGB_reconstruction=False,complete_source_trajectory_success=False,
        neural_updates_at_freeze=0,new_neural_checkpoints_at_freeze=0,goal_complete=False,
        next='Fresh CUDA one-step response WM, first healthy10 checkpoint readback/ETA then pause as standing user request. No Actor or A1/A2/physical improvement claim.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(summary=file_pin(out/'summary.json'),retained_files=len(retained),pins=len(pins))))


if __name__=='__main__':main()
