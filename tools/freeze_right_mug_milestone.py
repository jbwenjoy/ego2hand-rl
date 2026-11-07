"""Retain small right-mug evidence; simulator binaries/arrays stay in runs."""
from pathlib import Path
import importlib.metadata,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin,ArtifactRegistry


def main():
    out=ROOT/'provenance/right_mug_frames'
    if (out/'summary.json').exists():raise ValueError('Do not overwrite retained milestone')
    stage=ROOT/'runs/right_mug_frames_v4';review=ROOT/'runs/right_mug_saved_review_v2'
    report=json.loads((stage/'report.json').read_text());check=json.loads((review/'report.json').read_text())
    if report['passed']!=15 or check['passed']!=15:raise ValueError('Incomplete stage/review')
    for pin in report['sources']+list(report['outputs'].values()):verify_file_pin(pin)
    for pin in json.loads((ROOT/'runs/right_mug_proximity_v1/report.json').read_text())['source_pins']:verify_file_pin(pin)
    verify_file_pin(check['source'])
    frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
    difference=subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT,text=True)
    if difference.strip():raise ValueError('Frozen phone training source/config/archive changed')
    tests=(out/'unit_test_terminal.log').read_text()
    if 'Ran 81 tests' not in tests or '\nOK' not in tests or 'Exit code: 0' not in tests:raise ValueError('Missing actual test terminal')
    registry=ArtifactRegistry(stage/'artifacts.toml');index=json.loads((stage/'index_report.json').read_text())
    for name,pin in index['verified'].items():
        if file_pin(registry.resolve(name))!=pin:raise ValueError('Changed indexed artifact')
    paths={
        'frame_contract.json':stage/'report.json','frame_intent.json':stage/'intent.json',
        'scene_binding.json':stage/'binding.json','artifacts.toml':stage/'artifacts.toml',
        'index_report.json':stage/'index_report.json','saved_review.json':review/'report.json',
        'source_proximity.json':ROOT/'runs/right_mug_proximity_v1/report.json',
        'source_proximity_intent.json':ROOT/'runs/right_mug_proximity_v1/intent.json',
        'figures.json':ROOT/'runs/right_mug_figures_v1/report.json',
        'right_mug_geometry.png':ROOT/'runs/right_mug_figures_v1/right_mug_geometry.png',
        'right_mug_motion_proximity.png':ROOT/'runs/right_mug_figures_v1/right_mug_motion_proximity.png'}
    for version in [1,2,3]:
        p=ROOT/f'runs/right_mug_frames_v{version}'
        paths[f'prefinal_v{version}_intent.json']=p/'intent.json'
        paths[f'prefinal_v{version}_terminal.json']=p/'terminal.json'
        intent=json.loads((p/'intent.json').read_text())
        for pin in intent['source_pins']:
            name=Path(pin['path']).name
            if name in ['prepare_right_mug_reference.py','hand2_scene.py']:
                snapshot=p/'source_snapshot'/name
                current=file_pin(snapshot)
                if (current['bytes'],current['sha256'])!=(pin['bytes'],pin['sha256']):raise ValueError('Changed retained executed source')
                paths[f'prefinal_v{version}_{name}.snapshot']=snapshot
    copied=[]
    for name,path in paths.items():
        target=out/name
        if target.exists():raise ValueError('Fresh retained destination required')
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        copied.append(dict(original=file_pin(path),retained=file_pin(target)))
    summary=dict(schema='right_mug_frames_milestone:1',source=file_pin(__file__),
        stage_checks=[15,15],saved_checks=[15,15],unit_tests_passed=81,artifact_entries=len(index['verified']),
        retained=copied,tests=file_pin(out/'unit_test_terminal.log'),
        environment=dict(interpreter=sys.executable,versions={n:importlib.metadata.version(n) for n in ['numpy','torch','mujoco','scipy','trimesh']}),
        source_angle_limit_exceedances=dict(frame_parameter_pairs=21,zero_based_columns=[3,19],counts=[13,8],max_rad=[.020856648683547974,.026575922966003418],clamped=False),
        current_turn_neural_updates=0,current_turn_integration_steps=0,
        phone_training_closure_changed=False,physical_grasp_qualified=False,ego_RGB_reconstruction=False,
        source_gravity_known=False,goal_complete=False,
        next='Verify gravity; actual Hand2 finger-fit/collision-aware IK, wrench aggregation, new task contracts and no-aid physical baselines before any new learning.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=file_pin(out/'summary.json'),retained_files=len(copied),checks=summary['stage_checks'],saved=summary['saved_checks'],tests=81)))


if __name__=='__main__':main()
