"""Retain support diagnostics and failed native trials without binary data."""
from pathlib import Path
import importlib.metadata,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main():
    out=ROOT/'provenance/mug_support'
    if out.exists():raise ValueError('Fresh retained milestone only')
    names=dict(support='mug_support_v2',native='mug_support_native_v1',equilibrium='mug_equilibrium_v1',
        placement='mug_gravity_placement_v1',restoration='mug_gravity_restoration_v1',gravity_hold='mug_gravity_hold_v1',
        engagement='mug_gravity_engagement_v2',EFC_review='mug_support_saved_review_v1',
        trial_review='mug_support_trials_review_v1',figures='mug_support_figures_v1')
    roots={k:ROOT/'runs'/v for k,v in names.items()};reports={k:json.loads((v/'report.json').read_text()) for k,v in roots.items()}
    for report in reports.values():
        for pin in report.get('sources',[]):verify_file_pin(pin)
        if 'source' in report:verify_file_pin(report['source'])
    support=reports['support'];review=reports['trial_review'];efc=reports['EFC_review']
    if (support['LP_attempts'],support['linprog_invocations'],support['cone6D_feasible_count'])!=(192,176,1):raise ValueError('Support diagnostic scope')
    if (review['cases'],review['checks_passed'],review['checks_total'])!=(20,260,260):raise ValueError('Saved native review')
    if not all(efc['checks'].values()) or efc['primal_witnesses_checked']!=79 or efc['native_EFC_unit_column_queries']!=172:raise ValueError('Native EFC review')
    total=sum(reports[k]['integration_steps'] for k in ['native','equilibrium','gravity_hold'])
    if total!=33801:raise ValueError('Actual integration count')
    tests=ROOT/'runs/mug_support_tests.log';test_text=tests.read_text(encoding='utf8')
    if 'Ran 93 tests' not in test_text or '\nOK' not in test_text or 'FAILED' in test_text:raise ValueError('Missing full tests')
    frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
    difference=subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT,text=True)
    if difference.strip():raise ValueError('Frozen phone source/config/archive changed')
    out.mkdir(parents=True);retained=[]
    def retain(path,name):
        path=Path(path);target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
        retained.append(dict(original=file_pin(path),retained=file_pin(target)))
    for name,folder in roots.items():
        retain(folder/'report.json',name+'.json')
        if (folder/'intent.json').exists():retain(folder/'intent.json',name+'_intent.json')
    retain(tests,'unit_tests.log');retain(roots['figures']/'mug_support_results.png','mug_support_results.png')
    # Preserve both executed pre-correction sources; never rewrite their reports.
    old=ROOT/'runs/mug_support_v1';old_report=json.loads((old/'report.json').read_text())
    expected=next(p for p in old_report['sources'] if Path(p['path']).name=='diagnose_mug_support.py')
    snapshot=old/'source_snapshot/diagnose_mug_support.py'
    if file_pin(snapshot)['sha256']!=expected['sha256']:raise ValueError('Lost v1 executed source')
    retain(snapshot,'prefinal_support_counter.py.snapshot');retain(old/'report.json','prefinal_support_counter_report.json')
    failed=ROOT/'runs/mug_gravity_engagement_v1';failed_intent=json.loads((failed/'intent.json').read_text())
    expected=next(p for p in failed_intent['sources'] if Path(p['path']).name=='preload_gravity_mug_pose.py')
    snapshot=failed/'source_snapshot/preload_gravity_mug_pose.py'
    if file_pin(snapshot)['sha256']!=expected['sha256']:raise ValueError('Lost failed preload source')
    retain(snapshot,'prefinal_preload_margin.py.snapshot');retain(failed/'terminal.json','prefinal_preload_terminal.json');retain(failed/'intent.json','prefinal_preload_intent.json')
    # Verify every persisted candidate/geometry/trace used by this milestone.
    for name in ['placement','restoration']:
        for item in reports[name]['results']:
            for key in ['candidate','geometry']:verify_file_pin(item[key])
    for item in reports['equilibrium']['results']:
        for key in ['state','geometry','proposals']:verify_file_pin(item[key])
        for case in item['native_cases']:verify_file_pin(case['report']);verify_file_pin(case['trace'])
    for name in ['native','gravity_hold']:
        for case in reports[name]['cases']:verify_file_pin(case['case']);verify_file_pin(case['trace'])
    for item in reports['engagement']['results']:
        verify_file_pin(item['geometry']);raw=np.load(verify_file_pin(item['raw']))
        # The source tool checks actual joint bounds; independently check actual
        # actuator ctrlrange here as well, without editing a persisted command.
        import mujoco
        if 'model' not in locals():model=mujoco.MjModel.from_binary_path(str(ROOT/'runs/right_mug_frames_v4/hand2_mug.mjb'))
        if np.any(raw['ctrl20']<model.actuator_ctrlrange[:,0]) or np.any(raw['ctrl20']>model.actuator_ctrlrange[:,1]):raise ValueError('Persisted preload command was not legal')
    original=np.load(ROOT/'runs/right_mug_frames_v4/reference61.npz')['initial_qpos']
    previous=np.load(ROOT/'runs/mug_contact_restoration_v1/seed00/candidate.npz')['qpos27']
    current=np.load(roots['native']/'initial_state.npz')['qpos27']
    deviation=dict(description='Separate reference to q=0 assembly from previous fitted static grasp',
        q20_max_difference_from_unfitted_q0_rad=float(np.abs(current[:20]-original[:20]).max()),
        q20_max_difference_from_previous_static_grasp_rad=float(np.abs(current[:20]-previous[:20]).max()),
        object_root_delta_from_original_world_m=(current[20:23]-original[20:23]).tolist())
    (out/'pose_reference_clarification.json').write_text(json.dumps(deviation,indent=2)+'\n',encoding='utf8')
    summary=dict(schema='mug_support_milestone:1',source=file_pin(__file__),parent_commit='083c179',retained=retained,
        native_cases=20,physical_holding_passes=0,actual_integrations=33801,LP_attempts=192,linprog_invocations=176,
        source_snapshots=24,full6D_numerically_feasible=1,native_EFC_unit_columns=172,primal_witnesses_reviewed=79,
        saved_native_checks=[260,260],tests_passed=93,gravity_preload_actual_three_group_qualified=3,
        gravity_preload_full6D_feasible=0,neural_updates=0,new_checkpoints=0,phone_training_closure_changed=False,
        source_gravity_measured=False,source_trajectory_replayed=False,goal_complete=False,
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','mujoco','torch']},
        next='Wrench-conditioned contact topology/actual finger fit and separately pinned material-mode sensitivity. Preserve all source-trajectory, controlled cube/pen, WM/A1/A2, multi-trajectory, independent RGB and hardware requirements.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=file_pin(out/'summary.json'),retained_files=len(retained),native_cases=20,passed=0,tests=93)))


if __name__=='__main__':main()
