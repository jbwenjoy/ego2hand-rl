"""Retain physical projection failures and verified closed-loop input contract."""
from pathlib import Path
import importlib.metadata,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main():
    out=ROOT/'provenance/mug_native_projection'
    if out.exists():raise ValueError('Fresh retained milestone')
    roots={k:ROOT/'runs'/v for k,v in dict(projection='mug_native_projection_v2',acceleration='mug_acceleration_projection_v1',
        physical_review='mug_native_projection_review_v1',reference='mug_closed_loop_reference_v1',reference_review='mug_closed_loop_reference_review_v1').items()}
    reports={k:json.loads((v/'report.json').read_text()) for k,v in roots.items()};r=reports['physical_review'];ref=reports['reference'];rr=reports['reference_review']
    if (r['passed'],r['total'],r['trace_checks'],r['proposal_probes_checked'],r['total_unique_native_integrations'])!=(6,6,[156,156],7105,17950):raise ValueError('Physical projection readback incomplete')
    if (rr['passed'],rr['total'],ref['samples'],ref['control_intervals'],ref['time']['excluded_tail_ns'])!=(7,7,101,100,4434):raise ValueError('Reference readback incomplete')
    tests=ROOT/'runs/mug_native_reference_tests114.log';text=tests.read_text(encoding='utf8')
    if 'Ran 114 tests' not in text or '\nOK' not in text or 'FAILED' in text:raise ValueError('Full tests missing')
    frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
    if subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT,text=True).strip():raise ValueError('Frozen phone closure changed')
    old=ROOT/'runs/mug_native_projection_v1';old_intent=json.loads((old/'intent.json').read_text());alternatives={};redirects=[]
    for pin in old_intent['sources']:
        if Path(pin['path']).name in ['native_projection.py','project_mug_native_forces.py']:
            snapshot=file_pin(old/'source_snapshot'/Path(pin['path']).name)
            if snapshot['bytes']!=pin['bytes'] or snapshot['sha256']!=pin['sha256']:raise ValueError('Lost exact executed inverse source')
            alternatives[(pin['path'],pin['sha256'])]=snapshot;redirects.append(dict(original_pin=pin,executed_source_snapshot=snapshot))
    pins={}
    def check(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=value.keys():
                key=(value['path'],value['sha256']);resolved=alternatives.get(key,value)
                if key not in pins:verify_file_pin(resolved);pins[key]=dict(recorded=value,resolved=resolved)
            else:
                for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
    check(old_intent)
    for report in reports.values():check(report)
    for root in roots.values():
        if (root/'intent.json').exists():check(json.loads((root/'intent.json').read_text()))
    for row in reports['projection']['results']:check(json.loads(Path(row['solver']['path']).read_text()))
    contract=json.loads(verify_file_pin(ref['controller_contract']).read_text())
    if contract['learning_enabled'] or contract['executable_controller_and_native_observation_adapter_validated']:raise ValueError('Do not promote input contract to training integration')
    out.mkdir(parents=True);retained=[]
    def retain(path,name):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());retained.append(dict(original=file_pin(path),retained=file_pin(target)))
    for name,root in roots.items():
        retain(root/'report.json',name+'.json')
        if (root/'intent.json').exists():retain(root/'intent.json',name+'_intent.json')
    retain(old/'intent.json','interrupted/intent.json');retain(old/'terminal.json','interrupted/terminal.json')
    retain(old/'source_snapshot/native_projection.py','interrupted/native_projection.py.snapshot');retain(old/'source_snapshot/project_mug_native_forces.py','interrupted/runner.py.snapshot')
    for name in ['projection','acceleration']:
        rows=reports[name]['results']
        for i,row in enumerate(rows):
            folder=Path(row['solver']['path']).parent
            for filename in ['result.json','solver.json','proposals.jsonl']:retain(folder/filename,f'{name}/{i:02d}_{filename}')
            if row.get('tolerance_probe'):retain(verify_file_pin(row['tolerance_probe']),f'{name}/{i:02d}_tolerance_probe.json')
            for j,case in enumerate(row['native_cases']):retain(verify_file_pin(case['case']),f'{name}/{i:02d}_native{j:02d}.json')
    retain(verify_file_pin(ref['controller_contract']),'controller_contract_DRAFT.json');retain(tests,'unit_tests.log')
    (out/'verified_artifact_pins.json').write_text(json.dumps(list(pins.values()),indent=2)+'\n',encoding='utf8')
    cases=[c for name in ['projection','acceleration'] for row in reports[name]['results'] for c in row['native_cases']]
    summary=dict(schema='mug_native_projection_reference_milestone:1',source=file_pin(__file__),parent_commit='6e19de5',retained=retained,
        verified_pins=file_pin(out/'verified_artifact_pins.json'),verified_pin_count=len(pins),executed_source_redirects=redirects,
        unique_projections=6,all_final_geometry_and_candidate_region_matches=6,static_balance_passes=0,native_cases=12,holding_passes=sum(c['summary']['native_holding_gates_pass'] for c in cases),
        unique_integration_steps=17950,reused_inverse_steps=3459,new_force_projection_steps=6923,new_acceleration_steps=7568,
        objective_native_queries=7105,changed_contact_signature_columns=r['changed_signature_columns'],native_readback_states=6,
        review_forward_calls=14,review_inverse_calls=2,review_geometry_reruns=6,trace_checks=[156,156],physical_review=[6,6],
        baseline_finite_solver_probe=r['baseline_solver_tolerance_probe'],original_holding_solver_and_gates_changed=False,
        reference_source_frames=61,reference_samples=101,reference_control_intervals=100,reference_period_ns=20000000,excluded_source_tail_ns=4434,
        reference_time_stretched=False,reference_review=[7,7],reference_role=contract['dataset_role'],controller_and_observation_integration_complete=False,
        source_contact_truth_known=False,source_gravity_measured=False,full_source_trajectory_executed=False,independent_RGB_reconstruction=False,
        tests=114,neural_updates=0,new_checkpoints=0,frozen_phone_closure_changed=False,goal_complete=False,
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','mujoco','torch']},
        next='Executable bounded20D residual/EMA and native observation contract; closed-loop reference replay and new role-separated response interventions. Manual analytic static holding is not a learning prerequisite. Preserve full controlled cube/pen, WM/A1/A2/gradient, independent RGB/full trajectories, shared total cost and hardware scope; pause at first healthy new neural checkpoint.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=file_pin(out/'summary.json'),retained_files=len(retained),native_cases=12,holding_passes=0,reference_samples=101,tests=114)))


if __name__=='__main__':main()
