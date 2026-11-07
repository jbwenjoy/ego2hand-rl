"""Retain gravity targets, triplet hypotheses and failed native validations."""
from pathlib import Path
import importlib.metadata,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin


def main():
    out=ROOT/'provenance/mug_support_selection'
    if out.exists():raise ValueError('Fresh retained milestone')
    roots={k:ROOT/'runs'/v for k,v in dict(envelopes='mug_gravity_envelopes_v1',selection='mug_support_triplets_v1',
        fit='mug_support_fit_v1',review='mug_support_selection_review_v1',figures='mug_support_selection_figures_v1').items()}
    reports={k:json.loads((v/'report.json').read_text()) for k,v in roots.items()};review=reports['review'];fit=reports['fit'];selection=reports['selection']
    if (review['passed'],review['total'],review['triplet_primals'],review['native_EFC_columns'],review['trace_checks'])!=(6,6,710,52,[52,52]):raise ValueError('Saved review missing')
    if (selection['LP_calls'],len(selection['selected']),fit['reach_solves'],fit['coupled_seeds'],fit['geometry_pass_count'],fit['native_eligible_count'],fit['native_cases'],fit['integration_steps'])!=(4448,4,72,4,2,1,4,6400):raise ValueError('Study scope mismatch')
    tests=ROOT/'runs/mug_support_selected_tests.log';text=tests.read_text(encoding='utf8')
    if 'Ran 108 tests' not in text or '\nOK' not in text or 'FAILED' in text:raise ValueError('Test suite missing')
    frozen=['src/supermimic_research/runtime','src/supermimic_research/learning','src/chord_study','src/dex_supermimic','configs','archive']
    if subprocess.check_output(['git','diff','--name-only','9b38f97','--',*frozen],cwd=ROOT,text=True).strip():raise ValueError('Frozen phone source/config/archive changed')
    pins={}
    def check(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=value.keys():
                key=(value['path'],value['sha256'])
                if key not in pins:verify_file_pin(value);pins[key]=value
            else:
                for v in value.values():check(v)
        elif isinstance(value,list):
            for v in value:check(v)
    for report in reports.values():check(report)
    for root in roots.values():
        if (root/'intent.json').exists():check(json.loads((root/'intent.json').read_text()))
    check(json.loads((ROOT/'runs/mug_source_patches_v2/report.json').read_text()))
    out.mkdir(parents=True);retained=[]
    def retain(path,name):
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(Path(path).read_bytes());retained.append(dict(original=file_pin(path),retained=file_pin(target)))
    for name,root in roots.items():
        retain(root/'report.json',name+'.json')
        if (root/'intent.json').exists():retain(root/'intent.json',name+'_intent.json')
    for i,row in enumerate(fit['results']):
        folder=Path(row['candidate']['path']).parent
        for name in ['result.json','solver.json','reach.json','coupled_proposals.jsonl']:retain(folder/name,f'fit/{i:02d}_{name}')
        if (folder/'preload.json').exists():retain(folder/'preload.json',f'fit/{i:02d}_preload.json')
        for j,case in enumerate(row['native_cases']):retain(verify_file_pin(case['case']),f'fit/{i:02d}_native{j:02d}.json')
    retain(tests,'unit_tests.log');retain(roots['figures']/'mug_support_selection.png','mug_support_selection.png')
    closure=sorted(pins.values(),key=lambda p:(p['path'],p['sha256']))
    (out/'verified_artifact_pins.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf8')
    cases=[c for row in fit['results'] for c in row['native_cases']]
    summary=dict(schema='mug_support_selection_milestone:1',source=file_pin(__file__),parent_commit='cb1933d',retained=retained,
        verified_pins=file_pin(out/'verified_artifact_pins.json'),verified_pin_count=len(closure),
        gravity_angle_deg=reports['envelopes']['gravity_angle_deg'],gravity_feasibility_LP_attempts=36,gravity_separation_LP_attempts=6,
        gravity_source_measured=False,simulator_gravity_changed=False,triplet_LP_calls=4448,triples_per_hypothesis={'all30':896,'nonnegative21':216},
        primary_robot_gravity_feasible={'all30':130,'nonnegative21':28},selected_triplets=4,reach_solves=72,coupled_searches=4,
        geometry_and_actual_source_candidate_matches=2,native_cone_and_strict_actuator_feasible=1,
        native_cases=4,integration_steps=6400,complete_holding_passes=sum(c['summary']['native_holding_gates_pass'] for c in cases),
        native_durations_s=[c['summary']['actual_duration_s'] for c in cases],envelope_primals_reviewed=27,triplet_primals_reviewed=710,
        separators_reviewed=6,native_EFC_columns_reviewed=52,native_primals_reviewed=2,trace_checks=[52,52],saved_review=[6,6],
        distinct_primary_geometry_audits=6,distinct_review_geometry_audits=6,tests=108,
        source_contact_truth_known=False,independent_RGB_reconstruction=False,full_source_trajectory_replayed=False,
        neural_updates=0,new_checkpoints=0,frozen_phone_closure_changed=False,goal_complete=False,
        source_uncertainty='The native6D positive static case includes candidate13, local skin/face offset-0.389mm; nonnegative-offset branch does not pass native6D. No human-force/contact-truth validation.',
        startup_error='Before intent or output was created, np.load was once given a pin dict rather than its verified path. Corrected before experiment; input bytes unchanged.',
        versions={k:importlib.metadata.version(k) for k in ['numpy','scipy','mujoco','torch']},
        next='Actual native force/compliance, hand-equilibrium and controller-response consistency at the new6D-qualified contact configuration. Preserve full cube/pen, WM/A1/A2/gradients, ego/RGB/trajectory, total cost and hardware objective.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(dict(summary=file_pin(out/'summary.json'),retained_files=len(retained),pins=len(closure),native_cases=4,holding_passes=summary['complete_holding_passes'],tests=108)))


if __name__=='__main__':main()
