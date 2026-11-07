"""Verify current completed evidence and preserve the full unfinished active Goal."""
from pathlib import Path
import datetime,hashlib,json
P=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(r):assert pin(r['path'])==r,'Changed evidence '+r['path'];return Path(r['path'])
previous=P/'outputs/goal_extension_phone_WM_checks_milestone_v10.json'
for rec in read(previous)['evidence_pins'].values():checked(rec)
geo=P/'functional_tripod_support_v1/completed_v1/report.json';g=read(geo)
assert g['counts']['entries_returned']==4 and g['counts']['full_geometry_evaluations']==691
assert g['qualified_geometry_candidates']==0 and g['static_geometry_counts']['evaluations']==695
projection=P/'functional_tripod_projection_v1/completed_run1/report.json';q=read(projection)
assert q['qualified_geometry_candidates']==2 and q['counts']['full_geometry_evaluations']==27
assert q['counts']['separating_plane_point_jac_calls']==112 and q['geometric_QP_calls']==8
assert all(s['phone_pose_and_other8_x_parameters_unchanged'] and s['phase_error'] is None for s in q['summaries'])
for report in [g,q]:
    for c in report['candidates']:
        data=read(checked(c['candidate']));checked(c['state_npz'])
        assert data['geometry']['passed']==c['geometry_pass'] and data['geometry']['fresh_final_evaluation'] is True
        assert data['geometry']['phone_pair_count']==22 and data['geometry']['self_pair_count']==189
        assert data['geometry']['unknown']==[]
    assert report['counts']['integration_steps']==report['counts']['mj_forward']==report['counts']['NN_training_GPU']==0
proj_audit=P/'functional_tripod_projection_v1/saved_independent_v1/report.json';qa=read(proj_audit)
assert qa['passed']==qa['total']==10 and all(qa['checks'].values())
dynamic=P/'functional_tripod_dynamics_v1/projection_completed_run1/report.json';d=read(dynamic)
assert len(d['cases'])==12 and len(d['control_routes'])==6 and all(v['available'] for v in d['control_routes'])
assert all(c['termination']=='ABORT_LOSS_OF_HOLD' and not c['nominal_holding_dynamics_pass'] and not c['old_two_pad_pass']
    and not c['gates']['duration_complete'] and c['metrics']['tail_triple_pad_force_coverage'] is None for c in d['cases'])
assert d['counts']==dict(MJB_reads=1,MjData=14,preload_preparation_forward=2,preload_point_jacobians=6,
    initial_forward=12,integration_steps=6845,contact_force_reads=6136,neural_training_GPU=0)
assert sum(c['integration_steps'] for c in d['cases'])==6845
for c in d['cases']:checked(c['trace'])
dyn_audit=P/'functional_tripod_dynamics_v1/projection_saved_review_v1/report_v2.json';da=read(dyn_audit)
assert da['passed']==da['total']==14 and all(da['checks'].values()) and da['holding_qualified'] is False
momentum=P/'phone_momentum_contract_v1/completed_run2/report.json';m=read(momentum)
assert m['passed']==m['total']==14 and m['rows']==684 and all(m['checks'].values())
assert m['counts']['NN_parameter_updates']==m['counts']['native_model_reads']==m['counts']['physics_steps']==0
checked(m['outputs']['TRAIN684_closure'])
causal=P/'phone_causal_train_design_v1/generated_v1/report.json';c=read(causal)
assert c['contexts']==72 and c['available_native_sides']==2870 and c['unavailable_sides']==10
assert c['strict_zero_mask_true_count']==0 and c['fresh_final_families']==12 and c['exposure_policy']['fresh_final_not_acquired_or_fit'] is True
assert c['execution_counts']['new_models_FK_native_steps_NN_AD_training']==0
for rec in c['outputs'].values():checked(rec)
cv=read(P/'phone_causal_train_design_v1/validation_v1.json')
assert all(cv['checks'].values()) and cv['planned_executable_branches']==2942 and cv['planned_executable_steps']==282432
terminals={
    'bounded_geometry':P/'functional_tripod_support_v1/process_geometry_run1/terminal.json',
    'projection':P/'functional_tripod_projection_v1/process_run1/terminal.json',
    'actual_native_dynamics':P/'functional_tripod_dynamics_v1/process_projection_run1/terminal.json',
    'momentum_corrected':P/'phone_momentum_contract_v1/process_run2/terminal.json',
}
for path in terminals.values():
    t=read(path);assert t['returncode']==0
    for k in ['source','launcher','protocol','stdout','stderr']:checked(t[k])
paths=dict(previous_handoff=previous,plan=Path(r'D:\workspace\supermimic-chord-study\docs\research-plan-v0.30.md'),
    bounded_three_tip_geometry=geo,bounded_geometry_saved_arithmetic=P/'functional_tripod_support_v1/saved_arithmetic_diagnostics_v1/report.json',
    plane_helper_contract=P/'functional_tripod_support_v1/refinement_v1/SOURCE_CONTRACT_v1.json',
    geometric_projection=projection,geometric_projection_saved_audit=proj_audit,
    dynamics=dynamic,dynamics_saved_audit=dyn_audit,
    dynamics_audit_correction=P/'functional_tripod_dynamics_v1/projection_saved_review_v1/correction_v2.json',
    dynamics_audit_manifest=P/'functional_tripod_dynamics_v1/projection_saved_review_v1/final_manifest_v2.json',
    dynamics_source_preservation=P/'functional_tripod_dynamics_v1/source_preservation_v3.json',
    geometry_only_metadata_screen=P/'functional_tripod_dynamics_v1/actual_geometry_metadata_v3/report.json',
    projection_metadata_screen=P/'functional_tripod_dynamics_v1/projection_metadata_v3/report.json',
    saved_dynamics_description=P/'functional_tripod_dynamics_v1/saved_description_v1/report.json',
    saved_COM_figure=P/'functional_tripod_dynamics_v1/saved_description_v1/COM_abort_prefixes.png',
    saved_native_force_figure=P/'functional_tripod_dynamics_v1/saved_description_v1/native_PRE_forces.png',
    pose_visual_report=P/'functional_tripod_visual_v1/completed_v1/report.json',
    pose_visual_figure=P/'functional_tripod_visual_v1/completed_v1/saved_pose_comparison.png',
    momentum=momentum,momentum_core=P/'phone_momentum_contract_v1/momentum_contract.py',
    momentum_README=P/'phone_momentum_contract_v1/README.md',
    momentum_serialization_correction=P/'phone_momentum_contract_v1/serialization_correction_v1.json',
    momentum_source_review=P/'phone_causal_train_design_v1/momentum_source_review_v1.json',
    causal_design=causal,causal_design_validation=P/'phone_causal_train_design_v1/validation_v1.json',
    causal_delivery=P/'phone_causal_train_design_v1/delivery_v1.json',
    causal_README=P/'phone_causal_train_design_v1/README.md',
    upstream_interface_map=P/'phone_causal_train_design_v1/upstream_interface_map_v1.json')
paths.update({name+'_terminal':path for name,path in terminals.items()})
record=dict(schema='tripod_geometry_native_failure_and_momentum_goal_milestone:11',
    created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    previous_goal_turn_classification='progress',this_goal_turn_classification='progress',
    native_goal_status_last_verified='active',original_full_goal_complete=False,
    original_two_pad_results_and_success_gates_preserved=True,
    actual_new_native_integration_steps=6845,actual_new_native_preparation_and_initial_forward=14,
    geometric_candidates_all_retained=4,three_tip_geometric_candidates_qualified=2,
    native_control_routes_available=6,actual_dynamic_cases=12,new_successful_2s_holding_cases=0,
    native_geometric_qualification_is_actual_loaded_support=False,
    both_dt_abort_prefix_similarity_is_2s_consistency=False,
    new_control_or_CHORD_policy_benefit_proven=False,
    new_neural_parameter_training_started=False,new_neural_parameter_updates=0,
    new_neural_model_constructors=0,new_input_impulse_AD_Jacobians=2,
    momentum_interface_verified_eligible_TRAIN_rows=684,momentum_fixture_and_saved_checks_passed=14,
    all960_published_wrench_labels_contain_already_exposed_TEST=True,
    new_intervention_metadata_contexts=72,new_intervention_metadata_executable_branches=2942,
    new_intervention_branches_actually_acquired=0,new_final_families_predeclared_unacquired=12,
    strict_zero_causal_masks_issued=0,
    real_ego_generalization_or_calibrated_materials_or_hardware_contract_complete=False,
    standing_first_new_training_healthy_checkpoint_and_ETA_pause_triggered_this_turn=False,
    all_actual_numerical_executions_observed_terminal=True,visual_pose_and_two_dynamics_figures_inspected=True,
    preserved_implementation_failures=['Momentum run1 NumPy-bool JSON serialization, numerical output retained; corrected run2 formulas unchanged.',
        'Dynamic metadata v2 duplicate candidate field, no native import/physics; corrected v3 fixtures retained.',
        'Saved dynamic audit v1 batch quaternion reduction near acos1 differed by1.23e-10rad; v2 same-row observer replay, no threshold/data/physics changes.'],
    next_action='Build named native loaded-entry for all3 designated pads using compression-target geometry within unchanged hard windows, actual con.pos/frame Jacobians, pad-only COM wrench, real ctrl/tau/no-nonpad/table/self constraints. Initial load is not 2s holding. Diagnose signed load directions and persistent contact/wrench support; current Fn targets/material anchors/remaining torque margin do not prove correct actual force directions or sustained hold. Then original2s/twodt noaid, explicit simulator-observation feedback baseline if needed, and executable reference before3pad data/geometry-contact causal model/Actor/CHORD same-budget integration. Preserve full61frame/sourcegravity/material/hardware/independentego scope and pause only on a future new learning run after first healthy checkpoint and ETA.',
    evidence_pins={name:pin(path) for name,path in paths.items()},finalizer=pin(__file__))
out=P/'outputs/goal_extension_tripod_momentum_milestone_v11.json'
with out.open('x',encoding='utf8') as f:json.dump(record,f,indent=2,allow_nan=False)
print(json.dumps(dict(output=pin(out),evidence_pins=len(paths),new_training=False,full_goal_complete=False,goal_remains_active=True)))
