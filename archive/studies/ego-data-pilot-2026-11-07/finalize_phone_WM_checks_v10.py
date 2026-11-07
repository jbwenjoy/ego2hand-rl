"""Finalize verified evaluation/response evidence; retain full unfinished Goal."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np

P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):assert pin(r['path'])==r,'Changed evidence '+r['path'];return Path(r['path'])

previous=P/'outputs/goal_extension_phone_WM_first_checkpoint_v9.json'
for r in read(previous)['evidence_pins'].values():checked(r)
evaluation=P/'phone_world_model_evaluation_v1/completed_v1/report.json';ev=read(evaluation)
assert ev['checkpoint']['sha256']=='f2e51432377bcd68b235b0f5dbeb6953a3a411f586e008e0ee4a08111a3b1f1f'
checked(ev['checkpoint'])
restore=P/'phone_action_gradient_v2/independent_restore_review_v1/actual_fixture_v1/report.json';rc=read(restore)
assert rc['passed']==rc['total']==4 and rc['counts']['integration_steps']==384 and all(rc['checks'].values())
native=P/'phone_action_gradient_v2/native_completed_v1/report.json';na=read(native)
assert na['states']==12 and na['new_action_branches']==476 and na['partial_branches']==0
assert na['counts']['integration_steps']==46848 and na['counts']['initial_forward']==na['counts']['observer_forward']==0
native_audit=P/'phone_action_gradient_v2/native_saved_review_v1/report.json';nar=read(native_audit)
assert nar['engineering_qualified'] is True and nar['matched_context_data_qualified'] is True
assert nar['protocol_conformance'] is False and nar['protocol_order_all12_baselines_before_any_new_side'] is False
assert nar['passed']==12 and nar['total']==13
comparison=P/'phone_action_gradient_v2/comparison_completed_v1/report.json';co=read(comparison)
assert co['counts']==dict(checkpoint_loads=1,model_constructors=1,NN_forward=500,action_AD_jacobians=12,
    NN_parameter_updates=0,native_steps=0,TEST_arrays_read=0)
comparison_audit=P/'phone_action_gradient_v2/independent_comparison_saved_review_v1/report.json';ca=read(comparison_audit)
assert ca['passed']==ca['total']==15 and all(ca['checks'].values())
with np.load(checked(co['array_output']),allow_pickle=False) as z:
    joints=z['joint'];sec=z['native_secant_F64'];pred=z['model_secant_F64_denominator']
    secondary=joints>=8;native_COM=np.linalg.norm(sec[:,40:43],axis=1);model_COM=np.linalg.norm(pred[:,40:43],axis=1)
assert secondary.sum()==288 and np.sum(native_COM[secondary]==0)==233
wrench=P/'phone_object_wrench_labels_v1/completed_v1/report.json';wr=read(wrench)
support=P/'phone_object_wrench_labels_v1/support_target_v2/completed_v2/report.json';su=read(support)
assert wr['rows']==su['rows']==960 and wr['complete_H_rows']==su['complete_rows']==912
assert wr['actual_partial_prefix_rows']==su['partial_rows']==48 and wr['native_PRE_ticks']==89679
assert wr['model_native_FK_NN_AD_physics_new_training_calls']==0
assert su['counts']['native_NN_newphysics']==0
for r in [*wr['outputs'].values(),*su['outputs'].values()]:checked(r)
terminals={
 'evaluation':P/'phone_world_model_evaluation_v1/process_v1/terminal.json',
 'native_branches':P/'phone_action_gradient_v2/process_native_run1/terminal.json',
 'model_native_comparator':P/'phone_action_gradient_v2/process_comparator_run1/terminal.json',
 'wrench_labels':P/'phone_object_wrench_labels_v1/process_v1/terminal.json',
 'support_labels_corrected':P/'phone_object_wrench_labels_v1/support_target_v2/process_v2/terminal.json',
}
for path in terminals.values():
    t=read(path);assert t['returncode']==0
    launch_key='runner' if 'runner' in t else 'launcher'
    for k in ['source',launch_key,'protocol','stdout','stderr']:checked(t[k])
paths={
 'plan':Path(r'D:\workspace\supermimic-chord-study\docs\research-plan-v0.29.md'),
 'previous_handoff':previous,'fixed5000_evaluation':evaluation,
 'evaluation_description':P/'phone_world_model_evaluation_v1/saved_description_v1/report.json',
 'restore_actual_contract':restore,'native_response_data':native,'native_saved_audit':native_audit,
 'primary8_design':P/'phone_action_gradient_v2/design_v1/design_contract_v1.json',
 'all20_addendum':P/'phone_action_gradient_v2/design_v2_all20/design_addendum_v2_all20.json',
 'all20_selection':P/'phone_action_gradient_v2/design_v2_all20/selection_v2_all20.json',
 'model_native_comparison':comparison,'model_native_comparison_audit':comparison_audit,
 'response_description':P/'phone_action_gradient_v2/saved_description_v1/report.json',
 'response_figure':P/'phone_action_gradient_v2/saved_description_v1/COM_response_magnitudes_v1.png',
 'wrench_labels':wrench,'wrench_label_interface':P/'phone_object_wrench_labels_v1/label_interface_v1.json',
 'static_pad_only_target_labels':support,
 'support_source_correction':P/'phone_object_wrench_labels_v1/support_target_v2/execution_correction_receipt_v3.json',
 'support_label_API':P/'phone_object_wrench_labels_v1/support_target_v2/README.md',
 'three_digit_functional_review':P/'functional_support_review_v1/memo_v1.json',
}
paths.update({f'{k}_terminal':v for k,v in terminals.items()})
record=dict(schema='phone_WM_held_family_and_full20_response_goal_milestone:10',
    created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    previous_goal_turn_classification='progress',this_goal_turn_classification='progress',
    native_goal_status_last_verified='active',original_full_goal_complete=False,
    new_learning_training_started=False,new_neural_parameter_updates=0,
    standing_first_new_training_checkpoint_pause_triggered_this_turn=False,
    fixed_evaluated_model_checkpoint_iteration=5000,
    reserved_TEST_eligible_rows_first_evaluated=114,reserved_TEST_families=6,
    TEST_is_still_unseen_for_future_model_development=False,
    new_native_steps=384+46848,new_native_model_reads=4,new_native_forward=0,new_native_LP_calls=0,
    new_native_matched_baselines=12,new_native_finite_action_sides=476,
    model_action_AD_contexts=12,model_action_Jacobian_shape=[12,59,20],
    new_holding_qualification=False,new_SuperMimic_or_CHORD_policy_benefit_proven=False,
    native_response_numerical_data_engineering_qualified=True,
    strict_native_experiment_order_protocol_conformance=False,
    protocol_order_deviation_preserved='Per-state baseline before own perturbations; all12 baselines ultimately matched. Design all12-first requirement was not followed.',
    secondary12_sides=288,secondary12_exact_zero_native_endpoint_COM_response=233,
    secondary12_model_COM_response_norm_median_m_per_rad=float(np.median(model_COM[secondary])),
    model_internal_AD_agreement_is_physics_gradient_qualification=False,
    object_frame_contact_wrench_label_rows=960,complete_wrench_rows=912,partial_wrench_rows=48,
    observed_native_wrenches_are_real_force_ground_truth=False,
    static_support_target_is_new_holding_admission_gate=False,
    real_materials_or_functional_pad_or_source_gravity_calibrated=False,
    three_digit_functional_support_proposal_added=True,three_digit_physical_feasibility_qualified=False,
    all_current_actual_numerical_executions_observed_terminal=True,
    saved_support_export_first_schema_failure_preserved=True,
    final_response_figure_visually_inspected=True,
    next_action='Pursue native three-designated-tip functional-support feasibility with real middle q8:12 freedom/material triangles, original2pad results retained, strict2s/bothdt/noexternal/table/non-designated-link gates and explicit new3pad coverage. Learning side: restore geometry/contact features and task/Actor/CHORD semantics, causal TRAIN same-state intervention plus contact-wrench/momentum structured residuals. Keep current476 VAL responses out of TRAIN and use fresh final-check families after TEST exposure. Do not equate dense80 baseline/internally correct AD with physical control benefit; preserve full61frame/material/hardware scope. Any future new learning run still pauses after first healthy checkpoint and ETA.',
    evidence_pins={k:pin(v) for k,v in paths.items()},finalizer=pin(__file__))
out=P/'outputs/goal_extension_phone_WM_checks_milestone_v10.json'
with out.open('x',encoding='utf8') as f:json.dump(record,f,indent=2,allow_nan=False)
print(json.dumps(dict(output=pin(out),evidence_pins=len(record['evidence_pins']),
    new_native_steps=record['new_native_steps'],new_training=False,goal_remains_active=True)))
