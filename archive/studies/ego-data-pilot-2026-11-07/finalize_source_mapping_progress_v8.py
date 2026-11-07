"""Saved-evidence goal handoff, not completion or a new physics execution."""
from pathlib import Path
import datetime,hashlib,json

P=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def checked(r):
    assert pin(r['path'])==r,'Changed evidence '+r['path'];return Path(r['path'])

oldpath=P/'outputs/goal_extension_assembly_milestone_v7.json'
old=read(oldpath)
for r in old['evidence_pins'].values():checked(r)
branches=['exact_source_original_hand','source1mm_opposed_side_original_hand']
reports=[];finals=[];attempts=returned=0
for b in branches:
    h=P/'surface_assembly_v11'/f'{b}_completed_v1';r=read(h/'report.json')
    assert r['anchor_branch']==b and r['new_steps_training_GPU']==0
    assert r['model_audit']['passed'] is True and r['cheap_model_audit']['passed'] is True
    assert r['holding_diagnostic_candidates']==0
    for s in r['seed_results']:
        attempts+=s['full_attempts'];returned+=s['full_returned']
        for rec in s['finals']:
            v=read(checked(rec['result']))
            assert v['candidate']['assembly_gate'] and v['final_geometry']['geometry_gate_pass']
            assert max(abs(x) for x in v['candidate']['eq10_scaled'])<=1e-5
            assert v['strict_native_support_LP']['solver_status']==2
            assert not v['ideal_native_support_candidate'] and not v['holding_diagnostic_eligible']
            finals.append(rec['result'])
    reports.append(r)
dynpath=P/'hand2_source_mapping_dynamics_v11/cpu_nominal_v1/report.json';dyn=read(dynpath)
assert dyn['status']=='COMPLETED_SOURCE_MAPPING_CONSTANT_U_DYNAMIC_DIAGNOSTIC'
assert len(dyn['cases'])==4 and dyn['counts']['integration_steps']==2959
assert sum(c['integration_steps'] for c in dyn['cases'])==2959
for c in dyn['cases']:
    checked(c['trace']);checked(c['source_candidate_state'])
    assert c['termination']=='ABORT_LOSS_OF_HOLD' and c['nominal_holding_dynamics_pass'] is False
    assert not c['gates']['duration_complete'] and c['metrics']['warning_count']==0
    assert c['metrics']['tail_linear_velocity_RMS_m_s'] is None
assert all(not c['dt_pair_consistency_pass'] for c in dyn['dt_pair_consistency'])
for rel,n in [('surface_assembly_v11/terminal_saved_review_v1/report_v2.json',11),
              ('hand2_source_mapping_dynamics_v11/saved_review_v1/report.json',11),
              ('phone_action_secants_v1/actual_step_secants_v2/report.json',7)]:
    proof=read(P/rel)
    assert proof['passed']==n and proof['total']==n and all(v is True for v in proof['checks'].values())
wm=read(P/'phone_world_model_v2/contract_v2.json')
assert wm['passed'] is True and len(wm['cases'])==3
terminals=[P/'surface_assembly_v11/process_exact_source_v1/terminal.json',
           P/'surface_assembly_v11/process_source1mm_v1/terminal.json',
           P/'hand2_source_mapping_dynamics_v11/process_run1/terminal.json']
for p in terminals:
    t=read(p);assert t['returncode']==0 and t['stderr']['bytes']==0
    for k in ('source','runner','protocol','stdout','stderr'):checked(t[k])
paths={
 'plan':Path(r'D:\workspace\supermimic-chord-study\docs\research-plan-v0.27.md'),
 'previous_handoff':oldpath,
 'source_mapping_build':P/'source_surface_mapping_v11/report_v1.json',
 'source_mapping_preregistration':P/'source_surface_mapping_v11/preregistration_v1.json',
 'source_mapping_saved_independent_review':P/'source_surface_mapping_v11/audit_saved_mapping_v1/report.json',
 'source_region_catalog':P/'source_surface_mapping_v11/source_phone_region_catalog_v1.json',
 'exact_source_anchors':P/'source_surface_mapping_v11/exact_source_original_hand_anchors_v1.json',
 'sensitivity_anchors':P/'source_surface_mapping_v11/source1mm_opposed_side_original_hand_anchors_v1.json',
 'source_region_figure':P/'source_surface_mapping_v11/source_regions_v1.png',
 'source_region_figure_generator':P/'source_surface_mapping_v11/plot_source_regions_v1.py',
 'solver_derivation':P/'surface_assembly_v11/source_derivation_v11.json',
 'root_execution_review':P/'surface_assembly_v11/root_execution_review_v1.json',
 'solver_saved_review':P/'surface_assembly_v11/terminal_saved_review_v1/report_v2.json',
 'exact_search_report':P/'surface_assembly_v11/exact_source_original_hand_completed_v1/report.json',
 'sensitivity_search_report':P/'surface_assembly_v11/source1mm_opposed_side_original_hand_completed_v1/report.json',
 'dynamic_result':dynpath,
 'dynamic_protocol':P/'hand2_source_mapping_dynamics_v11/protocol_CPUTrue_v1.json',
 'dynamic_source_derivation':P/'hand2_source_mapping_dynamics_v11/source_derivation_v11.json',
 'dynamic_source_preservation':P/'hand2_source_mapping_dynamics_v11/source_preservation_v11.json',
 'dynamic_source_independent_review':P/'hand2_source_mapping_dynamics_v11_review_v1/report.json',
 'dynamic_saved_review':P/'hand2_source_mapping_dynamics_v11/saved_review_v1/report.json',
 'phone_WM_interface':P/'phone_world_model_v2/report.json',
 'phone_WM_source_binding':P/'phone_world_model_v2/FINAL_SOURCE_BINDING.json',
 'phone_WM_numeric_contract':P/'phone_world_model_v2/contract_v2.json',
 'phone_action_secants':P/'phone_action_secants_v1/actual_step_secants_v2/report.json',
}
paths.update({f'execution_terminal_{i}':p for i,p in enumerate(terminals)})
paths.update({f'final_candidate_{i}':r['path'] for i,r in enumerate(finals)})
summary=dict(schema='ego_source_region_mapping_goal_extension_milestone:8',
 created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 previous_goal_turn_classification='progress',this_goal_turn_classification='progress',
 status='PROGRESS_SOURCE_GEOMETRY_IMPROVED_DYNAMIC_HOLDING_FAILED',native_goal_status_last_verified='active',
 original_full_goal_complete=False,new_learning_training_started=False,
 new_training_first_healthy_checkpoint_pause_condition_triggered=False,
 new_native_MJB_reads=sum(r['counts']['MJB_reads'] for r in reports)+dyn['counts']['MJB_reads'],
 new_native_forward_calls=sum(r['counts']['mj_forward_returned'] for r in reports)+dyn['counts']['initial_forward'],
 new_LP_calls=sum(r['LP_calls'] for r in reports),new_strict_box_LP_calls=len(finals),
 new_static_FK_calls=sum(r['static_counts']['evaluations'] for r in reports),
 new_cheap_FK_calls=sum(r['cheap_FK_counts']['query_returned'] for r in reports),
 full_search_attempts=attempts,full_search_returned=returned,
 geometry_and_hard_assembly_valid_final_records=len(finals),records_are_independent_unique_grasps=False,
 ideal_strict_support_candidates=0,new_native_holding_diagnostic_cases=4,
 new_native_integration_steps=dyn['counts']['integration_steps'],new_holding_pass_cases=0,
 new_2s_dt_consistency_pass_pairs=0,dynamic_case_exit_times_are_stable_holding_durations=False,
 phone_WM_interface_contract_qualified=True,phone_WM_trained_or_accurate=False,
 saved_action_secants_actual_step=112,saved_action_secants_nominal_point01rad=110,
 saved_action_secants_actuator_coverage=list(range(8)),full20_action_or_native_physics_Jacobian_qualified=False,
 old322_transition_samples_alias_connected_prefix_groups=1,independent_ego_validation_qualified=False,
 phone_state58_Markov_or_full_solver_context_qualified=False,
 original_physical_model_wrist_scale_gains_material_solver_preserved=True,
 dynamic_dt_is_only_explicit_model_option_variant=True,
 functional_facets_compliance_friction_mass_inertia_calibrated=False,
 source_GT_gravity_registration_verified=False,
 all_known_execution_processes_observed_terminal=True,
 figure_final_PNG_visually_inspected=True,
 first_pad_threshold_event_ms_by_branch=[2.5,8.4375],
 first_pad_loss_precedes20ms_control_boundary=True,
 next_action='Implement actual phone H20ms Reader/state58/contact PRE timing and independent simulation episode acquisition, grouped across dt/clones. Preserve per-substep PRE origins and interval pad minimum/coverage/first-event labels, not just endpoint force. Integrate new phone physical-delta52 prediction/normalization/loss, initial-action anticipation/contact recovery, and finite-action secant checks. No repeat of the same static search as main work; preserve full SuperMimic/CHORD, two-second/two-dt/material, trajectory/ego and hardware scope. Any new learning run pauses only after first healthy checkpoint and ETA as requested.',
 evidence_pins={k:pin(v) for k,v in paths.items()},finalizer=pin(__file__))
dest=P/'outputs/goal_extension_source_mapping_milestone_v8.json'
with dest.open('x',encoding='utf8') as f:json.dump(summary,f,indent=2,allow_nan=False)
print(json.dumps(dict(output=pin(dest),evidence_pins=len(summary['evidence_pins']),
                     integration_steps=summary['new_native_integration_steps'],training_started=False)))
