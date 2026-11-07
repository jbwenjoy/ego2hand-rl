"""Freeze the reviewable research handoff before requesting Goal pause."""
from pathlib import Path
import datetime,hashlib,json
P=Path(r'D:\workspace\ego-data-pilot-2026-11-07'); M=Path(r'D:\workspace\supermimic-chord-study')
W=P/'phone_geometry_momentum_wm_v1'; R=W/'cpu_run1'
def pin(p):
    p=Path(p).resolve(strict=True); b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
health=read(R/'health_0250.json'); result=read(R/'result.json')
assert health['passed']==health['total']==15 and result['iteration']==250 and health['process_stops_here'] is True
refs=dict(previous_handoff=P/'outputs/goal_extension_tripod_momentum_milestone_v11.json',
    plan=M/'docs/research-plan-v0.31.md',study_index=M/'README.md',
    this_turn_evidence=P/'outputs/goal_extension_tripod_momentum_evidence_v12.json',
    training_source=W/'train_geometry_momentum_wm.py',training_recipe=W/'recipe_v1.json',training_delivery=W/'delivery_v1.json',
    training_SOURCE_review=W/'independent_source_review_v1/report.json',training_protocol=W/'protocol_CPUTrue_run1.json',training_launcher=W/'execute_once_v1.py',
    training_initial=R/'fresh_initial_state.pt',training_checkpoint=R/'checkpoint_0250.pt',training_health=R/'health_0250.json',training_readback=R/'health_readback_0250.npz',
    training_result=R/'result.json',training_log=R/'updates.jsonl',training_terminal=W/'process_run1/terminal.json',
    training_saved_review=W/'saved_run_review_v1/report.json',training_payload_review=W/'root_saved_payload_review_v1/report.json',
    freezer_source=Path(__file__))
pins={k:pin(v) for k,v in refs.items()}
indexed=read(refs['this_turn_evidence'])['files']
assert len(indexed)==148 and all(pin(v['path'])==v for v in indexed.values())
for label in ['training_saved_review','training_payload_review']:
    review=read(refs[label]); assert review['passed']==review['total'] and all(review['checks'].values()),label
assert pins['training_checkpoint']==health['checkpoint']==result['checkpoint']
assert read(refs['training_terminal'])['returncode']==0
value=dict(schema='tripod_loaded_contact_causal_geometry_momentum_WM_goal_milestone:12',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    previous_goal_turn_classification='progress',this_goal_turn_classification='progress',original_full_goal_complete=False,
    native_goal_status_last_verified='active',goal_pause_due=True,pause_reason='Standing user instruction: new training first genuine healthy checkpoint and ETA now available. Root requests native Goal paused immediately after this archive; this file does not assert the API has already returned paused.',
    original_physics_success_gates_and_old_results_preserved=True,
    new_native_loaded_conditions=2,new_static_preload_forward=62,new_static_preload_contact_Jacobians=186,
    new_dynamic_cases=16,new_dynamic_native_steps=9200,new_dynamic_initial_forward=16,new_successful_2s_holding_cases=0,
    dynamic_outcome='All16 ABORT_LOSS_OF_HOLD; exit-prefix112.1875..125ms, not stable duration or2s dt consistency. Initial3pad loads are now real; sustained designated wrench/control unresolved.',
    causal_native_branches_plan=2952,causal_native_branches_executed=2942,causal_native_branches_unavailable=10,causal_native_steps=282432,
    all72_original_TRAIN_baselines_bitmatched_before_any_side=True,causal_new_action_CHECK_is_unseen_state=False,fresh12_families_acquired=False,
    original_TRAIN_complete_absolute_rows=684,causal_FIT_available_sides_added_once=1434,absolute_training_rows=2118,CF_training_pairs=1434,
    current_geometry_features_and_momentum_decode_implemented=True,finite_direction_descriptor_is_exact_distance=False,structured_momentum_is_full_contact_pose_consistency=False,
    real_ego_reference_frames=61,real_ego_reference_kind='HOT3D annotated wrist-relative object-motion intent, not image reconstruction, robot trajectory execution or human-force labels',
    new_neural_parameter_training_started=True,new_neural_parameter_updates=250,new_neural_target_updates=5000,
    neural_worker_counts=result['counts'],extra_Root_CPU_own_payload_loads=1,extra_Root_NN_or_native_calls=0,
    checkpoint_health_passed=15,checkpoint_health_total=15,remaining_update_only_ETA_s=health['estimated_remaining_s'],resume_setup_heuristic_s=health['estimated_resume_setup_s'],
    resume_end_to_end_observed_rate_scenarios_s=health['estimated_resume_end_to_end_s_rate_scenarios'],
    first_new_training_healthy_checkpoint_and_ETA_pause_triggered_this_turn=True,worker_terminal_exit0=True,background_training_continues=False,
    native_gradient_or_Actor_or_CHORD_or_real_ego_benefit_qualified=False,materials_calibrated_or_hardware_contract_complete=False,
    full_goal_remaining=['Resume this own checkpoint at251 through fixed5000; same recipe/source/data/normalizers/Adam/RNG.',
        'Known-state new-action response CHECK, own AD/FD chart checks and native finite-response comparison; collect predeclared fresh families without relabeling exposed TEST as unseen.',
        'Integrate verified current geometry, structured WM, replay/Actor/reward and object-frame CHORD support; matched budgets and ablations. Manual2s controller success is not a learning prerequisite.',
        'Original2s/two-dt/noaid task evaluation and full61-frame robot retargeting; source gravity, geometry/dynamics uncertainty/calibration, independent ego, hardware observability and deployment remain.'],
    independently_rechecked_indexed_file_pins=len(indexed),evidence_pins=pins)
out=P/'outputs/goal_extension_tripod_momentum_milestone_v12.json'
with out.open('x',encoding='utf8') as f: json.dump(value,f,indent=2,allow_nan=False)
assert all(pin(v['path'])==v for v in pins.values())
print(json.dumps(dict(milestone=pin(out),evidence_pins=len(pins),original_full_goal_complete=False,pause_due=True)))
