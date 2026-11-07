"""Pin a progress handoff after this bounded native-contact experiment batch."""
from pathlib import Path
from datetime import datetime,timezone
import json
import hashlib

P=Path(__file__).resolve().parent
M=P.parent/'supermimic-chord-study'


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def main():
    files={
      'plan':M/'docs/research-plan-v0.24.md',
      'previous_handoff':P/'outputs/goal_extension_support_milestone_v4.json',
      'native_forward_source':P/'native_equilibrium_v6/evaluator_native_v6.py',
      'cold_contract':P/'native_equilibrium_v6/four_forward_contract_v1/report.json',
      'preload':P/'native_equilibrium_v6/preload_v1/report.json',
      'native_search':P/'native_equilibrium_v6/completed_v1/report.json',
      'directional_derivatives':P/'native_equilibrium_v6/directional_derivative_probe_v1/actual_v1/report.json',
      'native_wrench_source_review':P/'native_equilibrium_source_audit_v1/report.json',
      'soft_friction_source_review':P/'native_soft_friction_audit_v1/report.json',
      'sixD_LP_fixtures':P/'wrench_residual_v6/preparation_v1/report.json',
      'sixD_saved_v5':P/'wrench_residual_v6/saved_v5_diagnostic_v1.json',
      'COM_time_contract':P/'phone_state_time_contract_v1/report.json',
      'strict_holding_adapter_preflight':P/'hand2_holding_v4/metadata_preflight_v1.json',
      'constant_u':P/'hand2_constant_u_baseline_v1/cpu_v1/report.json',
      'constant_u_saved_review':P/'hand2_constant_u_baseline_v1/saved_recheck_v2/report.json',
      'branch_fixture':P/'native_mpc_v1/two_branch_contract_v1/report.json',
      'MPC_source':P/'native_mpc_v1/run_mpc_feasibility_v1.py',
      'MPC_protocol':P/'native_mpc_v1/mpc_feasibility_protocol_v1.json',
      'MPC_result':P/'native_mpc_v1/feasibility_fine_v1/report.json',
      'fixed_contact_wrench_upper_bound':P/'native_equilibrium_v6/wrench_upper_bound_saved_v1/actual_v1/report.json',
    }
    # Extra saved-only independent reviews are explicitly passed by their producer.
    extra=P/'native_mpc_v1/final_extra_evidence_v1.json'
    if extra.exists():
        for key,p in read(extra).items():
            if pin(p['path'])!=p:raise ValueError('Changed extra evidence '+key)
            if key in files:raise ValueError('No evidence shadowing')
            files[key]=Path(p['path'])
        files['extra_evidence_index']=extra
    terminal={
      'native_static':P/'native_equilibrium_v6/run1_process_terminal.json',
      'constant_u':P/'hand2_constant_u_baseline_v1/actual_process_terminal_v1.json',
      'MPC':P/'native_mpc_v1/pilot_process_terminal_v1.json'}
    for name,path in terminal.items():
        r=read(path)
        if r.get('returncode',r.get('exit_code'))!=0 or r['stderr']['bytes']!=0:
            raise ValueError('Unexpected process terminal '+name)
        for key in ('stdout','stderr'):
            if pin(r[key]['path'])!=r[key]:raise ValueError('Changed process log '+name)
        files[name+'_terminal']=path
    baseline=read(files['constant_u']);mpc=read(files['MPC_result'])
    if (baseline['counts']['integration_steps']!=1176 or len(baseline['cases'])!=2
        or baseline['all_original_cases_and_dt_pairs_pass'] is not False
        or any(c['nominal_holding_dynamics_pass'] for c in baseline['cases'])):
        raise ValueError('Preserve both failed baseline cases')
    if (mpc['actual_integration_steps']!=822 or mpc['planning_counts']['planning_steps_returned']!=41216
        or mpc['planning_counts']['branches_returned']!=161
        or mpc['original_2s_single_dt_holding_pass'] is not False or mpc['two_dt_qualified'] is not False
        or mpc['all_complete_selected_intervals_raw_bit_equal'] is not True
        or mpc['training_or_policy_benefit_qualified'] is not False):
        raise ValueError('Preserve failed bounded MPC scope')
    if read(files['branch_fixture'])['passed']!=42:raise ValueError('Branch contract')
    upper=read(files['fixed_contact_wrench_upper_bound'])
    if (upper['actual_LP_solves']!=2 or len(upper['results'])!=2
        or upper['new_native_model_forward_step_NN_GPU_calls']!=0
        or not all(r['success'] and r['recomputed_scaled_Linf']>.28 for r in upper['results'])):
        raise ValueError('Fixed saved point geometry LP scope')
    for key in ('constant_u_saved_review',):
        r=read(files[key])
        if r['passed']!=r['total']:raise ValueError('Saved review incomplete '+key)
    record={
      'schema':'ego_phone_goal_extension_progress:5','created_UTC':datetime.now(timezone.utc).isoformat(),
      'previous_goal_turn_classification':'progress','this_goal_turn_classification':'progress',
      'status':'ACTIVE_NATIVE_SOFT_CONTACT_AND_MPC_MEASURED_HOLDING_FAILED',
      'native_goal_status_last_verified':'active','original_full_goal_complete':False,
      'real_ego_reference_and_official_object_asset_used':True,
      'ego_policy_or_world_model_training_started':False,'new_learning_training_started':False,
      'first_healthy_new_training_checkpoint_pause_condition_triggered':False,
      'original_model_wrist_scale_actuator_material_solver_preserved':True,
      'mass_friction_COM_inertia_calibrated':False,'functional_pad_semantics_calibrated':False,
      'source_clip_GT_gravity_registration_verified':False,
      'strict_zero_speed_acceleration_gate_is_required_for_dynamic_holding':False,
      'native_static_search_forward_calls':420,'native_static_search_strict_candidates':0,
      'new_actual_holding_cases':3,'new_actual_holding_passes':0,
      'new_actual_native_steps':1998,'new_clone_planning_native_steps':41728,
      'MPC_complete_selected_actual_prediction_intervals_bit_equal':6,
      'MPC_partial_final_interval_actual_steps':54,
      'new_fixed_contact_upper_bound_LP_solves':2,
      'current_fixed_contact_ideal_wrench_balance_qualified':False,
      'three_N_cap_is_sole_cause_of_fixed_contact_wrench_failure':False,
      'all_known_root_processes_observed_terminal':True,
      'next_action':[
        'Use saved contact/control/cost diagnostics and current-contact wrench upper bound to choose a new reference/contact candidate, without repeating the failed initialization blindly.',
        'Construct explicit CAD convex patch/normal/extent candidates and compare ego near-contact regions under true per-finger force/torque/control limits, retaining unknown functional-pad semantics.',
        'Separate contact establishment, preload and bounded dynamic holding; preserve original2s gates and fixed-wrist model. New mounting or contact-source conditions require named matched branches.',
        'After nominal unassisted holding, complete timestep/parameter-prior robustness and61frame temporal reference with new COM/time/full-orientation phone contracts.',
        'Complete original SuperMimic/CHORD same-budget learning, WM contact/gradient checks, independent trajectories, real parameter and hardware contracts; next new training obeys checkpoint+ETA pause condition.'
      ],'evidence_pins':{k:pin(v) for k,v in files.items()},'finalizer':pin(__file__)}
    target=P/'outputs/goal_extension_native_mpc_milestone_v5.json'
    with target.open('x',encoding='utf8') as f:json.dump(record,f,indent=2,allow_nan=False)
    print(json.dumps({'status':record['status'],'handoff':pin(target),'pins':len(files)}))


if __name__=='__main__':main()
