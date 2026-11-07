"""Goal evidence handoff after bounded joint-support experiments terminate."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib
P=Path(__file__).resolve().parent;M=P.parent/'supermimic-chord-study'


def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    files={
      'plan':M/'docs/research-plan-v0.23.md',
      'previous_milestone':P/'outputs/goal_extension_contact_milestone_v3.json',
      'static_interface_contract':P/'retarget_gravity_v3/static_contract_v1/report.json',
      'residual_LP_fixtures':P/'gravity_support_design_v3/preparation_v3/report.json',
      'joint_v3':P/'retarget_gravity_v3/completed_v1/report.json',
      'joint_v3_saved_recheck':P/'retarget_gravity_v3/completed_v1_saved_recheck/report.json',
      'closure_seed_preparation':P/'retarget_contact_polish_v4/seed_preparation/report.json',
      'closure_v4':P/'retarget_contact_polish_v4/completed_v1/report.json',
      'closure_v4_saved_recheck':P/'retarget_contact_polish_v4/completed_v1_saved_recheck/report.json',
      'PD_family_counterfactual':P/'retarget_contact_polish_v4/ideal_torque_counterfactual_v1/report.json',
      'support_v5':P/'retarget_gravity_v3/support_refinement_v5_run1/report.json',
      'support_v5_saved_recheck':P/'retarget_gravity_v3/support_refinement_v5_saved_recheck/report.json',
      'holding_v3':P/'hand2_holding_v3/cpu_nominal_v1/report.json',
      'holding_v3_saved_recheck':P/'hand2_holding_v3/saved_recheck_v1/report.json',
      'holding_v3_actual_prefix_figure':P/'hand2_holding_v3/saved_recheck_v1/holding_v3_actual_prefix.png',
      'pad_semantics_source_review':P/'hand2_holding_v3/pad_semantics_source_review_v1.json',
      'source_gravity_registration':P/'gravity_reference_audit_v1/registration_source_followup_v1/report.json',
      'full_left_surface_proximity':P/'contact_region_audit_v1/report.json',
      'right_surface_proximity':P/'right_hand_contact_audit_v1/report.json',
      'temporal_state_bridge_source_audit':P/'temporal_bridge_audit_v1/report.json',
    }
    terminal={
      'v3':P/'retarget_gravity_v3/run1_process_terminal.json',
      'v4':P/'retarget_contact_polish_v4/run1_process_terminal.json',
      'v5':P/'retarget_gravity_v3/v5_run1_process_terminal.json',
      'holding_v3':P/'hand2_holding_v3/cpu_nominal_v1_process_terminal.json',
    }
    for k,v in terminal.items():
        r=read(v)
        if r['returncode']!=0 or r['stderr']['bytes']!=0:raise ValueError('Unexpected process state '+k)
        if pin(r['stdout']['path'])!=r['stdout'] or pin(r['stderr']['path'])!=r['stderr']:raise ValueError('Changed terminal logs')
        files[k+'_terminal']=v
    v4=read(files['closure_v4']);v5=read(files['support_v5']);holding=read(files['holding_v3'])
    if v4['geometry_pass_count']!=2 or v4['bounded_dynamic_diagnostic_eligible_count']!=0:raise ValueError('Unexpected closure scope')
    if v5['qualified_support_proposal_count']!=0 or v5['counts']['integration_training_GPU']!=0:raise ValueError('Unexpected strict support qualification')
    if (holding['counts']['integration_steps']!=1153 or len(holding['cases'])!=2
            or holding['all_original_cases_and_dt_pairs_pass'] is not False
            or any(v['nominal_holding_dynamics_pass'] for v in holding['cases'])):raise ValueError('Preserve actual holding failures')
    for k in ('joint_v3_saved_recheck','closure_v4_saved_recheck','support_v5_saved_recheck','holding_v3_saved_recheck'):
        a=read(files[k])
        if a['passed']!=a['total']:raise ValueError('Incomplete saved audit '+k)
    out={
      'schema':'ego_phone_goal_extension_progress:4',
      'created_UTC':datetime.now(timezone.utc).isoformat(),
      'previous_goal_turn_classification':'progress','this_goal_turn_classification':'progress',
      'status':'ACTIVE_GEOMETRY_CLOSED_HOLDING_FAILED_CONTACT_PATCH_AND_NATIVE_EQUILIBRIUM_NEXT',
      'native_goal_status_last_verified':'active','original_full_goal_complete':False,
      'no_hand_bench_qualified_under_assumptions':True,
      'real_HOT3D_left_reference_and_right_proximity_used':True,
      'ego_policy_or_world_model_training_started':False,'new_learning_training_started':False,
      'new_training_first_healthy_checkpoint_pause_condition_triggered':False,
      'original_fixed_robot_world_wrist_scale_and_parameters_preserved':True,
      'robot_functional_pad_geometry_semantics_verified':False,
      'source_clip_GT_gravity_registration_verified':False,
      'real_mass_inertia_COM_contact_material_calibrated':False,
      'v4_geometric_closure_count':2,'v5_strict_support_proposal_count':0,
      'new_holding_cases':2,'new_holding_passes':0,'new_native_holding_steps':1153,
      'all_known_root_processes_observed_terminal':True,
      'current_next_action':[
        'Fix base/FD cache registration in a new namespace; preserve old finite-search limits and outputs.',
        'Build explicit CAD-contact patch/normal/extent candidates and scaled six-dimensional wrench residual, without arbitrary torsion or per-sample force-budget multiplication.',
        'Validate native compliant-contact equilibrium and separate contact establishment/preload/holding under true controller limits; preregister any pad-compression assumptions and preserve nonpad/2s gates.',
        'After nominal unassisted holding: timestep/material-prior robustness, temporal61-frame reference and new20ms/64-or128substep COM-consistent phone task contract.',
        'Complete original same-budget SuperMimic/CHORD, WM contact/gradient, policy/heldout/generalization and hardware-contract scope; next new training obeys firsthealthycheckpoint+ETA pause condition.',
      ],
      'evidence_pins':{k:pin(v) for k,v in files.items()},'finalizer':pin(__file__),
    }
    target=P/'outputs/goal_extension_support_milestone_v4.json'
    with target.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'status':out['status'],'handoff':pin(target),'pins':len(files)}))


if __name__=='__main__':main()
