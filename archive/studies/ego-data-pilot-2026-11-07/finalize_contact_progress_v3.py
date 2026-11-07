"""Finalize this goal turn's evidence after all bounded jobs terminate."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
N = P.parent/'supermimic-full151-observation-study'
M = P.parent/'supermimic-chord-study'


def pin(p):
    p=Path(p).resolve()
    b=p.read_bytes()
    return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    files={
        'plan':M/'docs/research-plan-v0.22.md',
        'previous_milestone':P/'outputs/goal_extension_second_milestone_v2.json',
        'distance_probe':P/'distance_probe_v1/report.json',
        'geometry':P/'retarget_geometry_v2/report.json',
        'geometry_saved_recheck':P/'retarget_geometry_v2/seed2_saved_gate_recheck_v2.json',
        'support_preparation':P/'hand2_holding_v1/support_preparation_v1/actual_v1/report.json',
        'point_support_explanation':P/'hand2_holding_v1/support_preparation_v1/support_explanation_v1.json',
        'holding_protocol':P/'hand2_holding_v1/cpu_nominal_protocol_v2.json',
        'holding':P/'hand2_holding_v1/cpu_nominal_v1/report.json',
        'holding_terminal':P/'hand2_holding_v1/cpu_nominal_v1_process_terminal.json',
        'native_final1000_health':N/'outputs/full151105_native_public_only_final1000_saved_independent_v1/report.json',
        'native_common_eval':N/'outputs/full151105_native_public_only_common_evaluation_CPU_v1/report.json',
        'native_common_saved_recheck':N/'outputs/full151105_native_public_only_common_eval_saved_recheck_v1/report.json',
    }
    files.update({
        'holding_saved_recheck':P/'hand2_holding_v1/saved_recheck_v1/report.json',
        'gravity_alignment':P/'gravity_alignment_probe_v1/report.json',
    })
    geom=read(files['geometry']);hold=read(files['holding']);terminal=read(files['holding_terminal'])
    cases=[v for v in hold['cases'] if v.get('dt_s') is not None]
    if (not geom['seed_results'][2]['geometry_candidate_qualified'] or len(cases)!=4
            or any(c['nominal_holding_dynamics_pass'] for c in cases)
            or terminal['returncode']!=0 or terminal['stderr']['bytes']!=0
            or hold['counts']['integration_steps']!=sum(c['integration_steps'] for c in cases)):
        raise ValueError('Unexpected actual evidence state: do not finalize stale claims')
    audit=read(files['holding_saved_recheck'])
    if audit['passed'] != audit['total'] or audit['holding_qualified'] is not False or audit['steps_saved'] != 2011:
        raise ValueError('Saved arithmetic audit must have completed with holding failure preserved')
    out={
        'schema':'ego_phone_goal_extension_progress:3',
        'created_UTC':datetime.now(timezone.utc).isoformat(),
        'previous_goal_turn_classification':'progress',
        'this_goal_turn_classification':'progress',
        'status':'ACTIVE_GEOMETRY_QUALIFIED_NOMINAL_HOLD_FAILED_GRAVITY_SUPPORT_REQUIRED',
        'native_goal_status_last_verified':'active',
        'original_full_goal_complete':False,
        'no_hand_CPU_Warp_bench_qualified_under_assumptions':True,
        'real_HOT3D_reference_geometry_used':True,
        'ego_data_used_for_policy_or_world_model_training':False,
        'geometry_qualified_seed':2,
        'geometry_other_seeds_rejected':[0,1],
        'nominal_holding_attempts':4,
        'nominal_holding_passes':0,
        'holding_integration_steps':hold['counts']['integration_steps'],
        'holding_warnings':sum(c['metrics']['warning_count'] for c in cases),
        'new_learning_training_started':False,
        'new_training_first_checkpoint_pause_condition_triggered':False,
        'old_first_checkpoint_pause_condition_already_fulfilled':True,
        'mass_friction_COM_inertia_real_calibrated':False,
        'ego_trajectory_policy_generalization_hardware_qualified':False,
        'remaining_scope':'Contact and gravity feasible references, native unassisted holding, parameter robustness, temporal ego reference, same-budget SuperMimic/CHORD, WM contact/Jacobian and policy task checks, held-out data and hardware contracts.',
        'current_next_action':'Use gravity-alignment diagnosis to add support wrench feasibility and true torque bounds jointly with collision and reference fidelity, then perform a fresh frozen holding experiment. Do not tune torsion or force silently.',
        'evidence_pins':{k:pin(v) for k,v in files.items()},
        'finalizer':pin(__file__),
    }
    target=P/'outputs/goal_extension_contact_milestone_v3.json'
    with target.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'status':out['status'],'handoff':pin(target),'evidence_count':len(files)}))


if __name__=='__main__':main()
