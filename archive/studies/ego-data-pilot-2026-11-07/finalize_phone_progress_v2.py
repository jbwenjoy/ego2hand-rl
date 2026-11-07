"""Second wave: qualified no-hand model, real reference geometry, unresolved grasp."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))

def pin(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def main():
    cpu = read('simulation_fixedgrid/cpu_assessment_v4/report.json')
    warp = read('simulation_fixedgrid/warp_assessment_v4/report.json')
    parity = read('simulation_fixedgrid/parity_independent_v4/report.json')
    if not cpu['CPU_bench_qualified'] or not warp['stage4_engineering_qualified'] or not parity['G8_qualified'] or parity['G8_case_counts'] != {'PASS': 54}:
        raise ValueError('Completed model-bench gates not proven')
    files = ['protocols/phone_convex_contact_v2.json', 'protocols/phone_convex_timestep_v3.json',
             'protocols/phone_convex_fixedgrid_v4.json',
             'simulation_convex_v2/cpu_screen_assessment_v2/selection.json',
             'simulation_timestep_v3/cpu_screen_assessment_v3/selection.json',
             'simulation_timestep_v3/cpu_confirm_assessment_v3/report.json',
             'simulation_fixedgrid/cpu_v4/result.json', 'simulation_fixedgrid/cpu_assessment_v4/report.json',
             'simulation_fixedgrid/warp_v4/result.json', 'simulation_fixedgrid/warp_v4_process_terminal.json',
             'simulation_fixedgrid/warp_assessment_v4/report.json',
             'simulation_fixedgrid/parity_independent_v4/report.json',
             'hand2_stage1/handoff_v1.json', 'hand2_stage1/scene_loader_v1.py',
             'hand2_stage1/loader_reconstruction_v1.json',
             'hand2_stage1/staging_preview_v1.png', 'ego_hand_geometry_v1/report.json',
             'ego_hand_geometry_v1/geometry_reference_overlay.png', 'retarget_geometry_v1/report.json',
             'retarget_geometry_v1/retarget_protocol_v1.json',
             'retarget_geometry_v1/diagnostic_readback_v1.json',
             'retarget_geometry_v1/all3_geometry_candidates.png']
    result = {'schema': 'ego_phone_goal_extension_progress:2',
              'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'previous_goal_turn_classification': 'progress',
              'status': 'ACTIVE_NO_HAND_BENCH_QUALIFIED_REFERENCE_FK_AND_ROBOT_GEOMETRY_ATTEMPT_RECORDED',
              'goal_plan': pin(Path('D:/workspace/supermimic-chord-study/docs/research-plan-v0.21.md')),
              'original_full_goal_complete': False,
              'no_hand_bench': {'timeconst_s': .0025, 'dt_pair_s': [.0003125, .00015625],
                   'mass_kg': [.1, .2, .3], 'sliding_friction': [.3, .6, .9],
                   'CPU_cases': 54, 'Warp_cases': 54, 'steps_each_backend': 777600,
                   'CPU_bench_qualified': True, 'Warp_bench_qualified': True,
                   'independent_G8_qualified': True, 'maximum_Warp_penetration_m': .001402676688,
                   'maximum_final_CPU_Warp_orientation_difference_deg': .558304527,
                   'maximum_final_CPU_Warp_COM_difference_m': .000104245075,
                   'old_failures_and_null_selections_preserved': True,
                   'parameters_are_assumed_not_calibrated': True},
              'reference_geometry': {'clip': '001856', 'frames': [42, 102], 'frame_count': 61,
                   'human_hand': 'left', 'original_angles_preserved': 22,
                   'official_FK': 'hand_tracking_toolkit@9064a2fe7fda51bb1f1451a0d3d0da7f3ef5e64e',
                   'robot_angle_mapping_performed': False, 'force_truth_inferred': False},
              'robot': {'scene_fixed_wrist': True, 'scale': 1., 'hinge_joints': 20,
                   'actuators': 20, 'identified_parameters_preserved': True,
                   'exact_loader_reconstruction_byte_equal_model_arrays': 482,
                   'geometric_candidate_only': True, 'selected_seed': 2,
                   'tip_vector_RMS_error_m': .0142397, 'index_gap_m': .007157632159420704,
                   'grasp_or_dynamics_qualified': False,
                   'distance_query_stability_issue': 'Native gap jumped under ~1.49e-8m perturbation in one saved example; unique cause unknown.'},
              'new_training_started': False, 'new_Hand2_integration_steps': 0,
              'ego_method_evaluation_count_this_pilot': 0, 'real_parameter_calibration': False,
              'known_native_workers': [{'session': 21092, 'phase': 'v3_CPU54', 'terminal': True, 'exit': 0},
                   {'session': 43007, 'phase': 'v4_CPU54', 'terminal': True, 'exit': 0},
                   {'session': 7040, 'phase': 'v4_Warp54', 'terminal': True, 'exit': 0}],
              'new_training_checkpoint_pause_condition': 'Still applies only when a new learning training job starts; historic checkpoint100 pause already fulfilled.',
              'next_actions': ['New geometry protocol: fixed +/-10um distance-interface stability probe with physical Lipschitz bound and independent reference where possible.',
                   'Use fixed pads/explicit nonpenetration/contact constraints and a meaningful differentiation scale/site Jacobian; preserve left-to-right anatomical mismatch and original failures.',
                   'Specify phone-hand contact pairs explicitly; original hand priority must not silently override material/contact assumptions. Check nominal friction-cone gravity/moment support.',
                   'Only then unassisted fixed-wrist holding, followed by trajectory/learning/model/gradient comparisons; preserve original native-only AFTER1000 readback/evaluation backlog.'],
              'artifacts': [pin(ROOT/name) for name in files], 'finalizer_source': pin(__file__)}
    destination = ROOT/'outputs/goal_extension_second_milestone_v2.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'path': str(destination), 'artifact_count': len(files)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
