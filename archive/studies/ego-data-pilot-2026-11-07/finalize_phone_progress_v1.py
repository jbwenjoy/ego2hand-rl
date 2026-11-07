"""Consolidate this goal extension and the completed first milestone evidence."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def pin(path):
    path = Path(path)
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def main():
    cpu = json.loads((ROOT / 'simulation_stage1/cpu_v1/result.json').read_text())
    warp = json.loads((ROOT / 'simulation_stage1/warp_v1/result.json').read_text())
    authorization = json.loads((ROOT / 'outputs/goal_extension_authorization_v1.json').read_text(encoding='utf-8'))
    if cpu['case_count'] != 54 or warp['case_count'] != 54 or cpu['step_count'] != 97200 or warp['world_steps_returned'] != 97200:
        raise ValueError('Incomplete physical benchmark coverage')
    names = ['outputs/goal_extension_authorization_v1.json', 'outputs/phone_candidate_windows_v1.json',
             'asset_stage1/geometry_report.json', 'asset_stage1/aabb_geometry_audit_v1/report.json',
             'asset_stage2_convex/completed_v2/report.json',
             'asset_stage2_convex/completed_v2/phone_asset.xml', 'outputs/convex_asset_native_compile_v1.json',
             'protocols/phone_physics_stage1_v1.json', 'simulation_stage1/execution_binding_cpu_v1.json',
             'simulation_stage1/cpu_v1/result.json', 'simulation_stage1/warp_v1/result.json',
             'simulation_stage1/cpu_saved_assessment_v1/report.json',
             'simulation_stage1/warp_saved_assessment_v1/report.json',
             'simulation_stage1/cpu_visualization_v3/render_manifest.json',
             'simulation_stage1/cpu_visualization_v3/nominal_physics_traces.png',
             'simulation_stage1/cpu_visualization_v3/nominal_CPU_state_replay.mp4',
             'simulation_stage1/visualize_phone_cpu_bench_v1.py']
    result = {'schema': 'ego_phone_goal_extension_progress:1',
              'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'GOAL_ACTIVE_FIRST_ASSET_AND_PHYSICS_BASELINE_COMPLETED_WITH_RECORDED_FAILURES',
              'goal_state_last_observed': authorization['goal_snapshot']['goal']['status'],
              'goal_execution_supplement': pin(Path('D:/workspace/supermimic-chord-study/docs/research-plan-v0.20.md')),
              'original_goal_complete': False,
              'geometry': {'AABB': 'FAIL_p95_2.836mm_max6.008mm',
                           'convex_hull': 'PASS_FINITE_SAMPLED_GEOMETRY_ONLY_p95_0.9066mm_max1.8528mm',
                           'convex_native_compile_frame_check': 'PASS_NO_INTEGRATION',
                           'convex_dynamics_qualified': False},
              'CPU': {'cases': 54, 'steps': 97200, 'drop_penetration_failures': 18,
                      'maximum_observed_penetration_m': max(c['metrics']['max_penetration_m'] for c in cpu['cases']),
                      'timestep_pairs_passed': 27, 'bench_qualified': False},
              'Warp': {'cases': 54, 'world_steps': 97200, 'batch_steps': 32400, 'final_sampling_forward_calls': 18,
                       'maximum_observed_penetration_m': max(c['metrics']['max_penetration_m'] for c in warp['cases']),
                       'drop_penetration_failures': 18, 'timestep_pairs_passed': 27,
                       'full_GPU_warning_ledger': 'UNKNOWN_NOT_EXPORTED', 'bench_qualified': False},
              'CPU_Warp_same_AABB_model_comparisons': {'pairs_passed': 54, 'worst_final_COM_distance_m': 0.000749171,
                                                     'does_not_clear_common_physical_failure': True},
              'new_training_started': False, 'new_Hand2_execution': False,
              'physical_parameters_calibrated': False, 'ego_method_evaluation_count_this_pilot': 0,
              'video_QA': {'final_version': 3, 'representative_native_frame_viewed': True,
                           'earlier_render_versions_retained': 'v1 dark; v2 overexposed; v3 lighting and checker support readable',
                           'saved_states_unchanged': True, 'rendering_integration_steps': 0},
              'next_action': 'Use a new protocol/version for the accepted convex asset and explicit contact time-scale/timestep verification; preserve thresholds and baseline failures; verify CPU/Warp, then Hand2 static grasp. Do not transplant old pen policy or AABB parity qualification. Preserve old AFTER1000 health/eval backlog.',
              'artifacts': [pin(ROOT / name) for name in names], 'finalizer_source': pin(__file__)}
    destination = ROOT / 'outputs/goal_extension_first_milestone_v1.json'
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'path': str(destination),
                      'goal_state_last_observed': result['goal_state_last_observed'],
                      'artifact_count': len(result['artifacts'])}, ensure_ascii=False))

if __name__ == '__main__':
    main()
