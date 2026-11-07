"""Create the static temporal-bridge receipt. No backend/model/array imports."""
from pathlib import Path
import datetime
import hashlib
import json
import sys

OUT = Path(__file__).resolve().parent
P = OUT.parent
N = Path('D:/workspace/supermimic-full151-observation-study')
D = Path('D:/workspace/dex-supermimic-research')
RL = D / 'src/dex_supermimic/rl/dex_supermimic'

SOURCES = {
    'Manager_control_and_physics_loop': (D / '.venv/Lib/site-packages/mjlab/envs/manager_based_rl_env.py', [(263, 270), (395, 409), (435, 490)]),
    'pen_environment_cadence': (D / 'src/dex_supermimic/tasks/pen_spin/pen_spin_env_cfg.py', [(217, 223)]),
    'native_full151_guard': (N / 'src/full151_native/native.py', [(41, 66), (183, 184)]),
    'learning_guard': (N / 'src/full151_native/learning_v2.py', [(61, 66), (290, 299), (394, 399)]),
    'full151_bundle_cadence': (N / 'src/full151_native/bundle.py', [(43, 63), (84, 93)]),
    'full151_clock_and_readback': (N / 'src/full151_native/probe.py', [(16, 60), (78, 87), (262, 264)]),
    'serial_fresh_worker': (N / 'src/full151_gradient/serial_probe105_calibration_v1.py', [(227, 238), (456, 475), (508, 516), (542, 544), (561, 590), (632, 648), (805, 807)]),
    'four_substep_impulse_dataset': (N / 'src/full151_gradient/boundary_impulse_dataset105_v1.py', [(89, 109), (138, 142), (182, 233)]),
    'controller_once_per_control': (N / 'src/full151_native/controller.py', [(133, 164), (201, 213)]),
    'native_state58_delta52': (RL / 'state.py', [(6, 38), (42, 46)]),
    'WM_discrete_transition': (RL / 'networks.py', [(11, 61)]),
    'reader_and_physical_context': (RL / 'sim.py', [(87, 110), (185, 240)]),
    'xi_DR_names': (RL / 'observations.py', [(14, 22)]),
    'PenMDP_sensor_frame_and_task_interface': (RL / 'pen.py', [(47, 69), (86, 117), (124, 164), (175, 180), (218, 240), (254, 284), (347, 375), (382, 409), (432, 433)]),
    'runner_buffer_and_loss_phase': (RL / 'runner.py', [(377, 417), (447, 462), (535, 544), (734, 739)]),
    'reference_frequency_guard': (D / 'src/dex_supermimic/tasks/pen_spin/mdp/commands.py', [(345, 353), (615, 621)]),
    'CHORD_fixed_legacy_geometry': (N / 'src/full151_intent/binding_v3.py', [(37, 49), (75, 86), (221, 232)]),
    'CHORD_once_per_imagined_control': (N / 'src/full151_native/prior_hook_v1.py', [(10, 15), (57, 80)]),
    'phone_physical_COM_and_solver': (P / 'simulation_convex_v2/phone_convex_common_v2.py', [(41, 48), (53, 68), (83, 85)]),
    'phone_holding_COM_reader': (P / 'hand2_holding_v1/run_cpu_holding_v1.py', [(14, 14), (130, 132)]),
    'HOT3D_clip_format': (P / 'sources/hot3d_clips_README.md', [(1, 15)]),
}


def pin(p):
    p = Path(p).resolve(strict=True)
    b = p.read_bytes()
    s = p.stat()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest(),
            'file_identity': {'st_dev': s.st_dev, 'st_ino': s.st_ino}}


def main():
    if (OUT / 'report.json').exists():
        raise FileExistsError('Keep the original audit receipt immutable')
    refs = {}
    identities = {}
    for name, (path, ranges) in SOURCES.items():
        p = Path(path).resolve(strict=True)
        rec = pin(p)
        key = (rec['file_identity']['st_dev'], rec['file_identity']['st_ino'], rec['sha256'])
        if key in identities and not p.samefile(identities[key]):
            raise ValueError('File identity collision')
        identities[key] = p
        text = p.read_text(encoding='utf-8').splitlines()
        refs[name] = {'source_pin': rec, 'line_ranges': ranges,
                      'selected_source_lines': [{'line': i, 'text': text[i-1]}
                                                for a, b in ranges for i in range(a, min(b, len(text)) + 1)]}
    H = .02
    rows = []
    for h, n in ((.005, 4), (.0003125, 64), (.00015625, 128)):
        if n * h != H:
            raise ValueError('Declared integer cadence mismatch')
        rows.append({'nominal_physics_dt_s': h, 'control_dt_s': H, 'substeps_per_control': n,
                     'last_solver_PRE_nominal_phase_s': H-h,
                     'example150_intervals_physics_steps': 150*n,
                     'example150_intervals_raw_state_rows': 150*n+1,
                     'example452_groups_physics_rows': 452*n,
                     'example152_restored_groups_physics_rows': 152*n,
                     'physics_step_multiplier_vs_4x5ms': n//4,
                     'scope': 'Planned integer arithmetic only, not an authorized phone acquisition plan or actual clock readback'})
    report = {
        'schema': 'phone_to_SuperMimic_CHORD_temporal_bridge_SOURCE_audit:1',
        'status': 'SOURCE_ONLY_MIGRATION_REQUIRED_NOT_EXECUTED_OR_QUALIFIED',
        'created_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': pin(__file__), 'description': pin(OUT / 'report.md'),
        'scope': {'source_files_read': len(SOURCES), 'unique_actual_source_files': len(identities),
                  'NPZ_or_checkpoint_reads': 0, 'model_or_Env_construction': 0, 'NN_FW_AD': 0,
                  'Warp_or_physics_calls': 0, 'training_calls': 0, 'GPU_calls': 0,
                  'original_source_edits': 0},
        'cadence_plan_arithmetic': rows,
        'central_decision': 'Keep one 20ms WM/action/reward control step and hold its actual u20 over n physical steps; new phone namespace/labels/qualification are required. Do not merely alter dt in old qualified full151 guards.',
        'state_shape_scope': {'native_state': 58, 'native_delta': 52, 'control': 20, 'physical_xi': 46,
                              'native_selfcollision_logits': 15, 'boundary_mode_auxiliary': 2,
                              '13_is_not_model_mode': 'The public evaluation partitions have13 cohorts; one fresh protocol also has13 inputaliases. Neither is a dynamics-mode dimension.'},
        'critical_non_temporal_mapping': 'Current native58 p/v are free-body-origin based. Old serial acquisition requires COM offset0, while phone inertial.pos is asset center. A new root-to-COM or all-COM contract must be explicit before reusing momentum labels.',
        'unchanged_control_level_quantities_if_H_remains_20ms': ['one Actor decision', 'one residual EMA/C2 application', 'one WM prediction', 'reward dt=.02', 'one replay transition', 'one command/reference advance', 'H8=8controlsteps=.16s'],
        'never_inherit': ['old pen physical/cache labels', '15ms cached sensor timestamp', 'old600/601 FP32 clock qualification', 'old192/4096 capacity qualification for a different contact topology', 'old J/K/r_saved or model-gradient qualification', 'old105/native-only checkpoint health as new phone task accuracy', 'H123 design intent as human observed contact'],
        'same_budget_contract': 'Both SM and CHORD share newphone model/h/n, initialization and commonwarmup/replay/learning origin, controls, control sample count and update counts. Also report actual physics steps16x/32x vs old4-substep job; equal old control count is not equal old physics cost.',
        'source_references': refs,
        'result_is_not': ['an implemented phone task', 'a physical-reference certificate', 'new ego learning/generalization or Actor benefit', 'authorization to run training or physics']}
    if any(name in sys.modules for name in ('torch', 'mujoco', 'warp')):
        raise RuntimeError('The source audit must not import a backend')
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT/'report.json'), 'description': pin(OUT/'report.md'), 'scope': report['scope']}))


if __name__ == '__main__':
    main()
