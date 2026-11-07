"""Saved-only smoke40 provenance/lifecycle/init/coverage/cost review.

No Torch, native factory, Reader, FK, simulator, WM or optimizer is imported.
Checkpoint health belongs to the separate Contact auditor.
"""
from pathlib import Path
import argparse
import csv
import datetime
import hashlib
import json
import os
import shlex
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def stamp(value):
    return datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out / 'review_source.py').write_bytes(Path(__file__).read_bytes())
    run = ROOT / 'outputs/full151_native_A1L_smoke40_v1'
    watch = run.with_name(run.name + '_watchdog')
    config = ROOT / 'configs/full151-a1l-smoke40-authorized-v1.json'
    files = {name: run / name for name in ('config.json', 'runtime.json', 'result.json',
        'training_completion.json', 'runtime_reset_readback.json', 'actual_context_readbacks.npz',
        'expected_context_fields.npz', 'fixed_physics_context.npz', 'nominal_model.mjb',
        'metrics.jsonl', 'full151_actualu_evidence.jsonl', 'capacity_observation.json')}
    files.update({name: watch / name for name in ('state.json', 'engineering_identity.json',
        'post_terminal_CIM.json', 'preflight.json', 'config_snapshot.json', 'supervisor_source.py')})
    files.update(config_authority=config, inventory=run / 'source/inventory.json',
                 execution_zip=run / 'source/execution_sources.zip')
    inputs = {key: pin(path) for key, path in files.items()}
    cfg = read(config); result = read(run / 'result.json'); runtime = read(run / 'runtime.json')
    state = read(watch / 'state.json'); identity = read(watch / 'engineering_identity.json')
    post = read(watch / 'post_terminal_CIM.json'); preflight = read(watch / 'preflight.json')
    bundle = read(cfg['input_bundle']['path']); reset = read(run / 'runtime_reset_readback.json')
    checks = {}; details = {}
    def check(name, condition): checks[name] = bool(condition)
    check('config_three_copies_bytes_exact', config.read_bytes() == files['config.json'].read_bytes() == files['config_snapshot.json'].read_bytes())
    check('config_method_role_and_budget', cfg['method'] == 'A1L' and cfg['role'] == 'engineering_smoke'
          and cfg['training']['max_iterations'] == 40 and cfg['training']['warmup_steps'] == 64
          and cfg['training']['wm_only_iters'] == 20 and cfg['training']['num_envs'] == 128)
    check('source_inventory_alias_exact', read(files['inventory']) == cfg['source_pins'])
    current = {key: Path(value['path']).is_absolute() and pin(value['path'])['sha256'] == value['sha256'] for key, value in cfg['source_pins'].items()}
    check('all145_source_input_pins_current', len(current) == 145 and all(current.values()))
    with zipfile.ZipFile(files['execution_zip']) as archive:
        members = archive.namelist(); zipped = {}
        for index, (key, value) in enumerate(cfg['source_pins'].items()):
            name = f'{index:04d}_{Path(value["path"]).name}'
            zipped[key] = (name in members and hashlib.sha256(archive.read(name)).hexdigest() == value['sha256'])
        check('all145_zip_members_exact_unique_no_extra', len(members) == len(set(members)) == 145 and all(zipped.values()))
    check('nominal_model_matches_bundle', pin(run / 'nominal_model.mjb')['sha256'] == bundle['model']['sha256'])
    command = state['command']; observed = state['actual_process_OS_observation']
    launch = state['launcher_pid']; pid = state['actual_pid']; ids = {launch, pid}
    line = shlex.split(observed['command_line'], posix=False)
    line = [value.strip('"') for value in line]
    python = identity['selected_python_identity']
    life = {}
    def lifecheck(name, condition): life[name] = bool(condition); check('identity_' + name, condition)
    lifecheck('completed_exit0_notimeout', state['status'] == 'completed' and state['exit_code'] == 0 and state['timeout'] is False)
    lifecheck('actual_OS_observed', observed['status'] == 'exact_OS_identity_observed')
    lifecheck('positive_owned_PID_fields', type(pid) is int and type(launch) is int and pid > 0 and launch > 0 and pid != launch)
    lifecheck('runtime_PID_PPID', runtime['pid'] == observed['pid'] == pid and runtime['parent_pid'] == observed['parent_pid'] == launch)
    expected = [python['chosen_python'], '-u', str(ROOT / 'scripts/execute_full151_learning_v1.py'),
                '--config', str(config), '--output', str(run), '--run']
    lifecheck('command_actual_routes', command == expected and line[1:] == expected[1:] and state['worker_output'] == str(run))
    lifecheck('chosen_venv_identity', os.path.samefile(python['chosen_python'], python['supervisor_sys_executable']))
    lifecheck('observed_base_samefile_alias', os.path.samefile(line[0], observed['executable_path'])
              and os.path.samefile(observed['executable_path'], python['base_executable_raw'])
              and os.path.samefile(python['base_executable_raw'], python['base_executable_resolved'])
              and pin(python['base_executable_resolved'])['sha256'] == python['base_executable_sha256'])
    lifecheck('runtime_source_plan', runtime['plan_sha256'] == pin(config)['sha256'] == preflight['config_sha256']
              and runtime['source_sha256'] == pin(expected[2])['sha256'] == preflight['entry_sha256'])
    captured = state['runtime_metadata']
    lifecheck('watched_runtime_bound', all(captured[key] == runtime[key] for key in ('pid', 'parent_pid', 'started_utc', 'plan_sha256', 'source_sha256')))
    lifecheck('creation_order', stamp(state['Popen_prelaunch_utc']) <= stamp(observed['creation_date_utc'])
              <= stamp(runtime['started_utc']) <= stamp(observed['observed_utc']) <= stamp(state['finished_utc']))
    lifecheck('postterminal_owned_PIDs_absent', post['status'] == 'NO_OWNED_PID_REMAINING'
              and set(post['queried_pids']) == ids and post['actual_CIM_rows'] == [])
    lifecheck('raw_reconstruction_agrees_identity12', len(identity['checks']) == 12
              and all(v is True for v in identity['checks'].values()) and identity['engineering_identity_qualified'] is True)
    check('worker_normal_completion', runtime['status'] == 'completed_full151_training'
          and result['status'] == 'COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY'
          and result['cleanup_errors'] == [] and result['full151_capacity_qualified_final'] is True)
    completion = read(run / 'training_completion.json')
    check('completion_result_exact', completion['schema'] == 'full151_native_worker_training_completion:1'
          and completion['config_sha256'] == pin(config)['sha256'] and completion['result'] == result)
    context = npz(bundle['context']['path']); reads = npz(run / 'actual_context_readbacks.npz')
    wanted = npz(run / 'expected_context_fields.npz'); fixed = npz(run / 'fixed_physics_context.npz')
    fields = ('body_mass', 'body_inertia', 'body_iquat', 'geom_size', 'geom_pos',
              'geom_friction', 'actuator_gainprm', 'actuator_biasprm')
    layouts = {}
    for field in fields:
        check('actual8_expected_' + field, np.array_equal(wanted[field], context[field][0]))
        for stage in ('after_copy', 'after_set_const_0'):
            got = reads[stage + '__' + field]
            check('actual8_' + stage + '_' + field, got.dtype == np.float32 and np.array_equal(got, np.broadcast_to(wanted[field], got.shape)))
        check('actual8_fixed_' + field, np.array_equal(fixed[field], reads['after_set_const_0__' + field]))
        layouts[field] = list(fixed[field].shape)
    check('xi46_128_exact', fixed['reader_context'].shape == (128, 46)
          and np.array_equal(fixed['reader_context'], np.broadcast_to(context['reader_context'][0], (128, 46))))
    init = npz(bundle['source_pins']['init_arrays']['path'])
    native = npz(bundle['source_pins']['native_data']['path'])
    def exact_reset(name, value):
        got = np.asarray(reset[name], np.float32)
        check('initial_' + name, got.shape == (128,) + value.shape and np.array_equal(got, np.broadcast_to(value, got.shape)))
    exact_reset('joint_pos_rad', init['q_init_requested_FP32_rad'])
    exact_reset('joint_vel_rad_s', init['qd_init_requested_FP32_rad_s'])
    exact_reset('object_pose_w', init['object_pose_init_requested_FP32_w'])
    exact_reset('actual_ctrl', init['u_planned0_FP32_rad'])
    exact_reset('cached_target', init['u_planned0_FP32_rad'])
    wrist = native['wrist_pose_w'][0].astype(np.float32)
    exact_reset('mocap_pos', wrist[:3][None]); exact_reset('mocap_quat', wrist[3:][None])
    twist = np.asarray(reset['object_twist_native'], np.float32)
    body_error = float(np.abs(twist[:, 3:] - init['object_angular_requested_FP32_body_rad_s']).max())
    check('initial_object_world_v_exact', np.array_equal(twist[:, :3], np.broadcast_to(init['object_bodyorigin_v_requested_FP32_world_m_s'], (128, 3))))
    check('initial_object_body_omega_API1e6', body_error <= 1e-6)
    check('initial_zero_phase_time_EMA_actions', all(not np.asarray(reset[key]).any() for key in ('phase', 'time', 'EMA_residual', 'action', 'previous_action')))
    check('initial_all128_ids_generation1', reset['env_ids'] == list(range(128)) and reset['generation'] == [1] * 128)
    # Saved actual A0 fullstate request uses the same immutable model/bundle and
    # analytic initializer; this is array crossbinding, not a new FK/Reader run.
    a0 = ROOT / 'outputs/full151_native_A0_v1'
    a0cfg = read(a0 / 'config.json'); a0request = npz(a0 / 'requested_initialization_and_control.npz')
    check('fullstate_reference_same_model_bundle', a0cfg['input_bundle'] == cfg['input_bundle']
          and pin(a0 / 'nominal_model.mjb')['sha256'] == pin(run / 'nominal_model.mjb')['sha256'])
    fullq = np.asarray(reset['full_qpos'], np.float32); fullv = np.asarray(reset['full_qvel'], np.float32)
    check('initial_full_qpos_exact', np.array_equal(fullq, np.broadcast_to(a0request['full_qpos_requested_FP32'], fullq.shape)))
    object_va = np.setdiff1d(np.arange(26), a0request['joint_dof_ids']); angular_va = object_va[3:]
    other_va = np.setdiff1d(np.arange(26), angular_va)
    check('initial_full_qvel_except_bodyomega_exact', np.array_equal(fullv[:, other_va], np.broadcast_to(a0request['full_qvel_requested_FP32'][other_va], (128, len(other_va)))))
    check('initial_full_qvel_bodyomega_tolerance', float(np.abs(fullv[:, angular_va] - a0request['full_qvel_requested_FP32'][angular_va]).max()) <= 1e-6)
    events = [json.loads(line) for line in (run / 'full151_actualu_evidence.jsonl').read_text().splitlines()]
    hist = np.sum([np.asarray(row['phase_histogram'], np.int64) for row in events], axis=0)
    check('phase_hist_source_rows41_contiguous', len(events) == 41 and [row['collect_call'] for row in events] == list(range(1, 42)))
    check('phase_hist_each_row_samplecount', all(len(row['phase_histogram']) == 150 and sum(row['phase_histogram']) == row['steps'] * 128 for row in events))
    check('phase_hist_result_coverage_exact', hist.tolist() == result['runtime_evidence']['phase_hist150']
          == result['runtime_evidence']['phase_coverage']['phase_hist150'] and int(hist.sum()) == 20992 and bool((hist > 0).all()))
    check('phase149_natural_end_count', int(hist[149]) == result['runtime_evidence']['replay_counts']['natural_end_rows'] == 71)
    capacity = read(run / 'capacity_observation.json')
    reset_events = [row for row in capacity['records'] if row['event'] == 'before_reset']
    partial = [row['selected_reset']['values'] for row in reset_events if row['selected_reset']['kind'] == 'native_explicit_reset_argument']
    check('partial_reset_actual_selected_ids_recorded', len(reset_events) == 50 and len(partial) == 49
          and all(0 < len(values) < 128 and len(set(values)) == len(values) and all(0 <= v < 128 for v in values) for values in partial))
    check('partial_reset_counts_not_assumed_disjoint_categories', max(result['runtime_evidence']['replay_counts']['natural_end_rows'], result['runtime_evidence']['replay_counts']['physical_terminated_rows']) <= sum(map(len, partial)) <= result['runtime_evidence']['replay_counts']['natural_end_rows'] + result['runtime_evidence']['replay_counts']['physical_terminated_rows'])
    metrics = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    check('cost_metrics40_iterations', len(metrics) == 40 and [row['iteration'] for row in metrics] == list(range(40)))
    check('actual_phase_counts', result['phase_calls'] == {'collect': 41, 'wm_train': 80, 'wm_holdout': 40, 'policy': 20, 'checkpoint': 2})
    check('native_profile_vs_recorded_calls_explained', metrics[-1]['profile/calls/collect'] == 40
          and metrics[-1]['profile/calls/wm'] == 40 and metrics[-1]['profile/calls/policy'] == 20)
    policy_trim = (metrics[-1]['profile/seconds/policy'] - metrics[22]['profile/seconds/policy']) / 17
    base_smoke = (metrics[19]['training/time'] - metrics[10]['training/time']) / 9
    phase_costs = {key: {'seconds': value, 'calls': result['phase_calls'][key],
                        'seconds_per_call_including_startup': value / result['phase_calls'][key]}
                   for key, value in result['phase_seconds'].items()}
    costs = {'schema': 'full151_smoke40_saved_cost_inputs:1', 'phase_costs': phase_costs,
        'native_call_wall_excluding_imports_s': result['native_call_wall_seconds_excluding_imports'],
        'worker_wall_s': runtime['wall_seconds'], 'watchdog_elapsed_s': state['elapsed_s'],
        'native_unphased_difference_s': result['native_call_wall_seconds_excluding_imports'] - sum(result['phase_seconds'].values()),
        'smoke_policy_native_profile_17call_trimmed_mean_s': policy_trim,
        'policy_trim_rule': 'native cumulative profile/seconds/policy at iteration39 minus iteration22 (third policy call), divided17; all20 mean remains separately recorded',
        'smoke_WM_only_base_proxy_s_per_iteration': base_smoke,
        'base_proxy_rule': '(training/time at19 minus10)/9; actual JSONL field is training/time, not wall_seconds',
        'checkpoint_proxy_s_per_save': phase_costs['checkpoint']['seconds_per_call_including_startup'],
        'future_primary100_formula': '1900*b_primary_last50_WMonly_training/time_slope +1800*p_policy_proxy +20*c_save +20s_unmeasured_overhead',
        'primary100_inputs_not_consumed_by_this_review': True,
        'scope': 'Wall-clock phase observations, not GPUbusy or method speed comparison; no current primary ETA is fabricated. Cold setup and warmup are not multiplied by50.'}
    details.update(source_pin_count=len(current), source_zip_member_count=len(zipped), actual8_storage_layout=layouts,
        initial_object_body_omega_max_abs_error_rad_s=body_error, phase_hist150_min=int(hist.min()),
        phase_hist150_max=int(hist.max()), observed_phases=np.flatnonzero(hist).tolist(),
        natural_end_rows=int(hist[149]), physical_terminated_rows=82,
        explicit_partial_reset_batches=len(partial), explicit_partial_reset_world_rows=sum(map(len, partial)),
        actual_pid=pid, launcher_pid=launch, actual_creation_date_UTC=observed['creation_date_utc'],
        saved_identity_recomputed_checks=life)
    check('all_input_files_unchanged', all(pin(files[key]) == value for key, value in inputs.items()))
    report = {'schema': 'full151_smoke40_saved_lifecycle_init_coverage_review:1',
        'status': 'PASS_SAVED_SCOPE' if all(checks.values()) else 'FAIL_SAVED_REVIEW',
        'passed': sum(checks.values()), 'total': len(checks), 'checks': checks, 'details': details,
        'source': pin(__file__), 'input_pins': inputs, 'bundle': cfg['input_bundle'],
        'additional_saved_inputs': {'A0_fullstate_request': pin(a0 / 'requested_initialization_and_control.npz'),
            'initialization': bundle['source_pins']['init_arrays'], 'native_fourfield_reference': bundle['source_pins']['native_data'],
            'actual_context': bundle['context']},
        'checkpoint_health': 'NOT_REPEATED_owned_by_Contact',
        'v1_harness_preserved': {'source':pin(ROOT/'scripts/review_full151_smoke40_lifecycle_saved_cpu.py'),'report':pin(ROOT/'outputs/full151_native_A1L_smoke40_lifecycle_independent_v1/report.json'),'corrections':['File SHA+absolute route rather than slash-dependent pin dict equality','Do not sum natural and physical termination categories as disjoint; full overlap row labels are not saved'],'actual_partial_reset_rows':sum(map(len,partial)),'natural_physical_intersection':'UNKNOWN_NOT_INDEPENDENTLY_RECONSTRUCTABLE_FROM_SAVED_SUMMARIES'},
        'scope': 'Saved source/ZIP/raw OS identity/actual8 dual readback/initial128state and phase-evidence aggregation only. Later partial-reset IDs are logged but their full per-reset physical state is not saved; do not call this independent raw reconstruction of all later resets or replay windows. Phase coverage is source-validated all-batch controller evidence, not independent ego demonstrations. No task-quality or physical-reference qualification.',
        'new_Env_Reader_FK_WM_optimizer_GPU_calls': 0, 'Torch_imported': False}
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    (out / 'cost_inputs.json').write_text(json.dumps(costs, indent=2, allow_nan=False) + '\n')
    with (out / 'phase_histogram150.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream); writer.writerow(['phase', 'legacy_source_frame', 'actual_transition_rows'])
        writer.writerows((k, k + 200, int(value)) for k, value in enumerate(hist))
    print(json.dumps({'status': report['status'], 'passed': report['passed'], 'total': report['total'],
        'failed': [key for key, value in checks.items() if not value], 'report': pin(out / 'report.json'),
        'cost_inputs': pin(out / 'cost_inputs.json'), 'trimmed_policy_s': policy_trim}, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__': main()
