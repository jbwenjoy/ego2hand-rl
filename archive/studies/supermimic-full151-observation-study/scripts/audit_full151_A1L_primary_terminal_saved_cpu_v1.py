"""Saved JSON/NPZ/ZIP lifecycle audit; never deserialize a checkpoint/model."""
from pathlib import Path
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
import shlex
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/full151_v3_primary2000_A1L_seed0_v1'
WATCH = RUN.with_name(RUN.name + '_watchdog')
LAUNCH = ROOT / 'outputs/full151_v3_primary2000_A1L_launch_v1'
OUT = ROOT / 'outputs/full151_v3_A1L_primary2000_saved_lifecycle_v1'
CONFIG = ROOT / 'configs/full151-v3-primary2000-A1L-seed0-v1.json'
FOUR_END = ROOT / 'outputs/full151_v3_A1L_terminal_fourPID_CIM_v1.json'
ETA_PATH = ROOT / 'outputs/full151_v3_A1L_primary100_live_pause_v1/ETA.json'
COUNTS = {'wm': 4000, 'fail_head': 4000, 'actor': 1800, 'critic': 7200}
PRIOR_KEYS = ('policy_update_calls', 'objective_forward_calls', 'active_rows', 'inactive_rows', 'dt_applications')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def valid_pin(value):
    return (type(value) is dict and set(value) == {'path', 'sha256'}
            and Path(value['path']).is_absolute() and sha(value['path']) == value['sha256'])


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def stamp(value):
    value = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    assert value.utcoffset() is not None
    return value


def tokens(line):
    return [v.strip('"') for v in shlex.split(line, posix=False)]


def json_rows(path):
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b'\n'):
        raise ValueError('Partial terminal JSONL: ' + str(path))
    return [json.loads(v) for v in raw.splitlines() if v.strip()]


def normal_counts(value, expected):
    return (set(value) <= set(COUNTS)
            and all(type(v) is int and v >= 0 for v in value.values())
            and all(value.get(k, 0) == v for k, v in expected.items()))


def main():
    OUT.mkdir(exist_ok=False)
    own = pin(__file__)
    (OUT / 'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    cfg = read(CONFIG); runtime = read(RUN / 'runtime.json'); result = read(RUN / 'result.json')
    marker = read(RUN / 'training_completion.json'); state = read(WATCH / 'state.json')
    identity = read(WATCH / 'engineering_identity.json'); post = read(WATCH / 'post_terminal_CIM.json')
    launch = read(LAUNCH / 'launch.json'); initial = read(LAUNCH / 'initial_CIM_recovered.json'); four_end = read(FOUR_END)
    cap = read(RUN / 'capacity_observation.json'); ctl = read(RUN / 'controller_descriptor.json')
    checks = {}; detail = {}

    def need(name, value): checks[name] = bool(value)

    need('schema2_A2_primary2000_fixed_budget_authorized_no_retry',
         cfg['schema'] == 'full151_knownrough_native_training_run:2' and cfg['role'] == 'primary_matched'
         and cfg['method'] == 'A1L' and type(cfg['seed']) is int and cfg['seed'] == 0
         and cfg['training']['max_iterations'] == 2000 and cfg['training']['wm_only_iters'] == 200
         and cfg['training']['warmup_steps'] == 64 and cfg['training']['num_envs'] == 128
         and cfg['GPU_run_authorized'] is True and cfg['automatic_retry'] is False)
    need('config_run_and_watch_snapshots_exact', read(RUN / 'config.json') == cfg
         and read(WATCH / 'config_snapshot.json') == cfg
         and runtime['plan_sha256'] == launch['config_sha256'] == sha(CONFIG))
    inventory = read(RUN / 'source/inventory.json')
    current = {k: valid_pin(v) for k, v in cfg['source_pins'].items()}
    need('source498_inventory_equals_actual_config_and_current_hashes',
         len(inventory) == 498 and inventory == cfg['source_pins'] and all(current.values()))
    archived = {}
    with zipfile.ZipFile(RUN / 'source/execution_sources.zip') as z:
        names = []
        for i, (key, value) in enumerate(inventory.items()):
            name = f'{i:04d}_{Path(value["path"]).name}'; names.append(name)
            archived[key] = hashlib.sha256(z.read(name)).hexdigest() == value['sha256']
        need('all498_preexecution_ZIP_bytes_exact', z.namelist() == names and all(archived.values()))
    need('worker_and_supervisor_actual_source_pins_current',
         runtime['source_sha256'] == launch['worker_sha256'] == sha(ROOT / 'scripts/execute_full151_learning_v3.py')
         and launch['supervisor_sha256'] == identity['supervisor_sha256'] == sha(ROOT / 'scripts/run_full151_learning_v3.py'))

    observed = state['actual_process_OS_observation']; py = identity['selected_python_identity']
    need('watch_exit0_no_timeout7200_no_retry', state['status'] == 'completed' and state['exit_code'] == 0
         and state['timeout'] is False and state['hard_seconds'] == 7200 and state['automatic_retry'] is False)
    need('runtime_terminal2000_and_result_completed', runtime['status'] == 'completed_full151_training_v2'
         and runtime['iteration'] == result['final_iteration'] == 2000
         and result['schema'] == 'full151_native_training_completion:2'
         and result['status'] == runtime['result_status'] == 'COMPLETED_ENGINEERING_TRAINING_NOT_TASK_QUALITY'
         and result['cleanup_errors'] == [] and result['from_scratch'] is True
         and result['old_checkpoint_migration'] is False and result['physical_reference_qualified'] is False)
    need('worker_marker_schema2_config_exact_final_result', marker['schema'] == 'full151_native_worker_training_completion:2'
         and marker['config_sha256'] == sha(CONFIG) and marker['result'] == result)
    need('launch4_PID_identity_chain_and_capture_pin', launch['schema'] == 'full151_owned_hidden_supervisor_launch:2'
         and launch['method'] == 'A1L' and launch['hidden'] is True and launch['automatic_retry'] is False
         and valid_pin(launch['initial_CIM'])
         and [v['pid'] for v in initial] == [29736, 34456, 36500, 16096]
         and launch['retained_launcher_pid'] == initial[0]['pid']
         and initial[1]['parent_pid'] == initial[0]['pid']
         and initial[2]['parent_pid'] == initial[1]['pid']
         and initial[3]['parent_pid'] == initial[2]['pid']
         and state['supervisor_pid'] == initial[1]['pid']
         and state['launcher_pid'] == initial[2]['pid']
         and runtime['pid'] == state['actual_pid'] == initial[3]['pid']
         and runtime['parent_pid'] == initial[3]['parent_pid'])
    need('launch_creation_order_and_watch_actual_OS_match',
         stamp(launch['launched_utc']) <= stamp(initial[0]['creation_utc']) <= stamp(initial[1]['creation_utc'])
         <= stamp(state['Popen_prelaunch_utc']) <= stamp(initial[2]['creation_utc']) <= stamp(initial[3]['creation_utc'])
         <= stamp(runtime['started_utc']) <= stamp(observed['observed_utc'])
         and observed['pid'] == initial[3]['pid'] and observed['parent_pid'] == initial[3]['parent_pid']
         and observed['creation_date_utc'] == initial[3]['creation_utc']
         and observed['command_line'] == initial[3]['command_line']
         and observed['executable_path'] == initial[3]['executable_path'])
    worker_tail = ['-u', str(ROOT / 'scripts/execute_full151_learning_v3.py'), '--config', str(CONFIG), '--output', str(RUN), '--run']
    supervisor_tail = ['-u', str(ROOT / 'scripts/run_full151_learning_v3.py'), '--config', str(CONFIG), '--run']
    need('all4_raw_OS_command_routes_exact',
         all(tokens(row['command_line'])[1:] == (supervisor_tail if i < 2 else worker_tail)
             for i, row in enumerate(initial))
         and state['command'][1:] == worker_tail)
    need('all4_raw_OS_executables_samefile_expected_venv_base_alias',
         all(os.path.samefile(tokens(row['command_line'])[0], row['executable_path'])
             and os.path.samefile(row['executable_path'], py['chosen_python'] if i in (0, 2) else py['base_executable_raw'])
             for i, row in enumerate(initial))
         and os.path.samefile(py['base_executable_raw'], py['base_executable_resolved'])
         and sha(py['base_executable_raw']) == py['base_executable_sha256'])
    need('watch_runtime_identity_and_source_snapshot_match',
         all(state['runtime_metadata'][k] == runtime[k] for k in ('pid', 'parent_pid', 'started_utc', 'plan_sha256', 'source_sha256'))
         and identity['actual_pid'] == runtime['pid'] and identity['launcher_pid'] == state['launcher_pid']
         and len(identity['checks']) == 12 and all(v is True for v in identity['checks'].values())
         and identity['engineering_identity_qualified'] is True)
    need('saved_worker_pair_postterminal_CIM_empty', post['status'] == 'NO_OWNED_PID_REMAINING'
         and sorted(post['queried_pids']) == [16096, 36500] and post['actual_CIM_rows'] == [])
    need('fresh_root4PID_postterminal_CIM_empty_only_queried_scope',
         four_end['schema'] == 'full151_owned_fourPID_terminal_CIM:1'
         and four_end['queried_pids'] == [29736, 34456, 36500, 16096]
         and four_end['actual_CIM_rows'] == [] and four_end['all_queried_PID_absent'] is True
         and stamp(four_end['observed_UTC']) > stamp(state['finished_utc']))

    need('all_saved_hook_optimizer_counts_match_budget', result['actual_optimizer_steps'] == runtime['actual_optimizer_steps'] == COUNTS
         and result['sim_steps'] == 648192
         and result['phase_calls'] == {'collect': 2001, 'wm_train': 4000, 'wm_holdout': 2000, 'checkpoint': 21, 'policy': 1800})
    index = json_rows(RUN / 'checkpoint_index.jsonl')
    need('fixed_100step_index20_plus_final_not_quality_selected', len(index) == 21
         and [row['iteration'] for row in index] == list(range(100, 2001, 100)) + [2000]
         and [Path(row['checkpoint']).name for row in index] == [f'model_{i}.pt' for i in range(100, 2001, 100)] + ['model_final.pt']
         and all(row['selection'] == 'declared iteration only; no quality selection' for row in index))
    index_valid = True
    for row in index:
        i = row['iteration']; expected = {'wm': i * 2, 'fail_head': i * 2, 'actor': max(0, i - 200), 'critic': max(0, i - 200) * 4}
        path = Path(row['checkpoint'])
        index_valid = (index_valid and path.parent.resolve() == RUN / 'train' and sha(path) == row['sha256']
                       and read(path.with_suffix('.meta.json')) == row
                       and normal_counts(row['actual_optimizer_steps'], expected))
    need('each_declared_index_file_SHA_sidecar_and_count_scope_exact', index_valid)
    need('final2000_file_identity_bound_to_terminal_not_besthistory', valid_pin(result['checkpoint'])
         and Path(result['checkpoint']['path']).resolve() == RUN / 'train/model_final.pt'
         and result['checkpoint']['sha256'] == index[-1]['sha256'] == runtime['checkpoint']['sha256']
         and runtime['checkpoint'] == index[-1] and index[-1]['sim_steps'] == 648192)
    real_rows = json_rows(RUN / 'full151_actualu_evidence.jsonl')
    need('all2001_collect_records_account648192_real_transitions',
         len(real_rows) == 2001 and sum(v['real_transitions'] for v in real_rows) == 648192
         and real_rows[0]['steps'] == 64 and sum(v['steps'] for v in real_rows) == 5064
         and all(row['collect_call'] == i + 1 and row['worlds'] == 128
                 and row['real_transitions'] == row['steps'] * 128 and row['inactive_real_actions'] == 0
                 and row['all_batch_exact'] is True and row['all_finite'] is True
                 and all(v == 0 for v in row['max_abs_errors'].values()) for i, row in enumerate(real_rows)))

    bundle = read(cfg['input_bundle']['path']); root = ctl['root_descriptor']
    strip = {k: v for k, v in ctl.items() if k not in ('reference_controller_signature', 'latest_actualu_evidence')}
    need('bundle_controller105_root_source_signature_exact', valid_pin(cfg['input_bundle'])
         and root['input_bundle'] == cfg['input_bundle'] and root['trained_method'] == 'A1L'
         and root['training_seed'] == 0 and root['run_role'] == 'primary_matched'
         and root['config_sha256'] == sha(CONFIG) and root['execution_source_pins'] == cfg['source_pins']
         and ctl['format_version'] == 105 and ctl['margin_rad'] == .020
         and ctl['reference_controller_signature'] == result['reference_signature']
         == hashlib.sha256(canonical(strip).encode()).hexdigest()
         and bundle['physical_reference_qualified'] is False and bundle['known_rough_reference'] is True)
    with np.load(bundle['source_pins']['control_arrays']['path'], allow_pickle=False) as z:
        baseline = z['u_postC2_150_FP32_rad'].copy()
    need('baseline150_FP32_buffer_bound_not_changed_by_prior', baseline.shape == (150, 20)
         and baseline.dtype == np.float32 and np.isfinite(baseline).all()
         and hashlib.sha256(np.ascontiguousarray(baseline).tobytes()).hexdigest() == ctl['baseline_u150_FP32_sha256'])
    prior = result['full151_prior_evidence']; meta = read(RUN / 'prior_metadata.json'); rng = read(RUN / 'objective_build_rng.json')
    need('A1L_prior_NONE_no_spec_or_objective_and_all_counters_zero',
         prior['schema']=='full151_imagination_prior_checkpoint_evidence:1'
         and prior['method']=='A1L' and prior['status']=='NONE' and prior['spec_pin'] is None
         and prior['objective_metadata'] is None and all(prior[k]==0 for k in PRIOR_KEYS)
         and cfg['contact_binding'] is None and result['contact_objective'] is None
         and prior['real_collect_shaped'] is False and prior['WMfit_shaped'] is False
         and prior['dead_end_rule']=='unchanged native full151 phase<150 hardmask after reward'
         and prior['logging_errors']==[])
    need('A1L_initial_prior_metadata_and_RNG_NOT_CONSTRUCTED_preserved',
         meta==prior and rng==prior['build_rng'] and rng['status']=='NOT_CONSTRUCTED'
         and rng['method']=='A1L' and rng['objective']=='NONE' and rng['RNG_capture_required'] is False
         and not (RUN/'full151_prior_evidence.jsonl').exists())
    metrics=json_rows(RUN/'metrics.jsonl')
    need('all2000_metrics_have_no_prior_fields',len(metrics)==2000
         and not any(any(k.startswith('prior/') for k in row) for row in metrics))
    recovery=read(LAUNCH/'launch_record_recovery.json')
    failed_source=Path(recovery['failed_source']);failed_text=failed_source.read_text(encoding='utf-8-sig')
    need('post_Start_launch_record_error_source_preserved_no_relaunch',
         sha(failed_source)==recovery['failed_source_sha256']
         and failed_text.count('Start-Process -FilePath')==1
         and '-Algorithm SHA1L56' in failed_text
         and failed_text.index('Start-Process -FilePath')<failed_text.index('-Algorithm SHA1L56')
         and recovery['relaunch_performed'] is False and recovery['process_modified'] is False
         and launch['launch_record_recovered_after_source_error'] is True
         and valid_pin(launch['failure_recovery']))
    need('recovered_launch_time_honest_original_launcher_OS_creation',
         launch['launched_utc']==initial[0]['creation_utc']
         and launch['launch_timestamp_source']=='Retained original launcher OS CreationDate; precise script invocation UTC was not saved before recording failure'
         and recovery['original_four_PID_chain']==[29736,34456,36500,16096])

    records = cap['records']; raw_or = 0; unknown = 0; record_valid = True
    events = Counter()
    for i, record in enumerate(records):
        events[record['event']] += 1
        try:
            flags = np.asarray(record['raw_flags_values'], dtype=np.dtype(record['raw_flags_dtype']))
            good = flags.dtype.kind in 'iu' and flags.size > 0 and list(flags.shape) == record['raw_flags_shape']
            bits = 0
            for value in flags.reshape(-1): bits |= int(value)
            good = good and record['raw_bitmask'] == bits and record['read_status'] == ('OK' if bits == 0 else 'OVERFLOW')
            raw_or |= bits
        except Exception:
            good = False
        if not good: unknown += 1
        record_valid = record_valid and good and record['sequence'] == i + 1
        record_valid = record_valid and record['flags_cleared_by_observer'] is False
    need('final_capacity_raw_OR_unknown_records_and_final_once', record_valid and raw_or == unknown == 0
         and cap['record_count'] == len(records) and cap['before_reset_read_count'] == events['before_reset']
         and cap['final_read_count'] == events['final'] == 1 and records[-1]['event'] == 'final'
         and cap['status'] == 'OK' and cap['unknown_read_count'] == cap['cumulative_OR_bitmask'] == 0
         and cap['finalized'] is True and cap['entire_run_finalized'] is True
         and cap['evidence_kind'] == 'entire_run_final' and cap['prefix_at_checkpoint_capacity_qualified'] is False
         and result['full151_capacity_qualified_final'] is True and valid_pin(result['full151_capacity_observation']))
    need('capacity_final_source_identity_matches_execution',
         all(valid_pin(v) and any(v == p for p in cfg['source_pins'].values()) for v in cap['source_pins'].values()))

    eta = read(ETA_PATH); predicted = stamp(eta['estimated_finish_Asia_Shanghai']); actual = stamp(runtime['finished_utc'])
    eta_error = (predicted - actual).total_seconds()
    need('ETA_same_A2_source_anchor_and_point_error_recomputed', eta['method'] == 'A1L'
         and eta['anchor_checkpoint_iteration'] == 100
         and Path(eta['same_method_smoke_run']).resolve() == ROOT / 'outputs/full151_v2_smoke40_A1L_seed0_v1'
         and math.isclose(eta_error, 30.386901, abs_tol=1e-6)
         and stamp(eta['finish_range_Asia_Shanghai'][0]) <= actual <= stamp(eta['finish_range_Asia_Shanghai'][1]))
    need('cost_scopes_nested_not_added', 0 < result['native_call_wall_seconds_excluding_imports']
         < runtime['wall_seconds'] < state['elapsed_s'] < cfg['hard_process_seconds'])
    need('sources_and_config_still_current_after_saved_audit',
         all(valid_pin(v) for v in cfg['source_pins'].values()) and pin(__file__) == own)
    ok = bool(checks) and all(checks.values())
    detail.update(source_pin_count=len(inventory), current_source_matches=current, ZIP_member_matches=archived,
                  prior_counts={k: prior[k] for k in PRIOR_KEYS}, capacity_event_counts=dict(events),
                  worker_wall_seconds=runtime['wall_seconds'], watchdog_wall_seconds=state['elapsed_s'],
                  native_wall_seconds_excluding_imports=result['native_call_wall_seconds_excluding_imports'],
                  exact_actual_worker_PID=runtime['pid'], launch_four_PID_chain=[v['pid'] for v in initial],
                  source_backed_optimizer_hook_counts=result['actual_optimizer_steps'],
                  ETA={'predicted_finish_Asia_Shanghai': eta['estimated_finish_Asia_Shanghai'],
                       'actual_worker_finish_UTC': runtime['finished_utc'],
                       'prediction_late_seconds': eta_error,
                       'scope': 'One A1L run descriptive error, not general ETA calibration or task qualification.'})
    report = {'schema': 'full151_A1L_primary_saved_lifecycle_independent:1',
              'status': 'PASS_SAVED_PRIMARY_TERMINAL_LIFECYCLE_NOT_TASK_QUALITY' if ok else 'FAIL_SAVED_LIFECYCLE_REVIEW',
              'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
              'source_pins': {'auditor': own},
              'artifact_pins': {k: pin(p) for k, p in {
                  'config': CONFIG, 'runtime': RUN / 'runtime.json', 'result': RUN / 'result.json',
                  'marker': RUN / 'training_completion.json', 'source_inventory': RUN / 'source/inventory.json',
                  'source_archive': RUN / 'source/execution_sources.zip', 'controller': RUN / 'controller_descriptor.json',
                  'prior_metadata':RUN/'prior_metadata.json','objective_build_rng':RUN/'objective_build_rng.json','launch_recovery':LAUNCH/'launch_record_recovery.json','failed_launch_source':failed_source,'actualu_log': RUN / 'full151_actualu_evidence.jsonl',
                  'capacity_final': RUN / 'capacity_observation.json', 'watch_state': WATCH / 'state.json',
                  'identity': WATCH / 'engineering_identity.json', 'launch': LAUNCH / 'launch.json',
                  'initial_CIM': LAUNCH / 'initial_CIM_recovered.json', 'fresh_fourPID_terminal_CIM': FOUR_END,
                  'ETA': ETA_PATH, 'final_checkpoint': RUN / 'train/model_final.pt'}.items()},
              'details': detail, 'Goal_turn_classification': 'progress',
              'claim_scope': [
                  'Only source bytes, saved JSON/NPZ/ZIP and file hashes were reviewed. No checkpoint tensor deserialization or model/Env/GPU/physics construction.',
                  'Optimizer counts and whole-prefix actualu/RNG/prior summaries are saved native-hook evidence corroborated across logs/markers, not independently reexecuted optimizer or RNG state measurements.',
                  'All4 launch identities are source-bound; current absence is established only for the4 PID values in the saved fresh Root CIM sidecar, as of that observation. No other process/machine idleness inferred.',
                  'A1L completed one known-rough full151 primary training. Full scientific Goal and matched method benefit remain incomplete; physical task quality is not assessed here.',
                  'Launch recovery preserves the original post-Start source error and original four identities; launched_utc is OS launcher creation, not a fabricated invocation timestamp. Fixed final2000 is selected by protocol/index, not a history search. Learning curves, Actor performance, model accuracy and physical gradient qualification require their separate evidence.'
              ], 'execution': {'checkpoint_or_model_loads': 0, 'forward': 0, 'Env': 0, 'GPU': 0, 'physics': 0, 'new_OS_queries': 0}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'status': report['status'],
                      'passed': report['passed'], 'total': report['total'],
                      'failed': [k for k, v in checks.items() if not v]}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
