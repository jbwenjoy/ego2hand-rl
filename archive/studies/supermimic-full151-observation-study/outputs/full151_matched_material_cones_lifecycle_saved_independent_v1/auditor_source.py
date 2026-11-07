"""Saved-only matched302 process/source/coverage review. No LP or native imports."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shlex
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151_matched_material_cones_lifecycle_saved_independent_v1'
RUN = ROOT / 'outputs/full151_matched_actual_material_cones_CPU_v1'
WATCH = RUN.with_name(RUN.name + '_watchdog')
ENROLL = ROOT / 'outputs/full151_matched_material_cone_enrollment_v1'
CONFIG = ROOT / 'configs/full151-matched-material-cones-CPU-authorized-v1.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha(path)}


def norm(path):
    return os.path.normcase(str(Path(path).resolve()))


def stamp(value):
    value = datetime.datetime.fromisoformat(value.replace('Z', '+00:00'))
    assert value.utcoffset() is not None
    return value


def main():
    OUT.mkdir(exist_ok=False)
    own_pin = pin(__file__)
    (OUT / 'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    p = read(CONFIG)
    report = read(RUN / 'report.json')
    runtime = read(RUN / 'runtime.json')
    state = read(WATCH / 'state.json')
    identity = read(WATCH / 'engineering_identity.json')
    post = read(WATCH / 'post_terminal_CIM.json')
    enrollment = read(ENROLL / 'report.json')
    inventory = read(ENROLL / 'source_inventory.json')
    rows = read(RUN / 'frame_results.json')
    checks = {}

    def need(name, condition):
        checks[name] = bool(condition)

    need('authorized_fixed_CPU_only_protocol',
         p['CPU_run_authorized'] is True and p['training_authorized'] is False
         and p['GPU_used'] is False and p['automatic_retry'] is False
         and p['max_LP_calls'] == 302 and p['soft_worker_seconds'] == 30
         and p['hard_process_seconds'] == 45)
    need('authorized_protocol_exact_expected_SHA',
         sha(CONFIG) == '1fee2990715e6a5a2a7ae351ebf67ee41633b5bf014a10f71f0deb35ce0b17f8')
    need('config_snapshots_JSON_exact', read(RUN / 'config.json') == p
         and read(WATCH / 'protocol_snapshot.json') == p)
    need('enrollment_protocol_archive_inventory_pins_match',
         enrollment['protocol'] == pin(CONFIG)
         and all(sha(enrollment[k]['path']) == enrollment[k]['sha256']
                 for k in ('archive', 'inventory', 'environment')))
    need('inventory_equals_all63_protocol_pins', inventory == p['source_pins'] and len(inventory) == 63)
    current = {k: sha(v['path']) == v['sha256'] for k, v in inventory.items()}
    need('all63_immutable_source_input_pins_current', all(current.values()))
    unique = {}
    archived_checks = {}
    with zipfile.ZipFile(enrollment['archive']['path']) as archive:
        need('archive_authorized_protocol_byte_exact', archive.read('authorized_protocol.json') == CONFIG.read_bytes())
        expected = {'authorized_protocol.json'}
        for index, (key, value) in enumerate(inventory.items()):
            path_key = norm(value['path'])
            if path_key in unique:
                need('duplicate_alias_SHA_consistency_' + key, unique[path_key]['sha256'] == value['sha256'])
                continue
            unique[path_key] = value
            member = f'{index:03d}_{Path(value["path"]).name}'
            expected.add(member)
            archived_checks[key] = hashlib.sha256(archive.read(member)).hexdigest() == value['sha256']
        need('exact58_unique_archive_members_plus_protocol', len(unique) == 58 and set(archive.namelist()) == expected)
    need('all58_preexecution_ZIP_bytes_match_frozen_sources_inputs', all(archived_checks.values()))
    need('enrollment_precedes_actual_launch', stamp(enrollment['created_UTC']) <= stamp(state['Popen_prelaunch_utc']))
    need('worker_execution_copy_byte_exact', sha(RUN / 'execution_source.py') == p['source_pins']['worker']['sha256'])
    need('supervisor_execution_copy_byte_exact', sha(WATCH / 'supervisor_source.py') == p['source_pins']['supervisor']['sha256'])

    supervisor = Path(p['source_pins']['supervisor']['path']).read_text(encoding='utf-8')
    v1 = (ROOT / 'scripts/run_full151_matched_material_cones_cpu_v1.py').read_text(encoding='utf-8')
    # The scientific watch tail remains byte-identical; only default-preflight cleanup changed.
    need('v2_preflight_retained_Popen_and_bounded_cleanup',
         'process=subprocess.Popen(' in supervisor and 'process.communicate(timeout=15)' in supervisor
         and 'd.lifecycle().stop_owned(process)' in supervisor
         and 'process.communicate(timeout=1)' in supervisor
         and "raise RuntimeError('CPU preflight interrupted/timeout; no LP permitted;" in supervisor)
    need('v2_original45s_watch_tail_unchanged',
         supervisor[supervisor.index("    if proof['status']"):] == v1[v1.index("    if proof['status']"):])
    need('v2_supervisor_exact_execution_SHA', p['source_pins']['supervisor']['sha256']
         == '8a4a8ade24a3c88b0003d769577c66d2ef24675a3292fd396218a0c42329d842')

    observed = state['actual_process_OS_observation']
    command = state['command']
    need('watch_completed_exit0_no_timeout_no_retry', state['status'] == 'completed'
         and type(state['exit_code']) is int and state['exit_code'] == 0
         and state['timeout'] is False and state['automatic_retry'] is False
         and state['hard_seconds'] == 45)
    need('actual_launcher_supervisor_PID_binding', state['actual_pid'] == 44492
         and state['launcher_pid'] == 32800 and state['supervisor_pid'] == 41600
         and runtime['pid'] == observed['pid'] == state['actual_pid']
         and runtime['parent_pid'] == observed['parent_pid'] == state['launcher_pid'])
    expected = [command[0], '-u', p['source_pins']['worker']['path'], '--protocol',
                str(CONFIG), '--output', str(RUN), '--run']
    need('actual_watch_command_exact8tokens', command == expected and norm(state['worker_output']) == norm(RUN))
    os_tokens = [v.strip('"') for v in shlex.split(observed['command_line'], posix=False)]
    need('raw_OS_command_exact_worker_protocol_output_tokens', len(os_tokens) == 8
         and os_tokens[1:] == command[1:])
    py = identity['selected_python_identity']
    need('actual_OS_base_executable_alias_samefile_and_SHA',
         os.path.samefile(os_tokens[0], observed['executable_path'])
         and os.path.samefile(observed['executable_path'], py['base_executable_raw'])
         and os.path.samefile(observed['executable_path'], py['base_executable_resolved'])
         and sha(observed['executable_path']) == py['base_executable_sha256'])
    need('chosen_venv_samefile_supervisor_identity', os.path.samefile(py['chosen_python'], py['supervisor_sys_executable'])
         and py['venv_selected_samefile'] is True)
    need('raw_OS_creation_exact_and_ordered', observed['status'] == 'exact_OS_identity_observed'
         and observed['creation_date_utc'] == '2026-11-04T18:56:06.6813620Z'
         and stamp(state['Popen_prelaunch_utc']) <= stamp(observed['creation_date_utc'])
         <= stamp(runtime['started_utc']) <= stamp(observed['observed_utc'])
         <= stamp(runtime['finished_utc']) <= stamp(state['finished_utc']))
    need('runtime_source_protocol_binding', runtime['plan_sha256'] == sha(CONFIG)
         and runtime['source_sha256'] == p['source_pins']['worker']['sha256'])
    need('watch_captured_initial_runtime_identity_matches_terminal',
         all(state['runtime_metadata'][k] == runtime[k]
             for k in ('pid', 'parent_pid', 'started_utc', 'plan_sha256', 'source_sha256')))
    need('saved_postterminal_exact_owned_PIDs_absent', post['status'] == 'NO_OWNED_PID_REMAINING'
         and post['queried_pids'] == [32800, 44492] and post['actual_CIM_rows'] == [])
    need('recorded12_identity_checks_agree_with_raw_saved_fields',
         len(identity['checks']) == 12 and all(v is True for v in identity['checks'].values())
         and identity['engineering_identity_qualified'] is True
         and identity['supervisor'] == p['source_pins']['supervisor'])

    need('terminal_report_runtime_same_fixed302', report['status'] == runtime['status'] == 'COMPLETED_MATCHED_CONE_CPU_SCREEN'
         and report['LP_calls'] == runtime['LP_calls'] == 302
         and report['accounted_slots'] == runtime['completed_slots'] == 302
         and report['error'] is None and runtime['rowid'] == 301)
    need('all302_rowids_unique_and_frame_major', len(rows) == 302
         and all(r['rowid'] == i and r['frame'] == i // 2
                 and r['condition'] == ('H123', 'H12345')[i % 2] for i, r in enumerate(rows)))
    with np.load(RUN / 'matrices_and_witnesses.npz', allow_pickle=False) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    with np.load(p['cues']['path'], allow_pickle=False) as saved:
        times = saved['time_s'].copy()
        geometry = saved['geometry_valid_mask'].copy()
    need('row_time_and_geometry_membership_exact', all(r['time_s'] == float(times[r['frame']])
         and r['available_finger_ids'] == ([1, 2, 3] if r['condition'] == 'H123' else [1, 2, 3, 4, 5]) for r in rows)
         and geometry.shape == (151, 5) and geometry.dtype == np.bool_ and geometry.all()
         and np.array_equal(arrays['geometry_valid_mask'], geometry))
    need('all302_completed_mask_true', arrays['completed'].shape == (151, 2)
         and arrays['completed'].dtype == np.bool_ and arrays['completed'].all())
    dimensions = {'A_H123_N_equivalent': (151, 6, 18), 'A_H12345_N_equivalent': (151, 6, 30),
                  'lambda_H123_normal_N': (151, 18), 'lambda_H12345_normal_N': (151, 30)}
    need('every_claimed_feasible_row_has_complete_finite_witness',
         all(arrays[k].shape == shape and np.isfinite(arrays[k]).all() for k, shape in dimensions.items())
         and all(r['status'] == 'FEASIBLE_CONDITIONAL_WITNESS' and r['solver_status'] == 0
                 and all(type(r[k]) in (int, float) and np.isfinite(r[k]) for k in
                         ('lambda_sum_normal_N', 'lambda_min_N', 'residual_inf_N_equivalent',
                          'force_residual_inf_N', 'torque_residual_inf_Nm')) for r in rows))
    counts = {name: {status: sum(r['condition'] == name and r['status'] == status for r in rows)
                     for status in sorted({r['status'] for r in rows})} for name in ('H123', 'H12345')}
    need('reported_status_counts_equal_all_raw_rows', counts == report['condition_counts']
         == {'H123': {'FEASIBLE_CONDITIONAL_WITNESS': 151}, 'H12345': {'FEASIBLE_CONDITIONAL_WITNESS': 151}})
    need('no_qualification_upgrade', report['physical_reference_qualified'] is False
         and report['GPU_Env_physics_WM_optimizer'] is False
         and identity['physical_outcome_qualified_by_supervisor'] is False)
    need('worker_cost_nested_in_watch_and_within_declared_budget',
         0 < report['wall_seconds'] == runtime['wall_seconds'] < state['elapsed_s'] < 45)
    need('source_inputs_unchanged_after_audit', all(sha(v['path']) == v['sha256'] for v in inventory.values())
         and pin(__file__) == own_pin)

    result = {
        'schema': 'full151_matched_material_cones_lifecycle_saved_independent:1',
        'status': 'PASS_SAVED_LIFECYCLE_SOURCE_AND_COVERAGE' if all(checks.values()) else 'FAIL_SAVED_REVIEW',
        'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
        'auditor': own_pin,
        'evidence': {k: pin(v) for k, v in {
            'protocol': CONFIG, 'report': RUN / 'report.json', 'rows': RUN / 'frame_results.json',
            'raw_arrays': RUN / 'matrices_and_witnesses.npz', 'runtime': RUN / 'runtime.json',
            'watch_state': WATCH / 'state.json', 'identity': WATCH / 'engineering_identity.json',
            'saved_post_terminal_CIM': WATCH / 'post_terminal_CIM.json', 'enrollment': ENROLL / 'report.json',
            'archive': ENROLL / 'preexecution_sources_and_inputs.zip', 'inventory': ENROLL / 'source_inventory.json'
        }.items()},
        'source_pin_count': len(inventory), 'unique_archived_paths': len(unique),
        'current_source_hash_matches': current, 'ZIP_member_hash_matches': archived_checks,
        'row_status_counts': counts,
        'process': {'actual_pid': runtime['pid'], 'launcher_pid': state['launcher_pid'],
                    'supervisor_pid': state['supervisor_pid'],
                    'OS_creation_UTC': observed['creation_date_utc'],
                    'worker_seconds': runtime['wall_seconds'], 'watch_seconds': state['elapsed_s'],
                    'cost_scope': 'Worker interval nested in watchdog; no external whole-CLI wall claimed.'},
        'scope': [
            'Only saved JSON/NPZ/ZIP, hashes and source text reviewed; production modules not imported.',
            'A-lambda primal reconstruction, optimality and material cone algebra belong to separate Contact review and were not repeated.',
            'All302 saved rows are conditional witnesses for two declared contact sets under one fixed material/pyramid and one reference demand. Neither winner selection nor physical feasibility follows.',
            '151 frames are temporally dependent from one seen source;302 LP slots are not independent experimental samples.',
            'Finite hypothetical Frisvad contact frames, no normal-force cap, no hand/actuator reachability, no actual contact-force or dynamics certificate.',
            'Saved postterminal CIM rows were rechecked as evidence, not queried afresh. No new OS process measurement or fault-injection cleanup test.'
        ],
        'execution': {'LP_calls': 0, 'production_imports': 0, 'model_loads': 0,
                      'native_queries': 0, 'Env': 0, 'physics': 0, 'GPU': 0,
                      'owner_source_or_data_modified': False}
    }
    (OUT / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'passed': result['passed'], 'total': result['total'],
                      'report': pin(OUT / 'report.json'),
                      'failed': [k for k, v in checks.items() if not v]}, indent=2))
    raise SystemExit(0 if all(checks.values()) else 1)


if __name__ == '__main__':
    main()
