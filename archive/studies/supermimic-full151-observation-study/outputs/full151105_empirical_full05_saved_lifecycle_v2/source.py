"""Saved-only full05 execution/source/scope audit; no native/model imports."""
from pathlib import Path
from datetime import datetime
import ast
import hashlib
import json
import shlex
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/full151105_gradient_empirical_gradient05_v2'
WATCH = RUN.with_name(RUN.name + '_watchdog')
OUT = ROOT / 'outputs/full151105_empirical_full05_saved_lifecycle_v2'
CONFIG = ROOT / 'configs/full151105-gradient-empirical_gradient05-v2.json'
LABELS = [f'{p}/{k}' for p in ('A0', 'A2') for k in (0, 49, 99, 149)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def same(a, b):
    return a['sha256'] == b['sha256'] and Path(a['path']).samefile(b['path'])


def main():
    OUT.mkdir(exist_ok=False); (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    cfg = read(CONFIG); r = read(RUN / 'runtime.json'); p = read(RUN / 'physical_report.json')
    w = read(WATCH / 'state.json'); identity = read(WATCH / 'engineering_identity.json'); post = read(WATCH / 'post_terminal_CIM.json')
    manifest = read(cfg['foreign_inventory_review']['path']); discovery = read(cfg['inventory_discovery_health']['path'])
    snapshots = read(RUN / 'snapshot_records.json'); completion = read(RUN / 'completion.json'); result = read(RUN / 'result.json')
    inputs = {k: pin(path) for k, path in {'config': CONFIG, 'runtime': RUN / 'runtime.json',
        'producer_report': RUN / 'physical_report.json', 'result': RUN / 'result.json', 'completion': RUN / 'completion.json',
        'watch_state': WATCH / 'state.json', 'identity': WATCH / 'engineering_identity.json', 'post_CIM': WATCH / 'post_terminal_CIM.json',
        'snapshot_records': RUN / 'snapshot_records.json', 'source_inventory': RUN / 'source/inventory.json',
        'source_zip': RUN / 'source/execution_sources.zip'}.items()}
    checks = {}
    def check(name, value): checks[name] = bool(value)
    check('fixed05_config_namespace_authorization_and_finite_budget', cfg['schema'] == 'full151105_same_warp_physical_probe_run:2'
          and cfg['role'] == 'full151105_serial_physical_probe' and cfg['probe_mode'] == 'serial_full'
          and type(cfg['evaluation_seed']) is int and cfg['evaluation_seed'] == 0 and cfg['GPU_run_authorized'] is True
          and cfg['training_authorized'] is cfg['automatic_retry'] is cfg['compile_fallback'] is False
          and (cfg['soft_worker_seconds'], cfg['hard_process_seconds']) == (180, 240) and Path(cfg['output']).samefile(RUN))
    check('saved_config_copies_same_SHA', all(pin(path)['sha256'] == inputs['config']['sha256'] for path in (RUN / 'config.json', WATCH / 'config_snapshot.json')))
    source = read(RUN / 'source/inventory.json')
    check('immutable_execution_inventory_exact_config_map', source == cfg['source_pins'])
    live_ok = True; archive_ok = True
    with zipfile.ZipFile(RUN / 'source/execution_sources.zip') as archive:
        check('ZIP_member_count_equals_frozen_pin_table', len(archive.namelist()) == len(source))
        for i, (key, value) in enumerate(source.items()):
            live_ok &= pin(value['path'])['sha256'] == value['sha256']
            archive_ok &= hashlib.sha256(archive.read(f'{i:04d}_{Path(value["path"]).name}')).hexdigest() == value['sha256']
    check('every_pinned_source_or_input_current_bytes_match', live_ok)
    check('every_archived_source_or_input_bytes_match', archive_ok)
    for key in ('gradient_entry', 'gradient_supervisor', 'gradient_builder', 'physical_serial_probe', 'transaction105'):
        ast.parse(Path(source[key]['path']).read_text(encoding='utf-8'))
    check('current_worker_entry_supervisor_transaction_exact_reviewed_bytes', all(source[k]['sha256'] == v for k, v in {
        'gradient_entry': '517c102bf3415c0a34523556bceab5255137c76daedc4f60ae9027101f6375a8',
        'gradient_supervisor': '06580f1e8a56a1ebb71811fcd7b24c9e83d723433f38bfb0e31119ee233c620b',
        'physical_serial_probe': 'a24c89da47f4aad7acfc9d9dfd1f053518cdc878230716bb2c857127ec13e592',
        'transaction105': 'f1bd0eafed58783a6efbb78a0ba40da086b36851119eddd00e29fcddb972a55c'}.items()))
    check('watch_completed_exit0_without_timeout_or_retry', w['status'] == 'completed' and w['exit_code'] == 0
          and w['timeout'] is w['automatic_retry'] is False and w['hard_seconds'] == 240)
    check('saved_OS12_all_positive', identity['engineering_identity_qualified'] is True
          and len(identity['checks']) == 12 and all(v is True for v in identity['checks'].values()))
    obs = w['actual_process_OS_observation']; chosen = identity['selected_python_identity']
    check('actual_worker_launcher_runtime_saved_PID_PPID_identity', obs['status'] == 'exact_OS_identity_observed'
          and obs['pid'] == r['pid'] == w['actual_pid'] == identity['actual_pid'] == 37196
          and obs['parent_pid'] == r['parent_pid'] == w['launcher_pid'] == identity['launcher_pid'] == 33208)
    check('executable_alias_and_selected_venv_bound_without_venv_base_confusion', Path(chosen['chosen_python']).samefile(chosen['supervisor_sys_executable'])
          and Path(obs['executable_path']).samefile(chosen['base_executable_raw'])
          and Path(chosen['base_executable_raw']).samefile(chosen['base_executable_resolved'])
          and pin(obs['executable_path'])['sha256'] == chosen['base_executable_sha256'])
    command = [x.strip('"') for x in shlex.split(obs['command_line'], posix=False)]
    check('actual_command_fixed8tokens_worker_config_output_run', len(command) == 8 and command[1] == '-u'
          and Path(command[0]).samefile(obs['executable_path']) and command[2:] == w['command'][2:]
          and Path(command[2]).samefile(source['gradient_entry']['path']) and command[3] == '--config'
          and Path(command[4]).samefile(CONFIG) and command[5] == '--output' and Path(command[6]).samefile(RUN) and command[7] == '--run')
    dt = lambda value: datetime.fromisoformat(value.replace('Z', '+00:00'))
    bounds = w['actual_creation_conservative_bounds']
    check('saved_creation_and_terminal_chronology', dt(bounds['lower_utc']) <= dt(obs['creation_date_utc']) <= dt(bounds['upper_utc'])
          and dt(obs['creation_date_utc']) <= dt(r['started_utc']) <= dt(r['finished_utc']) <= dt(w['finished_utc']))
    check('postterminal_saved_query_only_exact_owned_two_PIDs_empty', set(post['queried_pids']) == {37196, 33208}
          and post['actual_CIM_rows'] == [] and post['status'] == 'NO_OWNED_PID_REMAINING')
    check('runtime_source_and_plan_bound_completed', r['status'] == 'completed_full151_gradient_probe'
          and r['plan_sha256'] == inputs['config']['sha256'] and r['source_sha256'] == source['gradient_entry']['sha256'])
    check('completion_and_result_same_config_and_saved_producer_payload', completion['config_sha256'] == inputs['config']['sha256']
          and completion['result'] == result == p)
    check('producer_reported_full_planned_coverage_engineering_not_independent_recount', p['schema'] == 'full151105_serial_physical_probe_worker_report:3'
          and p['status'] == 'COMPLETED_PHYSICAL_PROBE' and p['engineering_qualified'] is p['baseline_all_passed'] is True
          and p['episode_step_attempted'] == p['episode_step_returned'] == p['scheduled_substeps'] == 1808
          and p['completed_branch_count'] == 152 and p['completed_snapshot_count'] == 8)
    check('producer_reported_nontraining_nonmodel_scope', all(p[k] == 0 for k in ('optimizer_updates', 'WM_forwards', 'Actor_forwards', 'model_loads', 'Runner_instances')))
    check('producer_capacity_finalization_claim_separate_from_Contact_raw_audit', p['capacity_finalized'] is p['capacity_qualified'] is True)
    check('manifest_exact_review2_and_all8labels_not_union', manifest['schema'] == 'full151105_foreign_inventory_per_snapshot_source_review:2'
          and manifest['approved_probe_modes'] == ['serial_full'] and manifest['allowed_snapshot_labels'] == LABELS
          and list(manifest['per_snapshot']) == LABELS and manifest['phase0_only'] is False)
    check('manifest_input_worker_transaction_current_aliases', same(manifest['worker'], source['physical_serial_probe'])
          and same(manifest['transaction'], source['transaction105']) and same(manifest['input_bundle'], cfg['input_bundle']))
    check('manifest_actual_discovery_top4_provenance_join', same(manifest['source_inventory_saved_review'], cfg['inventory_discovery_health'])
          and same(manifest['source_inventory_config'], discovery['config']) and same(manifest['source_inventory_report'], discovery['producer_report'])
          and Path(manifest['source_inventory_run']).samefile(discovery['run_dir']))
    check('manifest_all102_inner_sources_archived_and_current', all(any(same(value, candidate) for candidate in source.values())
          for value in manifest['source_pins'].values()))
    current_labels = []; label_counts = {}
    for record in snapshots:
        label = record['path'] + '/' + str(record['k']); current_labels.append(label)
        table = manifest['per_snapshot'][label]; oldpins = discovery['inventory_snapshot_pins'][label]
        check(label + '_source_discovery_description_topology_joins', all(same(table['inventory'][k], oldpins[k]) for k in ('description', 'topology')))
        artifacts = record['inventory_archive']
        check(label + '_fresh05_recorded_description_topology_SHA_match', all(pin(artifacts[k]['path'])['sha256'] == artifacts[k]['sha256'] for k in ('description', 'topology')))
        for key in ('description', 'topology'):
            inputs['fresh05_' + label.replace('/', '_') + '_' + key] = artifacts[key]
        desc = read(artifacts['description']['path']); top = read(artifacts['topology']['path'])
        actual = {(v['path'], v['type']) for v in desc['opaque_identity_only']}; allowed = {(v['path'], v['type']) for v in table['allowed_identity_only']}
        check(label + '_fresh05_current_exact_inventory_matches_only_own_table', desc['label'] == label and actual == allowed
              and len(actual) == len(desc['opaque_identity_only']) == len(table['allowed_identity_only'])
              and top['worker_source'] == manifest['worker'] and top['transaction_source'] == manifest['transaction'])
        check(label + '_local_empirical_flags_do_not_claim_full_hidden_restore', record['baseline3_qualified'] is True
              and record['full_native_restore_qualified'] is False
              and ('whole_graph_hidden_state_qualified' not in record or record['whole_graph_hidden_state_qualified'] is False)
              and 'UNKNOWN' in record['native_restore_qualification_scope'])
        label_counts[label] = len(actual)
    check('fresh05_exact8labels_order', current_labels == LABELS)
    check('overall_hidden_gradient_reference_qualification_stay_false', p['gradient_qualified'] is p['whole_graph_hidden_state_qualified'] is p['physical_reference_qualified'] is False
          and manifest['whole_graph_hidden_state_qualified'] is manifest['physical_gradient_qualified'] is False)
    check('primary_controlled_post20_goal_k_not_native15', p['scalar_contract']['primary'] == 'matched_post20_gpu_score'
          and p['scalar_contract']['primary_state_phase'] == 'postintegration20ms' and p['scalar_contract']['goal_phase'] == 'preaction k')
    check('review_no_framework_import_or_execution', not any(k.split('.')[0] in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic', 'full151_native', 'full151_gradient') for k in sys.modules))
    check('all_reviewed_terminal_metadata_unchanged', all(pin(v['path'])['sha256'] == v['sha256'] for v in inputs.values()))
    limits = ['Engineering completion and visited inventory/source/OS identity are distinct from whole-hidden-state restoration or a true physical Jacobian.',
        'Graph executable cache and graph-owned temporary workspaces were not exhaustively serialized/proved independent; same-input baseline replay is empirical and local.',
        'Fresh donors replay fixed A0 zero/A2 saved actions; this is neither archived-state reproduction nor fresh closed-loop Actor inference.',
        'Primary scalar is controlled GPU FP32 formula on raw20ms/goal_k. Original RewardManager derived15ms score and k+1 rescore are secondary; self15 sensor labels do not mean hand-object contact.',
        '152 branches,24baseline repeats and8dependent snapshots from one seen source/seed are not independent sampleN, method-benefit evidence or physical-reference qualification.',
        'No live CIM was queried here. Absence applies only to two PIDs in the saved post-terminal query, not machine-wide idleness.',
        'Counts, raw physics equations, baseline errors, force/label reconstruction and cumulative raw capacity ledger arithmetic remain the separate Contact audit. This report checks producer claims and exact provenance, not a duplicated raw numerical regrade.',
        'Memory RNG/weights/normalizers are not independently remeasured. File SHA/source contracts and zero model-load/inference claims are distinct evidence scopes.']
    checklist = ['Contact raw1500+? sample/count audit against1808expected,452intervals/152branches/8snapshots/24baselines; preserve all failure/NOT_RUN masks',
        'Verify16initial serializer gets+152restored gets/8setters,152begin/finish/nativeRestore counts without summing nested backend API counters',
        'External raw capacity OR/UNKNOWN across every before-restore/substep/branch/donor/final; no sticky reset erased by snapshot',
        'Actual8/context/reset and original EMA/source-action/cache/absolute-u/domain checks for every interval; intervention only cached target after one process',
        'Primary native-formula GPU F32term/sum actual20ms plus correct goal_k, rawfound/self15 lastsolver15ms and original RM15secondary separate',
        'All8perlabel3freshbaseline numerical gates before directions, positive/negative/effective-F32 endpoints andd_eff/cancellation/noise/epsilon classifications retained',
        'No true-gradient, HO-force-gradient, fullhiddenstate, full-motion success or method advantage inferred; any CPU WM consumption preregistered/separate']
    # Correct wording without touching scientific input: checklist is prospective, not a count assertion.
    checklist[0] = 'Contact raw sample/count audit against1808expected,452intervals/152branches/8snapshots/24baselines; preserve all failure/NOT_RUN masks'
    report = {'schema': 'full151105_full05_saved_lifecycle_and_scope_review:1',
        'status': 'PASS_SAVED_SOURCE_LIFECYCLE_LOCAL_EMPIRICAL_SCOPE_ONLY' if all(checks.values()) else 'FAIL_SAVED_SOURCE_LIFECYCLE_CHECK',
        'checks': checks, 'passed': sum(checks.values()), 'total': len(checks), 'source': pin(__file__), 'inputs': inputs,
        'source_pin_count': len(source), 'per_label_opaque_counts': label_counts,
        'per_snapshot_whole_hidden_flag': {v['path'] + '/' + str(v['k']): v.get('whole_graph_hidden_state_qualified', 'ABSENT_NOT_SERIALIZED') for v in snapshots},
        'scope_authority': 'Producer top-level whole_graph_hidden_state_qualified False plus actual per-snapshot full_native_restore_qualified False and UNKNOWN empirical scope; absent is not replaced by false.',
        'actual_OS': obs, 'saved_terminal_queried_PIDs': post['queried_pids'],
        'costs_nested_not_additive': {'worker_wall_seconds': r['wall_seconds'], 'watch_wall_seconds': w['elapsed_s'],
            'native_wall_seconds_excluding_imports': p['native_wall_seconds_excluding_imports'], 'scope': 'Nested observed scopes; no GPUbusy or whole-process kernel-count inference.'},
        'interpretation_limits': limits, 'independent_raw_review_checklist': checklist,
        'native_full_hidden_restore_qualified': False, 'physical_gradient_qualified': False,
        'execution': {'production_imports': 0, 'live_CIM_queries': 0, 'model_or_weight_deserialization': 0,
                      'Env': 0, 'WM_Actor_forward': 0, 'physics': 0, 'GPU': 0}}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'checks': [report['passed'], report['total']], 'status': report['status']}, indent=2))


if __name__ == '__main__':
    main()
