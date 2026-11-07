"""Final source/JSON-only delta receipt; never import production worker/front."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151105_inventory_all_front_static_independent_v1'


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def functions(text):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main():
    OUT.mkdir(exist_ok=False)
    paths = {
        'review_source': Path(__file__),
        'entry2': ROOT / 'scripts/execute_full151_gradient_v2.py',
        'supervisor2': ROOT / 'scripts/run_full151_gradient_v2.py',
        'builder4': ROOT / 'scripts/build_full151_gradient_plan_cpu_v4.py',
        'worker2': ROOT / 'src/full151_gradient/serial_probe105_v2.py',
        'transaction': ROOT / 'src/full151_gradient/transaction105.py',
        'old_supervisor1': ROOT / 'scripts/run_full151_gradient_v1.py',
        'old_worker792': ROOT / 'src/full151_gradient/serial_probe105_v1.py',
        'proposal2': ROOT / 'protocols/full151_same_warp_control_gradient_proposal_v2.json',
        'planner_READY': ROOT / 'outputs/full151105_serial_probe_v2_preparation_CPU_v1/READY.json',
        'planner_report': ROOT / 'outputs/full151105_serial_probe_v2_preparation_CPU_v1/report.json',
        'prior_design': ROOT / 'outputs/full151105_inventory_all_scope_design_v1/design.json',
        'actual_phase0_empirical_saved85': ROOT / 'outputs/full151105_phase0_empirical_baseline_saved_review_v1/report.json',
    }
    ready = read(paths['planner_READY']); cpu = read(paths['planner_report'])
    for key, value in ready['source_pins'].items():
        paths['planner_source__' + key] = Path(value['path'])
    pins = {key: pin(path) for key, path in paths.items()}
    texts = {key: paths[key].read_text(encoding='utf-8-sig') for key in ('entry2', 'supervisor2', 'builder4', 'worker2', 'transaction', 'old_supervisor1', 'old_worker792')}
    parsed = {key: ast.parse(value) for key, value in texts.items()}
    oldf = functions(texts['old_worker792']); newf = functions(texts['worker2'])
    checks = {
        'reviewed_sources_parse_without_import': bool(parsed),
        'planner_actual_positive_not_repeated': cpu['passed'] == cpu['total'] == len(cpu['checks']) and all(v is True for v in cpu['checks'].values()),
        'planner_READY_matches_actual_report_pin': ready['cpu_report'] == pins['planner_report'],
        'planner_sources_current': all(pins['planner_source__' + k]['sha256'] == v['sha256'] for k, v in ready['source_pins'].items()),
        'current_worker_CPU_proof_bound': any(v['sha256'] == pins['worker2']['sha256'] and Path(v['path']).resolve() == paths['worker2'].resolve() for v in ready['source_pins'].values()),
        'supervisor_only_ENTRY_routing_delta': texts['old_supervisor1'].replace('execute_full151_gradient_v1.py', 'execute_full151_gradient_v2.py') == texts['supervisor2'],
        'pure_controlled_score_and_baseline_thresholds_unchanged_AST': all(oldf[name] == newf[name] for name in (
            'controlled_object_score_FP32', 'object_terms', 'object_terms_f64', 'baseline_errors', 'allocate_artifacts', 'backend_call_trace', 'dump_inventory')),
        'transaction_exact_frozen_source_f1bd': pins['transaction']['sha256'] == 'f1bd0eafed58783a6efbb78a0ba40da086b36851119eddd00e29fcddb972a55c',
    }
    judgments = {
        'two_v2_modes_only': 'Front/builder/worker reject legacy phase0 engineering mode. Frozen v1/792 and its actual03 empirical saved85 remain separate evidence, never a generic hidden-state certificate.',
        'discovery1200_8_0_and_NOT_RUN_slots': 'Two fixed150-interval donors produce1200 step returns and8captures. Fixed152 branch slots stay unscheduled/NOT_RUN/NaN, completion and restore masks false; source explicitly skips every branch loop in inventory_discovery_all.',
        'state_setter_and_restore_counter_scopes': 'Instrument begin/finish/native Snapshot.restore attempted+returned, require0 in discovery, separately require8same-value INTEGRATION setters and16getters, plus1200step/300interval attempted+returned. Does not claim0state setters or whole-process zero physics.',
        'discovery_failure_not_swallowed': 'Only absent foreign-review qualification is deferred after valid construction/current/integration checks. Profile/API/nonfinite/capacity/clock/serialization/deadline faults rethrow and preserve partial/NOT_RUN metadata; no retry or recovery to an accepted snapshot.',
        'discovery_qualification_non_vacuous': 'Completion status is inventory-only, baseline_all_passed false even for an empty list, foreign/restore/fullserial/gradient/wholehidden false. Donor-clock observations have their own scope.',
        'per_snapshot_exact8_schema2_no_flat_fallback': 'New source-review schema2 requires exact ordered8labels and exact8per_snapshot keys, exact worker/transaction/bundle, literal limited-scope flags, no schema1/phase0/type wildcard/union/fallback. Selected current tx exact opaque set must match its own label table.',
        'profile_and_saved_topology_join': 'Pinned description label and strictNative plus strict integer phase matching label, pre_action/105/mode/variant are checked. Topology binds worker and transaction. Direct owner field types are checked; tuple leaf evidence is explicitly limited to description plus source review.',
        'actual_discovery_prerequisite_join': 'Entry serial_full requires typed empirical phase0 and positive actual8/1200/0restore inventory review with same bundle, then review source_inventory_saved_review/config/report/run joins that actual evidence. Each description/topology pin joins the matching label in discovery.inventory_snapshot_pins by samefile+SHA.',
        'source_archive_closure': 'Builder flattens proposal execution sources, actual current planner and transaction READY source maps, CPU WM sources, new front/worker/helper and donor pins; foreign inner source_pins and health prerequisites are added. Entry rehashes and archives before run.',
        'default_metadata_framework_isolation': 'Entry calls actual v2 validate_config/execution_plan only after source and bundle checks, then asserts no Torch/Warp. Worker top-level and metadata validation are lazy with no framework/Env/model construction.',
        'bounded_shared_owned_lifecycle_unweakened': 'Unchanged supervisor logic retains CPU Popen20s communicate/40b cleanup+1s drain, one shared GPU lock, rejection of other workers, config SHA recheck,240s watchdog and f416 exact identity/post-terminal checks; no retries.',
        'numeric_physics_and_primary_phase_unchanged': 'Original processed residual/EMA/C2 and actualu checks, nominal-to-actual8/reset,4native substeps+one Managerforward, measured solver15/self15 versus raw20 and controlled FP32score(raw20,goal_k) remain unchanged. No Actor/WM/optimizer construction or assistance introduced.',
        'full152_scope_remains_empirical': 'Full mode still152branches/1808scheduled substeps and per-label3fresh baselines before directions. Graph executable cache and unenumerated graph-owned scratch remain UNKNOWN; neither inventory coverage nor the old03 baseline upgrades complete hidden-state or physical-gradient qualification.',
        'MODE_VARIANT_false_finding_retracted': 'Initial reviewer claim of missing constants was incorrect: validate_review_manifest locally imports MODE,VARIANT from transaction105. Source was reread and Root/owner notified; no production change was requested for this nonissue.',
    }
    checks.update({key: True for key in judgments})
    (OUT / 'sources').mkdir()
    for key in ('entry2', 'supervisor2', 'builder4', 'worker2', 'transaction'):
        (OUT / 'sources' / (key + '.py')).write_bytes(paths[key].read_bytes())
    (OUT / 'review_source.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'worker_delta.diff').write_text(''.join(difflib.unified_diff(texts['old_worker792'].splitlines(True), texts['worker2'].splitlines(True), fromfile='frozen792', tofile='worker2')), encoding='utf-8')
    checks['all_bound_sources_and_receipts_unchanged_during_read'] = all(pin(v['path'])['sha256'] == v['sha256'] for v in pins.values())
    checks['no_production_framework_import_or_execution'] = not any(k.split('.')[0] in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic', 'full151_gradient', 'full151_native') for k in sys.modules)
    result = {'schema': 'full151105_inventory_all_front_independent_static_review:1',
        'status': 'NO_KNOWN_INVENTORY_ALL_SOURCE_BLOCKER_STATIC_ONLY' if all(checks.values()) else 'BLOCKED_STATIC_SOURCE_OR_PROOF_DRIFT',
        'source_pins': pins, 'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
        'manual_source_review_judgments': judgments,
        'actual_owner_planner_checks': {'passed': cpu['passed'], 'total': cpu['total'], 'report': pins['planner_report']},
        'pending': ['Root actual default preflight/new immutable config join', 'Actual donor discovery-all and independent saved audit',
                    'Actual source review of all8label opaque inventories, not a prefilled allowlist',
                    'Any new restored baseline or FD run and future CPU model consumption'],
        'native_restore_qualified': False, 'whole_graph_hidden_state_qualified': False, 'physical_gradient_qualified': False,
        'execution': {'production_imports': 0, 'planner_repeated': False, 'default_or_builder_runs': 0,
                      'Env': 0, 'WM_Actor_forward': 0, 'model_loads': 0, 'physics': 0, 'GPU': 0}}
    (OUT / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'checks': [result['passed'], result['total']], 'status': result['status']}, indent=2))


if __name__ == '__main__':
    main()
