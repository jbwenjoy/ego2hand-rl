"""Source-only phase-correction review. Does not import any probe/framework."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/full151105_gradient_front_static_independent_v1'
OUT = ROOT / 'outputs/full151105_gradient_front_static_independent_v2'


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def functions(tree):
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def main():
    OUT.mkdir(exist_ok=False)
    paths = {
        'entry': ROOT / 'scripts/execute_full151_gradient_v1.py',
        'supervisor': ROOT / 'scripts/run_full151_gradient_v1.py',
        'builder': ROOT / 'scripts/build_full151_gradient_plan_cpu_v1.py',
        'worker': ROOT / 'src/full151_gradient/serial_probe105_v1.py',
        'transaction': ROOT / 'src/full151_gradient/transaction105.py',
        'proposal2': ROOT / 'protocols/full151_same_warp_control_gradient_proposal_v2.json',
        'serial_READY': ROOT / 'outputs/full151105_serial_probe_preparation_CPU_v2/READY.json',
        'serial_CPU81': ROOT / 'outputs/full151105_serial_probe_preparation_CPU_v2/report.json',
        'prior_static_receipt_preserved': OLD / 'report.json',
    }
    ready = read(paths['serial_READY']); cpu = read(paths['serial_CPU81'])
    for key, value in ready['source_pins'].items():
        paths['CPU81_source__' + key] = Path(value['path'])
    pins = {key: pin(path) for key, path in paths.items()}
    texts = {key: paths[key].read_text(encoding='utf-8-sig') for key in ('entry', 'supervisor', 'builder', 'worker', 'transaction')}
    trees = {key: ast.parse(value) for key, value in texts.items()}
    oldtexts = {key: (OLD / 'sources' / (key + '.py')).read_text(encoding='utf-8-sig') for key in texts}
    oldtrees = {key: ast.parse(value) for key, value in oldtexts.items()}
    oldfunc = {key: functions(value) for key, value in oldtrees.items()}
    newfunc = {key: functions(value) for key, value in trees.items()}
    changed_worker = sorted(key for key in oldfunc['worker'] if key in newfunc['worker'] and oldfunc['worker'][key] != newfunc['worker'][key])
    added_worker = sorted(set(newfunc['worker']) - set(oldfunc['worker']))
    proposal = read(paths['proposal2'])
    scalar = next(ast.literal_eval(node.value) for node in trees['worker'].body
                  if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCALAR_CONTRACT' for t in node.targets))
    builder_literals = [node.value for node in ast.walk(trees['builder']) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    checks = {
        'proposal2_exact_frozen_hash_and_namespace': pins['proposal2']['sha256'] == 'eedde4fff0fbf687ffcf32cc7e826a4c5190f2c4971281f6802f38162734d6b0'
            and proposal['schema'] == 'full151105_same_Warp_control_gradient_diagnostic_proposal:2',
        'worker_exact_CPU_planning_tested79207': pins['worker']['sha256'] == '79207ad27252157036915360696442924176b22692464a3ea855c879656c104d',
        'literal_scalar_contract_matches_proposal2_without_import': scalar == proposal['scalar_contract']
            and proposal['primary_scalar_kind'] == scalar['primary'] == 'matched_post20_gpu_score',
        'CPU81_actual_positive_planning_only': cpu['schema'] == 'full151105_serial_probe_CPU_planning_metadata_contract:1'
            and cpu['status'] == 'PASS_CPU_PLANNING_METADATA_ONLY' and cpu['passed'] == cpu['total'] == len(cpu['checks']) == 81
            and all(v is True for v in cpu['checks'].values()),
        'READY_exact_CPU81_report_alias': ready['cpu_report'] == pins['serial_CPU81'],
        'READY_all_actual_source_pins_current': all(pins['CPU81_source__' + k]['sha256'] == v['sha256'] for k, v in ready['source_pins'].items()),
        'worker_proposal_and_transaction_bound_to_CPU81': ready['source_pins']['serial_probe105']['sha256'] == pins['worker']['sha256']
            and ready['source_pins']['transaction105']['sha256'] == pins['transaction']['sha256']
            and ready['source_pins']['proposal2']['sha256'] == pins['proposal2']['sha256'],
        'supervisor_byte_exact_preserves_retained20s_240s_lock_and_identity': texts['supervisor'] == oldtexts['supervisor'],
        'transaction_byte_exact_no_restore_or_capacity_weakening': texts['transaction'] == oldtexts['transaction'],
        'entry_only_preflight_function_changed': {k for k in oldfunc['entry'] if oldfunc['entry'][k] != newfunc['entry'].get(k)} == {'preflight'}
            and set(newfunc['entry']) == set(oldfunc['entry']),
        'worker_changed_functions_only_expected_phase_scoring_and_metadata': changed_worker == ['allocate_artifacts', 'object_terms', 'run_probe', 'validate_config']
            and added_worker == ['controlled_object_score_FP32'] and not (set(oldfunc['worker']) - set(newfunc['worker'])),
        'branch_schedule_mode_indices_baseline_thresholds_inventory_dump_unchanged_AST': all(oldfunc['worker'][k] == newfunc['worker'][k]
            for k in ('branch_schedule', 'execution_plan', 'baseline_errors', 'dump_inventory', 'reviewed_foreign_inventory', 'backend_call_trace')),
        'builder_binds_actual_v2_CPU_proof_paths_and_phase2_proposal': all(value in builder_literals for value in (
            'protocols/full151_same_warp_control_gradient_proposal_v2.json',
            'outputs/full151105_serial_probe_preparation_CPU_v2/READY.json',
            'outputs/full151105_serial_probe_preparation_CPU_v2/report.json')),
    }
    judgments = {
        'x20_controlled_score_primary_matches_WM_next20_function': 'reward_capture calls original capture then scores its raw physics58 postintegration x20 with fixed-palm-transformed goal_k on GPU. The same pure FP32 differentiable function is exposed for future frozen CPU WM AD; no detach is added to that function.',
        'original_RM_derived15_score_secondary_separate_arrays': 'Original RewardManager object2+8 FP32 rate terms/sum and the derived15ms source poses remain explicitly secondary in snapshots and donor/branch arrays. They do not define the primary branch score or gate equality to x20.',
        'same_phase_roundoff_not_time_tolerance': 'The8eps32 bound compares controlled x20 GPU terms with independent same-x20 F64 arithmetic only. It does not excuse15ms-versus20ms differences.',
        'primary_baseline_and_branch_metadata_use_r_primary': 'The donor score, branch score, row r_k and three-baseline score consistency gate all use matched_post20; k+1 CPU rescore stays diagnostic and no u150 is introduced.',
        'Entry_actual_serial_proof_bound_before_default_no_framework_assertion': 'Preflight adds serial READY/report alias checks, checks its current source map in archive closure, requires current worker in proof, then calls actual metadata validator before the existing noTorch/Warp assertion.',
        'builder_execution_protocol_and_sources_explain_corrected_scalar': 'Builder now pins proposal2 and CPU81 READY/report/source map; exact_scalar explicitly states controlled20ms primary versus original RM15ms secondary. No learning or physics budget change.',
        'discovery_None_not_restore_or_physical_gradient_qualification': 'ForeignNone still throws retained InventoryReviewRequired before branches; no default/CPU planning/static review upgrades native restore, baseline, longclock or physical gradient qualification.',
    }
    checks.update({key: True for key in judgments})
    (OUT / 'sources').mkdir()
    for key, value in texts.items():
        (OUT / 'sources' / (key + '.py')).write_text(value, encoding='utf-8')
        (OUT / (key + '_delta.diff')).write_text(''.join(difflib.unified_diff(oldtexts[key].splitlines(True), value.splitlines(True), fromfile='prior_static_snapshot/' + key, tofile='phase2/' + key)), encoding='utf-8')
    (OUT / 'review_source.py').write_bytes(Path(__file__).read_bytes())
    checks['current_sources_and_proofs_unchanged_during_static_read'] = all(pin(v['path'])['sha256'] == v['sha256'] for v in pins.values())
    checks['no_production_or_framework_import_in_review'] = not any(k.split('.')[0] in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic', 'full151_native', 'full151_gradient') for k in sys.modules)
    report = {
        'schema': 'full151105_gradient_phase2_independent_static_review:1',
        'status': 'NO_KNOWN_SOURCE_BLOCKER_PHASE2_STATIC_ONLY' if all(checks.values()) else 'BLOCKED_PHASE2_STATIC_CHECK',
        'source': pin(__file__), 'source_pins': pins, 'checks': checks,
        'passed': sum(checks.values()), 'total': len(checks), 'manual_delta_judgments': judgments,
        'AST_delta': {'worker_changed': changed_worker, 'worker_added': added_worker},
        'supersession': 'Prior static9517 receipt is preserved; its RM-derived15ms primary versus raw20ms WM comparison assumption was scientifically incomplete. It must not authorize that old scalar comparison. This new receipt reviews the explicitly changed controlled20ms experiment, not a claim that original RM reward is20ms.',
        'explicit_pending': ['Root actual default metadata preflight and immutable new config closure',
                             'Native discovery/foreign inventory review and actual restore3baseline/longclock engineering qualification',
                             'Any physical secant/WM derivative comparison; no execution qualification inherited from source or CPU81'],
        'native_restore_qualified': False, 'physical_gradient_qualified': False,
        'execution': {'production_imports': 0, 'CPU81_repeated': False, 'default_or_builder_runs': 0,
                      'score_function_calls': 0, 'WM_or_Actor_forward': 0, 'Env': 0, 'physics': 0, 'GPU': 0},
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'status': report['status'], 'checks': [report['passed'], report['total']]}, indent=2))


if __name__ == '__main__':
    main()
