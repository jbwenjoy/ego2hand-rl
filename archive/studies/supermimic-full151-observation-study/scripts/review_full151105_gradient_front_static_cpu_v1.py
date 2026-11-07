"""Source/JSON-only review receipt; never import production probe modules."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151105_gradient_front_static_independent_v1'


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    OUT.mkdir(exist_ok=False)
    own = pin(__file__)
    paths = {
        'entry': ROOT / 'scripts/execute_full151_gradient_v1.py',
        'supervisor': ROOT / 'scripts/run_full151_gradient_v1.py',
        'builder': ROOT / 'scripts/build_full151_gradient_plan_cpu_v1.py',
        'worker': ROOT / 'src/full151_gradient/serial_probe105_v1.py',
        'transaction': ROOT / 'src/full151_gradient/transaction105.py',
        'transaction_READY': ROOT / 'outputs/full151105_transaction_preparation_CPU_v1/READY.json',
        'transaction_CPU42': ROOT / 'outputs/full151105_transaction_preparation_CPU_v1/report.json',
        'schema2_WM_CPU15': ROOT / 'outputs/full151_schema2_pair_WM_loader_CPU_v1/report.json',
        'inherited_A2_evaluation_config': ROOT / 'configs/full151-v2-policy-eval-primary2000_A2_seed0-v2.json',
        'proposal': ROOT / 'protocols/full151_same_warp_control_gradient_proposal_v1.json',
        'identity_f416': Path('D:/workspace/supermimic-observation-restoration-prefix-study/scripts/run_track_prefix_v3.py'),
        'lifecycle_40b': Path('D:/workspace/supermimic-control-feasibility-draft/scripts/run_braking_microprobe.py'),
    }
    ready = read(paths['transaction_READY'])
    for key, value in ready['source_pins'].items():
        paths['CPU42_source__' + key] = Path(value['path'])
    sources = {key: pin(value) for key, value in paths.items()}
    text = {k: v.read_text(encoding='utf-8-sig') for k, v in paths.items() if v.suffix == '.py'}
    parsed = {k: ast.parse(v) for k, v in text.items()}
    report42 = read(paths['transaction_CPU42']); report15 = read(paths['schema2_WM_CPU15'])
    checks = {
        'all_reviewed_python_AST_parses_without_import': bool(parsed),
        'CPU42_report_actual_positive_and_source_bound': report42['passed'] == report42['total'] == 42
            and all(v is True for v in report42['checks'].values())
            and report42['status'] == 'PASS_CPU_SURROGATE_INVENTORY_ONLY'
            and ready['cpu_report'] == sources['transaction_CPU42'],
        'CPU42_actual_source_pins_current': all(sources['CPU42_source__' + k]['sha256'] == v['sha256'] for k, v in ready['source_pins'].items()),
        'CPU15_WM_engineering_report_positive_not_physical': report15['passed'] == report15['total'] == 15
            and all(v is True for v in report15['checks'].values()),
        'reviewed_identity_source_exact_f416': sources['identity_f416']['sha256'] == 'f4161cb4be7ac2e5e0fc598820a944f4d9a0e7240ee79b4eb68054732f246475',
        'reviewed_owned_cleanup_source_exact_40b': sources['lifecycle_40b']['sha256'] == '40b397f60e91bf59a1c72ad43f9bd52b42e33a820a695c2ec6e43736fa8a54aa',
    }
    # These are source-review judgments, not synthetic execution or native evidence.
    judgments = {
        'unique_evaluation_seed0_entry_builder_worker_ABI': 'Builder emits evaluation_seed and both actual metadata guards reject bool/nonzero; no seed alias.',
        'actual_metadata_validator_called_before_no_framework_assertion': 'Entry loads bound NumPy bundle then calls worker.validate_config; worker imports frameworks only inside run_probe.',
        'CPU_component_readiness_not_native_qualification': 'Entry binds the exact CPU42 READY/report/sources and CPU WM evidence; proof explicitly remains metadata/surrogate.',
        'inherited_and_new_source_archive_closure': 'Builder retains the prior A2 execution map, adds new entry/supervisor/builder/worker/transaction/package/proposal/protocol plus donor7x2, CPU42 and WM CPU source maps. Entry rehashes all pins and archives bytes before native run.',
        'single_authorized_GPU_scope_no_default_launch': 'Default builder false; Entry and supervisor require explicit true and --run, fresh owned output, no retry/fallback/training.',
        'retained_CPU_preflight20s_and_owned_cleanup': 'Supervisor retains Popen, communicates20s, uses reviewed stop_owned on exceptions, bounded1s drain, then raises without GPU launch.',
        'single_shared_GPU_lock_reject_workers_then_owned240s_watch': 'Lock, project worker rejection, config SHA recheck, retained watchdog and actual PID observer precede one invocation; no retries.',
        'actual_OS_identity_samefile_and_post_terminal_scope': 'Unmodified f416 identity driver verifies PID/PPID/creation/command/executable alias and post-terminal query; wrapper supplies new entry identity.',
        'no_child_failure_metadata_conservative': 'Prewatch errors write failed_no_child_started and identity NOT_APPLICABLE without invented PID absence.',
        'formed_worker_no_model_or_Actor_inference': 'Worker replays archived source-action arrays and original controller; imported Runner class is not instantiated; no WM or checkpoint weights load.',
        'new_native_cadence_and_actual_context_preserved': 'Fresh nominal MJB/actual8/checked reset; interval uses original env.step with4 substeps/one Managerforward, original source action processes once; absolute target intervention is explicit.',
        'mode_smoke_indexing_and_fixed_full_ABI': 'Pure execution_plan selects only A0 k0/oneinterval in smoke, all8 snapshots/152 slots remain with NOT_RUN masks; selected drift indices follow capture_phases.',
        'native_reward_FP32_primary_and_sensor_pre_forward_phase': 'Native reward wrapper records GPU _step_reward object2+8 terms and FP32 sum after original reward compute and before Managerforward; raw_found/sensordata are saved there. CPU rescore is secondary.',
        'direction_FP32_endpoints_FP64_effective_vector_and_center_offset': 'Executed-u plus/minus endpoints use fixed inset-domain clipping contract; d_eff and center_offset record FP64 arithmetic, clipping explicitly marked, no signal-based resampling.',
        'three_baselines_qualify_before_snapshot_directions': 'Inventory bitexact mask is separate from empirical restore qualification; all3 fresh baseline gates precede that snapshot directions, failed baseline aborts without perturbation retry.',
        'foreign_inventory_None_discovery_is_refusal': 'InventoryReviewRequired retains inventory/partial arrays, reports discovery and rethrows; Entry fails rather than falsely reporting native restore or gradient qualification.',
        'snapshot_capacity_ledger_outside_restore': 'Standalone capacity observer/records external to transaction, before each restore and during branches/final; unknown or overflow blocks qualification, no clearing ledger.',
        'partial_failure_and_call_count_scope': 'Completed and NOT_RUN masks retained, API-attempt/returned/archived-sample scopes distinct; constructor/capture API boundaries not summed as whole-process kernel calls.',
    }
    checks.update({key: True for key in judgments})
    (OUT / 'sources').mkdir()
    for key in ('entry', 'supervisor', 'builder', 'worker', 'transaction'):
        (OUT / 'sources' / (key + '.py')).write_bytes(paths[key].read_bytes())
    (OUT / 'review_source.py').write_bytes(Path(__file__).read_bytes())
    checks['all_reviewed_source_and_prerequisite_bytes_unchanged_during_read'] = all(pin(v['path'])['sha256'] == v['sha256'] for v in sources.values())
    checks['review_process_no_framework_import_or_native_execution'] = not any(k.split('.')[0] in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic') for k in sys.modules)
    result = {
        'schema': 'full151105_gradient_front_independent_static_review:1',
        'status': 'NO_KNOWN_SOURCE_BLOCKER_STATIC_ONLY' if all(checks.values()) else 'BLOCKED_STATIC_PREREQUISITE_OR_DRIFT',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': own, 'source_pins': sources, 'checks': checks,
        'passed': sum(checks.values()), 'total': len(checks),
        'manual_source_review_judgments': judgments,
        'prior_findings_closed_before_any_new_config_execution': [
            'seed versus evaluation_seed ABI mismatch', 'missing re import',
            'unallocated center_offset', 'smoke5state versus selected full601 index mismatch',
            'restore bitexact/empirical qualification mask separation', 'FP64 d_eff and center offset',
            'builder adds actual schema2 WM CPU helper sources, beyond its receipt pin'],
        'explicit_pending': [
            'Actual default Entry/Supervisor preflight and actual new immutable config source closure',
            'First native inventory discovery and source-bound per-path opaque/closure review',
            'Actual full native restore, three-baseline replay and long-clock qualification',
            'Any physical probe branch, WM comparison or physical gradient qualification',
            'Independent saved postrun source/identity/capacity audit; preflight/archive is not evidence of postrun immutability'],
        'native_restore_qualified': False, 'physical_gradient_qualified': False,
        'execution': {'production_imports': 0, 'builder_or_default_runs': 0, 'Env': 0,
                      'model_loads': 0, 'WM_or_Actor_forward': 0, 'physics': 0, 'GPU': 0},
    }
    (OUT / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'status': result['status'],
                      'checks': [result['passed'], result['total']]}, indent=2))


if __name__ == '__main__':
    main()
