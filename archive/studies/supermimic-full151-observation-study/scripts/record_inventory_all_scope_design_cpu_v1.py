"""Record source-only interface design; no production/native module imports."""
from pathlib import Path
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151105_inventory_all_scope_design_v1'


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=False)
    labels = [f'{path}/{k}' for path in ('A0', 'A2') for k in (0, 49, 99, 149)]
    evidence = {
        'producer': pin(__file__),
        'worker792_phase2_reference_only': pin(ROOT / 'src/full151_gradient/serial_probe105_v1.py'),
        'transaction105_unchanged_reference': pin(ROOT / 'src/full151_gradient/transaction105.py'),
        'phase0_foreign_manifest': pin(ROOT / 'outputs/full151105_phase0_foreign_inventory_source_review_v1/foreign_review_v1.json'),
        'phase0_source_review': pin(ROOT / 'outputs/full151105_phase0_foreign_inventory_source_review_v1/report.json'),
        'phase0_execution_handle_limits': pin(ROOT / 'outputs/full151105_stream_graph_source_review_v1/report.json'),
        'phase0_discovery_saved38': pin(ROOT / 'outputs/full151105_native_inventory_discovery_saved_review_v1/report.json'),
        'parent_proposal2': pin(ROOT / 'protocols/full151_same_warp_control_gradient_proposal_v2.json'),
    }
    contract = {
        'schema': 'full151105_inventory_all_per_snapshot_source_design:1',
        'status': 'INTERFACE_DESIGN_ONLY_NOT_IMPLEMENTATION_OR_EXECUTION_QUALIFICATION',
        'GPU_authorized': False, 'training_authorized': False,
        'owner_interfaces_agreed': {
            'new_worker': 'full151_gradient.serial_probe105_v2 (Windows owner; source pin pending final READY)',
            'config_schema_revision': 2, 'execution_protocol_revision': 2, 'worker_report_revision': 3,
            'manifest_schema': 'full151105_foreign_inventory_per_snapshot_source_review:2',
            'Root_fronts': 'New front_v2/builder_v4 owned by Root; no changes made by this design'},
        'discovery_all': {
            'probe_mode': 'inventory_discovery_all',
            'donor_order': ['A0', 'A2'], 'phases': [0, 49, 99, 149], 'ordered_snapshot_labels': labels,
            'planned_donor_intervals': 300, 'planned_donor_state_rows': 1202,
            'planned_episode_substeps': 1200, 'planned_native_control_forward_returns': 300,
            'planned_capture_count': 8, 'native_snapshot_restore_calls': 0,
            'transaction_begin_calls': 0, 'planned_executed_branches': 0,
            'planned_INTEGRATION_set_get_roundtrips': 8,
            'state_write_scope': '0 Snapshot.restore/branch is not 0 state setters: each retained INTEGRATION roundtrip invokes set_state and must be counted separately.',
            'fixed152branchABI_if_retained': 'All slots scheduled=False, status NOT_RUN, completion/restore qualification masks false, no fake0-valued force/state data for unexecuted branches.',
            'completion_semantics': 'All8 valid inventories and donor data saved, source/capacity/clock/coverage retained. This only completes discovery, never restore/baseline/gradient qualification. Empty-baseline all([]) must not imply a pass.',
            'foreign_None_behavior': 'Only absent foreign-review gate is deferred. After valid snapshot construction/INTEGRATION/current verification save REVIEW_PENDING and continue next scheduled donor capture. No bypass for unknown unsupported mutable classes or invalid snapshot/profile.',
            'fatal_behavior': 'Profile/API/INTEGRATION/nonfinite/capacity/clock/I/O/deadline failure stops with partial records and explicit NOT_RUN slots. No retry, recapture fallback, opaque type wildcard or source mutation.',
            'clock_scope': 'Fresh-donor sequentialFP32 tick/50us checks can be reported in their observed donor scope. No branch-longclock or native restored-state qualification follows.',
            'cost_scope': '1200 is scheduled donor integration substeps only. Constructor, graph capture/warmup, checked reset, extra forward and INTEGRATION APIs are separately observed, not hidden or summed as episode steps.'},
        'per_snapshot_manifest': {
            'schema': 'full151105_foreign_inventory_per_snapshot_source_review:2',
            'required_top_fields': ['schema', 'worker', 'transaction', 'input_bundle', 'source_pins',
                                    'source_reviewed_closure_globals', 'approved_probe_modes',
                                    'allowed_snapshot_labels', 'per_snapshot'],
            'worker_transaction_bundle': 'Exact current immutable pins; full-run worker must match the worker that captured/reviewed the source inventory. Do not silently translate old792/source1 manifest.',
            'source_reviewed_closure_globals': 'True only after the new active hook/global/default/keyword-default source delta is reviewed; not whole foreign/CUDA state serialization.',
            'approved_probe_modes': ['serial_full'], 'allowed_snapshot_labels': labels,
            'per_snapshot_keys': labels,
            'per_snapshot_value': {
                'inventory': {'description': 'actual source pin for this label', 'topology': 'actual source pin for this label'},
                'allowed_identity_only': [{'path': 'exact observed path', 'type': 'exact observed qualified type', 'reason': 'per-item source-grounded conditional reason'}],
                'source_reviewed_closure_globals': 'literalTrue after review'},
            'strict_checks': [
                'Exactly8 ordered top labels and exactly8 per_snapshot keys; no missing/duplicate/unknown label.',
                'Each saved description.label/profile.phase/pre_action/mode105/bundle and topology worker+transaction identity matches its label/selected source.',
                'No duplicate(path,type), no extra allowed-item keys beyond path/type/reason, no wildcard or implicit type allowance.',
                'Each label allowed set exactly equals its saved actual opaque set and exactly equals the newly constructed runtime transaction opaque set for that label.',
                'Topology resolves the containing native entry, member and owner where represented. Tuple-contained leaf type/length is not always independently encoded: description supplies leaf type and explicit source review supplies owner semantics; do not claim full independent reconstruction from topology.',
                'No union of8sets, no stripping native_entry indices, no fallback to phase0 schema1 or previous label.',
                'Every source/document/inventory pin is current and all newly referenced flat source pins are included in execution archive closure.',
                'Keep manifest/config/review mappings outside env/reader to avoid introducing new DFS entries; select the label review after the actual inventory is captured.'],
            'required_limits': {'conditional_engineering_only': True, 'whole_graph_hidden_state_qualified': False,
                                'graph_hidden_scratch_dependence': 'UNKNOWN_NOT_EXHAUSTIVELY_PROVED_OR_RESTORED'},
            'cannot_fill_yet': 'This file defines an ABI, not any actual8inventory allowlist. Actual per-label descriptions/topologies do not yet exist for this new discovery-all run.'},
        'full152_after_review': {
            'mode': 'serial_full', 'planned_snapshot_count': 8, 'baseline_repeats_per_snapshot': 3,
            'planned_baseline_branches': 24, 'planned_direction_branches': 128,
            'planned_total_branches': 152, 'planned_episode_substeps': 1808,
            'qualification': 'Fresh run recaptures in-process pointers. Every label exact inventory and its3fresh baselines must pass before that label directions; old phase0 zero-error baseline is not a substitute. Source/sample/force/clock/labels stay same pinned proposal2.',
            'hidden_state_boundary': 'Observed inventory bytes and same-input baseline reproducibility are empirical scope only. Graph executable cache and hidden scratch remain unqualified as a universal physical/full-state restore certificate.',
            'not_authorized_here': True},
        'minimal_source_review_checklist': [
            'New mode schedules only fresh donors and8captures, no call path to tx.branch/begin/native.restore.',
            'Correct smoke/full/discovery interval and drift indices; no index600 against one-interval buffer.',
            'All8 per-label source/global/opaque tables selected strictly with no schema1 fallback.',
            'Discovery-only completion has0baseline/0branch and all native/gradient qualification claims false.',
            'Original processed residual/cache/actualu, four5ms steps plus one Managerforward, no state/force assistance unchanged.',
            'Main controlledGPUscore(raw20ms,goal_k), original RM15ms secondary andself15 lastsolver phase are preserved.',
            'Full body counters/attempted+returned scopes, external cumulative capacity OR/UNKNOWN and cleanup remain fail-closed.',
            'Default metadata import stays noTorch/Warp; actual CPU planning proof source-bound; retained20s preflight cleanup/240s watch/sharedlock/OS identity unchanged.',
            'No arbitrary model/WM/Actor loads or forward during physical discovery. Future CPU WM consumption stays a separate process/source/runplan.'
        ],
        'source_pins': evidence,
        'execution': {'production_imports': 0, 'model_loads': 0, 'Env': 0, 'WM_Actor_forward': 0,
                      'native_Reader_FK_collision_step_forward': 0, 'GPU': 0},
    }
    assert not any(k.split('.')[0] in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic', 'full151_gradient', 'full151_native') for k in sys.modules)
    (OUT / 'design.json').write_text(json.dumps(contract, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    report = {'schema': 'full151105_inventory_all_scope_source_design_receipt:1',
              'status': 'SOURCE_DESIGN_ONLY_PENDING_IMPLEMENTATION_AND_NATIVE_DISCOVERY',
              'design': pin(OUT / 'design.json'), 'source_pins': evidence,
              'actual_new_discovery': False, 'native_restore_qualified': False, 'physical_gradient_qualified': False,
              'execution': contract['execution']}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'report': pin(OUT / 'report.json'), 'design': pin(OUT / 'design.json')}, indent=2))


if __name__ == '__main__':
    main()
