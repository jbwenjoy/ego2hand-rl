"""Assemble exact source-reviewed phase0 opaque inventory. No native imports."""
from pathlib import Path
from collections import Counter
import argparse
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'outputs/full151105_gradient_restore_inventory_discovery02_v1'
SNAP = DATA / 'donor_A0/snapshot_0_inventory'
OUT = ROOT / 'outputs/full151105_phase0_foreign_inventory_source_review_v1'
SDK = Path('D:/workspace/dex-supermimic-research/.venv/Lib/site-packages')
NATIVE = Path('D:/workspace/dex-supermimic-research/src/dex_supermimic')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--execution-handles-review', type=Path, required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=False)
    (OUT / 'review_source.py').write_bytes(Path(__file__).read_bytes())
    cfg = read(DATA / 'config.json'); desc = read(SNAP / 'transaction_description.json'); top = read(SNAP / 'inventory_topology.json')
    peer3_path = ROOT / 'outputs/full151105_three_foreign_handles_source_review_v1/report.json'
    peer3 = read(peer3_path); handles = read(args.execution_handles_review)
    peer_scope_path = ROOT / 'outputs/full151105_three_foreign_handles_source_scope_v2/report.json'
    peer_scope = read(peer_scope_path)
    opaque = desc['opaque_identity_only']
    expected = {(v['path'], v['type']) for v in opaque}
    paths = {
        'source_review_producer': Path(__file__),
        'worker': ROOT / 'src/full151_gradient/serial_probe105_v1.py',
        'transaction': ROOT / 'src/full151_gradient/transaction105.py',
        'discovery_config': DATA / 'config.json',
        'discovery_description': SNAP / 'transaction_description.json',
        'discovery_topology': SNAP / 'inventory_topology.json',
        'input_bundle': Path(cfg['input_bundle']['path']),
        'peer_CPU_templates_no_grad': peer3_path,
        'peer_CPU_templates_no_grad_phase0_scope': peer_scope_path,
        'peer_execution_handles': args.execution_handles_review,
        'discovery_saved38': ROOT / 'outputs/full151105_native_inventory_discovery_saved_review_v1/report.json',
        'entity': SDK / 'mjlab/entity/entity.py',
        'scene': SDK / 'mjlab/scene/scene.py',
        'terrain_entity': SDK / 'mjlab/terrains/terrain_entity.py',
        'actuator': SDK / 'mjlab/actuator/actuator.py',
        'xml_actuator': SDK / 'mjlab/actuator/xml_actuator.py',
        'builtin_group': SDK / 'mjlab/actuator/builtin_group.py',
        'Manager_env': SDK / 'mjlab/envs/manager_based_rl_env.py',
        'command_manager': SDK / 'mjlab/managers/command_manager.py',
        'observation_manager': SDK / 'mjlab/managers/observation_manager.py',
        'reward_manager': SDK / 'mjlab/managers/reward_manager.py',
        'termination_manager': SDK / 'mjlab/managers/termination_manager.py',
        'SceneEntityCfg': SDK / 'mjlab/managers/scene_entity_config.py',
        'controller105': ROOT / 'src/full151_native/controller.py',
        'runtime105': ROOT / 'src/full151_native/runtime.py',
        'native105': ROOT / 'src/full151_native/native.py',
        'pure_C2': Path('D:/workspace/supermimic-c2-training-study/src/target_inset.py'),
        'native_pen': NATIVE / 'rl/dex_supermimic/pen.py',
        'native_rotations': NATIVE / 'rl/dex_supermimic/rotations.py',
        'native_actions': NATIVE / 'tasks/pen_spin/mdp/actions.py',
        'native_commands': NATIVE / 'tasks/pen_spin/mdp/commands.py',
        'native_observations': NATIVE / 'tasks/pen_spin/mdp/observations.py',
        'native_snapshot': NATIVE / 'rl/dex_supermimic/snapshot.py',
        'Mjs_type_binary_identity_only': SDK / 'mujoco/_specs.cp312-win_amd64.pyd',
        'WP_forward_scratch': SDK / 'mujoco_warp/_src/forward.py',
        'WP_derivative_scratch_initialization': SDK / 'mujoco_warp/_src/derivative.py',
        'WP_factor_solve_scratch': SDK / 'mujoco_warp/_src/smooth.py',
        'WP_solver_scratch': SDK / 'mujoco_warp/_src/solver.py',
        'WP_collision_scratch': SDK / 'mujoco_warp/_src/collision_driver.py',
    }
    for prefix, proof in (('peer3', peer3), ('peer3_scope', peer_scope), ('handles', handles)):
        for key, value in proof['source_pins'].items():
            paths[prefix + '__' + key] = Path(value['path'])
    pins = {key: pin(value) for key, value in paths.items()}
    for proof in (peer3, peer_scope, handles):
        for value in proof['source_pins'].values():
            assert pin(value['path'])['sha256'] == value['sha256'], 'Peer proof source drift'
    assert pins['worker']['sha256'] == '79207ad27252157036915360696442924176b22692464a3ea855c879656c104d'
    assert pins['transaction']['sha256'] == 'f1bd0eafed58783a6efbb78a0ba40da086b36851119eddd00e29fcddb972a55c'
    assert pins['input_bundle']['sha256'] == cfg['input_bundle']['sha256']
    assert desc['profile']['phase'] == 0 and desc['label'] == 'A0/0' and cfg['probe_mode'] == 'engineering_restore_smoke'
    assert len(opaque) == len(expected) == 198
    assert len(top['native_entries']) == desc['native_entries'] == 865
    # Resolve native value positions to the saved exact owner field, not a type wildcard.
    owner = {121: 'Scene', 123: "Scene.entities['terrain']", 128: "terrain.indexing",
             175: "Scene.entities['object']", 178: 'object.indexing',
             185: "Scene.entities['robot']", 190: 'robot.XmlActuator._mjs_actuators', 192: 'robot.indexing'}
    spec_types = {'mujoco._specs.' + name for name in ('MjSpec', 'MjsActuator', 'MjsBody', 'MjsGeom', 'MjsJoint', 'MjsLight', 'MjsMaterial', 'MjsSite', 'MjsTexture')}
    allowed = []; compiled = []
    for value in opaque:
        if value['type'] not in spec_types:
            continue
        match = re.fullmatch(r'native_entry\[(\d+)\]\.value\[(\d+)\](.*)', value['path'])
        assert match, value['path']
        i, j, suffix = int(match[1]), int(match[2]), match[3]
        assert i in owner
        entry = top['native_entries'][i]; member = entry['members'][j]
        semantic = owner[i] + ('.' + member['key'] if entry['kind'] == 'object' else '[' + member['key'] + ']') + suffix
        if value['type'] == 'mujoco._specs.MjSpec':
            detail = 'Compiled scene/entity specification retained after constructor. Scene construction and Entity.edit_spec/compile are not called by this fixed one-interval branch; runtime uses compiled WP model/data and already-created target/index tensors.'
        elif value['type'] == 'mujoco._specs.MjsActuator':
            detail = 'This exact XML actuator definition is retained for initialization/type and ctrl-ID lookup. XmlActuator.compute only selects an existing target tensor and Actuator.update is a no-op; no edit_spec/gain/limit write to the MjsActuator occurs in this branch.'
        else:
            detail = 'This exact retained spec element supplies immutable-in-this-branch name/id/topology through EntityIndexing or the fixed free/nonfree-joint list. Index tensors were created during Entity.initialize; per-substep Scene.write/update operates on EntityData/actuator tensors, not MJCF/spec edits. No renderer/spec compiler is invoked.'
        reason = semantic + ': ' + detail + ' Conditional identity-only for current worker79207, A0/0 engineering restore smoke; no general MuJoCo type exemption or foreign-internal byte-restore claim.'
        item = {'path': value['path'], 'type': value['type'], 'reason': reason}
        allowed.append(item); compiled.append({**item, 'semantic_owner_field': semantic,
            'evidence': ['entity.py:30-68,initialize689-744,_compute_indexing1250-1333', 'entity.py:862-896,1335-1342',
                         'scene.py:190-205', 'xml_actuator.py:73-126,128-134', 'actuator.py:269-292,415-435', 'native.py:54-83']})
    assert len(compiled) == 166
    assert handles['status'] == 'CONDITIONAL_PHASE0_IDENTITY_REVIEW_HIDDEN_STATE_UNKNOWN'
    assert len(peer3['items']) == 3 and len(handles['reviewed_exact_items']) == 29
    for values in (peer3['items'], handles['reviewed_exact_items']):
        for value in values:
            allowed.append({k: value[k] for k in ('path', 'type', 'reason')})
    assert len(allowed) == 198 and {(v['path'], v['type']) for v in allowed} == expected
    assert all(set(v) == {'path', 'type', 'reason'} and len(v['reason']) > 30 for v in allowed)
    # Preserve discovery order for an auditable one-to-one set, not observed process IDs.
    bykey = {(v['path'], v['type']): v for v in allowed}
    allowed = [bykey[(v['path'], v['type'])] for v in opaque]
    closures = {
        'scope': 'Source audit of the current phase0 branch call tree, not generic Python global serialization.',
        'saved_coverage': {'native_entries': desc['native_entries'], 'WP_arrays': desc['native_Warp_arrays'],
                          'Torch_tensors': desc['native_Torch_tensors'], 'explicit_arrays': desc['explicit_arrays'],
                          'explicit_containers': desc['explicit_containers'], 'closure_cells': desc['closure_cells'],
                          'local_generators': desc['local_generators']},
        'observed_closure_cells': top['closure_cells'],
        'active_hooks': [
            {'hook': 'env.step / full151_controller.env_step', 'review': 'Original Manager step and current process wrapper only. Module globals are frozen function handles/constants and torch math. Per-env process counts, pending records, action/target/memory/scene/manager tensors and containers are in native/sidecar inventory. Runner.collect/save/load, policy_update, constructors and context teardown are not called by the branch.'},
            {'hook': 'sim.step / worker.substep / backend_call_trace.counted', 'review': 'Native WP graph integrates only captured WP model/data. counted.__kwdefaults__ binds the original five reviewed backend function handles and immutable label strings; no data-bearing default is mutated. ctx/stage/counts/files/report are intentionally external. interval resets phase, override, branch_id, step_in_interval, previous_raw; process overwrites expected_u and nominal_prediction before any substep. Deadline and capacity ledger are intentionally monotone external gates, not restorable physics.'},
            {'hook': 'sim.forward / backend_call_trace.counted', 'review': 'Same fixed class-function handle/default label; no graph recapture/array expansion/set_param after capture. Graph.graph_exec may initialize a non-dynamical executable cache; it is explicitly NOT bitexact-restored. Pointer checks/full synchronizations and fresh-donor/three-baseline gates remain necessary.'},
            {'hook': 'reward_manager.compute / worker.reward_capture / PenSimReader.attach_capture.reward', 'review': 'All persistent reward/camera/sensor/overcurrent caches are reachable manager instances in native inventory. Pure scoring and rotations write no persistent global task state. Original derived15ms reward stays secondary; controlled score uses captured raw20ms physics and read-only goal tables. Measured-contact module and separate host-readback MjData produce logs/gates only, never state or force assistance.'},
            {'hook': 'action.process_actions / full151_controller.process / native residual EMA', 'review': 'Residual/cache/history/targets and masks are native tensors/containers. The frozen C2 function has no mutable global task state. Baseline, limits, bias and source-action tables are frozen/read-only inputs; absolute intervention is explicit and checked rather than clipped or replayed through EMA twice.'},
            {'hook': 'reader.memory', 'review': 'PEN_MEMORY_DIMS is a read-only module mapping; memory() creates output clones from captured manager/action/camera tensors. No module-global PEN state is mutated. Reader state/contact/rotations read current captured data; global camera/robot/object selector defaults are only read, not resolved/edited during the branch.'}],
        'globals_and_defaults_not_serialized': [
            'Generic _Sidecars walks function closures, not arbitrary __globals__/__defaults__/__kwdefaults__. This explicit source review does not claim otherwise.',
            'observations.ROBOT/OBJECT SceneEntityCfg default objects are mutable types, but object_pose_wrist_local/_reference_frame_pose only read name/body selection. Resolution/config edits complete before this snapshot; none are called inside a branch.',
            'Worker/backend function keyword defaults hold frozen callables/labels, while None/boolean/string defaults elsewhere do not carry evolving physics. Unused Runner no_grad context is separately reviewed.',
            'Torch/Warp module caches/allocator/stream/executable runtime internals are not universal cloned state; only the specified execution handles/cache are conditionally accepted. No physical Jacobian or complete serializer certificate follows.',
            'Graph-captured temporary workspace arrays can be unreachable from env/reader and are not included in the574 visited WP count. Local implicitfast used-output initialization and solver/SAP initialization patterns have source support, but there is NO exhaustive write-before-read proof for every graph-owned workspace. Hidden scratch bits are NOT all inventoried or restored, and whole-hidden-state dependence remains UNKNOWN. Actual same-input phase0 baseline gates are an empirical engineering experiment, not proof of arbitrary input/phase full-state equivalence.',
            'External NP/Torch goal/input tables are not modified by the frozen branch code. External host MjData/statebuf are measurement/INTEGRATION scratch, overwritten by readback/get_state and not consumed as WP integration state.',
            'Python/NumPy/Torch CPU/CUDA RNG snapshot/restore stays enabled; no arbitrary other-thread/module monkeypatch or source change is accepted.'],
        'known_counterexamples_in_this_fixed_phase0_source_path': [],
        'remaining_obstacles_to_complete_hidden_state_certificate': ['Captured graph-owned scratch is not enumerated/restored and exhaustive definite-assignment/read-mask proof is absent.', 'Snapshot-time graph executable warm/cold status and stream-handle internal state were not saved.'],
        'mandatory_future_re_review': ['Any full8snapshot/multi-donor scope or changed native-entry ordering',
            'Any CPU model/data stepping/write, renderer/variant synchronization, plugin, graph recapture/parameter update or array replacement',
            'Any new global/default mutable numerical state or changed active hooks, sources, donor, model, context, input bundle or source actions'],
    }
    details = {'schema': 'full151105_phase0_opaque_and_closure_source_review:1',
        'status': 'CONDITIONAL_PHASE0_SOURCE_ALLOW_HIDDEN_STATE_UNKNOWN',
        'source_pins': pins, 'compiled_spec_items': compiled, 'closure_globals_defaults_review': closures,
        'graph_cache_limit': 'graph_exec mutable/lazy, cold-warm status at snapshot UNKNOWN; not foreign-bitexact. Captured temporary workspaces may be graph-owned and not visited; partial source initialization support is not exhaustive write-before-read proof. Whole-hidden-state dependence UNKNOWN. Conditional engineering exploration requires fixed captured array addresses, no graph updates, full synchronization, integration/inventory checks and independent fresh3baseline gates; these do not certify all hiddenbits.',
        'execution_handle_review_remaining_scope': handles['graph_scratch_scope'],
        'future_path_limit': 'native_entry[index].value[index] is a discovery-order identity path. Later phases49/99/149 can add tensors/containers and shift first-visit paths. This phase0 set cannot authorize full8; new per-snapshot discovery/review or an explicitly reviewed semantic-path design is needed.',
        'execution': {'production_imports': 0, 'model_loads': 0, 'Env': 0, 'native_query_step_forward': 0, 'GPU': 0}}
    save(OUT / 'source_review_details.json', details)
    pins['source_review_details'] = pin(OUT / 'source_review_details.json')
    foreign = {'schema': 'full151105_foreign_inventory_source_review:1',
        'worker': pins['worker'], 'transaction': pins['transaction'], 'input_bundle': cfg['input_bundle'],
        'source_reviewed_closure_globals': True, 'allowed_identity_only': allowed, 'source_pins': pins,
        'approved_probe_modes': ['engineering_restore_smoke'], 'allowed_snapshot_labels': ['A0/0'], 'phase0_only': True,
        'conditional_engineering_only': True, 'whole_graph_hidden_state_qualified': False,
        'graph_hidden_scratch_dependence': 'UNKNOWN_NOT_EXHAUSTIVELY_PROVED_OR_RESTORED',
        'native_restore_qualified': False, 'physical_gradient_qualified': False, 'GPU_run_authorized': False,
        'scope': 'Exact current79207 worker/f1bd transaction and discovery02 phase0 only. Conditional identity acceptance is not restoration of opaque internals, not whole-foreign bitexact and not permission to run physics.',
        'implementation_limit': 'Current worker checks exact path/type/source table but extra mode/phase scope fields require Root builder/front guards. Do not consume this manifest for serial_full or later phases.'}
    assert all(pin(value['path'])['sha256'] == value['sha256'] for value in pins.values())
    assert not any(k.split('.')[0] in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic', 'full151_gradient', 'full151_native') for k in sys.modules)
    save(OUT / 'foreign_review_v1.json', foreign)
    checks = {'actual_phase0_discovery_bound': True, 'all198_unique_exact_path_type_covered': True,
              'compiletime166_semantic_field_review': True, 'execution_handle29_peer_source_join': True,
              'CPUtemplate_and_unused_decorator3_peer_source_join': True,
              'each_item_exact3keys_reason_not_type_wildcard': True, 'all_flat_sources_current': True,
              'active_closure_globals_and_kwdefaults_source_reviewed': True,
              'external_ledger_logs_and_branch_initialization_explicit': True,
              'mutable_graph_cache_not_claimed_readonly_or_bitexact': True,
              'phase0_only_mode_and_label_explicit': True, 'full8_path_drift_not_inherited': True,
              'native_restore_and_gradient_still_unqualified': True, 'no_production_import_or_execution': True}
    report = {'schema': 'full151105_phase0_foreign_inventory_source_enrollment:1',
        'status': 'PHASE0_CONDITIONAL_SOURCE_REVIEW_HIDDEN_STATE_UNKNOWN',
        'passed': len(checks), 'total': len(checks), 'checks': checks,
        'foreign_review': pin(OUT / 'foreign_review_v1.json'), 'details': pin(OUT / 'source_review_details.json'),
        'allowed_count': len(allowed), 'type_counts': dict(Counter(v['type'] for v in allowed)),
        'source_pin_count': len(pins), 'source_pins': pins,
        'source_reviewed_closure_globals_scope': closures['scope'],
        'known_source_counterexamples_this_fixed_phase0_path': [],
        'remaining_complete_state_certificate_blockers': closures['remaining_obstacles_to_complete_hidden_state_certificate'],
        'approved_probe_modes': ['engineering_restore_smoke'],
        'native_restore_qualified': False, 'physical_gradient_qualified': False,
        'execution': {'production_imports': 0, 'model_loads': 0, 'Env': 0, 'physics': 0, 'GPU': 0}}
    save(OUT / 'report.json', report)
    print(json.dumps({'report': pin(OUT / 'report.json'), 'foreign_review': pin(OUT / 'foreign_review_v1.json'),
                      'allowed': len(allowed), 'checks': [len(checks), len(checks)]}, indent=2))


if __name__ == '__main__':
    main()
