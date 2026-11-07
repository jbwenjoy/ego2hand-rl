"""Saved JSON/source-only per-snapshot review. No production or framework imports."""
from pathlib import Path
from collections import Counter, deque
import ast
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/full151105_eight_snapshot_foreign_source_review_v1'
LABELS = [f'{p}/{k}' for p in ('A0', 'A2') for k in (0, 49, 99, 149)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(value):
    assert type(value) is dict and set(value) == {'path', 'sha256'}
    path = Path(value['path']).resolve(strict=True)
    assert path.is_absolute() and pin(path)['sha256'] == value['sha256'], value['path']
    return path


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def rooted_names(top):
    """Resolve saved owner graph in each label; never normalize runtime indices."""
    entries = top['native_entries']; index = {v['identity']: v for v in entries}
    assert len(index) == len(entries)
    names = {}
    for typename, rootname in (('mjlab.envs.manager_based_rl_env.ManagerBasedRlEnv', 'env'),
                               ('dex_supermimic.rl.dex_supermimic.pen.PenSimReader', 'reader')):
        roots = [v for v in entries if v['type'] == typename]
        assert len(roots) == 1
        queue = deque([(roots[0]['identity'], rootname)])
        while queue:
            identity, name = queue.popleft()
            if identity in names:
                continue
            names[identity] = name
            if identity not in index:
                continue
            owner = index[identity]
            for member in owner['members']:
                key = member['key']
                suffix = '.' + key if owner['kind'] == 'object' else '[' + key + ']' if owner['kind'] == 'list' else '[' + repr(key) + ']'
                queue.append((member['identity'], name + suffix))
    return names


def semantic(path, top, names):
    match = re.fullmatch(r'native_entry\[(\d+)\]\.value\[(\d+)\](.*)', path)
    if not match:
        assert path.startswith('execution_hook['), path
        return path
    i, j, tail = int(match[1]), int(match[2]), match[3]
    owner = top['native_entries'][i]; assert owner['entry'] == i
    member = owner['members'][j]
    key = member['key']
    suffix = '.' + key if owner['kind'] == 'object' else '[' + key + ']' if owner['kind'] == 'list' else '[' + repr(key) + ']'
    return names[owner['identity']] + suffix + tail


def fnmap(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def main():
    OUT.mkdir(exist_ok=False)
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    health_path = ROOT / 'outputs/full151105_inventory_discovery_all_saved_review_v2/report.json'
    health = read(health_path); assert health['schema'] == 'full151105_inventory_discovery_all_saved_independent:1'
    assert health['status'] == 'PASS_SAVED_DONOR_AND_VISIBLE_INVENTORY_COVERAGE_ONLY'
    assert health['passed'] == health['total'] == len(health['checks']) == 750 and all(v is True for v in health['checks'].values())
    assert health['probe_mode'] == 'inventory_discovery_all' and health['completed_inventory_snapshots'] == 8
    assert health['episode_step_returned'] == 1200 and health['restored_branches'] == 0
    assert health['physical_gradient_qualified'] is health['whole_hidden_state_qualified'] is health['restored_baseline_qualified'] is False
    assert list(health['inventory_snapshot_pins']) == LABELS
    cfg = read(checked(health['config'])); assert cfg['probe_mode'] == 'inventory_discovery_all'
    old_foreign_path = ROOT / 'outputs/full151105_phase0_foreign_inventory_source_review_v1/foreign_review_v1.json'
    old = read(old_foreign_path)
    old_top = read(checked(old['source_pins']['discovery_topology'])); old_names = rooted_names(old_top)
    base_roles = {(v['type'], semantic(v['path'], old_top, old_names)) for v in old['allowed_identity_only']}
    assert len(base_roles) == 198
    delta_path = ROOT / 'outputs/full151105_inventory04_stream_delta_review_v1/report.json'
    delta = read(delta_path)
    extras = {(v['snapshot_label'], v['path'], v['type']): v for v in delta['additional_exact_stream_items']}
    assert len(extras) == 30
    source_pins = {'review_producer': pin(__file__), 'saved_review750': pin(health_path),
                   'source_inventory_config': health['config'], 'source_inventory_report': health['producer_report'],
                   'source_inventory_table': health['source_inventory'],
                   'old_phase0_source_evidence_only_not_runtime_fallback': pin(old_foreign_path),
                   'new5_stream_and_container_delta': pin(delta_path),
                   'worker2': pin(ROOT / 'src/full151_gradient/serial_probe105_v2.py'),
                   'transaction': pin(ROOT / 'src/full151_gradient/transaction105.py'),
                   'input_bundle': health['input_bundle']}
    for prefix, table in (('source_evidence', old['source_pins']), ('delta_source', delta['source_pins'])):
        for key, value in table.items():
            checked(value); source_pins[prefix + '__' + key] = value
    for key in ('auditor', 'pure_saved_helper'):
        source_pins['saved750_' + key] = health[key]
    assert source_pins['worker2']['sha256'] == 'a24c89da47f4aad7acfc9d9dfd1f053518cdc878230716bb2c857127ec13e592'
    assert source_pins['transaction']['sha256'] == 'f1bd0eafed58783a6efbb78a0ba40da086b36851119eddd00e29fcddb972a55c'
    oldf = fnmap(ROOT / 'src/full151_gradient/serial_probe105_v1.py'); newf = fnmap(source_pins['worker2']['path'])
    unchanged = sorted(k for k in oldf if k in newf and oldf[k] == newf[k])
    assert all(k in unchanged for k in ('controlled_object_score_FP32', 'object_terms', 'object_terms_f64', 'baseline_errors', 'backend_call_trace', 'allocate_artifacts', 'dump_inventory'))
    tables = {}; review_rows = {}; base_closures = None; occurrences = 0
    for label in LABELS:
        item = health['inventory_snapshot_pins'][label]
        desc = read(checked(item['description'])); top = read(checked(item['topology'])); names = rooted_names(top)
        profile = desc['profile']; phase = int(label.split('/')[1])
        assert desc['label'] == label and desc['strict_native_types_checked'] is True
        assert type(profile['phase']) is int and profile['phase'] == phase and profile['boundary'] == 'pre_action'
        assert profile['format_version'] == 105 and profile['mode'] == 'rough_observation_reference_full151_v1'
        assert profile['retarget_variant_id'] == 'legacy_H123_full151_SCP8_knownrough_v1'
        assert top['worker_source'] == source_pins['worker2'] and top['transaction_source'] == source_pins['transaction']
        observed = desc['opaque_identity_only']; expected_count = 198 if phase == 0 else 203
        assert len(observed) == expected_count and len({v['path'] for v in observed}) == expected_count
        allowed = []; semantic_items = []; observed_roles = set()
        for value in observed:
            role = semantic(value['path'], top, names); key = (value['type'], role); observed_roles.add(key)
            is_new = key not in base_roles
            if is_new:
                proof = extras[(label, value['path'], value['type'])]
                assert value['type'] == 'torch.cuda.streams.ExternalStream' and proof['structural_owner_field'] == role
            typename = value['type']
            if typename.startswith('mujoco._specs.'):
                assert key in base_roles
                detail = 'Exact retained compile-time spec/indexing/XML-actuator element. Runtime Scene.write/update and XmlActuator.compute use initialized index/target tensors, not edit_spec/compile or Mjs parameter writes; source entity.py/scene.py/xml_actuator.py/actuator.py reviewed. Only names/IDs may be read. This is not a MuJoCo type-wide exemption.'
            elif typename == 'torch.cuda.streams.ExternalStream':
                detail = 'Exact TorchArray._torch_stream wrapper for this named bridge field. Non-owning handle from wp.get_stream; lazy cache construction and pointer/value owners are captured in this label. No replacement/destruction/concurrent work is permitted. Full synchronization and actual per-label3baseline gates remain required; stream queue/hidden internals are not cloned or qualified.'
            elif typename == 'warp._src.context.Graph':
                assert key in base_roles
                detail = 'Exact Simulation graph execution handle. graph_exec is a mutable lazy executable cache, not read-only or bitexact-restored; actual internal cache bytes/cold-warm fields were not saved. No graph recapture/update/set_param or captured-array replacement occurs in frozen worker2. Graph-owned solver/SAP/implicit scratch can be outside visited WP arrays; exhaustive write-before-read and whole-hidden-state independence remain UNKNOWN. Conditional local empirical experiment only, with fixed captured pointers, synchronization and fresh3baselines at this label; never a complete restore certificate.'
            elif typename == 'warp._src.context.Device':
                assert key in base_roles
                detail = 'Exact Simulation.wp_device fixed CUDA execution-context handle. Worker selects one device; scoped calls restore the active context and no branch recreates or switches device. Preserve handle lifetime and synchronize all pending work. Allocator/runtime/stream hidden state is not serialized or certified.'
            elif typename == 'mujoco._structs.MjModel':
                assert key in base_roles
                detail = 'Exact Simulation host model template: nominal compilation/upload precedes snapshots. Branch integration uses restored WP model/data; CPU model is read-only for dimensions, IDs and independent host contact decoding. No CPU model update/recompile/viewer/variant synchronization in this branch. NanGuard is explicitly disabled.'
            elif typename == 'mujoco._structs.MjData':
                assert key in base_roles
                detail = 'Exact Simulation CPU initialization template, not the worker separate readback host. Branch does not CPU-step/forward/inverse/reset this object; disabled NanGuard returns before diagnostic CPU mutation. No claim to clone its C state or use it as restored branch integration state.'
            else:
                assert typename == 'torch.autograd.grad_mode.no_grad' and key in base_roles
                detail = 'Exact unused original Runner.collect decorator context factory. No Runner/collect is constructed/executed; decorator clone creates a new context rather than entering this retained object. The probe outer no_grad is separate. No blanket immutability or hidden thread-state restoration claim.'
            reason = f'{label}: {role}. {detail} Bound only to worker2a24c/transactionf1bd, this pinned description/topology and source-reviewed frozen hooks; later runtime exact path/type inventory and local baseline gates still mandatory.'
            allowed.append({'path': value['path'], 'type': typename, 'reason': reason})
            semantic_items.append({'path': value['path'], 'type': typename, 'semantic_owner': role, 'additional_lazy_stream': is_new})
        assert base_roles.issubset(observed_roles) and len(observed_roles - base_roles) == (0 if phase == 0 else 5)
        assert len(allowed) == len({(v['path'], v['type']) for v in allowed})
        closure_roles = {(semantic(v['path'], top, names), v['value_type']) for v in top['closure_cells']}
        if base_closures is None:
            base_closures = closure_roles
        assert closure_roles == base_closures and len(closure_roles) == desc['closure_cells'] == 44
        np_roles = [semantic(v['path'], top, names) for v in top['array_values'] if 'path' in v]
        assert len(np_roles) == desc['explicit_arrays'] == (8 if phase == 0 else 9)
        if phase:
            assert 'reader.reward_snapshot[\'raw_sensordata\']' in np_roles
        tables[label] = {'inventory': item, 'allowed_identity_only': allowed, 'source_reviewed_closure_globals': True}
        review_rows[label] = {'phase': phase, 'opaque_count': len(allowed), 'type_counts': dict(Counter(v['type'] for v in allowed)),
            'semantic_items': semantic_items, 'closure_roles': sorted(closure_roles), 'explicit_NP_roles': np_roles,
            'native_entries': desc['native_entries'], 'WP': desc['native_Warp_arrays'], 'Torch': desc['native_Torch_tensors'],
            'explicit_containers': desc['explicit_containers'],
            'source_scope': 'Exact label source review, not global index normalization for the runtime validator; all original paths are retained.'}
        for field, value in item.items():
            source_pins['inventory_' + label.replace('/', '_') + '_' + field] = value
        occurrences += len(allowed)
    assert occurrences == 1614
    detail = {'schema': 'full151105_eight_snapshot_opaque_source_semantic_review:1',
        'status': 'SOURCE_REVIEWED_EXACT_LABELS_CONDITIONAL_HIDDEN_STATE_UNKNOWN',
        'per_label': review_rows, 'unchanged_worker1_functions_AST': unchanged,
        'source_globals_delta': [
            'Current worker2 pure controlled-score/control/backend helpers are source-equivalent to792; new modes select captures/branches and new manifest validators only read pinned JSON/source data.',
            'instrument_transaction_calls stores original bound begin/finish/Snapshot.restore in keyword defaults and increments external report counters only. Those wrappers do not add a numerical global state; snapshot payload is deliberately fixed for each transaction.',
            'All8 normalized active closure path/type sets are identical44. Declared deadline/name external differences and descriptor maps are input/control-flow bookkeeping, not restored simulation state.',
            'NP8→9 is captured reader.reward_snapshot raw_sensordata copy after first interval, not a new opaque/global state. Native snapshot does not copy NP itself; explicit sidecar does.',
            'A2 explicit containers546 versus A0184 is exactly362 extra execution_source_pins subcontainers (499 versus137); 47 non-execution-map container roles match. Frozen descriptor maps are read-only during branches, no model/objective is instantiated from them.',
            'New5lazy stream roles are each source-reviewed at all6later labels. They wrap actuator_force/cvel/qacc/qacc_warmstart/subtree_com arrays, not5independent CUDA queues.',
            'Globals/defaults review from prior conditional source evidence remains limited to current frozen active hooks: spec/selector maps and callable defaults are not mutated; ledger/log/time/counters intentionally stay external; every branch initializes ctx phase/override/previous state and processes its original source action before simulation.',
            'No generic globals serializer, all-graph-scratch inventory, opaque-internal byte restore or true physical Jacobian is claimed. Full run must recapture and independently pass3baselines per label.'],
        'unknowns': ['Graph executable cache/stream/device internals not cloned', 'Graph-owned scratch not exhaustively inventoried or proven write-before-read',
                     'No actual restored baseline/FD occurs in discovery04; passing local future baselines would remain empirical, input-specific evidence'],
        'source_pins': source_pins, 'item_occurrences_not_independent_objects_or_samples': occurrences,
        'execution': {'production_imports': 0, 'model_loads': 0, 'Env': 0, 'WM_Actor_forward': 0, 'physics': 0, 'GPU': 0}}
    save(OUT / 'source_review_details.json', detail)
    source_pins['source_review_details'] = pin(OUT / 'source_review_details.json')
    foreign = {'schema': 'full151105_foreign_inventory_per_snapshot_source_review:2',
        'worker': source_pins['worker2'], 'transaction': source_pins['transaction'], 'input_bundle': health['input_bundle'],
        'source_reviewed_closure_globals': True, 'approved_probe_modes': ['serial_full'], 'allowed_snapshot_labels': LABELS,
        'phase0_only': False, 'per_snapshot': tables, 'source_pins': source_pins,
        'source_inventory_saved_review': pin(health_path), 'source_inventory_config': health['config'],
        'source_inventory_report': health['producer_report'], 'source_inventory_run': health['run_dir'],
        'conditional_engineering_only': True, 'whole_graph_hidden_state_qualified': False,
        'graph_hidden_scratch_dependence': 'UNKNOWN_NOT_EXHAUSTIVELY_PROVED_OR_RESTORED',
        'native_restore_qualified': False, 'physical_gradient_qualified': False, 'GPU_run_authorized': False,
        'scope': 'Exact8label current-source review for a proposed local empirical finite-interval diagnostic; not union/fallback, cross-process native-handle transplant, complete hidden-state certificate or run authorization.'}
    assert all(pin(v['path'])['sha256'] == v['sha256'] for v in source_pins.values())
    assert not any(k.split('.')[0] in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic', 'full151_gradient', 'full151_native') for k in sys.modules)
    save(OUT / 'foreign_review_v2.json', foreign)
    report = {'schema': 'full151105_eight_snapshot_foreign_source_review_enrollment:1',
        'status': 'EXACT8_SOURCE_REVIEW_CONDITIONAL_HIDDEN_STATE_UNKNOWN',
        'foreign_review': pin(OUT / 'foreign_review_v2.json'), 'details': pin(OUT / 'source_review_details.json'),
        'inventory_labels': LABELS, 'per_label_opaque_counts': {k: len(v['allowed_identity_only']) for k, v in tables.items()},
        'reviewed_item_occurrences': occurrences, 'source_pin_count': len(source_pins), 'source_pins': source_pins,
        'source_inventory_saved_review': pin(health_path), 'native_restore_qualified': False,
        'whole_graph_hidden_state_qualified': False, 'physical_gradient_qualified': False,
        'no_new_type_or_unreviewed_semantic_owner': True, 'all44_closure_roles_per_label_matched': True,
        'runtime_validation_not_executed': True, 'execution': detail['execution']}
    save(OUT / 'report.json', report)
    print(json.dumps({'report': pin(OUT / 'report.json'), 'foreign_review': pin(OUT / 'foreign_review_v2.json'),
                      'labels': len(tables), 'occurrences': occurrences}, indent=2))


if __name__ == '__main__':
    main()
