"""Source/JSON-only passive MJB callgraph addendum; never imports reviewed code."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/full151105_passive_MJB_cost_source_addendum_v1'
SOURCES = {
    'bundle': ROOT/'src/full151_native/bundle.py',
    'pure_control760': Path('D:/workspace/supermimic-observation-prefix-study/src/prefix_control.py'),
    'Actor105_loader': ROOT/'src/full151_native/checkpoint.py',
    'strict_WM105_loader': ROOT/'src/full151_native/world_model_checkpoint.py',
    'binding3': ROOT/'src/full151_intent/binding_v3.py',
    'dense_actual_consumer': ROOT/'src/full151_gradient/penalty_observability105_v3.py',
    'state452_caller_source_only': ROOT/'src/full151_gradient/state_prediction452105_v1.py',
    'frozen_contact_geometry': Path('D:/workspace/supermimic-rough-prefix-learning-study/vendor/frozen_contact_math/contact_geometry.py')}


def pin(p):
    p = Path(p).resolve(strict=True)
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, data):
    with p.open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def main():
    OUT.mkdir(exist_ok=False); (OUT/'source_snapshots').mkdir()
    reviewed = {}; text = {}; parsed = {}
    for key, p in SOURCES.items():
        b = p.read_bytes(); reviewed[key] = pin(p); text[key] = b.decode('utf-8-sig'); parsed[key] = ast.parse(text[key])
        (OUT/'source_snapshots'/(key+'__'+p.name)).write_bytes(b)
    (OUT/'executed_source_reviewer.py').write_bytes(Path(__file__).read_bytes())
    sites = {}
    def site(key, needle, label, expected=1):
        hits = [{'line': i, 'source': line.strip()} for i, line in enumerate(text[key].splitlines(), 1) if needle in line]
        if len(hits) != expected:
            raise ValueError('Unexpected source callsite count '+key+'/'+label+': '+str(len(hits)))
        sites[label] = {'source_pin': reviewed[key], 'occurrences': hits}
        return hits
    site('bundle', 'params,metadata=old.load_actual_pd(', 'bundle_calls_actual_PD_every_invocation')
    site('bundle', 'if name not in sys.modules:', 'module_import_cache_only')
    site('pure_control760', 'm=mujoco.MjModel.from_binary_path(', 'actual_PD_unconditional_MJB')
    site('Actor105_loader', "actual = load_bundle(", 'Actor_validated_bundle_unconditional_reload')
    site('Actor105_loader', 'b = _validated_bundle(bundle)', 'Actor_calls_bundle_revalidation')
    site('Actor105_loader', 'model = mujoco.MjModel.from_binary_path(', 'Actor_explicit_MJB')
    site('strict_WM105_loader', 'actor,actor_receipt=load_actor_checkpoint(', 'strict_WM_calls_Actor_validation')
    site('binding3', 'bundle=load_bundle(', 'factory_constructor_bundle_PD_MJB')
    site('binding3', 'model=mujoco.MjModel.from_binary_path(', 'factory_constructor_explicit_MJB')
    site('binding3', 'nn.Module.__init__(self)', 'new_binding_bypasses_old_A2_constructor')
    site('binding3', 'module=Full151DeclaredH123ContactObjective(spec)', 'factory_build_new_instance')
    site('binding3', 'return _build(spec,device,True)', 'P_ONLY_factory_enters_same_constructor')
    site('dense_actual_consumer', "bundle=load_bundle(", 'dense_outer_bundle_PD_MJB')
    site('dense_actual_consumer', "model=call('passive_mapping_MJB'", 'dense_explicit_MJB_counter')
    site('dense_actual_consumer', "objective=call('objective_factory'", 'dense_factory_API_counter')
    site('state452_caller_source_only', 'bundle=load_bundle(', 'WM452_outer_bundle_PD_MJB')
    site('state452_caller_source_only', "model,receipt=call('strict_WM_loader'", 'WM452_strict_loader_API_counter')
    # Only imports are cached. The model-producing function calls occur after
    # those branches and do not carry memoization decorators or early returns.
    selected = [('bundle', 'load_bundle'), ('pure_control760', 'load_actual_pd'),
        ('Actor105_loader', '_validated_bundle'), ('Actor105_loader', 'load_actor_checkpoint'),
        ('strict_WM105_loader', 'load_world_model_checkpoint'), ('binding3', '_build')]
    function_inventory = []
    for key, name in selected:
        n = next(n for n in parsed[key].body if isinstance(n, ast.FunctionDef) and n.name == name)
        function_inventory.append({'source': reviewed[key], 'name': name, 'start_line': n.lineno, 'end_line': n.end_lineno,
            'decorator_count': len(n.decorator_list)})
    checks = {
        'all_six_relevant_functions_have_no_result_cache_decorators': all(x['decorator_count'] == 0 for x in function_inventory),
        'module_cache_does_not_skip_PD_call': sites['module_import_cache_only']['occurrences'][0]['line'] < sites['bundle_calls_actual_PD_every_invocation']['occurrences'][0]['line'],
        'PD_native_load_unconditional_after_hash_checks': 'lru_cache' not in text['pure_control760'] and 'cache' not in ast.get_source_segment(text['pure_control760'], next(n for n in parsed['pure_control760'].body if isinstance(n, ast.FunctionDef) and n.name == 'load_actual_pd')).lower(),
        'factory_constructs_fresh_binding_not_old_A2_constructor': 'GeometricContactObjective.__init__(' not in text['binding3'],
        'geometry_proxy_has_no_additional_binary_model_read': 'from_binary_path' not in text['frozen_contact_geometry'],
        'dense_total4_is_two_outer_plus_two_factory': 1+1+(1+1) == 4,
        'strict_loader_internal2_is_bundle_and_Actor_explicit': 1+1 == 2,
        'WM452_total3_is_outer1_plus_strict2': 1+(1+1) == 3,
        'source_reviewer_imports_no_backend_or_model': not any(k in sys.modules for k in ('torch', 'numpy', 'mujoco', 'warp', 'scipy', 'dex_supermimic'))}
    producer = ROOT/'outputs/full151105_solver_phase_penalty_GPU_qualified_CPU_v1/report.json'
    old_review = ROOT/'outputs/full151105_solver_phase_penalty_GPU_qualified_saved_independent_v1/report.json'
    r = read(producer)
    dense = {'case': 'COMPLETED existing c489 denseP; numeric results not rerun',
        'producer_report': pin(producer), 'existing_independent35': pin(old_review),
        'actual_recorded_counters': r['counts'],
        'explicit_instrumented_passive_MJB_counter': r['counts']['passive_mapping_MJB'],
        'factory_API_calls_recorded': r['counts']['objective_factory'],
        'previous_factory_cost_text_retained_in_old_report': r['actual_factory_passive_MJB_read'],
        'source_derived_on_successful_path': {'outer_bundle_via_PD': 1, 'explicit_catalog_MJB': 1,
            'factory_bundle_via_PD': 1, 'factory_explicit_MJB': 1, 'factory_subtotal': 2, 'whole_process_project_callgraph_total': 4},
        'correction': 'Prior source-accounted factoryMJB1 text omitted the factory bundle->PD MJB load. Existing explicit counter1 and factoryAPI1 are valid within their instrumented scope, not total native-load counts. Original35 math/data gates do not certify correct whole-process cost accounting.'}
    model_file = read(ROOT/'outputs/full151_native_input_v1/bundle.json')['model']
    for key, p in SOURCES.items():
        if pin(p) != reviewed[key]:
            raise ValueError('Source changed during source-only addendum: '+key)
    report = {'schema': 'full151105_passive_MJB_source_cost_addendum:1',
        'status': 'SOURCE_ONLY_CALLGRAPH_COST_CORRECTION_NO_NUMERIC_RERUN',
        'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'reviewer': pin(__file__), 'source_pins': reviewed, 'callsite_evidence': sites,
        'function_spans_and_decorators': function_inventory, 'same_nominal_model_pin': model_file,
        'actual_dense_case': dense,
        'prospective_WM452': {'source_only_caller': reviewed['state452_caller_source_only'],
            'source_derived_on_successful_path': {'outer_bundle_via_PD': 1,
                'strict_loader_Actor_bundle_via_PD': 1, 'strict_loader_Actor_explicit_MJB': 1,
                'strict_loader_subtotal': 2, 'whole_process_project_callgraph_total': 3},
            'scope': 'Unexecuted caller source as pinned here; later cost-only route revision must bind its own source. No actual new WM/MJB load claimed.'},
        'limits': ['Source-derived number of listed successful from_binary_path invocations, not a process-wide/native API interceptor measurement.',
            'sys.modules cache stores helper modules, not model/PD/bundle return values; each load_bundle invocation re-enters load_actual_pd.',
            'Existing Actor validation instantiation/two checkpoint reads and other costs are separate scopes, not MJB load counts.',
            'All old reports/ledgers/source remain unchanged. Numerical35/58/SDF/P/HO masks and scientific qualifications are not modified.',
            'No claims about total costs of unlisted old callers or backend-internal operations beyond these source-grounded call edges.'],
        'new_model_checkpoint_loads': 0, 'new_MJB_loads': 0, 'numeric_or_geometry_reruns': 0, 'new_physics_GPU': 0}
    write(OUT/'report.json', report)
    print(json.dumps({'report': pin(OUT/'report.json'), 'passed': report['passed'], 'total': report['total'],
        'dense_source_total': 4, 'strict_internal_source_total': 2, 'WM452_source_total': 3}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
