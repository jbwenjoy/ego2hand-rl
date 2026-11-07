"""Source/AST-only preservation proof; no native backend or state execution."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py'
V2 = ROOT.parent/'hand2_holding_v2/run_cpu_support_holding_v2.py'
V3 = ROOT/'run_cpu_support_holding_v3.py'


def pin(p):
    b = p.read_bytes()
    return {'path': str(p.resolve()), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def one(tree, cls, predicate):
    found = [n for n in ast.walk(tree) if isinstance(n, cls) and predicate(n)]
    assert len(found) == 1, (cls.__name__, len(found))
    return found[0]


def same(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


texts = [p.read_text(encoding='utf8') for p in (V1, V2, V3)]
trees = [ast.parse(s) for s in texts]
mains = [one(t, ast.FunctionDef, lambda n: n.name == 'main') for t in trees]
checks = {}
for label, tree, main in zip(('v1', 'v2'), trees, mains):
    checks[label+'_complete_tick_loop_exact'] = same(
        one(main, ast.For, lambda n: ast.unparse(n.target) == 'tick'),
        one(mains[2], ast.For, lambda n: ast.unparse(n.target) == 'tick'))
    for name in ('DTS', 'DURATION', 'GATES'):
        pred = lambda n: any(isinstance(v, ast.Name) and v.id == name for v in n.targets)
        checks[label+'_'+name+'_exact'] = same(one(tree, ast.Assign, pred), one(trees[2], ast.Assign, pred))
    for name in ('COM', 'angle'):
        checks[label+'_'+name+'_exact'] = same(
            one(main, ast.FunctionDef, lambda n: n.name == name),
            one(mains[2], ast.FunctionDef, lambda n: n.name == name))
    for name in ('tail_start_tick', 'tail', 'pre_tail', 'qvio', 'lin', 'omega', 'metrics', 'gates'):
        pred = lambda n: any(isinstance(v, ast.Name) and v.id == name for v in n.targets)
        checks[label+'_'+name+'_exact'] = same(one(main, ast.Assign, pred), one(mains[2], ast.Assign, pred))
    pred = lambda n: ast.unparse(n.target) == '(k, v)' and ast.unparse(n.iter) == 'metrics.items()'
    checks[label+'_all_individual_metric_gate_comparisons_exact'] = same(one(main, ast.For, pred), one(mains[2], ast.For, pred))
checks['v2_dt_pair_function_exact'] = same(
    one(trees[1], ast.FunctionDef, lambda n: n.name == 'dt_pair_summary'),
    one(trees[2], ast.FunctionDef, lambda n: n.name == 'dt_pair_summary'))

# Main's executable physics is identical to v2; only exact descriptive labels
# and result namespace differ. No numerical constant or expression is erased.
labels = {
    'hand2_phone_support_CPU_holding_result:3': 'hand2_phone_support_CPU_holding_result:2',
    'BLOCKED_NO_SUPPORT_POSE_SEED_ELIGIBLE_NO_MODEL_OR_STEPS': 'BLOCKED_NO_V4_SEED_ELIGIBLE_NO_MODEL_OR_STEPS',
    'COMPLETED_ALL_ENROLLED_SUPPORT_POSE_SEEDS_BOTH_DT_UNASSISTED_DIAGNOSTIC': 'COMPLETED_ALL_ENROLLED_V4_SEEDS_BOTH_DT_UNASSISTED_DIAGNOSTIC',
    'Exact finite support-pose verified pose and static zero speed': 'Exact finite v4 verified pose and static zero speed',
}
normalized = ast.parse(texts[2])
normalized_main = one(normalized, ast.FunctionDef, lambda n: n.name == 'main')
for node in ast.walk(normalized_main):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node.value = labels.get(node.value, node.value).replace(
            'All joint-support-pose force-hard/geometry-eligible seeds', 'All v4 force-hard/geometry-eligible seeds')
checks['whole_main_v2_AST_exact_after_only_enumerated_label_normalization'] = same(mains[1], normalized_main)
assert all(checks.values()), {k:v for k,v in checks.items() if not v}
diff = ''.join(difflib.unified_diff(texts[1].splitlines(True), texts[2].splitlines(True),
    fromfile=str(V2), tofile=str(V3)))
dp = ROOT/'metadata_adapter_v2_to_v3.diff'
with dp.open('x', encoding='utf8') as f:
    f.write(diff)
report = {
    'schema': 'holding_v3_cross_version_source_preservation:1',
    'status': 'PASS_SOURCE_AST_ONLY_NOT_EXECUTION_QUALIFICATION',
    'source_pins': {'v1':pin(V1), 'v2':pin(V2), 'v3':pin(V3), 'checker':pin(Path(__file__))},
    'diff':pin(dp), 'checks':checks, 'passed':sum(checks.values()), 'total':len(checks),
    'adapter_delta': [
        'strict fixed_wrist_gravity_contact_pose_result:3/protocol:3 and four support_pose semantic inputs',
        'same compiled_binding/model pin required in static producer inputs and physics inputs',
        'derive enrollment from physical geometry, unknown empty, valid force-hard LP/caps/torque bounds/control and selected actual COM-below',
        'original strict candidate moment flag only recorded, never used as an enrollment condition',
        'both seed records retained and every enrolled seed receives both original dt cases'],
    'MJB_loads':0, 'native_imports':0, 'forward_or_steps':0,
}
rp = ROOT/'source_preservation_v1.json'
with rp.open('x', encoding='utf8') as f:
    json.dump(report, f, indent=2)
print(json.dumps({'report':pin(rp), 'runner':pin(V3), 'passed':report['passed'], 'total':report['total']}))
