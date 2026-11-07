"""Pure AST/source preservation audit: no backend/model/state/physics imports."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py'
NEW = ROOT/'run_cpu_support_holding_v2.py'


def pin(path):
    data = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def nodes(tree, cls, predicate):
    return [n for n in ast.walk(tree) if isinstance(n, cls) and predicate(n)]


def unique(tree, cls, predicate):
    found = nodes(tree, cls, predicate)
    assert len(found) == 1, (cls.__name__, len(found))
    return found[0]


def same(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


old_tree = ast.parse(OLD.read_text(encoding='utf8'))
new_tree = ast.parse(NEW.read_text(encoding='utf8'))
tick_old = unique(old_tree, ast.For, lambda n: ast.unparse(n.target) == 'tick')
tick_new = unique(new_tree, ast.For, lambda n: ast.unparse(n.target) == 'tick')
checks = {'full_native_tick_loop_AST_exact_original': same(tick_old, tick_new)}
for name in ('DTS', 'DURATION', 'GATES'):
    find = lambda n: any(isinstance(v, ast.Name) and v.id == name for v in n.targets)
    checks[name+'_exact_original'] = same(unique(old_tree, ast.Assign, find), unique(new_tree, ast.Assign, find))
for name in ('COM', 'angle'):
    checks[name+'_function_AST_exact_original'] = same(
        unique(old_tree, ast.FunctionDef, lambda n: n.name == name),
        unique(new_tree, ast.FunctionDef, lambda n: n.name == name))
for name in ('tail_start_tick', 'tail', 'pre_tail', 'qvio', 'lin', 'omega', 'metrics', 'gates'):
    find = lambda n: any(isinstance(v, ast.Name) and v.id == name for v in n.targets)
    checks[name+'_AST_exact_original'] = same(unique(old_tree, ast.Assign, find), unique(new_tree, ast.Assign, find))
gate_old = unique(old_tree, ast.For, lambda n: ast.unparse(n.target) == '(k, v)' and ast.unparse(n.iter) == 'metrics.items()')
gate_new = unique(new_tree, ast.For, lambda n: ast.unparse(n.target) == '(k, v)' and ast.unparse(n.iter) == 'metrics.items()')
checks['all_original_metric_gate_comparisons_AST_exact'] = same(gate_old, gate_new)
assert all(checks.values()), checks
report = {
    'schema': 'holding_v2_source_preservation_only:1',
    'status': 'PASS_SOURCE_AST_ONLY_NOT_EXECUTION_QUALIFICATION',
    'source_pins': {'original_runner': pin(OLD), 'new_runner': pin(NEW), 'source_checker': pin(Path(__file__))},
    'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
    'intentional_delta': [
        'new four-input protocol/result adapter with complete two-seed enrollment screening',
        'per-variant selected qpos, zero velocity, COM0/quaternion0 and proposed ctrl20',
        'trace additionally saves initial_COM_world3; summary additionally binds seed/state and predecessor residual',
        'saved terminal dt comparison with normalized-F64 quaternion observer and both-complete requirement',
        'new result namespace, complete screening and blocked-no-eligible route'],
    'actual_model_data_reads': 0, 'native_backend_imports': 0,
    'forward_or_integration_steps': 0, 'automatic_execution_or_retry': False,
}
path = ROOT/'source_preservation_v2.json'
with path.open('x', encoding='utf8') as f:
    json.dump(report, f, indent=2)
print(json.dumps({'report': pin(path), 'passed': report['passed'], 'total': report['total']}))
