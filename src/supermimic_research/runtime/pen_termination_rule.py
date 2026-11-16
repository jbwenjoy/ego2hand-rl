"""Pinned AST qualification and exclusive process-local termination binding.

This module does not import a simulator/CUDA until explicitly requested. It
does not transfer old native/Python state or qualify a physical transition.
"""
import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from ..artifacts import file_pin, verify_file_pin

_ACTIVE = None
RULE = 'DROP_IMPROVEMENT_ONLY_EXIT_KEEP_GRAD_DECREMENT_CAP_V1'


def _function(path, name):
    module = ast.parse(Path(path).read_text(encoding='utf-8'))
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(functions) != 1:
        raise ValueError('Exactly one expected top-level function required: ' + name)
    return functions[0]


def _loaded_code_matches(function, path):
    compiled = compile(Path(path).read_text(encoding='utf-8'), str(path), 'exec', dont_inherit=True)
    expected = [c for c in compiled.co_consts if isinstance(c, type(compiled)) and c.co_name == function.__name__]
    return len(expected) == 1 and function.__code__ == expected[0]


def qualify_sources(installed_pin, candidate_path=None):
    """Require exactly one AST expression change; preserve every other node."""
    installed = verify_file_pin(installed_pin)
    candidate_path = Path(candidate_path or Path(__file__).with_name('pen_termination_rule_kernels.py'))
    original = _function(installed, '_solve_done')
    candidate = _function(candidate_path, 'candidate_solve_done')
    if ast.dump(_function(installed, '_rescale').body[0], include_attributes=False) != ast.dump(
            ast.parse('return value / (meaninertia * float(nv))').body[0], include_attributes=False):
        raise ValueError('Pinned scaling expression changed')
    assignments = [n for n in original.body if isinstance(n, ast.Assign) and
                   len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'done']
    if len(assignments) != 1:
        raise ValueError('Single original done assignment required')
    expected_original = ast.parse('done = (improvement < tolerance) or (gradient < tolerance) or (model_improvement < tolerance)').body[0]
    if ast.dump(assignments[0], include_attributes=False) != ast.dump(expected_original, include_attributes=False):
        raise ValueError('Original OR termination rule changed')
    assignments[0].value = ast.parse('(gradient < tolerance) or (model_improvement < tolerance)', mode='eval').body
    original.name = candidate.name
    if ast.dump(original, include_attributes=False) != ast.dump(candidate, include_attributes=False):
        raise ValueError('Candidate changes more than the single done expression')
    verify_file_pin(installed_pin)
    return dict(installed_solver=installed_pin, candidate=file_pin(candidate_path),
                adapter=file_pin(__file__), single_expression_change=True, rule=RULE)


def implementation_identity(collision_backend_id, installed_pin):
    if len(collision_backend_id) != 64 or any(c not in '0123456789abcdef' for c in collision_backend_id):
        raise ValueError('Exact qualified collision backend digest required')
    identity = dict(schema='prospective_pen_solver_rule_backend:1',
                    collision_backend_id=collision_backend_id, **qualify_sources(installed_pin))
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(backend_id=hashlib.sha256(payload).hexdigest(), identity=identity,
                physically_qualified=False, old_source_transfer_qualified=False)


@contextmanager
def kernel_binding(module, installed_pin, candidate, *, synchronize):
    """Bind only inside an exclusive graph build/replay scope, restore on error.

    Caller supplies synchronization and must keep graph/context owners alive.
    This helper does not approve runtime/model/old-source metadata migration.
    """
    global _ACTIVE
    proof = qualify_sources(installed_pin)
    if module.__name__ != 'mujoco_warp._src.solver' or file_pin(module.__file__) != installed_pin:
        raise ValueError('Wrong installed solver module')
    original = module._solve_done
    if original.func.__name__ != '_solve_done' or not _loaded_code_matches(original.func, installed_pin['path']):
        raise ValueError('Unmodified original kernel required')
    if (candidate.func.__name__ != 'candidate_solve_done'
            or not _loaded_code_matches(candidate.func, proof['candidate']['path'])
            or candidate.func.__globals__['_rescale'] is not module._rescale
            or not _loaded_code_matches(module._rescale.func, installed_pin['path'])):
        raise ValueError('Qualified candidate kernel required')
    if _ACTIVE is not None:
        raise ValueError('Exclusive termination scope required')
    audit = dict(rule=RULE, source_proof=proof, entered=True, bindings_restored=False)
    _ACTIVE = audit
    module._solve_done = candidate
    try:
        yield audit
        if module._solve_done is not candidate or _ACTIVE is not audit:
            raise ValueError('Termination binding changed inside scope')
    finally:
        try:
            synchronize()
        finally:
            module._solve_done = original
            _ACTIVE = None
            audit['bindings_restored'] = module._solve_done is original
            qualify_sources(installed_pin)
