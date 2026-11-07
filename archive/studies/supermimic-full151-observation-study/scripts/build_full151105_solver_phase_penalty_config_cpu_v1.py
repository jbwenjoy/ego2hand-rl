"""Metadata-only authorizer for one saved same-solver-phase CPU diagnostic.

Default prints a declaration and writes nothing. --authorize creates one fresh
Root-owned CPU config; it never launches the diagnostic. No model/NPZ loading.
Consumer import occurs only after current-source, executed CPU READY checks.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/full151105_solver_phase_penalty_observability_CPU_prereg_v2.json'
PROTOCOL_SHA = '1ec18db963bc2931920bedce3f2bf43e345544798ee4ac1e0f856ac119cea13c'
CONFIG = ROOT / 'configs/full151105-solver-phase-penalty-CPU-v1.json'
OUTPUT = ROOT / 'outputs/full151105_solver_phase_penalty_CPU_v1'
OWN = {'consumer': ROOT / 'src/full151_gradient/penalty_observability105_v1.py',
       'CLI': ROOT / 'scripts/diagnose_solver_phase_penalty_cpu_v1.py',
       'checker': ROOT / 'scripts/check_solver_phase_penalty_cpu_v1.py'}
SOURCE_REVIEW = ROOT / 'outputs/full151105_solver_phase_observability_protocol_independent_v1/report.json'
SOURCE_REVIEW_SHA = 'd1331b67206eb066766f45c4d6e7c26f65114775c9b510e1dee57ed8df77e1b7'
SAVED_MATH = ROOT / 'outputs/full151105_observability_saved_auditor_preparation_v1/report.json'
KEYS = {'schema', 'role', 'protocol', 'output', 'CPU_run_authorized', 'cpu_threads', 'source_pins', 'cpu_ready'}
FORBIDDEN = ('torch', 'warp', 'mujoco', 'mujoco_warp', 'scipy')


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def pin(path):
    p = Path(path).resolve(strict=True)
    require(p.is_file(), 'Pinned regular file required')
    return {'path': str(p), 'sha256': digest(p)}


def checked(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Strict flat path/SHA pin required')
    require(type(value['path']) is str and Path(value['path']).is_absolute(), 'Absolute input pin required')
    require(type(value['sha256']) is str and len(value['sha256']) == 64
            and all(c in '0123456789abcdef' for c in value['sha256']), 'Canonical SHA256 required')
    p = Path(value['path']).resolve(strict=True)
    require(p.is_file() and digest(p) == value['sha256'], 'Changed pinned file: ' + str(p))
    return p


def same_pin(left, right):
    """Retain original strings; intended identity is SHA plus actual samefile."""
    return left['sha256'] == right['sha256'] and checked(left).samefile(checked(right))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def positive(report, passed_key='passed', total_key='total'):
    checks = report.get('checks')
    require(type(checks) is dict and checks and all(type(v) is bool and v for v in checks.values()), 'All source/CPU checks must be positive')
    require(type(report.get(passed_key)) is int and type(report.get(total_key)) is int
            and report[passed_key] == report[total_key] == len(checks), 'Positive receipt counts differ')


def load_executed_ready(path, protocol_pin):
    """No consumer import before this executed-source gate is closed."""
    ready = read(path)
    require(ready.get('schema') == 'full151105_dense_penalty_CPU_contract_READY:1'
            and ready.get('status') == 'PASS_SYNTHETIC_NATIVE_STATE_AND_HOOK_ONLY', 'Actual new dense CPU READY required')
    require(same_pin(ready['protocol'], protocol_pin), 'READY belongs to another protocol')
    report = read(checked(ready['cpu_report'])); positive(report)
    require(report.get('schema') == 'full151105_dense_penalty_CPU_contract_proof:1', 'Wrong CPU proof namespace')
    require(ready.get('cpu_checks') == {'passed': report['passed'], 'total': report['total']}, 'READY/proof counts differ')
    scope = report['scope']
    for key in ('actual_physical_state_conversions', 'actual_MJB_loads', 'actual_objective_constructs', 'actual_full05_data_reads'):
        require(type(scope.get(key)) is int and scope[key] == 0, 'Synthetic proof scope elevated: ' + key)
    for key, current in OWN.items():
        actual = pin(current)
        require(same_pin(ready['source_pins'][key], actual) and same_pin(report['source_pins'][key], actual), 'CPU-tested source drift: ' + key)
    for table in (ready['source_pins'], report['source_pins']):
        require(type(table) is dict and table, 'Flat executed source table missing')
        for value in table.values(): checked(value)
    require(not any(name in sys.modules for name in FORBIDDEN), 'Metadata READY checks imported a native/model framework')
    return ready, report


def build_candidate(ready_path, authorize=False):
    own_before = pin(__file__)
    protocol_pin = pin(PROTOCOL); require(protocol_pin['sha256'] == PROTOCOL_SHA, 'Frozen dense protocol changed')
    protocol = read(PROTOCOL)
    require(protocol.get('schema') == 'full151105_penalty_solver_phase_observability_CPU_preregistration:1'
            and type(protocol.get('revision')) is int and protocol['revision'] == 2
            and protocol.get('CPU_run_authorized') is False and protocol.get('GPU_run_authorized') is False
            and protocol.get('training_authorized') is False, 'Source-only protocol2 required')
    ready_path = Path(ready_path).resolve(strict=True)
    ready, proof = load_executed_ready(ready_path, protocol_pin)
    # This is the current CPU-tested module, not an unfinished draft. Its
    # required_sources function reads JSON and hashes; it opens no NPZ/model.
    sys.path.insert(0, str(ROOT / 'src'))
    module = importlib.import_module('full151_gradient.penalty_observability105_v1')
    require(Path(module.__file__).samefile(OWN['consumer']), 'Foreign cached consumer module')
    require(module.SCHEMA == 'full151105_solver_phase_penalty_CPU_run:1'
            and module.ROLE == 'same_solver_phase_penalty_observability' and module.KEYS == KEYS, 'Actual frozen consumer ABI differs')
    sources = module.required_sources(protocol)
    require(type(sources) is dict and sources, 'Current required_sources must return a flat nonempty map')
    # Preserve all recorded path spellings and transitive histories, including
    # original physical sources and previous bonus1711 dependencies. Extra
    # evidence aliases archive the current authorization path, not a new score.
    sources = dict(sources)
    for key, value in protocol['inputs'].items(): sources['dense_prereg_input__' + key] = value
    for key, value in ready['source_pins'].items(): sources['dense_CPU_ready_source__' + key] = value
    for key, value in proof['source_pins'].items(): sources['dense_CPU_proof_source__' + key] = value
    source_review_pin = pin(SOURCE_REVIEW)
    require(source_review_pin['sha256'] == SOURCE_REVIEW_SHA, 'Independent24source review changed')
    review = read(SOURCE_REVIEW); positive(review, 'passed_source_checks', 'total_source_checks')
    require(review.get('schema') == 'full151105_solver_phase_penalty_protocol_independent_source_review:1'
            and same_pin(review['protocol'], protocol_pin) and review.get('numerical_tests_performed') == 0, 'Protocol source review namespace/scope mismatch')
    algebra = read(SAVED_MATH); positive(algebra)
    require(algebra.get('schema') == 'full151105_observability_saved_algebra_preparation:1'
            and algebra.get('synthetic_only') is True and algebra.get('actual_dense_outputs_read') is False
            and algebra.get('actual_converter_calls') == algebra.get('model_loads') == algebra.get('physics_steps') == 0,
            'Saved-algebra11 proof is not synthetic/source preparation')
    require(same_pin(algebra['source_pins']['preregistration'], protocol_pin), 'Saved algebra belongs to another protocol')
    for prefix, table in (('dense_source_review__', review['source_pins']), ('dense_saved_math__', algebra['source_pins'])):
        for key, value in table.items(): sources[prefix + key] = value
    sources.update(dense_authorizer_builder=own_before, dense_protocol=protocol_pin, dense_CPU_READY=pin(ready_path),
                   dense_CPU_contract=ready['cpu_report'], dense_protocol_source_review=source_review_pin,
                   dense_saved_math_preparation=pin(SAVED_MATH))
    for value in sources.values(): checked(value)
    candidate = {'schema': module.SCHEMA, 'role': module.ROLE, 'protocol': protocol_pin, 'output': str(OUTPUT),
                 'CPU_run_authorized': bool(authorize), 'cpu_threads': 1, 'source_pins': sources, 'cpu_ready': pin(ready_path)}
    require(set(candidate) == module.KEYS, 'Authorizer introduced an unconsumed/unknown field')
    module.validate_header(candidate)
    require(not CONFIG.exists() and not OUTPUT.exists(), 'Fresh fixed config and output required; no overwrite/retry')
    require(pin(__file__) == own_before, 'Builder source changed while constructing config')
    require(not any(name in sys.modules for name in FORBIDDEN), 'Default authorizer crossed model/native boundary')
    return candidate, {'schema': 'full151105_dense_penalty_builder_declaration:1',
        'status': 'AUTHORIZED_CPU_CONFIG_TO_BE_WRITTEN' if authorize else 'DECLARED_NOT_AUTHORIZED_NOT_WRITTEN',
        'builder': own_before, 'config_path': str(CONFIG), 'output_path': str(OUTPUT), 'protocol': protocol_pin,
        'CPU_run_authorized': bool(authorize), 'source_pin_count': len(sources), 'CPU_ready': pin(ready_path),
        'CPU_contract_report': ready['cpu_report'], 'module': ready['source_pins']['consumer'],
        'GPU_executed': False, 'Torch_imported': False, 'Warp_imported': False, 'NPZ_loaded': 0, 'native_model_loads': 0,
        'state_conversions': 0, 'objective_constructs': 0, 'proxy_scores': 0,
        'scope': 'Current executed CPU proof and full transitive hashes only. No actual dense converter/score is launched by this builder.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready', required=True, type=Path)
    parser.add_argument('--authorize', action='store_true')
    args = parser.parse_args()
    candidate, declaration = build_candidate(args.ready, authorize=args.authorize)
    if args.authorize:
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG.open('x', encoding='utf-8') as stream:
            json.dump(candidate, stream, indent=2, ensure_ascii=False, allow_nan=False); stream.write('\n')
        declaration.update(status='AUTHORIZED_CPU_CONFIG_WRITTEN_NOT_EXECUTED', config=pin(CONFIG))
    print(json.dumps(declaration, indent=2, allow_nan=False))


if __name__ == '__main__': main()
