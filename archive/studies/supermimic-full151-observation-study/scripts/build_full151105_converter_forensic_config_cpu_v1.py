"""Create a fresh arithmetic-only forensic config after explicit authorization.

Default: JSON/hash checks and declaration only. This builder never opens NPZ,
loads MJB/models, converts states, constructs objectives, or invokes run().
Executed dense builder/config/failed output remain separate and unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'protocols/full151105-converter-arithmetic-forensic-cpu-v1.json'
PROTOCOL_SHA = '06c5b055403962d963e8b8cc6f34ceba69399bef46afe0fb8582eeacb60daa13'
READY = ROOT / 'outputs/full151105_converter_forensic_preparation_CPU_v1/READY.json'
READY_SHA = '04ed31cea7ccebdfeaced27a703a5f8e2cfa3c8eacd982953ee6be6b829f6af7'
CONSUMER = ROOT / 'src/full151_gradient/converter_forensic105_v1.py'
CONSUMER_SHA = 'a5acf8583664b92285512ca62125c82530e73b1e923887ab3130d0042bdee52e'
CONFIG = ROOT / 'configs/full151105-converter-forensic-CPU-v1.json'
OUTPUT = ROOT / 'outputs/full151105_converter_forensic_CPU_v1'
BUILD_REVIEW = ROOT / 'outputs/full151105_converter_forensic_builder_source_only_v1/READY_source_only.json'
FREEZE = ROOT / 'outputs/full151105_converter_arithmetic_source_review_v1/FREEZE.json'
FREEZE_SHA = '2e1b9af72d694fc11772395460869c5eb26af7b65965b1d223b841a2d83f689e'
FAILURE_REVIEW = ROOT / 'outputs/full151105_solver_phase_converter_failure_saved_review_v2/report.json'
FAILURE_REVIEW_SHA = 'c1ec07d1b2a54539722e312d8ad239d10f22a5d7faa58304d01def006b8996be'
MATH_READY = ROOT / 'outputs/full151105_converter_forensic_saved_math_preparation_v1/READY.json'
MATH_READY_SHA = '3b018385bb00534f03e411985cd20a13681545b6375494c2a49fa664bed11271'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(path):
    path = Path(path)
    require(path.is_file(), 'Missing file: ' + str(path))
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def checked(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'}, 'Exact path/SHA pin required')
    require(type(value['path']) is str and Path(value['path']).is_absolute(), 'Absolute pin required')
    require(type(value['sha256']) is str and len(value['sha256']) == 64, 'SHA256 required')
    require(pin(value['path'])['sha256'] == value['sha256'], 'Source/evidence drift: ' + value['path'])
    return Path(value['path'])


def same_pin(left, right):
    a, b = checked(left), checked(right)
    return left['sha256'] == right['sha256'] and a.samefile(b)


def positive_report(report, schema, expected):
    require(report.get('schema') == schema, 'Wrong CPU/evidence schema')
    checks = report.get('checks')
    require(type(checks) is dict and len(checks) == expected and all(type(x) is bool and x for x in checks.values()), 'Incomplete/failed checks')
    require(type(report.get('passed')) is int and type(report.get('total')) is int and report['passed'] == report['total'] == expected, 'Wrong check counts')


def add_sources(target, prefix, sources):
    require(type(sources) is dict and bool(sources), 'Missing flat source map')
    for key, value in sources.items():
        require(type(key) is str, 'Source alias must be a string')
        checked(value)
        alias = prefix + key
        require(alias not in target, 'Source alias collision: ' + alias)
        target[alias] = value  # Preserve original recorded path spelling.


def build(ready_path, authorized=False):
    require('torch' not in sys.modules and 'warp' not in sys.modules and 'mujoco' not in sys.modules, 'Fresh metadata process required')
    require(type(authorized) is bool, 'Explicit authorization bool required')
    require(same_pin(pin(ready_path), {'path': str(READY), 'sha256': READY_SHA}), 'Use exact executed forensic READY')
    require(pin(PROTOCOL)['sha256'] == PROTOCOL_SHA and pin(CONSUMER)['sha256'] == CONSUMER_SHA, 'Frozen forensic source/protocol drift')
    ready, protocol = read(ready_path), read(PROTOCOL)
    require(ready.get('schema') == 'full151105_converter_forensic_CPU_READY:1' and ready.get('status') == 'PASS_ARITHMETIC_ONLY_SOURCE_AND_SMALL_FIXTURES', 'Wrong READY scope')
    require(ready.get('actual_forensic_authorized') is False and ready.get('objective_or_scoring_authorized') is False, 'READY is not scientific authorization')
    require(ready.get('old_failure_not_overwritten') is True, 'Original failure must remain')
    require(same_pin(ready['protocol'], pin(PROTOCOL)), 'READY/protocol mismatch')
    proof = read(checked(ready['cpu_report']))
    positive_report(proof, 'full151105_converter_forensic_CPU_proof:1', 38)
    require(ready['cpu_checks'] == {'passed': 38, 'total': 38}, 'READY counts mismatch')
    require(all(type(proof[k]) is int and proof[k] == 0 for k in ('actual_physical_state_conversions', 'actual_MJB_or_objective_or_model_loads', 'actual_dense_data_reads', 'Env_physics_GPU_AD')), 'Fixture scope elevated')
    require(proof['old_gate_relaxed'] is False and proof['new_dense_eligibility'] is False, 'Fixture changed eligibility')
    require(set(ready['source_pins']) == set(proof['source_pins']) == {'consumer', 'CLI', 'checker'}, 'Wrong executed source keyset')
    for key, value in ready['source_pins'].items():
        require(same_pin(value, proof['source_pins'][key]), 'CPU proof/source mismatch: ' + key)
    require(same_pin(ready['source_pins']['consumer'], pin(CONSUMER)), 'Wrong executed consumer')
    require(protocol.get('schema') == 'full151105_converter_arithmetic_forensic_protocol:1', 'Wrong forensic protocol schema')
    require(all(protocol.get(k) is False for k in ('CPU_run_authorized', 'GPU_run_authorized', 'proxy_scoring_authorized')), 'Protocol declaration is not authorization')
    require(protocol['fixed_budget']['native_state_calls'] == 1 and protocol['fixed_budget']['native_state_rows'] == 2260 and protocol['fixed_budget']['factory_calls'] == 0 and protocol['fixed_budget']['objective_WM_Actor_forward'] == 0, 'Forensic budget drift')

    # The import occurs only after its actual immutable CPU source proof is joined.
    # Its top level and required_sources() are metadata-only; never call run().
    sys.path.insert(0, str(ROOT / 'src'))
    consumer = importlib.import_module('full151_gradient.converter_forensic105_v1')
    require(same_pin(pin(consumer.__file__), pin(CONSUMER)), 'Unexpected imported consumer')
    require(consumer.SCHEMA == ready['api']['schema'] and consumer.ROLE == ready['api']['role'] and sorted(consumer.KEYS) == ready['api']['keys'], 'Consumer/READY ABI drift')
    require(same_pin(pin(ready['api']['entry']), ready['source_pins']['CLI']) and same_pin(pin(ready['api']['module']), pin(CONSUMER)), 'READY entry/module mismatch')
    sources = {}
    add_sources(sources, '', consumer.required_sources(protocol))
    add_sources(sources, 'tested__', ready['source_pins'])
    sources['forensic_cpu_ready'] = pin(ready_path)
    sources['forensic_cpu_report'] = ready['cpu_report']
    sources['forensic_protocol'] = pin(PROTOCOL)

    require(pin(FREEZE)['sha256'] == FREEZE_SHA, 'Pre-error interpretation freeze drift')
    freeze = read(FREEZE)
    require(freeze['schema'] == 'full151105_converter_arithmetic_interpretation_freeze:1' and freeze['new_forensic_raw_arrays_or_errors_read'] is False and freeze['operative_acceptance_rule'] is False and freeze['old_gate_changed'] is False, 'Wrong conditional interpretation scope')
    require(all(type(x) is bool and x for x in freeze['source_pin_checks'].values()) and freeze['passed'] == freeze['total'] == 10, 'Interpretation source checks incomplete')
    add_sources(sources, 'conditional_context__', freeze['artifact_pins'])
    add_sources(sources, 'conditional_source__', freeze['source_pins'])
    sources['conditional_interpretation_freeze'] = pin(FREEZE)

    require(pin(FAILURE_REVIEW)['sha256'] == FAILURE_REVIEW_SHA, 'Original failure review drift')
    failure = read(FAILURE_REVIEW)
    positive_report(failure, 'full151105_solver_phase_converter_failure_saved_independent:1', 15)
    require(failure['original_converter_qualified'] is False and failure['proxy_scored_rows'] == 0 and failure['objective_factory_calls'] == 0, 'Old FAIL must remain FAIL')
    require(same_pin(failure['config'], protocol['failed_config']) and same_pin(failure['failure'], protocol['failed_record']), 'Wrong old failed case')
    sources['old_failure_saved_review15'] = pin(FAILURE_REVIEW)
    sources['old_failure_saved_checker'] = failure['checker']
    checked(failure['checker'])

    require(pin(MATH_READY)['sha256'] == MATH_READY_SHA, 'Independent math READY drift')
    math_ready = read(MATH_READY)
    math_proof = read(checked(math_ready['cpu_report']))
    positive_report(math_proof, 'full151105_converter_forensic_saved_math_preparation:1', 6)
    require(math_ready['cpu_checks'] == {'passed': 6, 'total': 6} and math_proof['synthetic_only'] is True and math_proof['actual_new_forensic_outputs_read'] is False, 'Wrong independent math scope')
    require(math_proof['actual_converter_calls'] == math_proof['models'] == 0 and math_proof['GPU'] is False, 'Independent math executed science')
    add_sources(sources, 'independent_math6__', math_ready['source_pins'])
    for key, value in math_ready['source_pins'].items():
        require(same_pin(value, math_proof['source_pins'][key]), 'Math proof/source mismatch')
    sources['independent_math6_ready'] = pin(MATH_READY)
    sources['independent_math6_proof'] = math_ready['cpu_report']

    build_review = read(BUILD_REVIEW)
    require(build_review['schema'] == 'full151105_converter_forensic_builder_source_only:1' and build_review['AST_parse_only'] is True and build_review['builder_executed'] is False, 'Wrong builder AST-only review')
    require(same_pin(build_review['source_pins']['builder'], pin(__file__)), 'Builder source changed since AST review')
    sources['forensic_config_builder'] = pin(__file__)
    sources['forensic_builder_AST_source_review'] = pin(BUILD_REVIEW)
    require(not CONFIG.exists() and not OUTPUT.exists(), 'New config/output must both be fresh; no rerun/replacement')
    config = {'schema': consumer.SCHEMA, 'role': consumer.ROLE, 'protocol': pin(PROTOCOL), 'output': str(OUTPUT), 'CPU_run_authorized': authorized, 'cpu_threads': 1, 'source_pins': sources, 'cpu_ready': pin(ready_path)}
    require(set(config) == consumer.KEYS, 'Actual forensic eight-key ABI mismatch')
    consumer.header(config)
    require('torch' not in sys.modules and 'warp' not in sys.modules and 'mujoco' not in sys.modules, 'Metadata import reached a backend')
    for value in sources.values():
        checked(value)
    if authorized:
        with CONFIG.open('x', encoding='utf-8', newline='\n') as out:
            json.dump(config, out, indent=2)
            out.write('\n')
    return {'schema': 'full151105_converter_forensic_builder_declaration:1', 'status': 'CREATED_EXPLICITLY_AUTHORIZED_ARITHMETIC_ONLY_CONFIG' if authorized else 'DECLARED_NO_CONFIG_WRITTEN', 'config': pin(CONFIG) if authorized else {'path': str(CONFIG), 'written': False}, 'output': str(OUTPUT), 'protocol': pin(PROTOCOL), 'cpu_ready': pin(ready_path), 'source_pin_count': len(sources), 'CPU_run_authorized': authorized, 'Torch_imported': False, 'Warp_imported': False, 'NPZ_opened': 0, 'MJB_loads': 0, 'state_conversions': 0, 'objective_factory': 0, 'model_or_proxy_forwards': 0, 'GPU_physics_Env': 0, 'new_eligibility': False, 'old_failure_changed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ready', type=Path, required=True)
    parser.add_argument('--authorize', action='store_true', help='Explicitly create one fresh arithmetic-only CPU config; does not run it.')
    args = parser.parse_args()
    print(json.dumps(build(args.ready, args.authorize), indent=2))


if __name__ == '__main__':
    main()
