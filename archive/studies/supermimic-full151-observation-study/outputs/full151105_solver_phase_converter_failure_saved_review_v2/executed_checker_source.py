"""Saved-only failed converter audit. Stdlib; no data conversion or model load."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

CONSUMER_SHA = '00e1aea5163c7f6d07ea75755e076d808f171d0a766acbf6a683db8a3c00b22c'
CONFIG_SHA = '160c2b8d27012631ffab821f04cdd8ae115c84765b5051daa0289926192020a3'
PROTOCOL_SHA = '1ec18db963bc2931920bedce3f2bf43e345544798ee4ac1e0f856ac119cea13c'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {'path': str(Path(p).resolve(strict=True)), 'sha256': sha(p)}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def checked(p):
    path = Path(p['path']).resolve(strict=True)
    if sha(path) != p['sha256']:
        raise ValueError('Pin mismatch: '+str(path))
    return path


def write(p, x):
    with p.open('x', encoding='utf-8') as f:
        json.dump(x, f, indent=2, allow_nan=False)


def main(config_path, run, out):
    out.mkdir(exist_ok=False)
    source = pin(__file__)
    (out/'executed_checker_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'source_before_execution.json', {'checker': source, 'UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'EQUATIONS_FROZEN_BEFORE_FAILED_CASE_REVIEW', 'expected': 'Original8eps32 FAIL retained; no missing arrays reconstructed'})
    checks = {}
    def gate(k, v):
        checks[k] = bool(v)
    cfg = read(config_path); failed = read(run/'failure.json'); proto = read(run/'protocol.json')
    gate('exact_root_config_and_savedcopy_bound', sha(config_path) == CONFIG_SHA  # Saved config is independently pinned JSON reserialization.
         and cfg == read(run/'config.json') and failed['config']['sha256'] == CONFIG_SHA
         and Path(failed['config']['path']).samefile(config_path) and Path(cfg['output']).samefile(run))
    gate('original_prereg2_unchanged_no_gate_relaxation', cfg['protocol']['sha256'] == PROTOCOL_SHA
         and sha(checked(cfg['protocol'])) == sha(run/'protocol.json') == PROTOCOL_SHA
         and proto['revision'] == 2 and proto['CPU_run_authorized'] is False)
    inventory = read(run/'source_inventory.json')
    gate('all2478_source_inventory_entries_match_config', inventory == cfg['source_pins'] and len(inventory) == 2478)
    sourcebytes = None
    with zipfile.ZipFile(run/'execution_sources.zip') as z:
        zip_ok = len(z.namelist()) == len(inventory)
        for i, p in enumerate(inventory.values()):
            path = checked(p)
            b = z.read('%04d_%s' % (i, path.name))
            zip_ok &= hashlib.sha256(b).hexdigest() == p['sha256']
            if path.name == 'penalty_observability105_v1.py':
                if p['sha256'] != CONSUMER_SHA:
                    raise ValueError('Wrong immutable consumer')
                sourcebytes = b
    gate('all2478_current_source_and_archived_bytes_exact', zip_ok and sourcebytes is not None)
    for p in proto['inputs'].values():
        checked(p)
    gate('all_preregistered_input_pins_still_exact', True)
    gate('original_failure_status_phase_error_retained',
         failed['schema'] == 'full151105_same_solver_phase_penalty_CPU_failure:1'
         and failed['status'] == 'FAILED_PARTIAL_PRESERVED_NO_RETRY'
         and failed['phase'] == 'passive_constants_and_native_converter'
         and failed['error'] == "ValueError('all452post20 p/R/v/omega exceeds frozen8eps32 bound')")
    expected = {'passive_mapping_MJB': 1, 'native_state_conversion': 1, 'objective_factory': 0, 'objective_forward': 0}
    gate('reported_attempt_return1MJB1conversion0factory0forward', failed['counts'] == {k: {'attempted': v, 'returned': v} for k, v in expected.items()})
    gate('zero_scored_rows_no_invariants_or_newphysics_assertions', failed['completed_scored_rows'] == 0
         and failed['remaining_rows'] == 'NOT_RUN' and failed['invariants_available'] == {} and failed['new_physics_GPU'] == 0)
    tree = ast.parse(sourcebytes)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    src = sourcebytes.decode()
    validate = ast.get_source_segment(src, functions['validate_x58'])
    arithmetic = ast.get_source_segment(src, functions['arithmetic_check'])
    run_source = ast.get_source_segment(src, functions['run_analysis'])
    gate('q_qd_exact_guard_precedes_arithmetic_failure_source_path',
         validate.index('not bitexact') < validate.index('return arithmetic_check')
         and 'a[:,:40].tobytes()==b[:,:40].tobytes()' in validate)
    gate('unchanged_output_scaled8eps32_equation_source',
         '8*2**-23*np.maximum(1.,np.maximum(abs(a),abs(b)))' in arithmetic
         and 'np.all(abs(a-b)<=bound)' in arithmetic)
    gate('all2260_return_precedes_first452gate_and_all_output_array_saves',
         run_source.index('converted=call(') < run_source.index("'all452post20'")
         < run_source.index("'converted_state58.npz'") < run_source.index("phase='one_actual_P_ONLY_factory'"))
    names = {p.name for p in run.iterdir() if p.is_file()}
    expected_files = {'config.json', 'protocol.json', 'source_inventory.json', 'execution_sources.zip', 'failure.json'}
    gate('exact_failed_case_file_inventory_no_hidden_dense_results', names == expected_files)
    gate('no_converter_arrays_gates_chunks_or_score_artifacts_saved',
         not any((run/name).exists() for name in ('converted_state58.npz', 'converter_qualification.json', 'native_depth_before_proxy.json',
             'proxy_sample_buffers.npz', 'factory_receipt.json', 'all1808_proxy_scores.npz', 'rows_all1808.json', 'statistics.json', 'report.json', 'output_pins.json'))
         and not any(run.glob('chunk_*.npz')))
    gate('no_backend_native_or_model_imported_by_checker', not bool({'torch', 'warp', 'mujoco', 'numpy', 'scipy'} & set(sys.modules)))
    gate('checker_and_original_config_unchanged', pin(__file__) == source and sha(config_path) == CONFIG_SHA)
    report = {'schema': 'full151105_solver_phase_converter_failure_saved_independent:1',
        'status': 'PASS_SAVED_FAILURE_REVIEW_ORIGINAL_CONVERTER_FAIL_RETAINED' if all(checks.values()) else 'FAILED_SAVED_FAILURE_REVIEW',
        'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'checker': source, 'config': pin(config_path), 'saved_config_copy': pin(run/'config.json'),
        'saved_config_relationship': 'Strict parsed JSON equality, independently pinned reserialization; no claim of identical bytes.',
        'failure': pin(run/'failure.json'),
        'protocol': cfg['protocol'], 'consumer_sha256': CONSUMER_SHA, 'source_pin_count': len(inventory),
        'source_ZIP': pin(run/'execution_sources.zip'), 'reported_counts': failed['counts'],
        'original_converter_qualified': False, 'proxy_scored_rows': 0, 'objective_factory_calls': 0, 'objective_forward_calls': 0,
        'physical_gradient_qualified': False, 'whole_hidden_state_qualified': False,
        'evidence_levels': {'q_qd_guard_passed': 'Source-mediated path inference from thrown arithmetic error, not saved-array numerical recomputation.',
            'converted2260_returned': 'Instrumented1 native conversion return after source-enforced2260 batch; dense arrays were not saved.',
            'per_component_error_magnitude': 'UNKNOWN_NOT_SAVED', '160_anchor_gate': 'NOT_REACHED', '1808_solver_pose_gate': 'NOT_REACHED',
            'RNG_weights_invariance': 'UNAVAILABLE_FACTORY_NOT_REACHED'},
        'next_forensic_contract': ['New condition/output/source only; never rerun or reinterpret this FAIL as pass.',
            'Save all2260 converted58, raw qpos/qvel and authoritative Reader palms before diagnostics; always return before factory.',
            'Save all452 post58 original reference, all160 pre-anchor identities/reference, all1808 solverpreposes.',
            'Preserve original q/qd exact and8eps32 masks/abs errors/bounds by component and phase, without output-row dropping.',
            'Independent NumPy transforms may diagnose arithmetic/cancellation but do not replace native results or relax original gate.'],
        'model_loads_by_auditor': 0, 'native_queries': 0, 'new_conversions': 0, 'physics_steps': 0, 'GPU_used': False}
    write(out/'report.json', report)
    print(json.dumps({'report': pin(out/'report.json'), 'passed': report['passed'], 'total': report['total'], 'original_case': 'FAIL_RETAINED'}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    try:
        raise SystemExit(main(a.config, a.run_dir, a.output))
    except Exception as e:
        if a.output.exists() and not (a.output/'failure.json').exists():
            write(a.output/'failure.json', {'error': repr(e), 'checker': pin(__file__), 'saved_only': True, 'old_outputs_unchanged': True})
        raise
