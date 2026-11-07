"""Independent saved-only all-row forensic review; never calls the converter.

Original q/qd bits and output-scaled8eps32 masks remain the historical rule.
F64 formula/operand recalc comparison uses 64eps64*max(1,abs(operands),abs
of either compared value), solely to compare equivalent arithmetic evaluation
orders. It is NOT a new scientific eligibility/error bound. No dense/proxy
qualification follows from successful forensic audit completion.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT/'src/full151_gradient/converter_forensic_saved_math_v1.py'
MATH_SHA = 'd27319855ee6e4eaa55694da176975c64e2476635ba4bb4dde278f1b5bd358e5'
CONSUMER_SHA = 'a5acf8583664b92285512ca62125c82530e73b1e923887ab3130d0042bdee52e'
PROTOCOL_SHA = '06c5b055403962d963e8b8cc6f34ceba69399bef46afe0fb8582eeacb60daa13'
BLOCKS = {'q': (0, 20), 'qd': (20, 40), 'position': (40, 43),
          'rotation': (43, 52), 'linear_velocity': (52, 55), 'angular_velocity': (55, 58)}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def pin(p):
    return {'path': str(Path(p).resolve(strict=True)), 'sha256': sha(p)}


def checked(p):
    path = Path(p['path']).resolve(strict=True)
    if sha(path) != p['sha256']:
        raise ValueError('Pin mismatch: '+str(path))
    return path


def same(a, b):
    return a['sha256'] == b['sha256'] and Path(a['path']).samefile(b['path'])


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def qmat(q, dtype):
    q = np.asarray(q, dtype=dtype)
    if not np.isfinite(q).all() or np.any(abs(np.linalg.norm(q.astype('f8'), axis=-1)-1) > 1e-5):
        raise ValueError('Invalid saved quaternion')
    unit = q/np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = unit.T
    r = np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(-1, 3, 3)
    return unit, r


def shadow(q, v, pp, R, handq, handv, oq, ov):
    """F64 saved-operands expression, explicit multiply+sum (no BLAS GEMM)."""
    q, v, pp, R = (a.astype('f8') for a in (q, v, pp, R))
    unit, obj = qmat(q[:, oq+3:oq+7], np.float64)
    inv = R.swapaxes(-1, -2); pos = q[:, oq:oq+3]; dp = pos-pp
    vel, body = v[:, ov:ov+3], v[:, ov+3:ov+6]
    mv = lambda A, b: (A*b[..., None, :]).sum(-1)
    mm = lambda A, B: (A[..., :, :, None]*B[..., None, :, :]).sum(-2)
    world = mv(obj, body); directq, directv = q[:, handq], v[:, handv]
    x = np.concatenate((directq, directv, mv(inv, dp), mm(inv, obj).reshape(-1, 9), mv(inv, vel), mv(inv, world)), -1)
    operands = np.concatenate((abs(directq), abs(directv), mv(abs(inv), abs(dp)),
        mm(abs(inv), abs(obj)).reshape(-1, 9), mv(abs(inv), abs(vel)), mv(abs(inv), mv(abs(obj), abs(body)))), -1)
    return {'F64_shadow_native_definition_NOT_ACCEPTED_STATE': x, 'raw_object_quaternion': q[:, oq+3:oq+7],
        'normalized_quaternion_F64': unit, 'object_rotation_F64': obj, 'inverse_actual_palm_rotation_F64': inv,
        'object_world_position_F64': pos, 'position_difference_F64': dp, 'world_linear_velocity_F64': vel,
        'body_angular_velocity_F64': body, 'world_angular_velocity_F64': world,
        'absolute_operand_sums_F64': operands, 'position_coordinate_operand_scale_F64': mv(abs(inv), abs(pos)+abs(pp)),
        'cancellation_ratios_DIAGNOSTIC_ONLY': operands/np.maximum(abs(x), 1e-30)}


def recalc_close(a, b, operand=None):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return False, None
    scale = np.maximum(1., np.maximum(abs(a), abs(b)))
    if operand is not None:
        scale = np.maximum(scale, abs(operand))
    error = abs(a-b)
    return bool(np.all(error <= 64*np.finfo('f8').eps*scale)), float(error.max(initial=0.))


def block_summary(d, labels):
    result = {}
    for name, (start, stop) in BLOCKS.items():
        e = abs(d['difference_F64'][:, start:stop])
        row, column = map(int, np.unravel_index(np.argmax(e), e.shape))
        result[name] = {'failed_elements': int((~d['old_flags'][:, start:stop]).sum()),
            'failed_rows': int((~d['old_flags'][:, start:stop]).any(-1).sum()),
            'maximum_absolute_difference': float(e[row, column]), 'max_error_row': row,
            'max_error_feature': start+column, 'row_identity': labels[row]}
    return result


def audit(config_path, run, out):
    out.mkdir(exist_ok=False)
    source = pin(__file__)
    (out/'executed_checker_source.py').write_bytes(Path(__file__).read_bytes())
    write(out/'equations_before_numeric_read.json', {'checker': source, 'math': pin(MATH),
        'actual_numeric_forensic_arrays_read': False, 'equations': __doc__})
    if sha(MATH) != MATH_SHA:
        raise ValueError('Independent frozen residual math differs')
    spec = importlib.util.spec_from_file_location('forensic_math', MATH)
    math = importlib.util.module_from_spec(spec); spec.loader.exec_module(math)
    checks, details = {}, {}
    def gate(k, v):
        checks[k] = bool(v)
    cfg, report = read(config_path), read(run/'report.json')
    proto = read(checked(cfg['protocol'])); parent = read(checked(proto['parent_protocol'])); inputs = parent['inputs']
    gate('root_config_exactpin_vs_independent_JSON_copy_pin', same(report['config'], pin(config_path))
         and cfg == read(run/'config_copy.json') and same(report['config_copy'], pin(run/'config_copy.json'))
         and Path(cfg['output']).samefile(run) and cfg['CPU_run_authorized'] is True and cfg['cpu_threads'] == 1
         and cfg['schema'] == 'full151105_converter_arithmetic_forensic_CPU_run:1')
    gate('forensic_protocol_source_inventory_copy_binding', cfg['protocol']['sha256'] == PROTOCOL_SHA
         and report['source_pins'] == cfg['source_pins'] == read(run/'source_inventory.json')
         and same(report['protocol'], cfg['protocol']) and proto == read(run/'protocol.json')
         and any(Path(p['path']).name == 'converter_forensic105_v1.py' and p['sha256'] == CONSUMER_SHA for p in cfg['source_pins'].values()))
    pins = [*cfg['source_pins'].values(), *inputs.values(), cfg['protocol'], proto['parent_protocol'], proto['failed_config'], proto['failed_record']]
    for p in pins:
        checked(p)
    gate('all_sources_and_preregistered_original_inputs_current', True)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:
        zipok = len(z.namelist()) == len(cfg['source_pins'])
        for i, p in enumerate(cfg['source_pins'].values()):
            zipok &= hashlib.sha256(z.read('%04d_%s' % (i, Path(p['path']).name))).hexdigest() == p['sha256']
    gate('all_execution_ZIP_bytes_exact', zipok)
    outputpins = read(run/'output_pins.json')
    for p in outputpins.values():
        checked(p)
    gate('all_saved_output_pins_exact', True)
    a = load(run/'raw_conversion_all2260.npz'); condition = load(run/'conditioning_secondary_all2260.npz')
    saved = {name: load(run/(name+'_old_rule.npz')) for name in ('endpoint452', 'leading160', 'solver_position1808', 'solver_rotation1808')}
    labels, alabels = read(run/'endpoint_identities.json'), read(run/'anchor_identities.json')
    samples, rewards = jsonl(checked(inputs['physical_samples'])), jsonl(checked(inputs['reward_snapshots']))
    rewardmap = {(r['path'], r['k'], r['branch_id']): r for r in rewards}; groups = {}
    for i, s in enumerate(samples):
        groups.setdefault((s['path'], s['phase'], s['branch_id']), []).append(i)
    identityok = len(samples) == 1808 and len(groups) == len(rewardmap) == 452 and len(labels) == 452 and len(alabels) == 160
    q = np.asarray([s['pre_qpos'] for s in samples], dtype='f4'); v = np.asarray([s['pre_qvel'] for s in samples], dtype='f4')
    lastids = np.asarray([ids[-1] for ids in groups.values()], dtype='i8')
    postq = np.asarray([samples[i]['post_qpos'] for i in lastids], dtype='f4'); postv = np.asarray([samples[i]['post_qvel'] for i in lastids], dtype='f4')
    pp = np.empty((1808, 3), dtype='f4'); R = np.empty((1808, 3, 3), dtype='f4'); gpp = []; gR = []; refs = []; elabels = []
    idx, arefs, anchorlabels = [], [], []; oldarrays = load(checked(inputs['physical_arrays']))
    for key, ids in groups.items():
        rw = rewardmap[key]
        p, r = np.asarray(rw['state20_reader_palm_pos'], dtype='f4')[0], np.asarray(rw['state20_reader_palm_rot'], dtype='f4')[0]
        pp[ids] = p; R[ids] = r; gpp.append(p); gR.append(r); refs.append(rw['physics'])
        elabels.append({'path': key[0], 'k': key[1], 'branch_id': key[2], 'last_sample_index': ids[-1]})
        identityok &= len(ids) == 4 and ids == list(range(ids[0], ids[0]+4)) and [samples[i]['substep'] for i in ids] == [1, 2, 3, 4]
        if key[2] is not None or key[1] in (0, 49, 99, 149):
            sid = ('A0', 'A2').index(key[0])*4+(0, 49, 99, 149).index(key[1])
            idx.append(ids[0]); arefs.append(oldarrays['snapshot_x'][sid])
            anchorlabels.append({'path': key[0], 'k': key[1], 'branch_id': key[2], 'pre_index': ids[0], 'snapshot_id': sid})
    allq, allv = np.concatenate((q, postq)), np.concatenate((v, postv))
    allpp, allR = np.concatenate((pp, np.asarray(gpp, dtype='f4'))), np.concatenate((R, np.asarray(gR, dtype='f4')))
    refs, arefs, idx = np.asarray(refs, dtype='f4'), np.asarray(arefs, dtype='f4'), np.asarray(idx, dtype='i8')
    gate('all452_endpoints_and160_anchor_identities_original_phase_complete', identityok and labels == elabels and alabels == anchorlabels
         and exact(idx, a['anchor_pre_indices']) and exact(lastids, a['endpoint_last_sample_indices']))
    gate('raw2260_operands_and452_refs_160anchors_match_original_saved_inputs',
         all(exact(a[k], value) for k, value in {'pre_qpos': q, 'pre_qvel': v, 'post20_qpos': postq,
             'post20_qvel': postv, 'post20_reference': refs, 'anchor_references': arefs,
             'palm_pos_all2260': allpp, 'palm_rot_all2260': allR}.items()))
    native = a['converted_native58']; oq, ov = int(a['object_qpos_start']), int(a['object_dof_start'])
    hq, hv = a['hand_qpos'], a['hand_dof']
    gate('all2260_native_q_qd_direct_named_mapping_and_finite_shape', native.shape == (2260, 58) and native.dtype == np.float32 and np.isfinite(native).all()
         and hq.shape == hv.shape == (20,) and hq.dtype.kind in 'iu' and hv.dtype.kind in 'iu'
         and len(set(hq.tolist())) == len(set(hv.tolist())) == 20 and 0 <= oq <= 20 and 0 <= ov <= 20
         and set(hq).isdisjoint(range(oq, oq+7)) and set(hv).isdisjoint(range(ov, ov+6))
         and exact(native[:, :20], allq[:, hq]) and exact(native[:, 20:40], allv[:, hv]))
    solver = np.asarray([s['object_solver_pose'] for s in samples], dtype='f4')
    gate('all1808_solver_prepose_raw_original_match', solver.shape == (1808, 7) and exact(solver, a['object_solver_pose']))
    original_rules = {}
    for name, count, x, y, state58 in (
        ('endpoint452', 452, native[1808:], refs, True), ('leading160', 160, native[idx], arefs, True),
        ('solver_position1808', 1808, solver[:, :3], q[:, oq:oq+3], False),
        ('solver_rotation1808', 1808, saved['solver_rotation1808']['actual'], saved['solver_rotation1808']['reference'], False)):
        d = math.error_arrays(x, y); flags = d['original8eps32_pass_mask'].copy()
        if state58:
            flags[:, :40] = d['FP32_exact_bit_mask'][:, :40]
        reconstructed = {'actual': x, 'reference': y, 'difference_F64': d['signed_error_F64'], 'old_bound_F64': d['bound_original8eps32'], 'old_flags': flags}
        gate(name+'_every_actual_reference_difference_bound_flag', len(x) == count and set(saved[name]) == set(reconstructed)
             and all(exact(saved[name][k], val) for k, val in reconstructed.items()))
        original_rules[name] = {**reconstructed, 'old_all_pass': bool(flags.all())}
        details[name] = {'rows': count, 'failed_elements': int((~flags).sum()), 'failed_rows': int((~flags).any(-1).sum()),
            'maximum_abs_difference': float(abs(d['signed_error_F64']).max()), 'original_rule_all_pass': bool(flags.all())}
    u32, R32 = qmat(allq[:, oq+3:oq+7], np.float32)
    u64, R64 = qmat(allq[:, oq+3:oq+7], np.float64)
    fp32_checks = {}
    for key, expected in [('normalized_quaternion_CPU_FP32', u32), ('object_rotation_CPU_FP32', R32)]:
        actual = condition[key]
        delta = abs(actual.astype('f8')-expected.astype('f8')); bound = 8*np.finfo('f4').eps*np.maximum(1., np.maximum(abs(actual), abs(expected)))
        fp32_checks[key] = bool(actual.shape == expected.shape and actual.dtype == np.float32 and np.isfinite(actual).all() and np.all(delta <= bound))
        details[key+'_NumPy_arithmetic_maxdiff'] = float(delta.max())
    gate('all2260_CPU_FP32_quaternion_normalization_rotation_arithmetic', all(fp32_checks.values()))
    gate('all1808_solver_rotation_algebra_from_raw_wxyz_no_phase_repair',
         np.all(abs(saved['solver_rotation1808']['actual']-qmat(solver[:, 3:], np.float32)[1].reshape(1808, 9)) <= 8*np.finfo('f4').eps)
         and exact(saved['solver_rotation1808']['reference'], condition['object_rotation_CPU_FP32'][:1808].reshape(1808, 9)))
    ownshadow = shadow(allq, allv, allpp, allR, hq, hv, oq, ov)
    ownshadow['converted_minus_F64_shadow'] = native.astype('f8')-ownshadow['F64_shadow_native_definition_NOT_ACCEPTED_STATE']
    ownshadow['endpoint_GPU_reference_minus_F64_shadow'] = refs.astype('f8')-ownshadow['F64_shadow_native_definition_NOT_ACCEPTED_STATE'][1808:]
    # Ratio is ill-conditioned near an exactly cancelled output. Verify its
    # declared arithmetic relation to the SAVED shadow/operand values rather
    # than demanding invariance to another valid F64 reduction ordering.
    ownshadow['cancellation_ratios_DIAGNOSTIC_ONLY'] = condition['absolute_operand_sums_F64']/np.maximum(abs(condition['F64_shadow_native_definition_NOT_ACCEPTED_STATE']), 1e-30)
    shadowok = set(condition) == set(ownshadow) | {'normalized_quaternion_CPU_FP32', 'object_rotation_CPU_FP32'}
    shadow_diffs = {}
    for key, expected in ownshadow.items():
        actual = condition[key]
        operand = ownshadow['absolute_operand_sums_F64'] if expected.shape == (2260, 58) else None
        if expected.shape == (452, 58):
            operand = ownshadow['absolute_operand_sums_F64'][1808:]
        close, maximum = recalc_close(actual, expected, operand)
        shadowok &= close and actual.dtype == np.float64
        shadow_diffs[key] = maximum
    gate('all2260_F64_shadow_normalizedquat_operands_coordinate_scale_ratios_secondary', shadowok)
    details['independent_F64_recalculation_max_abs_differences'] = shadow_diffs
    expected_summary = {'endpoints': block_summary(original_rules['endpoint452'], labels),
        'leadinganchors': block_summary(original_rules['leading160'], alabels),
        'new_replay_old_endpoint_rule_all_pass': original_rules['endpoint452']['old_all_pass'],
        'new_replay_old_leading_rule_all_pass': original_rules['leading160']['old_all_pass'],
        'solver_position_old_flags_all_pass': original_rules['solver_position1808']['old_all_pass'],
        'solver_rotation_old_flags_all_pass': original_rules['solver_rotation1808']['old_all_pass'],
        'original_failed_gate_unchanged': 'FAIL', 'replacement_gate_defined': False, 'dense_scoring_eligibility': False}
    gate('all_block_counts_worst_component_identity_and_oldmask_summaries', expected_summary == report['summary'] == read(run/'summary.json'))
    oldfailure = read(checked(proto['failed_record']))
    gate('historical_original8eps_failure_preserved_not_reinterpreted', same(report['old_failure_preserved'], proto['failed_record'])
         and oldfailure['error'] == "ValueError('all452post20 p/R/v/omega exceeds frozen8eps32 bound')"
         and oldfailure['counts']['objective_factory']['attempted'] == 0 and oldfailure['completed_scored_rows'] == 0)
    gate('source_instrumented1passive_MJB_1conversion2260_no_factory', report['counts'] == {k: {'attempted': 1, 'returned': 1} for k in ('passive_MJB', 'native_state_conversion')}
         and report['converted_rows'] == 2260 and report['endpoint_rows'] == 452 and report['leading_anchor_comparisons'] == 160
         and report['solver_pose_comparisons'] == 1808 and report['factory_calls'] == report['proxy_or_WM_Actor_forward'] == report['weights_AD_optimizer_Env_physics_nativeFK_collision_GPU'] == 0)
    gate('saved_readonly_invariants_bound_source_assertions_not_rawRNG', bool(report['invariants']) and all(v is True for v in report['invariants'].values()))
    gate('forensic_completion_never_dense_or_physics_qualification', report['schema'] == 'full151105_converter_arithmetic_forensic_result:1'
         and report['status'] == 'COMPLETED_ARITHMETIC_FORENSIC_NOT_DENSE_ELIGIBILITY'
         and report['physical_gradient_qualified'] is report['wholehiddenstate_qualified'] is report['dense_scoring_eligibility'] is False)
    for p in pins:
        checked(p)
    for p in outputpins.values():
        checked(p)
    gate('all_pins_and_frozen_independent_math_unchanged', pin(__file__) == source and sha(MATH) == MATH_SHA)
    gate('no_model_converter_geometry_framework_imports', not bool({'torch', 'warp', 'mujoco', 'scipy'} & set(sys.modules)))
    result = {'schema': 'full151105_converter_forensic_saved_independent:1',
        'status': 'PASS_SAVED_FORENSIC_REVIEW_NOT_ELIGIBILITY' if all(checks.values()) else 'FAILED_SAVED_FORENSIC_REVIEW',
        'passed': sum(checks.values()), 'total': len(checks), 'checks': checks, 'checker': source, 'math': pin(MATH),
        'config': pin(config_path), 'config_copy': pin(run/'config_copy.json'), 'producer_report': pin(run/'report.json'),
        'protocol': cfg['protocol'], 'run_dir': str(run.resolve()), 'source_pin_count': len(cfg['source_pins']),
        'all_row_diagnostics': details, 'summary': expected_summary,
        'original_gate_status': 'FAIL_RETAINED', 'dense_scoring_eligibility': False, 'physical_gradient_qualified': False,
        'limitations': ['All saved original-rule masks/component values reconstructed; no old gate relaxation or replacement acceptance.',
            'F64 shadow is an arithmetic descriptor, not an accepted converter or error theorem.',
            'Source native mm/mv are elementwise multiply+sum; this review does not attribute differences uniquely to TF32/hardware/cancellation.',
            'Passive MJB mapping/topology checks and gate-save execution order are source-bound; no new MJB/Reader/Env call.',
            'Invariants/counts are source-instrumented assertions; producer saved no independent raw RNG byte states here.',
            'Root config pin and reserialized JSON copy are independently hashed with strict parsed equality; their bytes need not match.'],
        'new_converter_calls': 0, 'model_loads': 0, 'native_queries': 0, 'physics_steps': 0, 'GPU_used': False}
    write(out/'report.json', result)
    print(json.dumps({'report': pin(out/'report.json'), 'passed': result['passed'], 'total': result['total'], 'summary': expected_summary}, indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        raise SystemExit(audit(args.config, args.run_dir, args.output))
    except Exception as e:
        if args.output.exists() and not (args.output/'failure.json').exists():
            write(args.output/'failure.json', {'checker': pin(__file__), 'error': repr(e), 'saved_only': True, 'original_outputs_unchanged': True})
        raise
