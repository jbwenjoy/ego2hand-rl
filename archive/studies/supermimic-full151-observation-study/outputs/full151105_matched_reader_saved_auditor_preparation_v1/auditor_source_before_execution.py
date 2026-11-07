"""Independent saved-only matched-native GPU/B1 Reader audit; no backend imports.

Source is frozen before actual replay arrays are opened. All original q/qd
exact bits and other18 old8eps32 predicates are mandatory. GPU-vs-CPU2260
differences are descriptive only, with device and batch changes confounded.
Core runtime.json pin is explicitly mutable by entry finalization; its raw
terminal identity is rechecked instead. All other saved core pins stay strict.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT/'src/full151_gradient/matched_reader_saved_math_v1.py'
MATH_SHA = 'a3301167f5711ddb4750068e0f530823f9e5ddb26edd5654d1854812242405e4'
CORE_SHA = '9ff9aef6956ca917068834f89524861d6b9ad8492e58cf6504ca404393a403b3'
PROTOCOL_SHA = 'fddcf970d7a8e44732fe93e6c83d1918fe017d4d5df159ae5c851a77e15fca6c'


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
    q = Path(p['path']).resolve(strict=True)
    if sha(q) != p['sha256']:
        raise ValueError('Pin mismatch: '+str(q))
    return q


def same(a, b):
    return a['sha256'] == b['sha256'] and Path(a['path']).samefile(b['path'])


def dump(p, v):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(v, f, indent=2, allow_nan=False)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode()+a.tobytes()).hexdigest()


def math_module():
    if sha(MATH) != MATH_SHA:
        raise ValueError('Independent frozen math changed')
    spec = importlib.util.spec_from_file_location('independent_matched_reader_math', MATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def rule_evidence(m, saved, actual, reference, state58):
    d = m.old_rule(actual, reference, state58)
    expected = {'actual': actual, 'reference': reference, **{k: d[k] for k in ('difference_F64', 'old_bound_F64', 'old_flags')}}
    ok = set(saved) == set(expected) and all(bits(saved[k], v) for k, v in expected.items())
    return ok, d


def raw_rng_same(before, after):
    keys = {'Python_pickle', 'NumPy_pickle', 'Torch_CPU', 'Torch_CUDA_device0'}
    return set(before) == set(after) == keys and all(a.dtype == np.uint8 and a.ndim == 1 and a.size > 0 for a in [*before.values(), *after.values()]) and all(bits(before[k], after[k]) for k in keys)


def audit(config_path, run, out):
    out.mkdir(exist_ok=False)
    auditor = pin(__file__)
    (out/'executed_auditor_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_frozen_before_actual_read.json', {'auditor': auditor, 'math': pin(MATH),
        'actual_replay_outputs_read': False, 'equations': __doc__})
    m = math_module(); checks = {}; details = {}
    def gate(k, v):
        checks[k] = bool(v)
    cfg, r = read(config_path), read(run/'report.json')
    proto = read(checked(cfg['protocol'])); inputs = proto['inputs']
    watchdir = run.with_name(run.name+'_watchdog')
    watch, ident, post = (read(watchdir/name) for name in ('state.json', 'engineering_identity.json', 'post_terminal_CIM.json'))
    rt, completion = read(run/'runtime.json'), read(run/'completion.json')
    gate('Root_config_source_protocol_bound_and_bytecopied_snapshot', cfg['schema'] == 'full151105_matched_native_GPU_reader_replay_run:1'
         and cfg['role'] == 'full151105_native_GPU_state_replay' and cfg['GPU_run_authorized'] is True
         and cfg['soft_worker_seconds'] == 90 and cfg['hard_process_seconds'] == 120
         and cfg['protocol']['sha256'] == PROTOCOL_SHA and same(r['protocol'], cfg['protocol'])
         and same(r['config'], pin(config_path)) and sha(run/'config.json') == sha(config_path)
         and Path(cfg['output']).samefile(run) and Path(watch['command'][4]).samefile(config_path))
    inventory = read(run/'source/inventory.json')
    gate('actual_v2_core_and_native_reader_source_closure', r['source_pins'] == cfg['source_pins'] == inventory
         and any(p['sha256'] == CORE_SHA and Path(p['path']).name == 'state_replay105_v2.py' for p in inventory.values())
         and all(any(same(p, q) for q in inventory.values()) for p in proto['source_pins'].values())
         and sha(watch['command'][2]) == rt['source_sha256'])
    sourcepins = [*inventory.values(), *inputs.values(), *proto['source_pins'].values(), cfg['protocol'], cfg['CPU_ready'], cfg['CPU_contract_report']]
    for p in sourcepins:
        checked(p)
    gate('all_current_source_input_CPU_ready_pins_exact', True)
    with zipfile.ZipFile(run/'source/execution_sources.zip') as z:
        zipok = len(z.namelist()) == len(inventory)
        for i, p in enumerate(inventory.values()):
            zipok &= hashlib.sha256(z.read('%04d_%s' % (i, Path(p['path']).name))).hexdigest() == p['sha256']
    gate('all_archived_source_bytes_exact', zipok)
    obs, py, command = watch['actual_process_OS_observation'], ident['selected_python_identity'], watch['command']
    stamp = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
    expected_parent = watch['supervisor_pid'] if watch['actual_pid'] == watch['launcher_pid'] else watch['launcher_pid']
    exitcode = 0 if r['all_original_gates_passed'] else 1
    gate('terminal_result_exit_scope_matches_and_no_timeout', watch['exit_code'] == exitcode and watch['timeout'] is False
         and rt['status'] == 'completed_full151_native_GPU_state_replay' and rt['exit_code'] == exitcode
         and read(run/'result.json') == r and not (run/'failure.json').exists() and not (run/'core_failure.json').exists())
    gate('OS12_source_recorded_checks_normal_or_gatefailure_scope', len(ident['checks']) == 12
         and all(v is True for k, v in ident['checks'].items() if k != 'watchdog_completed_exit0_no_timeout')
         and ident['checks']['watchdog_completed_exit0_no_timeout'] == (exitcode == 0)
         and ident['engineering_identity_qualified'] == (exitcode == 0))
    gate('raw_PID_parent_executable_command_SourceCreation_rebound', obs['status'] == 'exact_OS_identity_observed'
         and obs['pid'] == rt['pid'] == watch['actual_pid'] > 0 and obs['parent_pid'] == rt['parent_pid'] == expected_parent
         and os.path.samefile(obs['executable_path'], py['base_executable_raw'])
         and obs['command_line'] == '"'+py['base_executable_raw']+'" '+' '.join(command[1:])
         and rt['plan_sha256'] == sha(config_path) and Path(command[6]).samefile(run)
         and stamp(watch['Popen_prelaunch_utc']) <= stamp(obs['creation_date_utc']) <= stamp(rt['started_utc']) <= stamp(obs['observed_utc'])
         and all(watch['runtime_metadata'][k] == rt[k] for k in ('pid', 'parent_pid', 'started_utc', 'plan_sha256', 'source_sha256')))
    gate('postterminal_exact_owned_PIDs_absent_saved_CIM', post['status'] == 'NO_OWNED_PID_REMAINING' and post['actual_CIM_rows'] == []
         and post['queried_pids'] == sorted(set([watch['actual_pid'], watch['launcher_pid']])))
    raw = load(checked(inputs['raw_conversion_all2260']))
    q, v = np.concatenate((raw['pre_qpos'], raw['post20_qpos'])), np.concatenate((raw['pre_qvel'], raw['post20_qvel']))
    expected_upload = {'qpos': q, 'qvel': v, 'palm_pos': raw['palm_pos_all2260'], 'palm_rot': raw['palm_rot_all2260'],
        'hand_qpos': raw['hand_qpos'], 'hand_dof': raw['hand_dof'],
        'object_qpos': np.arange(int(raw['object_qpos_start']), int(raw['object_qpos_start'])+7, dtype='i8'),
        'object_dof': np.arange(int(raw['object_dof_start']), int(raw['object_dof_start'])+6, dtype='i8')}
    upload = load(run/'verified_upload_readback.npz'); receipt = read(run/'upload_receipt.json')
    gate('all2260_FP32_input_upload_readback_and_integer_mapping_bytes_exact', set(upload) == set(expected_upload)
         and all(bits(upload[k], v) for k, v in expected_upload.items())
         and receipt['before'] == receipt['GPU_readback'] == {k: array_hash(expected_upload[k]) for k in ('qpos', 'qvel', 'palm_pos', 'palm_rot')})
    gpu = load(run/'native_GPU_all2260_raw.npz'); x = gpu['native_GPU_state58_raw']
    gate('GPU_F32_all2260_raw_saved_complete_finite', x.shape == (2260, 58) and x.dtype == np.float32 and np.isfinite(x).all()
         and gpu['completed_mask'].dtype == bool and gpu['completed_mask'].shape == (2260,) and gpu['completed_mask'].all())
    evidence = read(run/'call_evidence_all2260.json')
    callok = len(evidence) == 2260
    for i, e in enumerate(evidence):
        expected = {'row': i, 'batch_size': 1, 'q_shape': [1, 27], 'qvel_shape': [1, 26], 'palm_pos_shape': [1, 3], 'palm_R_shape': [1, 3, 3],
            'input_dtype': 'torch.float32', 'input_device': 'cuda:0', 'index_dtype': 'torch.int64',
            'q_stride': [27, 1], 'qvel_stride': [26, 1], 'palm_pos_stride': [3, 1], 'palm_R_stride': [9, 3, 1],
            'q_storage_offset': i*27, 'output_shape': [1, 58], 'output_stride': [58, 1], 'output_dtype': 'torch.float32', 'output_device': 'cuda:0',
            'requires_grad': False, 'autocast_enabled_CUDA_inside': False, 'grad_enabled_inside': False}
        callok &= e == expected
    gate('every_native_call_B1_F32_same_device_index_noautocast_noAD', callok)
    gate('all2260_q_qd_exact_direct_named_index_mapping', bits(x[:, :20], q[:, raw['hand_qpos']]) and bits(x[:, 20:40], v[:, raw['hand_dof']]))
    rules = {}
    for name, count, actual, reference, state58 in (
        ('endpoint452_original_rule', 452, x[1808:], raw['post20_reference'], True),
        ('anchor160_original_rule', 160, x[raw['anchor_pre_indices']], raw['anchor_references'], True)):
        saved = load(run/(name+'.npz')); good, d = rule_evidence(m, saved, actual, reference, state58)
        gate(name+'_all_components_exact_recomputed', len(actual) == count and good)
        rules[name] = d
    for key in ('solver_position1808_old_rule', 'solver_rotation1808_old_rule'):
        original = load(checked(inputs[key])); saved = load(run/(key+'_rechecked.npz'))
        good, d = rule_evidence(m, saved, original['actual'], original['reference'], False)
        gate(key+'_all1808_inherited_raw_flags_recomputed', len(original['actual']) == 1808 and good and set(original) == set(saved)
             and all(bits(original[k], saved[k]) for k in original))
        rules[key] = d
    gate('all452_and160_original_identity_tables_preserved', read(run/'endpoint_identities.json') == read(checked(inputs['endpoint_identities']))
         and read(run/'anchor_identities.json') == read(checked(inputs['anchor_identities'])))
    allpass = all(d['all_passed'] for d in rules.values())
    gate('all_original_qualification_booleans_independently_recomputed',
         r['all_original_gates_passed'] == r['future_scoring_inputs_qualified'] == r['state_replay_qualified'] == allpass
         and r['endpoint_failed_elements'] == rules['endpoint452_original_rule']['failed_elements']
         and r['anchor_failed_elements'] == rules['anchor160_original_rule']['failed_elements'])
    packetpath = run/'prestate1808_qualified_for_future_scoring.npz'
    if allpass:
        packet = load(checked(r['qualified_prestate']))
        wanted = {'native_GPU_prestate58': x[:1808], 'completed_mask': gpu['completed_mask'][:1808], 'source_pre_row': np.arange(1808, dtype='i8'),
            'palm_pos': raw['palm_pos_all2260'][:1808], 'palm_rot': raw['palm_rot_all2260'][:1808],
            **{k: raw[k] for k in ('hand_qpos', 'hand_dof', 'object_qpos_start', 'object_dof_start')}}
        packetok = Path(r['qualified_prestate']['path']).samefile(packetpath) and set(packet) == set(wanted) and all(bits(packet[k], value) for k, value in wanted.items())
    else:
        packetok = r['qualified_prestate'] is None and not packetpath.exists()
    gate('qualified_prestate_packet_only_allgatepass_exact1808_input', packetok)
    validation = {'execution_complete': True, 'all_original_gates_passed': allpass, 'future_scoring_inputs_qualified': allpass, 'exit_code': exitcode}
    gate('entry_completion_result_validation_same_independent_scope', completion['schema'] == 'full151105_native_GPU_state_replay_worker_completion:1'
         and same(completion['config'], pin(config_path)) and completion['result_validation'] == validation
         and rt['all_original_gates_passed'] == rt['future_scoring_inputs_qualified'] == allpass)
    before, after = read(run/'backend_before.json'), read(run/'backend_after.json')
    flags = before['flags']
    gate('actual_backend_flags_before_after_held_source_settings', flags == after and flags['threads'] == 4 and flags['allow_tf32'] is True
         and flags['autocast_enabled_CUDA'] is False and before['device'] == 'cuda:0'
         and isinstance(flags['environment'], dict) and flags['environment']['CUDA_VISIBLE_DEVICES'] != '-1'
         and {'CUDA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'CUBLAS_WORKSPACE_CONFIG'} <= set(flags['environment']))
    rng_before, rng_after = load(run/'RNG_before.npz'), load(run/'RNG_after.npz')
    gate('raw_CPU_CUDA_and_serialized_host_RNG_bytes_unchanged', raw_rng_same(rng_before, rng_after))
    gate('producer_invariant_assertions_bound_not_replacing_raw_checks', bool(r['invariants']) and all(v is True for v in r['invariants'].values()))
    delta, descriptions = m.all2260_gpu_cpu_description(x, raw['converted_native58'])
    np.savez_compressed(out/'independent_GPU_minus_CPU_all2260_F64.npz', difference_F64=delta)
    details.update(GPU_vs_CPU_descriptive_by_unit=descriptions,
        original_rule_comparisons={k: {key: d[key] for key in ('all_passed', 'failed_elements', 'failed_rows')} for k, d in rules.items()},
        actual_flags=flags, serialized_raw_RNG_byte_equal=True)
    gate('all_flat_counts_native_calls2260_and_zero_prohibited_scope', r['counts'] == proto['expected_counts'] and len(evidence) == r['counts']['native_state_returned'] == 2260
         and all(type(v) is int for v in r['counts'].values()))
    gate('oldCPUFAIL_and_no_scoring_physics_hidden_qualification', r['original_CPU_failure_unchanged'] == 'FAIL'
         and r['dense_scoring_eligibility'] is r['proxy_scoring_authorized'] is r['physical_gradient_qualified'] is r['whole_hidden_state_qualified'] is False
         and r['status'] == ('COMPLETED_NATIVE_GPU_STATE_REPLAY_QUALIFIED' if allpass else 'COMPLETED_RAW_NOT_QUALIFIED'))
    corepins = read(run/'output_pins_core.json'); mutable = {}; immutable_count = 0
    for name, p in corepins.items():
        if name == 'runtime.json':
            mutable[name] = {'core_preterminal_pin': p, 'terminal_pin': pin(run/name), 'hash_equal': sha(run/name) == p['sha256'],
                'scope': 'Entry updates runtime after core output manifest; independently raw terminal identity checked above.'}
        else:
            checked(p); immutable_count += 1
    gate('all_immutable_core_output_pins_exact_runtime_explicitly_mutable', 'runtime.json' in corepins and immutable_count > 0)
    details.update(core_immutable_output_pin_count=immutable_count, mutable_core_pins=mutable,
        evidence_levels={'state_calls_B1_dtypes': 'Per-row source-recorded2260 records plus output/input bytes.',
            'RNG': 'Independent equality of four raw uint8 state/serialization vectors; no unpickling or RNG restoration.',
            'noMJB_model_factory_physics': 'Frozen reachable-source/count assertions; no independent process-wide call interceptor.',
            'solverpose': 'Inherited forensic23 raw actual/reference rechecked by NumPy, no new GPUquat/CPUstate/model.',
            'flags': 'Source-recorded current runtime observations, not retrospective flags or proof of each hardware instruction.'})
    for p in sourcepins:
        checked(p)
    gate('all_source_inputs_auditor_math_unchanged_postreview', pin(__file__) == auditor and sha(MATH) == MATH_SHA)
    gate('no_reloaded_nativeReader_Torch_model_GPU_or_geometry', not any(k in sys.modules for k in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'mjlab', 'dex_supermimic', 'scipy')))
    result = {'schema': 'full151105_matched_native_GPU_reader_saved_independent:1',
        'status': 'PASS_SAVED_NATIVE_REPLAY_REVIEW' if all(checks.values()) else 'FAILED_SAVED_NATIVE_REPLAY_REVIEW',
        'engineering_qualified': all(checks.values()), 'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'auditor': auditor, 'math': pin(MATH), 'config': pin(config_path), 'producer_report': pin(run/'report.json'), 'run_dir': str(run.resolve()),
        'source_pin_count': len(inventory), 'details': details, 'state_replay_qualified': allpass, 'future_scoring_inputs_qualified': allpass,
        'qualified_prestate': r['qualified_prestate'], 'dense_scoring_eligibility': False, 'physical_gradient_qualified': False, 'whole_hidden_state_qualified': False,
        'limitations': ['Original full58 q/qd bits+8eps32 predicates unchanged. New CUDA+B1 qualification never changes historical CPU+B2260 FAIL.',
            'All2260 GPU-vs-CPU differences retained separately by physical units, not evidence isolating device, shape, TF32 or unique mechanism.',
            'Native mm/mv use elementwise multiplication+sum, not GEMM. Historical flags/hidden graph state not inferred.',
            'Even allpass grants only specifically matched-state replay inputs for a separately authorized future scoring condition, not scoring/physics/geometry/gradient/task qualification.',
            'Repeated saved paths/branches/components are not independent observations or method comparisons.'],
        'new_native_reader_calls': 0, 'model_loads': 0, 'Env_created': False, 'physics_steps': 0, 'GPU_used': False}
    dump(out/'report.json', result)
    print(json.dumps({'report': pin(out/'report.json'), 'passed': result['passed'], 'total': result['total'], 'state_replay_qualified': allpass}, indent=2))
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
            dump(args.output/'failure.json', {'error': repr(e), 'auditor': pin(__file__), 'saved_only': True, 'old_source_outputs_unchanged': True})
        raise
