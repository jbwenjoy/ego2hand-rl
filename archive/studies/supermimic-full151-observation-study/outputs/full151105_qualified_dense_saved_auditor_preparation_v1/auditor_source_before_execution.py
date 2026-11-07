"""Saved-only qualified-GPU-input dense penalty audit, NumPy/stdlib only.

No producer module, nativeReader, Torch, model, MJB, FK, or geometry is loaded.
Frozen independent P/HO and F64 alignment math are used on captured arrays.
RNG/buffer before-after booleans are source-backed, not raw RNG replay proof.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CORE_SHA = 'c4892a933860c0415e4a229c181a94f88755e6c529bd27c5f79bdc56be8da47a'
PROTOCOL_SHA = 'c9a746f929d5943a03067c67d5b28f23e09b093084c2cde07cfae05f5db60028'
DEPENDENCIES = {
    'proxy_math': (ROOT/'src/full151_gradient/observability_saved_math_v1.py', '08c9d1d5d4f7c9d6866cdb51292a904ae7eb167cbb2bf5c093c70648622ff7a1'),
    'rows': (ROOT/'src/full151_gradient/dense_saved_rows_v1.py', 'e16645e366157abec974fd33de045a0f1923ae32ec445b27cef2412eac03388d'),
    'alignment': (ROOT/'src/full151_gradient/dense_packet_alignment_saved_math_v1.py', 'f5cb37ff938fcf5b4742e1f563c0a2493675f352426d4c55bec685dd08683335'),
    'old_rule': (ROOT/'src/full151_gradient/matched_reader_saved_math_v1.py', 'a3301167f5711ddb4750068e0f530823f9e5ddb26edd5654d1854812242405e4')}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def pin(path):
    return {'path': str(Path(path).resolve(strict=True)), 'sha256': sha(path)}


def checked(p):
    path = Path(p['path']).resolve(strict=True)
    if sha(path) != p['sha256']:
        raise ValueError('Pin mismatch: '+str(path))
    return path


def same(a, b):
    return a['sha256'] == b['sha256'] and Path(a['path']).samefile(b['path'])


def dump(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, indent=2, allow_nan=False)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def independent_modules():
    modules = {}
    for name, (path, digest) in DEPENDENCIES.items():
        if sha(path) != digest:
            raise ValueError('Frozen independent dependency changed: '+name)
        spec = importlib.util.spec_from_file_location('independent_'+name, path)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); modules[name] = m
    return modules


def CPU_ready_metadata_passed(ready, proof):
    checks = proof.get('checks')
    return (ready.get('schema') == 'full151105_qualified_dense_penalty_CPU_contract_READY:1'
        and proof.get('schema') == 'full151105_qualified_dense_penalty_CPU_contract_proof:1'
        and ready.get('status') == proof.get('status') == 'PASS_SYNTHETIC_QUALIFIED_PACKET_ALIGNMENT_AND_HOOK_ONLY'
        and type(checks) is dict and len(checks) == 72 and all(v is True for v in checks.values())
        and type(proof.get('passed')) is int and type(proof.get('total')) is int
        and proof['passed'] == proof['total'] == 72
        and ready.get('cpu_checks') == {'passed': 72, 'total': 72})


def audit(config_path, run, out):
    out.mkdir(exist_ok=False)
    auditor = pin(__file__)
    (out/'executed_auditor_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_frozen_before_numeric_read.json', {'auditor': auditor,
        'independent_dependencies': {k: pin(p) for k, (p, _) in DEPENDENCIES.items()},
        'actual_dense_results_read': False, 'scope': __doc__})
    m = independent_modules(); checks = {}; details = {}
    def gate(k, v):
        checks[k] = bool(v)
    cfg, report = read(config_path), read(run/'report.json')
    proto = read(checked(cfg['protocol'])); inputs = proto['inputs']
    gate('actual_Root_config_exact_pin_and_reserialized_copy_strict_JSON', same(report['config'], pin(config_path))
         and cfg == read(run/'config.json') and Path(cfg['output']).samefile(run)
         and cfg['schema'] == 'full151105_solver_phase_penalty_qualified_GPU_CPU_run:1'
         and cfg['CPU_run_authorized'] is True and cfg['cpu_threads'] == 1)
    gate('protocol_rev4_frozen_core_inventory_bound', cfg['protocol']['sha256'] == PROTOCOL_SHA and proto['revision'] == 4
         and same(report['protocol'], cfg['protocol']) and proto == read(run/'protocol.json')
         and cfg['source_pins'] == report['source_pins'] == read(run/'source_inventory.json')
         and any(p['sha256'] == EXPECTED_CORE_SHA for p in cfg['source_pins'].values()))
    pins = [*cfg['source_pins'].values(), *inputs.values(), cfg['protocol'], cfg['cpu_ready']]
    for p in pins:
        checked(p)
    gate('all_current_sources_and_preregistered_inputs_exact', True)
    ready = read(checked(cfg['cpu_ready'])); proof = read(checked(ready['cpu_report']))
    gate('actual_frozen_consumer_CPU72_metadata_source_contract', CPU_ready_metadata_passed(ready, proof)
         and ready['source_pins'] == proof['source_pins']
         and ready['source_pins']['consumer']['sha256'] == EXPECTED_CORE_SHA
         and same(ready['protocol'], cfg['protocol'])
         and all(checked(p).exists() for p in ready['source_pins'].values()))
    with zipfile.ZipFile(run/'execution_sources.zip') as z:
        zipok = len(z.namelist()) == len(cfg['source_pins'])
        for i, p in enumerate(cfg['source_pins'].values()):
            zipok &= hashlib.sha256(z.read('%04d_%s' % (i, Path(p['path']).name))).hexdigest() == p['sha256']
    gate('all_archived_source_bytes_exact', zipok)
    outputpins = read(run/'output_pins.json')
    for p in outputpins.values():
        checked(p)
    gate('all_original_immutable_dense_output_pins_exact', True)
    replay_review = read(checked(inputs['GPU_replay_saved_independent']))
    replay = read(checked(inputs['GPU_replay_report']))
    replayrun = checked(inputs['GPU_replay_report']).parent
    gate('prior_independent29_full58_replay_proof_exact_not_producer_bool', replay_review['schema'] == 'full151105_matched_native_GPU_reader_saved_independent:1'
         and replay_review['status'] == 'PASS_SAVED_NATIVE_REPLAY_REVIEW' and replay_review['engineering_qualified'] is True
         and replay_review['passed'] == replay_review['total'] == 29 and all(v is True for v in replay_review['checks'].values())
         and same(replay_review['config'], inputs['GPU_replay_config']) and same(replay_review['producer_report'], inputs['GPU_replay_report'])
         and same(replay_review['qualified_prestate'], inputs['qualified_GPU_prestate1808'])
         and Path(replay_review['run_dir']).samefile(replayrun) and replay_review['state_replay_qualified'] is replay_review['future_scoring_inputs_qualified'] is True
         and replay_review['dense_scoring_eligibility'] is False)
    gate('saved_qualified_replay_join_record_exact', read(run/'qualified_replay_provenance.json') == report['qualified_replay_provenance']
         and same(report['qualified_replay_provenance']['independent_saved_review'], inputs['GPU_replay_saved_independent'])
         and same(report['qualified_replay_provenance']['producer_report'], inputs['GPU_replay_report'])
         and same(report['qualified_replay_provenance']['qualified_prestate'], inputs['qualified_GPU_prestate1808']))
    corepins = read(checked(inputs['GPU_replay_core_output_manifest']))
    exception = replay_review['details']['mutable_core_pins']['runtime.json']
    immutable = 0
    for name, p in corepins.items():
        if name == 'runtime.json':
            continue
        checked(p); immutable += 1
    runtime = read(checked(inputs['GPU_replay_terminal_runtime']))
    identity = read(checked(inputs['GPU_replay_identity']))
    watch = read(checked(inputs['GPU_replay_watchdog_state'])); post = read(checked(inputs['GPU_replay_post_terminal_CIM']))
    gate('GPU_core_immutable_pins_and_sole_explicit_mutable_runtime', corepins['runtime.json'] == exception['core_preterminal_pin']
         and same(exception['terminal_pin'], inputs['GPU_replay_terminal_runtime'])
         and exception['hash_equal'] == (corepins['runtime.json']['sha256'] == inputs['GPU_replay_terminal_runtime']['sha256'])
         and immutable == replay_review['details']['core_immutable_output_pin_count'])
    gate('GPU_replay_terminalflags_identity_OS12_and_route_remain_current', watch['status'] == 'completed' and watch['exit_code'] == 0 and watch['timeout'] is False
         and identity['engineering_identity_qualified'] is True and len(identity['checks']) == 12 and all(v is True for v in identity['checks'].values())
         and runtime['execution_complete'] is True and runtime['exit_code'] == 0 and runtime['all_original_gates_passed'] is runtime['future_scoring_inputs_qualified'] is True
         and runtime['pid'] == watch['actual_pid'] == identity['actual_pid'] and runtime['parent_pid'] == watch['launcher_pid']
         and Path(watch['command'][4]).samefile(inputs['GPU_replay_config']['path']) and Path(watch['command'][6]).samefile(replayrun)
         and runtime['plan_sha256'] == inputs['GPU_replay_config']['sha256'] and runtime['source_sha256'] == sha(watch['command'][2])
         and post['actual_CIM_rows'] == [] and set(post['queried_pids']) == {runtime['pid'], runtime['parent_pid']})
    details['GPU_runtime_progress_scope'] = {'last_progress_converted_rows': runtime.get('converted_rows'), 'last_progress_phase': runtime.get('phase'),
        'authority': 'Terminal completion flags + full raw2260/mask/call records and independent29, not possibly stale periodic progress counters.',
        'mutable_old_pin': corepins['runtime.json'], 'terminal_pin': inputs['GPU_replay_terminal_runtime']}
    packet = load(checked(inputs['qualified_GPU_prestate1808'])); consumed = load(run/'consumed_GPU_state58.npz')
    forensic = load(checked(replay['input_pins']['raw_conversion_all2260']))
    x = packet['native_GPU_prestate58']; q = forensic['pre_qpos']; v = forensic['pre_qvel']; pp, R = packet['palm_pos'], packet['palm_rot']
    gate('qualified_packet1808_complete_strict_FP32_int64_order', x.dtype == np.float32 and x.shape == (1808, 58) and np.isfinite(x).all()
         and packet['completed_mask'].dtype == bool and packet['completed_mask'].shape == (1808,) and packet['completed_mask'].all()
         and packet['source_pre_row'].dtype == np.int64 and bits(packet['source_pre_row'], np.arange(1808, dtype='i8'))
         and bits(pp, forensic['palm_pos_all2260'][:1808]) and bits(R, forensic['palm_rot_all2260'][:1808]))
    expected_consumed = {'pre_x': x, 'pre_qpos': q, 'pre_qvel': v, 'palm_pos': pp, 'palm_rot': R, 'source_pre_row': packet['source_pre_row'],
        **{k: packet[k] for k in ('hand_qpos', 'hand_dof', 'object_qpos_start', 'object_dof_start')}}
    gate('all_consumed_arrays_exact_qualified_input_not_CPU_reconversion', set(consumed) == set(expected_consumed)
         and all(bits(consumed[k], val) for k, val in expected_consumed.items())
         and bits(x[:, :20], q[:, packet['hand_qpos']]) and bits(x[:, 20:40], v[:, packet['hand_dof']]))
    samples, rewards = jsonl(checked(inputs['physical_samples'])), jsonl(checked(inputs['reward_snapshots']))
    groups = {}; rewardmap = {(r['path'], r['k'], r['branch_id']): r for r in rewards}
    clock = np.zeros(601, dtype='f4')
    for i in range(600):
        clock[i+1] = np.float32(clock[i]+np.float32(.005))
    timeline = len(samples) == 1808 and len(rewardmap) == len(rewards) == 452
    coverage = {'donor': 0, 'baseline': 0, 'direction': 0}
    for i, s in enumerate(samples):
        k, sub = s['phase'], s['substep']; key = (s['path'], k, s['branch_id'])
        groups.setdefault(key, []).append(i); coverage[s['mode']] += 1
        timeline &= (0 <= k < 150 and s['source_frame'] == 200+k and s['canonical_pre_tick'] == 4*k+sub-1
            and s['canonical_post_tick'] == 4*k+sub and s['pre_time'] == float(clock[4*k+sub-1])
            and s['post_time'] == float(clock[4*k+sub]) and s['external_force_max'] == 0)
    for key, ids in groups.items():
        timeline &= ids == list(range(ids[0], ids[0]+4)) and [samples[i]['substep'] for i in ids] == [1, 2, 3, 4]
        rw = rewardmap[key]
        timeline &= np.array_equal(pp[ids], np.broadcast_to(np.asarray(rw['state20_reader_palm_pos'], dtype='f4')[0], (4, 3)))
        timeline &= np.array_equal(R[ids], np.broadcast_to(np.asarray(rw['state20_reader_palm_rot'], dtype='f4')[0], (4, 3, 3)))
        for a, b in zip(ids, ids[1:]):
            timeline &= np.array_equal(np.asarray(samples[a]['post_qpos'], dtype='f4'), np.asarray(samples[b]['pre_qpos'], dtype='f4'))
            timeline &= np.array_equal(np.asarray(samples[a]['post_qvel'], dtype='f4'), np.asarray(samples[b]['pre_qvel'], dtype='f4'))
    gate('all1808_rows_452groups_fixed_palm_clock_phase_substeps_noassist', timeline and set(groups) == set(rewardmap)
         and coverage == {'donor': 1200, 'baseline': 96, 'direction': 512}
         and bits(q, np.asarray([s['pre_qpos'] for s in samples], dtype='f4')) and bits(v, np.asarray([s['pre_qvel'] for s in samples], dtype='f4')))
    poses = {}
    for name, width in [('forensic_solver_position1808', 3), ('forensic_solver_rotation1808', 9)]:
        record = load(checked(inputs[name])); d = m['old_rule'].old_rule(record['actual'], record['reference'], False)
        gate(name+'_actual_flat_ABI_and_all_original_flags_recomputed', record['actual'].shape == (1808, width) and record['actual'].dtype == np.float32
             and d['all_passed'] and all(bits(record[k], d[k]) for k in ('difference_F64', 'old_bound_F64', 'old_flags')))
        poses[name] = record['actual']
    worldp = poses['forensic_solver_position1808']; worldR = m['rows'].logical_world_rotation_from_saved_flat9(poses['forensic_solver_rotation1808'])
    gate('alignment_solver_world_position_original_prephase_row_identity',
         bits(worldp, np.asarray([s['object_solver_pose'][:3] for s in samples], dtype='f4'))
         and bits(forensic['object_solver_pose'], np.asarray([s['object_solver_pose'] for s in samples], dtype='f4')))
    aligned = m['alignment'].alignment(x, pp, R, worldp, worldR)
    saved_alignment = load(run/'GPU_prestate_solver_pose_alignment_all1808.npz')
    expected_alignment = {'actual_GPU_p_R_F64': aligned['actual_pose12_F64'], 'derived_reference_F64': aligned['reference_pose12_F64'],
        'difference_F64': aligned['difference_F64'], 'original_8eps_bound_F64': aligned['old_bound_F64'], 'original_8eps_flags': aligned['old_flags'],
        'palm_pos_FP32': pp, 'palm_rot_FP32': R, 'solver_world_p_FP32': worldp, 'solver_world_R_FP32': worldR}
    gate('all1808_new_F64_alignment_reference_delta_bound_masks_exact', set(saved_alignment) == set(expected_alignment)
         and all(bits(saved_alignment[k], val) for k, val in expected_alignment.items()) and aligned['alignment_all_passed'])
    gates = read(run/'state_input_qualification.json')
    gate('input_qualification_record_source_order_no_calls_before_gates', gates == report['state_input_gates']
         and gates['inherited_GPU_full58_replay_original_gates_passed'] is gates['independent_saved_replay_passed'] is gates['NEW_all1808_GPU_p_R_derived_alignment_passed'] is True
         and gates['factory_calls_before_gates'] == gates['CPU_state_conversions'] == 0)
    details['alignment'] = {k: aligned[k] for k in ('alignment_all_passed', 'position_maximum_absolute_difference_m', 'rotation_maximum_absolute_difference', 'failed_rows', 'failed_elements')}
    for path in ('A0', 'A2'):
        expected = load(checked(inputs[path+'__expected_context_fields.npz'])); actual = load(checked(inputs[path+'__actual_context_readbacks.npz'])); context = load(checked(inputs['actual_context']))
        ctxok = len(expected) == 8 and set(actual) == {stage+'__'+f for stage in ('after_copy', 'after_set_const_0') for f in expected}
        for f, value in expected.items():
            ctxok &= np.array_equal(value, context[f][0])
            for stage in ('after_copy', 'after_set_const_0'):
                got = actual[stage+'__'+f]; want = context[f][0]
                ctxok &= got.shape[1:] == want.shape and got.ndim == want.ndim+1 and np.array_equal(got, np.broadcast_to(want.astype(got.dtype), got.shape))
        gate(path+'_raw_actual8_double_readbacks_exact', ctxok)
    catalog = read(run/'geometry_ID_mapping.json'); n = catalog['native_ngeom']; geoms = catalog['native_geom_catalog']; bodies = catalog['native_body_catalog']
    bybody = {b['id']: b for b in bodies}; names = {b['name']: b for b in bodies}
    hand = {g['id'] for g in geoms if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)', g['name']) and (g['contype'] or g['conaffinity'])}
    objectid = names['object/cube']['id']; objects = {g['id'] for g in geoms if g['native_body_id'] == objectid}
    catalogok = type(n) is int and {g['id'] for g in geoms} == set(range(n)) and len(geoms) == n and len(bybody) == len(bodies)
    catalogok &= all(g['native_body_id'] in bybody and bybody[g['native_body_id']]['name'] == g['native_body_name'] for g in geoms)
    catalogok &= catalog['native_object_body_id'] == objectid and {g['id'] for g in catalog['native_hand_geoms']} == hand and {g['id'] for g in catalog['native_object_geoms']} == objects
    catalogok &= names['robot/mocap_base']['mocap_id'] == 0 and names['robot/right_palm_link']['parent_id'] == names['robot/mocap_base']['id']
    gate('complete_saved_native_catalog_rederives_masks_body_vs_mocap_IDs', catalogok)
    sample = load(run/'proxy_sample_buffers.npz'); ids = sample['sample_geom_ids']; local = sample['sample_body_ids']; pts = sample['sample_points_body']
    localmap = {b['local_FK_index']: b for b in catalog['FK_local_body_table']}; records = {g['id']: g for g in catalog['actual_proxy_hand_geometry_records']}
    sampleok = ids.dtype.kind in 'iu' and local.dtype.kind in 'iu' and ids.shape == local.shape == (1344,) and pts.shape == (1344, 3) and pts.dtype == np.float32 and np.isfinite(pts).all()
    sampleok &= len(np.unique(ids)) == len(records) == 21 and set(ids) == set(records) and all((ids == g).sum() == 64 for g in records)
    sampleok &= all(int(local[i]) in localmap and localmap[int(local[i])]['native_body_name'] == records[int(ids[i])]['body'] for i in range(1344))
    factory = read(run/'factory_receipt.json'); oldgeometry = read(checked(inputs['previous_bonus_factory_geometry_manifest']))['P_ONLY']['geometry']
    sampleok &= factory['geometry'] == oldgeometry and factory['geometry']['sampling_points_sha256'] == hashlib.sha256(np.ascontiguousarray(pts).tobytes()).hexdigest()
    gate('all1344_sample_buffers_nativegeom_vs_FKlocal_and_frozen_geometry', sampleok)
    score = load(run/'all1808_proxy_scores.npz')
    rebuilt = m['rows'].reconstruct_rows(samples, score, ids, hand, objects, n, m['proxy_math'])
    gate('full1808x1344_SDF_exact_P_BP_rowmin_pergeommin_source_arithmetic', score['sampled_sdf_m'].shape == (1808, 1344) and rebuilt['captured_proxy_arrays_match'])
    rows, stats = rebuilt['rows'], rebuilt['statistics']
    gate('all_native_category_HO_depths_match_exact_generated_mask', all(r['native_generated_HO_depth_m'] == s['measured_hand_object']['penetration_by_category_m']['hand_object'] for r, s in zip(rows, samples)))
    gate('all_rows_exact_tied_deepest_secondary_allpairs_boundary_flags', rows == read(run/'rows_all1808.json'))
    before = read(run/'native_depth_before_proxy.json'); beforeok = len(before) == 1808
    for r, b in zip(rows, before):
        wanted = {k: r[k] for k in ('native_generated_HO_depth_m', 'native_generated_HO_count', 'native_HO_depth_gt3mm', 'all_exact_tied_deepest_pairs', 'all_violating_pairs_SECONDARY')}
        wanted.update(primary_deepest_tie_coverage=None, secondary_all_violating_coverage=None, proxy_coverage_status='NOT_EVALUATED_BEFORE_FACTORY')
        beforeok &= wanted == b
    gate('all1808_native_before_factory_coverage_explicit_unknown', beforeok)
    gate('wholecohort_and_all_six_path_modes_statistics_recomputed', stats == read(run/'statistics.json') == report['statistics'])
    gate('native_depth_phase_positive_masks_allrows_exact', np.array_equal(score['native_HO_depth_m'], [r['native_generated_HO_depth_m'] for r in rows])
         and np.array_equal(score['phases'], [s['phase'] for s in samples]) and bits(score['proxy_P_positive'], rebuilt['recomputed_proxy']['P_positive']))
    chunksok = True
    for batch, start in enumerate(range(0, 1808, 128)):
        stop = min(start+128, 1808); chunk = load(run/('chunk_%02d.npz' % batch))
        chunksok &= np.array_equal(chunk['row_indices'], np.arange(start, stop)) and bits(chunk['geom_ids'], score['geom_ids'])
        chunksok &= all(bits(chunk[k], score[k][start:stop]) for k in ('sampled_sdf_m', 'P', 'BP', 'minimum_sdf_m', 'per_geom_min_sdf_m'))
    gate('all15_chunks_14x128plus16_exact_full_arrays', chunksok)
    with (run/'rows_all1808.csv').open(newline='', encoding='utf-8-sig') as f:
        csvok = True; count = 0
        for expected, actual in zip(rows, csv.DictReader(f), strict=True):
            count += 1
            for key, val in actual.items():
                want = expected[key]
                csvok &= val == ('' if want is None else str(want)) if isinstance(want, (str, bool)) or want is None else float(val) == want
    gate('all1808_CSV_rows_exact_frozen_fields', csvok and count == 1808)
    counts = {'passive_mapping_MJB': 1, 'objective_factory': 1, 'objective_forward': 15}
    gate('source_instrumented_passiveMJB1_factory1_FW15_not_new_conversions', report['counts'] == {k: {'attempted': v, 'returned': v} for k, v in counts.items()}
         and report['new_state_conversions'] == 0 and report['qualified_GPU_rows_consumed'] == report['scored_rows'] == 1808)
    gate('RNG_and_objective_buffer_invariants_source_assertions_bound', bool(report['invariants']) and all(v is True for v in report['invariants'].values()))
    gate('scientific_scope_no_new_gradient_hidden_or_task_qualification', report['schema'] == 'full151105_same_solver_phase_penalty_qualified_GPU_CPU_analysis:1'
         and report['status'] == 'COMPLETED_SAME_SOLVER_PHASE_PROXY_OBSERVABILITY_NOT_PHYSICS_CERTIFICATE'
         and report['physical_gradient_qualified'] is report['wholehiddenstate_qualified'] is False
         and report['WM_Actor_weight_loads'] == report['new_AD'] == report['Env_physics_nativeFK_collision_GPU'] == 0)
    for p in pins:
        checked(p)
    for p in outputpins.values():
        checked(p)
    gate('all_sources_inputs_outputs_auditor_unchanged_after_saved_review', pin(__file__) == auditor)
    gate('no_Torch_nativeReader_model_MJB_FK_geometry_GPU_imports', not any(k in sys.modules for k in ('torch', 'warp', 'mujoco', 'mujoco_warp', 'dex_supermimic', 'mjlab', 'scipy')))
    result = {'schema': 'full151105_qualified_solver_phase_penalty_saved_independent:1',
        'status': 'PASS_SAVED_ONLY_DENSE_OBSERVABILITY_REVIEW' if all(checks.values()) else 'FAILED_SAVED_ONLY_DENSE_REVIEW',
        'engineering_qualified': all(checks.values()), 'passed': sum(checks.values()), 'total': len(checks), 'checks': checks,
        'auditor': auditor, 'config': pin(config_path), 'producer_report': pin(run/'report.json'), 'protocol': cfg['protocol'], 'run_dir': str(run.resolve()),
        'source_pin_count': len(cfg['source_pins']), 'details': details, 'statistics': stats,
        'physical_gradient_qualified': False, 'whole_hidden_state_qualified': False,
        'limitations': ['Generated HO candidate depth is not exhaustive gap/contact/force truth; no positive-force primary filter.',
            'Exact deepest ties and all strict>3mm pairs yield overlapping evidence, never an exhaustive exclusive cause partition.',
            'FP32 scalar allowance boundary and nominal3mm are separately retained; no threshold change or zero reweighting.',
            'F64 p/R alignment supplements the separately qualified full58 Reader replay and never replaces it.',
            'Factory passive MJB read is source-accounted; call counters and before/after RNG/parameter/buffer booleans are source-backed, not raw RNG or weights replay.',
            'MJB model catalog copied from the producer is source-bound; masks/ID spaces and raw contact/SDF arithmetic are independently reconstructed from its saved constants.',
            'GPU runtime periodic progress can remain2176; completion/full2260 raw and independent29 are final count authority.',
            'No method benefit, independent-N, physical-gradient, hidden-state, human-label, force-closure or task-success qualification.'],
        'new_converter_calls': 0, 'new_models_MJB_nativeFK_geometry_queries': 0, 'physics_steps': 0, 'GPU_used': False}
    dump(out/'report.json', result)
    print(json.dumps({'report': pin(out/'report.json'), 'passed': result['passed'], 'total': result['total'], 'statistics': stats}, indent=2))
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
            dump(args.output/'failure.json', {'error': repr(e), 'auditor': pin(__file__), 'saved_only': True, 'original_outputs_unchanged': True})
        raise
