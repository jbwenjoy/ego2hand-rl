"""Saved-only independent all452 WM state-error audit. No model/backend imports."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT/'src/full151_gradient/state_prediction452_saved_math_v1.py'
MATH_SHA = 'e3e345c1d7abe40034625185eb59e1bed72803a96b71eebdea3bba16056f0b8f'
RULE = ROOT/'src/full151_gradient/matched_reader_saved_math_v1.py'
RULE_SHA = 'a3301167f5711ddb4750068e0f530823f9e5ddb26edd5654d1854812242405e4'
CORE_SHA = '5cd15a0a3fefd2f17bbb971944fbc7c47fa8faaab473a9164728583598e2e240'
PROTOCOL_SHA = '37652718ed150421ed84d25bcafc14b27a1e7390a7dee7e729a860b3d91ef411'


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


def dump(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def jsonl(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]


def bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def module(path, expected):
    if sha(path) != expected:
        raise ValueError('Independent math changed')
    spec = importlib.util.spec_from_file_location('independent_'+path.stem, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def raw_rng_equal(a, b):
    keys = {'Python_pickle', 'NumPy_pickle', 'Torch_CPU'}
    return set(a) == set(b) == keys and all(v.dtype == np.uint8 and v.ndim == 1 and v.size > 0 for v in [*a.values(), *b.values()]) and all(bits(a[k], b[k]) for k in keys)


def CPU38_valid(ready, proof):
    return (ready.get('schema') == 'full151105_state_prediction452_CPU_READY:1'
        and proof.get('schema') == 'full151105_state_prediction452_CPU_contract:1'
        and proof.get('status') == 'PASS_SYNTHETIC_GROUP_ERROR_AND_SINGLE_FORWARD_CONTRACT'
        and type(proof.get('passed')) is int and type(proof.get('total')) is int
        and proof['passed'] == proof['total'] == len(proof.get('checks', {})) == 38
        and all(v is True for v in proof['checks'].values()))


def audit(config_path, run, out):
    out.mkdir(exist_ok=False); auditor = pin(__file__)
    (out/'executed_auditor_source.py').write_bytes(Path(__file__).read_bytes())
    dump(out/'equations_before_actual_predictions_read.json', {'auditor': auditor, 'math': pin(MATH),
        'actual_predictions_read': False, 'rules': 'F64subtract first; all452+6pathmodes/perfeature/original-unit blocks; Rraw9 only; no mixed-unit scalar/angles/score/goal/AD.'})
    math, rule = module(MATH, MATH_SHA), module(RULE, RULE_SHA)
    checks = {}; details = {}
    def gate(k, value):
        checks[k] = bool(value)
    cfg, r = read(config_path), read(run/'report.json')
    proto = read(checked(cfg['protocol'])); inputs = proto['inputs']
    inventory = read(run/'source_inventory.json')
    gate('actual_config_protocol_fixed_model_role_source_identity', cfg['schema'] == 'full151105_state_prediction452_CPU_run:1'
         and cfg['CPU_run_authorized'] is True and cfg['cpu_threads'] == 4
         and cfg['protocol']['sha256'] == PROTOCOL_SHA and same(r['config'], pin(config_path))
         and cfg == read(run/'config.json') and proto == read(run/'protocol.json')
         and same(r['protocol'], cfg['protocol']) and Path(cfg['output']).samefile(run))
    ids = []
    for p in cfg['source_pins'].values():
        path = checked(p); st = path.stat(); ids.append((st.st_dev, st.st_ino or str(path).casefold()))
    gate('dedup_current_source_actualfile_identity_not_equalbytes', len(ids) == len(set(ids))
         and len(inventory) == len(cfg['source_pins']) == r['logical_source_pin_count'] == r['unique_source_file_count']
         and r['source_pins'] == inventory
         and all(any(same(p, q) for q in cfg['source_pins'].values()) for p in inventory.values())
         and any(p['sha256'] == CORE_SHA for p in inventory.values()))
    pins = [*inventory.values(), *inputs.values(), cfg['protocol'], cfg['cpu_ready']]
    for p in pins:
        checked(p)
    with zipfile.ZipFile(run/'execution_sources.zip') as z:
        zipok = len(z.namelist()) == len(inventory)
        for i, p in enumerate(inventory.values()):
            zipok &= hashlib.sha256(z.read('%04d_%s' % (i, Path(p['path']).name))).hexdigest() == p['sha256']
    gate('all_immutable_current_sources_and_archive_bytes_exact', zipok)
    outputpins = read(run/'output_pins.json')
    for p in outputpins.values():
        checked(p)
    gate('all_original_output_pins_exact', True)
    ready = read(checked(cfg['cpu_ready'])); proof = read(checked(ready['cpu_report']))
    gate('current_consumer_CPU38_scope_and_protocol_bound', CPU38_valid(ready, proof)
         and ready['source_pins'] == proof['source_pins'] and ready['source_pins']['consumer']['sha256'] == CORE_SHA
         and same(ready['protocol'], cfg['protocol']))
    replay = read(checked(inputs['GPU_replay_saved_independent']))
    health = read(checked(inputs['fixed_A2_final_health']))
    gate('GPU29_and_finalA2health196_bound_separate_qualifications', replay['passed'] == replay['total'] == 29
         and all(v is True for v in replay['checks'].values()) and replay['state_replay_qualified'] is True
         and health['passed'] == health['total'] == 196 and all(v is True for v in health['checks'].values())
         and health['method'] == 'A2' and health['role'] == 'primary_matched' and health['iteration'] == 2000 and health['final_iteration_scope'] is True
         and health['checkpoint'] == inputs['fixed_A2_checkpoint'] and health['auditor'] == inputs['fixed_A2_health_auditor']
         and same(health['config'], inputs['fixed_A2_training_config']))
    samples, rewards = jsonl(checked(inputs['physical_samples'])), jsonl(checked(inputs['reward_snapshots']))
    packet = load(checked(inputs['qualified_GPU_prestate1808'])); gpu = load(checked(inputs['GPU_raw_all2260']))
    forensic = load(checked(inputs['forensic_raw_all2260'])); endpointlabels = read(checked(inputs['GPU_endpoint_identities']))
    identities = read(run/'group_identities452.json'); math.validate_identity_order(identities)
    expected_ids = math.expected_identities()
    a, b = load(run/'matched_input_arrays.npz'), load(run/'prediction_arrays.npz')
    first = np.arange(0, 1808, 4, dtype='i8'); last = np.arange(3, 1808, 4, dtype='i8')
    targetrows = np.arange(1808, 2260, dtype='i8')
    allu = np.asarray([s['ctrl'] for s in samples], dtype='f4').reshape(452, 4, 20)
    expected = {'pre_x452_FP32': packet['native_GPU_prestate58'][first], 'target_x452_FP32': gpu['native_GPU_state58_raw'][1808:],
        'actual_u452_FP32': allu[:, 0].copy(), 'all_substep_u452x4x20_FP32': allu,
        'group_first_pre_indices': first, 'group_last_physical_indices': last, 'post20_GPU_row_indices': targetrows}
    gate('all452_assembly_pre_lasttarget_actualu_indices_exact_saved_sources', set(a) == set(expected)
         and all(bits(a[k], val) and bits(b[k], val) for k, val in expected.items())
         and bits(packet['native_GPU_prestate58'], gpu['native_GPU_state58_raw'][:1808])
         and gpu['completed_mask'].dtype == bool and gpu['completed_mask'].shape == (2260,) and gpu['completed_mask'].all())
    gate('all452_four_native_absolute_controls_bitexact_no_second_filter', math.held_controls_exact(allu, a['actual_u452_FP32']))
    rewardmap = {(v['path'], v['k'], v['branch_id']): v for v in rewards}
    refs = []; sourceok = len(samples) == 1808 and len(rewards) == len(rewardmap) == 452 and len(endpointlabels) == 452
    for i, wanted in enumerate(expected_ids):
        rows = samples[4*i:4*i+4]; key = (wanted['path'], wanted['k'], wanted['branch_id'])
        sourceok &= all((s['path'], s['phase'], s['branch_id']) == key and s['mode'] == wanted['mode'] and s['substep'] == j+1
            and s['canonical_pre_tick'] == 4*wanted['k']+j and s['canonical_post_tick'] == 4*wanted['k']+j+1
            and s['source_frame'] == wanted['source_frame'] for j, s in enumerate(rows))
        sourceok &= endpointlabels[i] == {'path': key[0], 'k': key[1], 'branch_id': key[2], 'last_sample_index': 4*i+3}
        refs.append(rewardmap[key]['physics'])
        hq, hv = packet['hand_qpos'], packet['hand_dof']
        sourceok &= bits(a['target_x452_FP32'][i, :20], np.asarray(rows[-1]['post_qpos'], dtype='f4')[hq])
        sourceok &= bits(a['target_x452_FP32'][i, 20:40], np.asarray(rows[-1]['post_qvel'], dtype='f4')[hv])
    gate('all452_source_order_pathmode_rawphase_terminal_and_endpoint_identity', sourceok)
    gate('all1808_source_inputs_match_qualified_forensic_raw_bytes',
         bits(np.asarray([s['pre_qpos'] for s in samples], dtype='f4'), forensic['pre_qpos'])
         and bits(np.asarray([s['pre_qvel'] for s in samples], dtype='f4'), forensic['pre_qvel'])
         and bits(packet['palm_pos'], forensic['palm_pos_all2260'][:1808]) and bits(packet['palm_rot'], forensic['palm_rot_all2260'][:1808]))
    refs = np.asarray(refs, dtype='f4'); target = a['target_x452_FP32']
    d = rule.old_rule(target, refs, True); savedrule = load(run/'target452_original_rule_alignment.npz')
    wanted_rule = {'actual': target, 'reference': refs, **{k: d[k] for k in ('difference_F64', 'old_bound_F64', 'old_flags')}}
    gate('all452_original_full58_target_rule_raw_recomputed_not_bool', set(savedrule) == set(wanted_rule)
         and all(bits(savedrule[k], val) for k, val in wanted_rule.items()) and d['all_passed'])
    pred = b['predicted_x452_FP32']; calculated = math.all_statistics(pred, target, identities)
    gate('all452x58_F64_signed_absolute_errors_exact_before_subtraction',
         bits(b['signed_error452x58_F64'], calculated['signed_error452x58_F64']) and bits(b['absolute_error452x58_F64'], calculated['absolute_error452x58_F64']))
    gate('every_feature_units_blocks_and_all_six_pathmode_stats_exact', calculated['statistics'] == r['statistics'] == read(run/'statistics.json'))
    with (run/'all452_errors.csv').open(newline='', encoding='utf-8') as f:
        csvok = True; count = 0
        for i, row in enumerate(csv.DictReader(f)):
            count += 1; want = dict(identities[i]); err = calculated['signed_error452x58_F64'][i]
            for name, (lo, hi, _) in math.BLOCKS.items():
                want[name+'_component_RMS'] = float(np.sqrt(np.mean(err[lo:hi]**2))); want[name+'_maxabs'] = float(abs(err[lo:hi]).max())
            csvok &= set(row) == set(want)
            for key, val in row.items():
                ref = want[key]
                csvok &= val == ('' if ref is None else str(ref)) if ref is None or isinstance(ref, (str, bool)) else float(val) == ref
    gate('all452_CSV_perrow_identity_and_sixunit_errors_exact', count == 452 and csvok)
    bundle = read(checked(inputs['input_bundle'])); context = load(checked(bundle['context']))
    actualcontext = load(checked(inputs['fixed_A2_fixed_physics_context']))
    xi = b['physical_xi46_FP32']; expectedxi = context['reader_context']; actualxi = actualcontext['reader_context']
    gate('physical_xi46_exact_fixed_native_context_not_privileged368', xi.dtype == np.float32 and xi.shape == (46,) and np.isfinite(xi).all()
         and bits(xi, expectedxi[0]) and actualxi.dtype == np.float32 and actualxi.shape == (128, 46)
         and bits(actualxi, np.broadcast_to(xi, actualxi.shape).copy()))
    aux = b['aux_self15_logits_FP32']
    gate('aux_native_self15_logits_raw_finite_shape_only_not_HO_or_labels', aux.dtype == np.float32 and aux.shape == (452, 15) and np.isfinite(aux).all())
    receipt = read(run/'loader_receipt.json'); trace = read(run/'checkpoint_read_trace.json')
    gate('strict119f_receipt_fixed_actual105_final_model_health_context',
         receipt['schema'] == 'full151_105_world_model_CPU_evaluation_load:1' and receipt['status'] == 'PASS_FROZEN_CPU_WORLD_MODEL_LOAD'
         and same(receipt['checkpoint'], inputs['fixed_A2_checkpoint']) and same(receipt['health'], inputs['fixed_A2_final_health'])
         and same(receipt['health_auditor'], inputs['fixed_A2_health_auditor'])
         and receipt['method'] == 'A2' and receipt['iteration'] == 2000 and receipt['seed'] == 0
         and receipt['state_dim'] == 58 and receipt['physical_xi_dim'] == 46 and receipt['self_collision_logits_dim'] == 15
         and receipt['amp_enabled_for_CPU_diagnostic'] is receipt['GPU_used'] is receipt['Env_or_Warp_created'] is False
         and receipt['physics_steps'] == 0 and receipt['loader']['sha256'] == '119fa7cebc0e9b0727b0160660b1a47822deee2a0e0e9fa5be69465ac5578052')
    gate('actual_recorded_two_checkpoint_CPU_reads_same_fixed_bytes', len(trace) == 2 and all(Path(t['path']).samefile(inputs['fixed_A2_checkpoint']['path'])
         and t['map_location'] == 'cpu' and t['weights_only'] is False and t['returned'] is True for t in trace))
    expectedcounts = {'strict_WM_loader': 1, 'checkpoint_torch_loads': 2, 'WM_forward': 1}
    gate('one_strictload_two_reads_one_fullbatch452_forward_source_counts', r['counts'] == {k: {'attempted': v, 'returned': v} for k, v in expectedcounts.items()}
         and r['prediction_rows'] == r['WM_batch_size'] == 452)
    sourcecost = r['passive_MJB_source_accounted']
    gate('passive_MJB_total3_separate_source_accounting_not_API_counter', all(sourcecost[k] == v for k, v in {
        'caller_bundle_load': 1, 'strict_loader_internal_bundle_load': 1, 'strict_loader_internal_Actor_explicit_load': 1,
        'strict_loader_internal_total': 2, 'process_path_total': 3}.items())
         and r['internal_loader_source_accounted']['Actor_validation_constructors'] == 1 and r['internal_loader_source_accounted']['Actor_forwards'] == 0)
    rb, ra = load(run/'RNG_before.npz'), load(run/'RNG_after.npz')
    gate('raw_serialized_host_and_TorchCPU_RNG_bytes_unchanged', raw_rng_equal(rb, ra))
    gate('runtime_weights_normalizers_buffers_invariants_bound_source_assertions', bool(r['invariants']) and all(v is True for v in r['invariants'].values()))
    gate('no_taskscore_goal_AD_factory_state_or_physics_qualification', r['schema'] == 'full151105_state_prediction452_CPU_analysis:1'
         and r['status'] == 'COMPLETED_FIXED_A2_ALL452_STATE_ERROR_NOT_MODEL_QUALIFICATION'
         and r['native_state_conversions'] == r['objective_factories_or_forwards'] == r['taskscore_or_goal_calls'] == r['AD_calls'] == r['new_training_Env_nativeFK_collision_Warp_GPU_physics'] == 0
         and all(r[k] is False for k in ('physical_gradient_qualified', 'HO_gradient_qualified', 'whole_hidden_state_qualified', 'task_quality_qualified')))
    for p in pins:
        checked(p)
    for p in outputpins.values():
        checked(p)
    gate('all_sources_input_checkpoint_and_outputpins_unchanged', pin(__file__) == auditor and sha(MATH) == MATH_SHA)
    gate('reviewer_no_Torch_loader_native_state_model_FK_MJB_or_GPU', not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp','dex_supermimic','mjlab','scipy')))
    details.update(native_target_original_rule_failed_elements=d['failed_elements'], raw_RNG_vectors_equal=True,
        passive_MJB_accounting='Successful source callgraph total3, not independently intercepted nativeAPI count.',
        weights_norm_invariant_evidence='Source-recorded before/after digest assertions, no checkpoint tensors reloaded by reviewer.')
    result = {'schema':'full151105_state_prediction452_saved_independent:1',
        'status':'PASS_SAVED_ALL452_STATE_ERROR_REVIEW_NOT_MODEL_QUALIFICATION' if all(checks.values()) else 'FAILED_SAVED_ALL452_REVIEW',
        'engineering_qualified':all(checks.values()),'passed':sum(checks.values()),'total':len(checks),'checks':checks,
        'auditor':auditor,'math':pin(MATH),'config':pin(config_path),'producer_report':pin(run/'report.json'),'run_dir':str(run.resolve()),
        'source_file_count':len(inventory),'statistics':calculated['statistics'],'details':details,
        'physical_gradient_qualified':False,'HO_gradient_qualified':False,'task_quality_qualified':False,
        'limitations':['FP32CPU ampFalse fixed-WM one-step error, not historical nativeGPU BF16 policy/imagination replay or hardware invariance.',
            'All452 rows and six pathmode groups remain dependent fixed-source observations; no model selection/CI or method benefit.',
            'R uses raw9 matrix elements only; q not wrapped, units never pooled to an overall58scalar; no taskscore/goal/angle computation.',
            'Aux15 is raw selfcollision output only, not native HO contact/force or a newly evaluated label metric.',
            'One strict loader includes two torch reads and validationActor construction; no Actor forward. PassiveMJB3 is static source-accounted.',
            'Raw serialized RNG equality is independently checked; model/normalizer before-after digest assertions remain source-backed.'],
        'new_models_or_Torch_loads':0,'new_Reader_FK_MJB_geometry_calls':0,'new_AD_physics_GPU':0}
    dump(out/'report.json',result)
    print(json.dumps({'report':pin(out/'report.json'),'passed':result['passed'],'total':result['total'],'all452_blocks':calculated['statistics']['all452']['blocks']},indent=2))
    return 0 if all(checks.values()) else 2


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:raise SystemExit(audit(a.config,a.run_dir,a.output))
    except Exception as e:
        if a.output.exists() and not (a.output/'failure.json').exists():dump(a.output/'failure.json',{'error':repr(e),'auditor':pin(__file__),'saved_only':True,'source_outputs_unchanged':True})
        raise
