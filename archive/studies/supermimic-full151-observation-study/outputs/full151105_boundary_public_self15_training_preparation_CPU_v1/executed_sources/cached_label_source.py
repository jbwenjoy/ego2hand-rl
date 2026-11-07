"""Saved-only self15 label evaluation. No weights/native/model/Torch imports.

The parent native evaluation remains drift-only. This new namespace evaluates
the original cached last-solver sensor target, NOT post20 collision geometry.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = Path('D:/workspace/supermimic-full151-observation-study')
SCHEMA = 'full151105_cached_self15_label_evaluation_CPU_protocol:1'
ROLE = 'all152_cached_native15_sensor_labels_saved_only'
PREFIX = 'full151105_boundary_cached_self15_saved_CPU_'
ROLES = ('unwrapped_baseline', 'shared_public_losses', 'shared_public_plus_cross_target_consistency')
FORWARD_ALIASES = dict(zip(ROLES, ('baseline_forward152', 'control_forward152', 'method_forward152')))
INPUTS = {'native_eval_protocol', 'native_eval_report', 'baseline_forward152', 'control_forward152', 'method_forward152',
          'native_eval_group152', 'target_protocol', 'target_report', 'target_group152', 'reward_snapshots',
          'sensor_mapping_A0', 'sensor_mapping_A2', 'physical_saved_review'}
CONTRACT = ROOT/'docs/full151_boundary_final_self15_evaluation_source_contract_v1.md'
CONTRACT_SHA = 'a535112324eeddb9113ebf47b2d68e379c3ddf5f191604eea498e756a30254f5'
MIDDLE = [1, 2, 6, 7, 11, 12, 16, 17, 21, 22]
DISTAL = [3, 8, 13, 18, 23]
TIPS = [4, 9, 14, 19, 24]
PHASES = (0, 49, 99, 149)
FEATURES = [f'finger{f}_link{j}' for f in range(1, 6) for j in (2, 3)]+[f'finger{f}_link4_OR_tip' for f in range(1, 6)]
REWARD_DT_F64 = 1/50  # Pen reward dt; NOT four raw WP timesteps summed in F64.
PREP = ROOT/'outputs/full151105_cached_self15_saved_preparation_CPU_v1'
OWN = {'consumer': ROOT/'src/full151_gradient/boundary_self15_cached_saved105_v1.py',
       'CLI': ROOT/'scripts/evaluate_full151_boundary_cached_self15_saved_cpu_v1.py',
       'checker': ROOT/'scripts/check_full151_boundary_cached_self15_saved_cpu_v1.py', 'source_contract': CONTRACT}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def pin(path):
    return {'path': str(Path(path).absolute()), 'sha256': sha(path)}


def checked(value):
    require(type(value) is dict and set(value) == {'path', 'sha256'} and type(value['path']) is str and
            Path(value['path']).is_absolute() and type(value['sha256']) is str and len(value['sha256']) == 64, 'Exact absolute pin')
    p = Path(value['path']).resolve(strict=True)
    require(sha(p) == value['sha256'], 'Pinned saved artifact changed')
    return p


def same(a, b):
    return a['sha256'] == b['sha256'] and checked(a).samefile(checked(b))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def allchecks(value):
    require(value['passed'] == value['total'] == len(value['checks']) and value['checks'] and
            all(v is True for v in value['checks'].values()), 'All existing saved-parent checks passed')


def validate_protocol(p, execute=False):
    require(p['schema'] == SCHEMA and p['role'] == ROLE and p['output_prefix'] == PREFIX, 'New saved self15 namespace')
    require(type(p['CPU_run_authorized']) is bool and p['GPU_run_authorized'] is False and p['training_authorized'] is False and
            p['automatic_retry'] is False and p['accuracy_gate'] is None, 'Saved descriptive CPU-only authorization/no selection')
    require(type(p['inputs']) is dict and set(p['inputs']) == INPUTS and p['source_pins'], 'Exact13 saved semantic inputs')
    require(same(p['source_contract'], {'path': str(CONTRACT), 'sha256': CONTRACT_SHA}), 'Frozen cached15/geometry phase contract')
    if execute:
        require(p['CPU_run_authorized'] is True, 'Root explicit saved evaluation authorization')
    return p


def metadata_preflight(protocol_path, output=None):
    require(not any(k in sys.modules for k in ('torch', 'mujoco', 'warp')), 'No neural/native imports')
    p = validate_protocol(read(protocol_path));i = p['inputs']
    for v in [*i.values(), *p['source_pins'].values()]:
        checked(v)
    for path in OWN.values():
        require(any(same(pin(path), v) for v in p['source_pins'].values()), 'Current source/contract membership')
    n = read(checked(i['native_eval_report']));np = read(checked(i['native_eval_protocol']))
    require(n['schema'] == 'full151105_boundary_component_final_evaluation_CPU_analysis:1' and
            n['status'] == 'COMPLETED_FIXED_INITIAL_AND_TWO_AFTER1000_EVAL_NOT_CONFIRMATION' and
            same(n['protocol'], i['native_eval_protocol']) and np['schema'] == 'full151105_boundary_component_final_evaluation_CPU_protocol:1' and
            n['counts']['native_FW_returned'] == 3 and n['counts']['predicted_rows'] == 456 and
            n['counts']['AD_optimizer_Actor_119f_MJB_Env_physics_GPU'] == 0, 'Completed pinned native evaluation')
    allchecks(n)
    for role, alias in FORWARD_ALIASES.items():
        require(same(n['outputs'][role+'_forward152.npz'], i[alias]), 'Exact parent saved role output')
    require(same(n['outputs']['group152_identities.json'], i['native_eval_group152']) and
            same(np['inputs']['target_report'], i['target_report']) and same(np['inputs']['target_identities'], i['target_group152']), 'Same target cohort copied by native producer')
    t = read(checked(i['target_report']));tp = read(checked(i['target_protocol']))
    require(t['schema'] == 'full151105_saved_boundary_impulse_target_dataset_CPU_analysis:1' and t['enrollment_qualified'] is True and
            all(v is True for v in t['guards'].values()) and same(t['protocol'], i['target_protocol']) and
            tp['schema'] == 'full151105_saved_boundary_impulse_target_dataset_CPU_protocol:1' and
            same(tp['inputs']['reward_snapshots'], i['reward_snapshots']), 'Original old152 target and cached reward provenance')
    require(checked(i['target_group152']).samefile(checked(i['target_report']).parent/'group152_identities.json') and
            i['target_group152']['sha256'] == i['native_eval_group152']['sha256'] and
            checked(i['target_group152']).read_bytes() == checked(i['native_eval_group152']).read_bytes(), 'Source/copy JSON bytes, not an incorrect samefile claim across copies')
    physical = read(checked(i['physical_saved_review']));allchecks(physical)
    require(physical['schema'] == 'full151105_full_serial_saved_independent:1' and physical['engineering_qualified'] is True and
            Path(physical['run_dir']).samefile(checked(i['reward_snapshots']).parent), 'Original full05 cached sensor source review')
    for path in ('A0', 'A2'):
        require(checked(i['sensor_mapping_'+path]).samefile(checked(i['reward_snapshots']).parent/('donor_'+path)/'sensor_label_mapping.json'), 'Actual per-path sensor mapping route')
        validate_mapping(read(checked(i['sensor_mapping_'+path])))
    out = None if output is None else Path(output).resolve()
    if out is not None:
        require(out.parent == ROOT/'outputs' and out.name.startswith(PREFIX) and not out.exists() and
                ('output' not in p or out == Path(p['output']).resolve()), 'Fresh fixed saved output')
    return p, out


def validate_mapping(m):
    require(m['middle'] == MIDDLE and m['distal'] == DISTAL and m['tips'] == TIPS and
            m['native_sensor'] == 'finger_collision' and m['state_phase'] == 'postintegration20ms' and
            m['sensor_phase'] == 'lastsolver preintegration15ms before Managerforward', 'Fixed source logical15 from raw25; not mode2 or HO')


def validate_plan_row(g):
    bid = g['row_id'];require(type(bid) is int and 0 <= bid < 152, 'Original branch row id')
    require(all(type(g[k]) is int for k in ('snapshot_id', 'k', 'branch_id', 'group_id', 'terminal_observation')) and
            type(g['physical_rows']) is list and all(type(v) is int for v in g['physical_rows']), 'Exact integer source identity fields')
    s, off = divmod(bid, 19);pi, si = divmod(s, 4);k = PHASES[si];group = 226*pi+150+19*si+off
    require(g['branch_id'] == bid and g['snapshot_id'] == s and g['path'] == ('A0', 'A2')[pi] and g['k'] == k and
            g['mode'] == ('baseline' if off < 3 else 'direction') and g['group_id'] == group and
            g['physical_rows'] == [4*group+x for x in range(4)] and g['terminal_observation'] == k+1, 'Exact original152 order/path/mode/group/phase')


def join_labels(plan, reward_rows, mappings):
    """Pure JSON join; can test two synthetic rows, real caller enforces152."""
    lookup = {}
    for r in reward_rows:
        key = (r['path'], r['k'], r['branch_id'])
        require(key not in lookup, 'Unique cached reward identity')
        lookup[key] = r
    labels = [];joined = []
    for g in plan:
        validate_plan_row(g);r = lookup[(g['path'], g['k'], g['branch_id'])];m = mappings[g['path']];validate_mapping(m)
        require(all(type(r[k]) is int for k in ('k', 'branch_id', 'source_frame', 'terminal_observation')) and
                r['mode'] == g['mode'] and r['source_frame'] == 200+g['k'] and r['terminal_observation'] == g['k']+1 and
                r['valid_next_policy_decision'] is (g['k'] < 149) and r['no_u150'] is True and
                r['label_time'] == 'lastsolver_preintegration~kdt+.015' and r['state_time'] == 'postintegration_kdt+.020' and
                r['capture'] == 'after_original_reward_compute_before_Manager_forward', 'Native cached15 versus post20 timing/terminal contract')
        c, raw = r['contact'], r['raw_found']
        require(type(c) is list and len(c) == 15 and all(type(v) in (int, float) and math.isfinite(v) and v in (0, 1) for v in c), 'Binary representable contact15')
        require(type(raw) is list and len(raw) == 1 and type(raw[0]) is list and len(raw[0]) == 25 and
                all(type(v) in (bool, int, float) and math.isfinite(v) and v >= 0 and int(v) == v for v in raw[0]), 'Nonnegative raw sensor-found25 counts')
        found = [v > 0 for v in raw[0]]
        reconstructed = [float(found[j]) for j in m['middle']]+[float(found[d] or found[t]) for d, t in zip(m['distal'], m['tips'])]
        require(all(struct.pack('<f', float(a)) == struct.pack('<f', b) for a, b in zip(c, reconstructed)), 'All15 raw-found logical mapping exact FP32 bytes')
        labels.append([float(v) for v in c]);joined.append({**g, 'contact15': c, 'raw_found25': raw[0],
            'label_time': r['label_time'], 'state_time': r['state_time'], 'source_frame': r['source_frame'],
            'valid_next_policy_decision': r['valid_next_policy_decision'], 'no_u150': True})
    return labels, joined


def score_matrix(logits, target, np):
    require(logits.dtype == np.float32 and logits.ndim == 2 and logits.shape[1] == 15 and logits.shape == target.shape and
            target.dtype == np.float32 and np.isfinite(logits).all() and np.isfinite(target).all() and np.isin(target, [0, 1]).all(), 'Finite original FP32 logits/labels15')
    l, y = logits.astype(np.float64), target.astype(np.float64)
    probability = np.exp(-np.logaddexp(0., -l))
    bce = np.maximum(l, 0.)-l*y+np.log1p(np.exp(-np.abs(l)))
    predicted = l > 0;truth = y > 0;ties = l == 0
    count = probability.sum(axis=1);true_count = y.sum(axis=1)
    return {'logits15_FP32': logits.copy(), 'probability15_F64': probability, 'BCE15_F64': bce,
            'Brier15_F64': (probability-y)**2, 'decision15_bool': predicted, 'truth15_bool': truth,
            'zero_logit_tie15_bool': ties, 'pred_intensity_F64': count, 'target_intensity_F64': true_count,
            'intensity_error_F64': count-true_count, 'pred_proxy_penalty_F64': -REWARD_DT_F64*count,
            'target_proxy_penalty_F64': -REWARD_DT_F64*true_count,
            'proxy_penalty_error_F64': -REWARD_DT_F64*(count-true_count)}


def cohorts(plan):
    require(len(plan) == 152 and [g['row_id'] for g in plan] == list(range(152)), 'All152 original rows, no selection')
    result = {'ALL': list(range(152))}
    for path in ('A0', 'A2'):
        result[path] = [i for i, g in enumerate(plan) if g['path'] == path]
    for path in ('A0', 'A2'):
        for k in PHASES:
            result[f'{path}/k{k}'] = [i for i, g in enumerate(plan) if g['path'] == path and g['k'] == k]
    for mode in ('baseline', 'direction'):
        result[mode] = [i for i, g in enumerate(plan) if g['mode'] == mode]
    require([len(v) for v in result.values()] == [152, 76, 76]+[19]*8+[24, 128], 'Exact native-eval13 overlapping cohorts')
    return result


def summary(scores, ids, np):
    p = scores['decision15_bool'][ids];y = scores['truth15_bool'][ids];tie = scores['zero_logit_tie15_bool'][ids]
    masks = {'TP': p & y, 'FP': p & ~y, 'FN': ~p & y, 'TN': ~p & ~y,
             'zero_logit_tie': tie, 'tie_positive_target': tie & y, 'tie_negative_target': tie & ~y}
    def describe(v):
        return {'mean': float(v.mean()), 'MAE': float(np.abs(v).mean()), 'RMS': float(np.sqrt((v*v).mean())), 'maxabs': float(np.abs(v).max())}
    return {'rows': len(ids), 'decisions': len(ids)*15,
            'BCE_mean_entry_F64': float(scores['BCE15_F64'][ids].mean()),
            'BCE_sum15_mean_rows_F64': float(scores['BCE15_F64'][ids].sum(axis=1).mean()),
            'Brier_mean_entry_F64': float(scores['Brier15_F64'][ids].mean()),
            'per_feature_BCE_F64': scores['BCE15_F64'][ids].mean(axis=0).tolist(),
            'per_feature_Brier_F64': scores['Brier15_F64'][ids].mean(axis=0).tolist(),
            'confusion_total': {k: int(v.sum()) for k, v in masks.items()},
            'confusion_per_feature': {k: v.sum(axis=0).astype(int).tolist() for k, v in masks.items()},
            'pred_intensity_mean': float(scores['pred_intensity_F64'][ids].mean()),
            'target_intensity_mean': float(scores['target_intensity_F64'][ids].mean()),
            'intensity_error': describe(scores['intensity_error_F64'][ids]),
            'Pen_proxy_penalty_error_F64': describe(scores['proxy_penalty_error_F64'][ids])}


def ready_guard():
    ready = read(PREP/'READY.json');proof = read(checked(ready['cpu_report']))
    require(ready['schema'] == 'full151105_cached_self15_saved_CPU_READY:1' and proof['schema'] == 'full151105_cached_self15_saved_CPU_contract:1' and
            proof['status'] == 'PASS_TINY_CACHED_LABEL_JOIN_AND_F64_SCORE_NO_ACTUAL_DATA' and
            ready['cpu_checks'] == {'passed': proof['passed'], 'total': proof['total']}, 'Executed Tiny saved-only proof')
    allchecks(proof)
    require(proof['actual_counts'] and all(type(v) is int and v == 0 for v in proof['actual_counts'].values()) and
            set(ready['source_pins']) == set(proof['source_pins']) == set(OWN) and
            all(same(pin(path), ready['source_pins'][k]) and same(ready['source_pins'][k], proof['source_pins'][k]) for k, path in OWN.items()), 'Current narrow proof sources and zero actual data')
    return pin(PREP/'READY.json'), ready['cpu_report']


def declaration():
    return {'schema': 'full151105_cached_self15_saved_declaration:1', 'status': 'METADATA_ONLY_NO_SAVED_ARRAY_OR_METRICS',
            'protocol_schema': SCHEMA, 'role': ROLE, 'inputs': sorted(INPUTS), 'roles': ROLES,
            'cohorts': 'ALL,A0,A2,eight_path_k,baseline,direction', 'actual_NPZ_weights_NN_FW_AD_physics': 0}


def run_saved_evaluation(protocol_path, output):
    p, out = metadata_preflight(protocol_path, output);validate_protocol(p, True);ready, proof = ready_guard()
    out.mkdir();counts = {'forward_NPZ_attempted': 0, 'forward_NPZ_returned': 0, 'selected_labels': 0,
                         'weights_models_NN_FW_AD_MJB_Env_physics_GPU': 0}
    phase = 'source_and_JSON_join'
    try:
        write(out/'protocol.json', p)
        for k, path in OWN.items():
            if path.suffix == '.py':(out/('executed_'+k+'.py')).write_bytes(path.read_bytes())
        i = p['inputs'];plan = read(checked(i['target_group152']))
        for g in plan:validate_plan_row(g)
        subsets = cohorts(plan)
        reward_rows = [json.loads(s) for s in checked(i['reward_snapshots']).read_text(encoding='utf-8').splitlines() if s.strip()]
        require(len(reward_rows) == 452, 'Existing full source records; derive only original152 branches')
        target, joined = join_labels(plan, reward_rows, {path: read(checked(i['sensor_mapping_'+path])) for path in ('A0', 'A2')})
        require(len(target) == 152, 'All152 labels, no repair/filter');counts['selected_labels'] = 152
        write(out/'joined152_cached_sensor_labels.json', joined)
        import numpy as np
        target = np.asarray(target, dtype=np.float32);saved = {'cached_self15_target_FP32': target};all_stats = {}
        rows = [];phase = 'three_saved_logits_and_F64_metrics'
        for role, alias in FORWARD_ALIASES.items():
            counts['forward_NPZ_attempted'] += 1
            with np.load(checked(i[alias]), allow_pickle=False) as z:
                require(set(z.files) == {'prediction58_FP32', 'self15_logits_FP32', 'z12_FP32', 'mode_logits2_FP32'}, 'Exact existing forward output ABI, mode2 not a target')
                logits = z['self15_logits_FP32'].copy()
            counts['forward_NPZ_returned'] += 1
            require(logits.shape == (152, 15), 'All152x15 predictions')
            scores = score_matrix(logits, target, np);saved.update({role+'__'+k: v for k, v in scores.items()})
            all_stats[role] = {name: summary(scores, ids, np) for name, ids in subsets.items()}
            for bid, identity in enumerate(plan):
                r = summary(scores, [bid], np)
                rows.append({'role': role, 'row_id': bid, 'snapshot_id': identity['snapshot_id'], 'path': identity['path'], 'k': identity['k'],
                             'mode': identity['mode'], 'group_id': identity['group_id'], 'BCE_mean15_F64': r['BCE_mean_entry_F64'],
                             'Brier_mean15_F64': r['Brier_mean_entry_F64'], **r['confusion_total'],
                             'pred_intensity_F64': float(scores['pred_intensity_F64'][bid]), 'target_intensity_F64': float(scores['target_intensity_F64'][bid]),
                             'Pen_proxy_error_F64': float(scores['proxy_penalty_error_F64'][bid])})
        np.savez_compressed(out/'all152_three_role_cached_self15_metrics.npz', **saved)
        write(out/'three_roles_13cohorts.json', all_stats);write(out/'all456_role_rows.json', rows)
        with (out/'all456_role_rows.csv').open('x', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        for v in [*p['inputs'].values(), *p['source_pins'].values()]:checked(v)
        require(not any(k in sys.modules for k in ('torch', 'mujoco', 'warp', 'dex_supermimic')), 'No model/native import')
        report = {'schema': 'full151105_cached_self15_saved_evaluation:1', 'status': 'COMPLETED_ALL152_CACHED_SELF15_LABEL_METRICS_NOT_GEOMETRY_ACCURACY',
                  'protocol': pin(protocol_path), 'source_contract': p['source_contract'], 'input_pins': i,
                  'source_pins': {k: pin(path) for k, path in OWN.items()}, 'cpu_ready': ready, 'cpu_report': proof,
                  'roles': ROLES, 'cohort_row_counts': {k: len(v) for k, v in subsets.items()}, 'counts': counts,
                  'feature_names': FEATURES, 'reward_dt_F64': REWARD_DT_F64,
                  'formulas': {'BCE': 'max(logit,0)-logit*label+log1p(exp(-abs(logit))), FP64', 'decision': 'logit>0; exact zero is negative and separately counted',
                               'Pen_proxy': '-F64(1/50)*sum(sigmoid_F64(logits)); difference versus same cached binary label intensity'},
                  'scope': ['Original152 only, all three roles/all15 channels/all13 overlapping cohorts. Cohorts are not independent sample N.',
                            'Truth is cached lastsolver~15ms finger sensor, not post20 geometry/HO/human. No native self15 training loss was added.',
                            'Relationship to physical_arrays.branch_self15 is producer/prior saved-review source evidence only; this evaluator never reads that NPZ.',
                            'F64 saved postprocessing differs from exact native FP32 loss/reward arithmetic. Pen proxy is not full reward or policy benefit.',
                            'Frozen native evaluation v1 remains drift-only; these are separate exploratory saved-label metrics, no winner or accuracy gate.'],
                  'geometry_physics_Actor_ego_or_wholehidden_qualified': False}
        write(out/'report.json', report);write(out/'output_pins.json', {f.name: pin(f) for f in out.iterdir() if f.is_file()})
        print(json.dumps({'status': report['status'], 'report': pin(out/'report.json'), 'counts': counts}))
        return report
    except BaseException as error:
        write(out/'failure.json', {'schema': 'full151105_cached_self15_saved_failure:1', 'phase': phase, 'error': repr(error),
                                  'counts': counts, 'no_retry_no_repair': True, 'protocol': pin(protocol_path)})
        raise
