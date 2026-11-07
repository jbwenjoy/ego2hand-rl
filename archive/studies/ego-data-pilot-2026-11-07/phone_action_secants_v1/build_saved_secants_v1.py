"""H1 finite action secants at captured identical integration states; saved only."""
from pathlib import Path
import ast
import collections
import csv
import hashlib
import json
import math
import numpy as np

R = Path(__file__).resolve().parent
P = R.parent
DATA = P / 'phone_transition_dataset_v1/dataset_v1'
REPORT_SHA = '98e533210bf47859db829a34a3a17c598ae0ce2cbd59b8ce523dfeb6a4757c2d'

def pin(p):
    p = Path(p).resolve(strict=True)
    b = p.read_bytes()
    return {'path': str(p), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def checked(p):
    if pin(p['path']) != p:
        raise ValueError('Changed saved source: '+p['path'])
    return Path(p['path'])

def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))

def save(p, x):
    with p.open('x', encoding='utf8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()

def literal_config_scalar(source, key):
    t = ast.parse(Path(source).read_text(encoding='utf8'))
    c = next(n.value for n in t.body if isinstance(n, ast.Assign)
             and any(isinstance(a, ast.Name) and a.id == 'CONFIG' for a in n.targets))
    return next(ast.literal_eval(v) for k, v in zip(c.keys, c.values) if ast.literal_eval(k) == key)

def require(value, message):
    if not value:
        raise ValueError(message)

def main():
    report = read(DATA/'report.json')
    require(pin(DATA/'report.json')['sha256'] == REPORT_SHA, 'Frozen dataset report')
    require(report['passed'] == report['total'] and all(v is True for v in report['checks'].values()), 'Parent all-row codec/source guards')
    for x in report['outputs'].values():
        checked(x)
    prereg = read(checked(report['preregistration']))
    overlap_path = DATA.parent/'overlap_audit_v1.json'
    overlap = read(overlap_path)
    require(overlap['inputs']['dataset_report'] == pin(DATA/'report.json')
            and overlap['cross_cohort_alias_group_count'] == 6
            and overlap['cohort_union_group_count'] == 1, 'Preserve original single-lineage cross-cohort aliases')
    for x in overlap['inputs'].values():
        checked(x)
    parent = read(checked(report['parent_report']))
    protocol = read(checked(parent['protocol']))
    controller = checked(protocol['inputs']['controller_source'])
    for x in prereg['inputs'].values():
        checked(x)
    step = literal_config_scalar(controller, 'FD_absolute_control_step_rad')
    improvement = literal_config_scalar(controller, 'cost_improvement_absolute_min')
    require(step == .01 and improvement == 1e-9, 'Exact original FD/base acceptance rules')
    names = parent['actuator_joint_names']
    require(len(names) == 20 and names[:8] == [f'right_finger{f}_joint{j}' for f in [1, 2] for j in range(1, 5)], 'Original first8 anatomical actuator order')
    ids = read(checked(report['outputs']['sample_identities322.json']))
    context = read(checked(report['outputs']['branch_H2_context161.json']))
    cby = {v['branch_id']: v for v in context}
    with np.load(checked(report['outputs']['transitions322_F64.npz']), allow_pickle=False) as z:
        a = {k: z[k].copy() for k in ['input_state58', 'control20', 'next_state58', 'physical_world_delta52', 'H2_step', 'branch_id', 'source_boundary_index']}
    require(len(ids) == 322 and a['input_state58'].shape == a['next_state58'].shape == (322, 58)
            and a['control20'].shape == (322, 20) and a['physical_world_delta52'].shape == (322, 52)
            and all(np.isfinite(v).all() for v in a.values()), 'All322 saved finite row arrays')
    h1 = collections.defaultdict(list)
    for i, row in enumerate(ids):
        require(row['sample_id'] == i and int(a['H2_step'][i]) == row['H2_step']
                and int(a['branch_id'][i]) == row['branch_id']
                and int(a['source_boundary_index'][i]) == row['source_boundary_index'], 'Every row identity-array join')
        if row['H2_step'] == 0:
            h1[row['branch_id']].append(i)
    require(len(h1) == 161 and all(len(v) == 1 for v in h1.values()), 'One H1 saved row per branch')
    pairs = []
    replay = []
    aliases = collections.defaultdict(list)
    sources = {'dataset_report': pin(DATA/'report.json'), 'dataset_preregistration': report['preregistration'],
               'MPC_report': report['parent_report'], 'MPC_protocol': parent['protocol'], 'controller': pin(controller),
               'dataset_arrays': report['outputs']['transitions322_F64.npz'], 'dataset_identities': report['outputs']['sample_identities322.json'],
               'dataset_context': report['outputs']['branch_H2_context161.json'], 'source_codec': report['codec'],
               'parent_cross_cohort_alias_audit': pin(overlap_path)}
    for boundary in range(7):
        interval_pin = prereg['inputs'][f'interval_{boundary:03d}']
        interval = read(checked(interval_pin))
        before_path = Path(interval_pin['path']).parent/f'selection_before_actual_{boundary:03d}.json'
        before = read(before_path)
        require(all(before[k] == interval[k] for k in ['interval', 'start_tick', 'evaluations', 'bounded_GN', 'selected_branch', 'selected_control20', 'selected_cost', 'source_digest_before_actual']), 'Before-actual selection bound to frozen interval')
        sources[f'interval_{boundary:03d}'] = interval_pin
        sources[f'selection_before_actual_{boundary:03d}'] = pin(before_path)
        branches = before['evaluations']
        require(len(branches) == 23 and len({b['branch_index'] for b in branches}) == 23, 'Exactly23 unique cached branches')
        byid = {b['branch_index']: b for b in branches}
        receipt = before['source_digest_before_actual']
        for b in branches:
            bid = b['branch_index']
            u = np.asarray(b['controls2x20'], dtype=np.float64)
            require(u.shape == (2, 20) and bits(u[0], u[1]) and b['source_state_before'] == b['source_state_after'] == receipt
                    and b['complete_clone_equal'] is True and b['actual_planning_steps'] == 256, 'Same complete native integration-state receipt and held controls')
            rowi = h1[bid][0]
            require(ids[rowi]['source_boundary_index'] == boundary and ids[rowi]['input_integration_sha256'] == receipt['integration_sha256']
                    and cby[bid]['source_clone_receipt_before'] == cby[bid]['source_clone_receipt_after'] == receipt
                    and bits(a['control20'][rowi], u[0]), 'Dataset H1 branch/control/complete receipt')
            aliases[bid].append({'kind': 'original_evaluation', 'purpose': b['purpose'], 'boundary': boundary})
        previous = [b for b in branches if b['purpose'] == 'previous_control']
        require(len(previous) == 1, 'Unambiguous previous-control base')
        best = previous[0]
        require(len(before['bounded_GN']) == 2, 'Two captured original GN iterations')
        for gn in before['bounded_GN']:
            it = gn['iteration']
            base = best
            base_id = base['branch_index']
            base_u = np.asarray(base['controls2x20'][0], dtype=np.float64)
            require(base['cost'] == gn['base_exact_cost'], 'Replay exact intended baseline cost, not nearest cost match')
            aliases[base_id].append({'kind': 'FD_intended_baseline', 'GN_iteration': it, 'boundary': boundary})
            declared_steps = np.asarray(gn['FD_steps_rad'], dtype=np.float64)
            require(declared_steps.shape == (8,) and np.isfinite(declared_steps).all(), 'Raw saved signed FD increments')
            for j in range(8):
                pairid = len(pairs)
                expected_label = f'FD_iter{it}_act{j}'
                expected_du = float(declared_steps[j])
                nominal = math.copysign(step, expected_du) if expected_du != 0 else 0.
                labeled = [b['branch_index'] for b in branches if b['purpose'] == expected_label]
                candidates = []
                off = np.arange(20) != j
                for b in branches:
                    u = np.asarray(b['controls2x20'][0], dtype=np.float64)
                    if bits(u[off], base_u[off]) and float(u[j]-base_u[j]) == expected_du and u[j] != base_u[j]:
                        candidates.append(b['branch_index'])
                status = 'DEFINED_CAPTURED_STATE_FINITE_ACTION_SECANT'
                if expected_du == 0:
                    status = 'UNDEFINED_ZERO_DECLARED_INCREMENT'
                elif len(candidates) == 0:
                    status = 'UNDEFINED_NO_EXACT_CONTROL_MATCH'
                elif len(candidates) > 1:
                    status = 'UNDEFINED_AMBIGUOUS_CONTROL_MATCH'
                elif labeled and labeled != candidates:
                    status = 'UNDEFINED_PURPOSE_ALIAS_CONFLICT'
                fd_id = candidates[0] if len(candidates) == 1 else None
                base_row = h1[base_id][0]
                fd_row = h1[fd_id][0] if fd_id is not None else None
                guard = {}
                du = None
                if fd_row is not None:
                    fd_u = a['control20'][fd_row]
                    du = float(fd_u[j]-a['control20'][base_row, j])
                    guard = {
                        'both_original_H1_step0': ids[fd_row]['H2_step'] == ids[base_row]['H2_step'] == 0,
                        'complete_integration_receipts_exact': byid[fd_id]['source_state_before'] == base['source_state_before'] == receipt,
                        'integration_SHA_exact': ids[fd_row]['input_integration_sha256'] == ids[base_row]['input_integration_sha256'] == receipt['integration_sha256'],
                        'all58_observed_input_bits_exact': bits(a['input_state58'][fd_row], a['input_state58'][base_row]),
                        'all19_other_control_bits_exact': bits(fd_u[off], a['control20'][base_row, off]),
                        'actual_increment_matches_original_GN_record': du == expected_du,
                        'selected_coordinate_is_literal_base_plus_signed_0p01': float(fd_u[j]) == float(base_u[j]+nominal),
                        'nonzero_denominator': du != 0,
                    }
                    if not all(guard.values()):
                        status = 'UNDEFINED_GUARD_FAILURE'
                    aliases[fd_id].append({'kind': 'FD_intended_perturbation', 'GN_iteration': it, 'actuator': j, 'boundary': boundary})
                pairs.append({'pair_id': pairid, 'source_boundary_index': boundary, 'source_boundary_cohort': f'SOURCE_BOUNDARY_{boundary:03d}',
                    'GN_iteration': it, 'actuator_index': j, 'actuator_joint_name': names[j], 'expected_purpose_alias': expected_label,
                    'intended_baseline_branch_id': base_id, 'intended_baseline_original_purpose': base['purpose'], 'baseline_sample_id': base_row,
                    'FD_branch_id': fd_id, 'FD_sample_id': fd_row, 'matching_control_candidates': candidates, 'original_purpose_matches': labeled,
                    'cached_alias_recovered_without_new_branch': len(labeled) == 0 and len(candidates) == 1,
                    'input_complete_integration_receipt': receipt, 'declared_original_signed_increment_rad': expected_du,
                    'nominal_signed_step_rad': nominal, 'actual_denominator_rad': du, 'guards': guard,
                    'defined': status == 'DEFINED_CAPTURED_STATE_FINITE_ACTION_SECANT', 'status': status,
                    'baseline_identity': ids[base_row], 'FD_identity': None if fd_row is None else ids[fd_row]})
            line_decisions = []
            require([l['scale'] for l in gn['line_exact_costs']] == [1., .5, .25], 'Original three line scales')
            for line in gn['line_exact_costs']:
                b = byid[line['branch_index']]
                require(b['cost'] == line['cost'], 'Line cost and exact cached branch')
                prior_id = best['branch_index']
                accepted = bool(b['original_horizon_safety_limits_pass'] and
                                (not best['original_horizon_safety_limits_pass'] or b['cost'] < best['cost']-improvement))
                if accepted:
                    best = b
                aliases[b['branch_index']].append({'kind': 'original_line_search', 'GN_iteration': it, 'scale': line['scale'], 'accepted': accepted, 'boundary': boundary})
                line_decisions.append({'branch': b['branch_index'], 'scale': line['scale'], 'cost': line['cost'],
                                       'prior_best': prior_id, 'accepted_by_original_rule': accepted, 'new_best': best['branch_index']})
            replay.append({'boundary': boundary, 'GN_iteration': it, 'FD_baseline_branch_id': base_id,
                           'base_exact_cost': gn['base_exact_cost'], 'line_acceptance_replay': line_decisions,
                           'best_after_iteration': best['branch_index']})
        require(best['branch_index'] == before['selected_branch'] and bits(best['controls2x20'][0], before['selected_control20']), 'Replay ends at original selected branch/control')
    require(len(pairs) == 112, 'All7x2x8 expected finite direction records retained')
    # Preserve missing/ambiguous rows as NaN plus defined mask; never drop raw pair records.
    out = R/'dataset_v1'
    if out.exists():
        raise ValueError('Fresh output only; no overwrite/retry')
    out.mkdir()
    arrays = {'baseline_input58_F64': np.full((112, 58), np.nan), 'FD_input58_F64': np.full((112, 58), np.nan),
              'baseline_target58_F64': np.full((112, 58), np.nan), 'FD_target58_F64': np.full((112, 58), np.nan),
              'baseline_control20_F64': np.full((112, 20), np.nan), 'FD_control20_F64': np.full((112, 20), np.nan),
              'baseline_physical_delta52_F64': np.full((112, 52), np.nan), 'FD_physical_delta52_F64': np.full((112, 52), np.nan),
              'finite_response52_F64': np.full((112, 52), np.nan), 'secant52_per_rad_F64': np.full((112, 52), np.nan),
              'actual_control_delta20_F64': np.full((112, 20), np.nan), 'actual_du_rad_F64': np.full(112, np.nan),
              'defined_mask': np.array([p['defined'] for p in pairs], dtype=bool)}
    for p in pairs:
        i, br, fr = p['pair_id'], p['baseline_sample_id'], p['FD_sample_id']
        for side, row in [('baseline', br), ('FD', fr)]:
            if row is not None:
                for dst, src in [('input58_F64', 'input_state58'), ('target58_F64', 'next_state58'), ('control20_F64', 'control20'), ('physical_delta52_F64', 'physical_world_delta52')]:
                    arrays[f'{side}_{dst}'][i] = a[src][row]
        if fr is not None:
            arrays['actual_control_delta20_F64'][i] = a['control20'][fr]-a['control20'][br]
            arrays['actual_du_rad_F64'][i] = p['actual_denominator_rad']
            arrays['finite_response52_F64'][i] = a['physical_world_delta52'][fr]-a['physical_world_delta52'][br]
            if p['defined']:
                arrays['secant52_per_rad_F64'][i] = arrays['finite_response52_F64'][i]/p['actual_denominator_rad']
    require(np.isfinite(arrays['secant52_per_rad_F64'][arrays['defined_mask']]).all(), 'Finite every defined component; no sample filtering')
    np.savez_compressed(out/'secants112_F64.npz', **arrays,
                        source_boundary_index=np.array([p['source_boundary_index'] for p in pairs], dtype=np.int64),
                        GN_iteration=np.array([p['GN_iteration'] for p in pairs], dtype=np.int64),
                        actuator_index=np.array([p['actuator_index'] for p in pairs], dtype=np.int64))
    save(out/'pairs112.json', pairs)
    save(out/'branch_aliases161.json', {str(b): v for b, v in aliases.items()})
    save(out/'baseline_selection_replay14.json', replay)
    save(out/'parent_cross_cohort_aliases6.json', {'source_audit': pin(overlap_path),
         'aliases': overlap['aliases'], 'cohort_union_groups': overlap['cohort_union_groups'],
         'scope': 'Retained parent alias evidence only; excluded H2 rows are not counted again as independent H1 secants.'})
    with (out/'coverage112.csv').open('x', encoding='utf8', newline='') as f:
        fields = ['pair_id', 'source_boundary_index', 'GN_iteration', 'actuator_index', 'actuator_joint_name',
                  'intended_baseline_branch_id', 'FD_branch_id', 'baseline_sample_id', 'FD_sample_id', 'actual_denominator_rad', 'defined', 'status']
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader();w.writerows({k: p[k] for k in fields} for p in pairs)
    schema = {'schema': 'phone_H1_same_integration_finite_action_secants:1',
        'primary_formula': '(saved physical_world_delta52_FD - saved physical_world_delta52_baseline) / actual signed control difference[j]',
        'rotation': 'Each physical delta uses Log(Rtarget @ Rcommon_input.T) in world chart. Difference of these two saved world rotation deltas; raw full58/R9 inputs and targets are retained. Not native WM chart52, no R projection or new Log evaluation.',
        'blocks': {'q20': {'slice': [0,20], 'response_unit': 'rad', 'secant_unit': 'rad/rad'},
                   'qd20': {'slice': [20,40], 'response_unit': 'rad/s', 'secant_unit': '(rad/s)/rad'},
                   'pCOM3': {'slice': [40,43], 'response_unit': 'm', 'secant_unit': 'm/rad'},
                   'world_rotation_Log3': {'slice': [43,46], 'response_unit': 'rad', 'secant_unit': 'rad/rad'},
                   'vCOM3': {'slice': [46,49], 'response_unit': 'm/s', 'secant_unit': '(m/s)/rad'},
                   'omega_world3': {'slice': [49,52], 'response_unit': 'rad/s', 'secant_unit': '(rad/s)/rad'}},
        'state58_same_input_scope': 'Complete captured integration-state SHA/size/signature and source-before/after receipts match; all observed58 input bytes match too. No raw hidden state was reconstructed and no state58 Markov certificate follows.',
        'coverage': '112 intended signed finite directions:7 source boundaries x2 distinct iteration bases x8 actuators. Remaining12 controls fixed, not probed. All H2_step1 excluded because branch-dependent state differs.',
        'undefined': 'All112 records retained. Undefined/missing/ambiguous raw rows are retained with NaN arrays and false defined mask; JSON status/identities remain explicit.',
        'scope': 'One consecutive failed-hold planning lineage, captured native simulator branches only; not independent episodes, heldout ego data, full20-action Jacobian, infinitesimal derivative, contact/force derivative or learning/holding qualification.'}
    save(out/'schema.json', schema)
    statuses = dict(collections.Counter(p['status'] for p in pairs))
    coverage = {str(j): {'possible': sum(p['actuator_index'] == j for p in pairs),
                         'defined': sum(p['actuator_index'] == j and p['defined'] for p in pairs),
                         'name': names[j]} for j in range(20)}
    outputs = {f.name: pin(f) for f in out.iterdir() if f.is_file()}
    save(out/'report.json', {'schema': 'phone_H1_same_integration_action_secant_build_report:1',
        'status': 'COMPLETE_ALL112_DIRECTION_RECORDS_WITH_EXPLICIT_DEFINED_MASK', 'source': pin(__file__), 'input_pins': sources,
        'outputs': outputs, 'counts': {'expected_directions':112, 'defined':sum(p['defined'] for p in pairs),
        'source_boundaries':7, 'iteration_bases':14, 'input_branches161':161, 'excluded_H2_step1_rows':161,
        'saved_NPZ_reads':1, 'new_model_FK_Log_solver_physics_NN_training_GPU':0},
        'status_counts': statuses, 'actuator_coverage': coverage,
        'declared_signed_step_range_rad': [min(p['declared_original_signed_increment_rad'] for p in pairs),max(p['declared_original_signed_increment_rad'] for p in pairs)],
        'checks': {'all322_source_rows_and_H1_branch_identities':True, 'all7_before_actual_selections_match_frozen_dataset_intervals':True,
        'all14_GN_baselines_replayed_from_original_acceptance_rule':True, 'all7_replays_match_original_selected_control':True,
        'all112_intended_direction_records_preserved':True, 'all_defined_have_exact_integration_and_other19_control_bytes':all(all(p['guards'].values()) for p in pairs if p['defined']),
        'first8_actuator_names_bound_to_original_native_metadata':True, 'no_H2_branch_state_secants_or_force_derivatives':True},
        'scope': schema['scope'], 'scientific_gradient_or_Markov_or_holding_qualified':False})
    print(json.dumps({'report': pin(out/'report.json'), 'status_counts': statuses, 'coverage': coverage}))

if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        failure = R/'failure_v1.json'
        if not failure.exists():
            save(failure, {'error':repr(e), 'source':pin(__file__), 'automatic_retry':False,
                           'new_model_FK_solver_physics_NN_training':0})
        raise
