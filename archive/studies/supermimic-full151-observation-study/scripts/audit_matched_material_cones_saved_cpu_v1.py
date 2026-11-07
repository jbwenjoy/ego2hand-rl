"""Independent NumPy-only saved 302-cone witness audit; no solver imports."""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'outputs/full151_matched_actual_material_cones_CPU_v1'
OUT = ROOT / 'outputs/full151_matched_material_cones_saved_numeric_independent_v1'


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def arrays(path, names=None):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in (z.files if names is None else names)}


def maximum(value):
    return float(np.max(np.abs(value)))


def rotation(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((
        1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), axis=-1).reshape(q.shape[:-1]+(3, 3))


def build_all(points_body, outward_body, com, radius, mu):
    """Direct six-ray construction independent of production matrix helper."""
    normal = -outward_body.astype(float)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    nx, ny, nz = np.moveaxis(normal, -1, 0)
    sign = np.where(nz >= 0, 1., -1.)
    a = -1/(sign+nz)
    b = nx*ny*a
    t1 = np.stack((1+sign*nx*nx*a, sign*b, -sign*nx), axis=-1)
    t2 = np.stack((b, sign+ny*ny*a, -ny), axis=-1)
    t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 /= np.linalg.norm(t2, axis=-1, keepdims=True)
    frame = np.stack((normal, t1, t2), axis=-1)
    force = np.stack((normal+mu[0]*t1, normal-mu[0]*t1,
                      normal+mu[1]*t2, normal-mu[1]*t2,
                      normal, normal), axis=-1)
    direct = np.zeros_like(force)
    direct[..., 4] = mu[2]*normal
    direct[..., 5] = -mu[2]*normal
    arm = points_body-com
    lever = np.cross(arm[..., None, :], np.moveaxis(force, -1, -2))
    lever = np.moveaxis(lever, -2, -1)
    moment = lever+direct
    return np.concatenate((force, moment/radius), axis=-2), frame, force, direct, lever


def summary(x):
    return {'minimum': float(np.min(x)), 'median': float(np.median(x)),
            'mean': float(np.mean(x)), 'maximum': float(np.max(x))}


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    checks = {}
    errors = {}
    def check(name, value):
        checks[name] = bool(value)
    protocol_path = ROOT/'configs/full151-matched-material-cones-CPU-authorized-v1.json'
    protocol = read(protocol_path)
    config = read(RUN/'config.json')
    result = read(RUN/'report.json')
    rows = read(RUN/'frame_results.json')
    raw = arrays(RUN/'matrices_and_witnesses.npz')
    cues = arrays(protocol['cues']['path'])
    target = arrays(protocol['old_H123_targets']['path'])
    old = arrays(protocol['demand_matrices']['path'])
    material = read(protocol['material_contract']['path'])
    old_report_path = Path(protocol['source_pins']['demand_report']['path'])
    old_report = read(old_report_path)
    ref_path = Path(protocol['source_pins']['old_demand__reference']['path'])
    ref = arrays(ref_path, ['time_s', 'mapped_object_pose_w', 'object_rotation_w',
                           'object_a_world_m_s2', 'object_omega_world_rad_s',
                           'object_alpha_world_rad_s2'])
    context_path = Path(protocol['source_pins']['material_input__actual_context']['path'])
    context = arrays(context_path, ['body_mass', 'body_inertia'])
    input_pins = {k: protocol[k] for k in ('cues', 'old_H123_targets', 'demand_matrices', 'material_contract')}
    input_pins.update({k: protocol['source_pins'][k] for k in
                       ('demand_report', 'old_demand__reference', 'material_input__actual_context')})
    check('exact_authorized_config_SHA', pin(protocol_path)['sha256'] == '1fee2990715e6a5a2a7ae351ebf67ee41633b5bf014a10f71f0deb35ce0b17f8')
    check('saved_config_exact_authorized_JSON', config == protocol)
    check('relevant_input_pins_current_exact', all(pin(v['path'])['sha256'] == v['sha256'] for v in input_pins.values()))
    check('completed_302_not_partial', result['status'] == 'COMPLETED_MATCHED_CONE_CPU_SCREEN' and result['LP_calls'] == result['accounted_slots'] == 302 and raw['completed'].shape == (151, 2) and raw['completed'].all())
    check('frame_major_full151_two_conditions', len(rows) == 302 and all(r['rowid'] == i and r['frame'] == i//2 and r['condition'] == ('H123', 'H12345')[i%2] and r['time_s'] == float(cues['time_s'][i//2]) for i, r in enumerate(rows)))
    check('all_saved_statuses_successful_conditional_witnesses', all(r['status'] == 'FEASIBLE_CONDITIONAL_WITNESS' and r['solver_status'] == 0 for r in rows))
    check('full_masks_no_missing_geometry', raw['geometry_valid_mask'].dtype == np.bool_ and raw['geometry_valid_mask'].shape == (151, 5) and raw['geometry_valid_mask'].all() and np.array_equal(raw['geometry_valid_mask'], cues['geometry_valid_mask']))
    check('human_observation_labels_remain_unknown', not cues['human_contact_known'].any() and not cues['human_contact_present'].any() and cues['human_contact_unknown_mask'].all() and not cues['declared_robot_intent_known'].any())
    check('time_pose_source_common_exact', np.array_equal(cues['time_s'], target['time_s']) and np.array_equal(cues['time_s'], ref['time_s']) and np.array_equal(cues['mapped_object_pose_w'], target['mapped_object_pose_w']) and np.array_equal(cues['mapped_object_pose_w'], ref['mapped_object_pose_w']))
    com = target['object_com_offset_body_m']
    radius = float(target['object_nominal_enclosing_radius_m'])
    mu = np.asarray(material['friction5'], float)
    local = np.zeros((6, 6))
    local[0] = 1
    for component in range(3):
        local[component+1, 2*component] = mu[component]
        local[component+1, 2*component+1] = -mu[component]
    check('condim4_actual_six_normal_force_one_columns_exact', material['condim'] == 4 and material['model_cone'] == 0 and np.array_equal(local, np.asarray(material['A_local6x6'])) and np.array_equal(local, raw['local_template_contact6x6']))
    check('H123_source_geometry_exact_old_target_prefix', np.array_equal(cues['object_surface_closest_body_m'][:, :3]-com, target['point_com_body_m']) and np.array_equal(-cues['object_surface_outward_normal_body'][:, :3], target['inward_normal_body']))
    rebuilt, frame, force, direct, lever = build_all(cues['object_surface_closest_body_m'], cues['object_surface_outward_normal_body'], com, radius, mu)
    errors['frame_orthonormal_max'] = maximum(np.swapaxes(frame, -1, -2)@frame-np.eye(3))
    check('hypothetical_Frisvad_frames_proper_orthonormal', errors['frame_orthonormal_max'] < 1e-14 and np.min(np.linalg.det(frame)) > 1-1e-14)
    check('source_normals_unit_valid', maximum(np.linalg.norm(cues['object_surface_outward_normal_body'], axis=-1)-1) < 1e-8)
    check('torsion_columns_keep_normal_force_and_lever', np.array_equal(force[..., 4], frame[..., 0]) and np.array_equal(force[..., 5], frame[..., 0]) and np.array_equal(direct[..., 4], mu[2]*frame[..., 0]) and np.array_equal(direct[..., 5], -mu[2]*frame[..., 0]) and maximum(lever[..., 4]) > 0)
    matrices = {}
    for name, count in (('H123', 3), ('H12345', 5)):
        matrix = rebuilt[:, :count].transpose(0, 2, 1, 3).reshape(151, 6, count*6)
        matrices[name] = matrix
        errors[name+'_matrix_max_abs'] = maximum(matrix-raw['A_'+name+'_N_equivalent'])
        check(name+'_all151_matrices_rebuilt_with_direct_and_lever_torque', errors[name+'_matrix_max_abs'] < 1e-14)
    check('H123_exact_18_column_prefix_of_H12345', np.array_equal(raw['A_H123_N_equivalent'], raw['A_H12345_N_equivalent'][:, :, :18]))
    check('same_frozen_demand_RHS_exact_both_conditions', np.array_equal(raw['required_wrench_N_equivalent'], old['required_wrench_N_equivalent']))
    R = ref['object_rotation_w'][::4]
    errors['quaternion_R_vs_reference_R_max'] = maximum(rotation(ref['mapped_object_pose_w'][:, 3:])-R)
    check('reference_R_matches_mapped_object_pose', errors['quaternion_R_vs_reference_R_max'] < 1e-14)
    body = int(old_report['object_body_id'])
    mass = float(context['body_mass'][0, body])
    Iprincipal = context['body_inertia'][0, body].astype(float)
    QI = rotation(np.asarray(old_report['actual_inertia_quaternion_wxyz']))
    inertia = QI@np.diag(Iprincipal)@QI.T
    check('actual_mass_inertia_saved_context_binding', mass == float(old['actual_mass_kg']) == old_report['actual_mass_kg'] and np.array_equal(inertia, old['I_body_kg_m2']) and np.array_equal(Iprincipal, np.asarray(old_report['actual_inertia_diagonal_kg_m2'])))
    gravity = old['gravity_world_m_s2']
    wb = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), ref['object_omega_world_rad_s'][::4])
    ab = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), ref['object_alpha_world_rad_s2'][::4])
    needed_force = np.einsum('nij,nj->ni', R.transpose(0, 2, 1), mass*(ref['object_a_world_m_s2'][::4]-gravity))
    needed_moment = ab@inertia.T+np.cross(wb, wb@inertia.T)
    W = raw['required_wrench_N_equivalent']
    errors['analytic_force_max_N'] = maximum(needed_force-W[:, :3])
    errors['analytic_torque_max_Nm'] = maximum(needed_moment-W[:, 3:]*radius)
    check('analytic_designed_inertial_force_recomputed', errors['analytic_force_max_N'] < 1e-14)
    check('analytic_designed_body_Euler_moment_recomputed', errors['analytic_torque_max_Nm'] < 1e-14)
    check('omega_alpha_body_conventions_recomputed', maximum(wb-old['omega_body_rad_s']) < 1e-14 and maximum(ab-old['alpha_body_rad_s2']) < 1e-14)
    check('COM_and_radius_normalization_once_exact_source', np.array_equal(com, np.zeros(3)) and radius == old_report['radius_m'] and np.array_equal(W[:, :3], old['required_force_body_N']) and maximum(W[:, 3:]-old['required_moment_body_Nm']/radius) == 0)
    sums = {}
    residuals = {}
    decoded_totals = {}
    per_contact_forces = {}
    for ci, (name, count) in enumerate((('H123', 3), ('H12345', 5))):
        lam = raw['lambda_'+name+'_normal_N']
        A = raw['A_'+name+'_N_equivalent']
        check(name+'_all_raw_coefficients_finite_nonnegative_unclipped', lam.shape == (151, count*6) and np.isfinite(lam).all() and np.min(lam) >= 0)
        residual = np.einsum('nij,nj->ni', A, lam)-W
        residuals[name] = residual
        errors[name+'_max_scaled_primal_N'] = maximum(residual)
        errors[name+'_max_force_primal_N'] = maximum(residual[:, :3])
        errors[name+'_max_torque_primal_Nm'] = maximum(residual[:, 3:]*radius)
        check(name+'_all151_primal_at_original_1e7_gate', np.max(np.abs(residual), axis=1).max() <= protocol['equality_residual_gate_N'])
        sums[name] = lam.sum(axis=1)
        L = lam.reshape(151, count, 6)
        decoded = L@local.T
        budget = np.abs(decoded[..., 1])/mu[0]+np.abs(decoded[..., 2])/mu[1]+np.abs(decoded[..., 3])/mu[2]
        errors[name+'_decoded_Fn_sum_roundoff_N'] = maximum(decoded[..., 0]-L.sum(axis=-1))
        check(name+'_each_contact_shared_L1_sliding_torsion_budget', np.max(budget-decoded[..., 0]) <= 1e-14 and errors[name+'_decoded_Fn_sum_roundoff_N'] <= 1e-14 and not decoded[..., 4:].any())
        F = np.einsum('nkij,nkj->nki', frame[:, :count], decoded[..., :3])
        Tdirect = np.einsum('nkij,nkj->nki', frame[:, :count], decoded[..., 3:])
        Tlever = np.cross(cues['object_surface_closest_body_m'][:, :count]-com, F)
        netforce = F.sum(axis=1)
        netmoment = (Tlever+Tdirect).sum(axis=1)
        decoded_totals[name] = np.concatenate((netforce, netmoment), axis=1)
        per_contact_forces[name] = decoded[..., 0]
        errors[name+'_independent_contact_sum_force_N'] = maximum(netforce-W[:, :3])
        errors[name+'_independent_contact_sum_moment_Nm'] = maximum(netmoment-W[:, 3:]*radius)
        check(name+'_independent_contact_force_and_COM_moment_balance', errors[name+'_independent_contact_sum_force_N'] <= 1e-7 and errors[name+'_independent_contact_sum_moment_Nm'] <= radius*1e-7)
        row_errors = []
        for k in range(151):
            saved = rows[k*2+ci]
            row_errors.append(max(abs(saved['lambda_sum_normal_N']-sums[name][k]), abs(saved['lambda_min_N']-lam[k].min()), abs(saved['residual_inf_N_equivalent']-np.abs(residual[k]).max()), abs(saved['force_residual_inf_N']-np.abs(residual[k, :3]).max()), abs(saved['torque_residual_inf_Nm']-np.abs(residual[k, 3:]*radius).max())))
        errors[name+'_saved_row_metrics_max_abs'] = float(max(row_errors))
        check(name+'_all151_saved_scalar_metrics_recomputed', max(row_errors) < 1e-12)
    padded = np.pad(raw['lambda_H123_normal_N'], ((0, 0), (0, 12)))
    errors['padded_H123_normal_sum_roundoff_N'] = maximum(padded.sum(axis=1)-sums['H123'])
    check('H123_witness_embeds_unchanged_in_five_cone', maximum(np.einsum('nij,nj->ni', raw['A_H12345_N_equivalent'], padded)-W) <= 1e-7 and errors['padded_H123_normal_sum_roundoff_N'] <= 1e-14)
    difference = sums['H12345']-sums['H123']
    check('same_frame_minFn5_le_minFn3_at_fixed_1e7N_allowance', float(np.max(difference)) <= 1e-7)
    check('old_proxy_23_results_preserved_not_reinterpreted_as_added_fingers', old_report['frame_status_counts'] == {'FEASIBLE': 23, 'INFEASIBLE': 128, 'SOLVER_UNKNOWN': 0})
    check('no_solver_native_model_Env_Torch_Warp_or_WM_import', not any(n.split('.')[0] in ('scipy', 'mujoco', 'mujoco_warp', 'mjlab', 'torch', 'warp') for n in sys.modules))
    np.savez_compressed(OUT/'independent_numeric_evidence.npz',
        rebuilt_A_H123=matrices['H123'], rebuilt_A_H12345=matrices['H12345'],
        independent_force_demand_body_N=needed_force, independent_moment_demand_body_Nm=needed_moment,
        residual_H123_N_equivalent=residuals['H123'], residual_H12345_N_equivalent=residuals['H12345'],
        H123_sum_normal_N=sums['H123'], H12345_sum_normal_N=sums['H12345'],
        H12345_minus_H123_normal_N=difference,
        H123_per_finger_normal_N=per_contact_forces['H123'],
        H12345_per_finger_normal_N=per_contact_forces['H12345'],
        independent_contact_net_H123_force_N_moment_Nm=decoded_totals['H123'],
        independent_contact_net_H12345_force_N_moment_Nm=decoded_totals['H12345'])
    with (OUT/'normal_force_by_frame.csv').open('x', newline='', encoding='utf-8') as handle:
        csvwriter = csv.writer(handle)
        csvwriter.writerow(['frame', 'time_s', 'H123_solver_minFn_N', 'H12345_solver_minFn_N', 'five_minus_three_N', 'three_minus_five_fraction'])
        for k in range(151):
            csvwriter.writerow([k, cues['time_s'][k], sums['H123'][k], sums['H12345'][k], difference[k], -difference[k]/sums['H123'][k]])
    report = {
        'schema': 'full151_matched_material_cones_saved_numeric_independent:1',
        'status': 'PASS_SAVED_NUMERIC_AUDIT' if all(checks.values()) else 'FAIL_SAVED_NUMERIC_AUDIT',
        'checks': checks, 'passed': sum(checks.values()), 'total': len(checks),
        'source': pin(__file__), 'authorized_config': pin(protocol_path),
        'saved_run_report': pin(RUN/'report.json'),
        'raw_arrays': pin(RUN/'matrices_and_witnesses.npz'),
        'raw_frame_results': pin(RUN/'frame_results.json'), 'relevant_input_pins': input_pins,
        'errors': errors,
        'normal_force_N': {'H123': summary(sums['H123']), 'H12345': summary(sums['H12345']),
            'five_minus_three': summary(difference),
            'fractional_reduction': summary(-difference/sums['H123']),
            'strictly_lower_frames': int(np.sum(difference < -1e-7)),
            'equal_within_1e7N_frames': int(np.sum(np.abs(difference) <= 1e-7)),
            'frame0_H123': float(sums['H123'][0]), 'frame0_H12345': float(sums['H12345'][0])},
        'numeric_evidence': pin(OUT/'independent_numeric_evidence.npz'),
        'normal_force_csv': pin(OUT/'normal_force_by_frame.csv'),
        'limits': [
            '302 saved primal witnesses audited, not 302 new LP calls. Solver-reported optimality was not independently dual-certified; minFn names retain solver semantics.',
            'A minimum normal-force cone solution is an idealized model lower bound, not a measured hand force, force upper capacity, actuator balance or reachable grasp.',
            'Both matched H123 and H12345 are feasible at all151 frames. The earlier H123 mu.1/9-unitforce-rays/no-direct-torsion cone differs in representation and material; its23/151 versus151/151 cannot be attributed to adding fingers.',
            'Five versus three force reductions are conditional on the same hypothetical source surface points/normals and Frisvad frame; allocations can be nonunique, and finite-pyramid frame choice matters.',
            'Demand is the fixed-palm task-redesign analytic reference with known simulation mass/inertia, not observed video force or velocity.',
            'Human contacts remain unknown. No target/mask was changed; no geometry/contact/dynamics/physical qualification is promoted. Source/lifecycle archive review is a separate audit.'
        ],
        'new_LP_calls': 0, 'native_queries_or_physics': 0, 'Env_WM_GPU': False,
        'physical_reference_qualified': False, 'wall_seconds': time.perf_counter()-started,
    }
    with (OUT/'report.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status': report['status'], 'passed': report['passed'], 'total': report['total'], 'errors': errors, 'normal_force_N': report['normal_force_N'], 'report': pin(OUT/'report.json')}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
