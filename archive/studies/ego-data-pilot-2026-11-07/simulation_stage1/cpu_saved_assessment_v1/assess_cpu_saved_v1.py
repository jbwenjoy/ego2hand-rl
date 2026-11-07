"""Independent saved NumPy arithmetic only. No MuJoCo/physics/model imports.

Frozen scientific rules: protocol8981 G1-G7, no threshold selection. Quaternion
vector rotation independently checks native COM/Jacobian/energy captures. Contact
table min-reduction independently checks every saved depth/count. G8 is NOT_RUN.
"""
import collections
import csv
import datetime
import hashlib
import itertools
import json
import math
import pathlib
import traceback

import numpy as np

BASE = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
PROTOCOL = BASE / 'protocols/phone_physics_stage1_v1.json'
PRODUCER = BASE / 'simulation_stage1/cpu_v1/result.json'
BINDING = BASE / 'simulation_stage1/execution_binding_cpu_v1.json'
EXPECTED_PROTOCOL = '8981e89aec0993887aad65db1722f80d27470d6553a7bd25760a6ba7556d0505'
EXPECTED_SOURCE = '3e3303f09d710c5d9de9e61324fcfc84776cb648b5d8cf3a8d32972953aa858c'


def pin(p):
    p = pathlib.Path(p).resolve()
    h = hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def check_pin(v):
    actual = pin(v['path'])
    assert actual['sha256'] == v['sha256'] and actual['bytes'] == v['bytes'], v['path']
    assert pathlib.Path(actual['path']).samefile(v['path'])
    return actual


def read(p):
    return json.loads(pathlib.Path(p).read_text(encoding='utf-8-sig'))


def write(p, value):
    with pathlib.Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def status(v):
    return 'PASS' if bool(v) else 'FAIL'


def close(a, b, atol=2e-11):
    return bool(np.allclose(a, b, rtol=2e-11, atol=atol, equal_nan=False))


def qvec(q, v):
    """Quaternion vector rotation, independent of producer matrix/Jacobian path."""
    q, v = np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64)
    u, w = q[..., 1:], q[..., :1]
    return v + 2*w*np.cross(u, v) + 2*np.cross(u, np.cross(u, v))


def quat_angle(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return 2*np.arccos(np.clip(np.abs(np.sum(a*b, axis=-1)), 0, 1))


def rms_vec(a):
    return float(np.sqrt(np.mean(np.sum(np.asarray(a, dtype=np.float64)**2, axis=-1))))


def force_schedule(test, n, dt):
    f = np.zeros(n+1, dtype=np.float64)
    t = np.arange(n, dtype=np.float64)*dt
    if test == 'force_push_release':
        f[1:] = np.where((t >= 1.0) & (t < 1.2), 2.0, 0.0)
    return f


def passive_energy(E, dt, test):
    windows = [(0.0, 3.0)] if test != 'force_push_release' else [(0.0, 1.0), (1.2, 3.0)]
    rows = []
    for lo, hi in windows:
        a, b = round(lo/dt), round(hi/dt)
        e0 = float(E[a])
        gain = float(np.max(E[a:b+1]-e0))
        limit = max(0.0005, 0.10*max(abs(e0), 0.005))
        rows.append({'interval_s': [lo, hi], 'indices_inclusive': [a, b],
                     'E_start_J': e0, 'maximum_positive_energy_gain_J': gain,
                     'limit_J': limit, 'status': status(gain <= limit)})
    return rows


def focused_closed_form():
    q = np.array([math.sqrt(.5), math.sqrt(.5), 0., 0.])
    center = np.array([0., .01, 0.])
    off = qvec(q, center)
    world_omega = qvec(q, [1., 0., 0.])
    v = np.array([.1, .2, .3]) + np.cross(world_omega, off)
    assert close(off, [0., 0., .01], 1e-14)
    assert close(v, [.1, .19, .3], 1e-14)
    assert np.count_nonzero(force_schedule('force_push_release', 1200, .0025)) == 80
    return {'quaternion_COM_offset_and_cross_sign': True, 'half_open_force_ticks': True,
            'no_model_or_physics': True}


def evaluate_case(c, protocol, binding):
    trace_pin = check_pin(c['trajectory'])
    with np.load(trace_pin['path'], allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    dt, mass, mu, test = c['dt_s'], c['mass_kg'], c['sliding_friction'], c['test']
    n = round(3.0/dt)
    shapes = {'qpos': (n+1, 7), 'qvel': (n+1, 6), 'com_world_m': (n+1, 3),
              'com_velocity_world_m_s': (n+1, 3), 'angular_velocity_world_rad_s': (n+1, 3),
              'physical_energy_U_T_J': (n+1, 2), 'energy_potential_kinetic_J': (n+1, 2),
              'time_s': (n+1,), 'tick': (n+1,), 'tick_time_s': (n+1,),
              'penetration_m': (n+1,), 'contact_count': (n+1,),
              'applied_com_force_world_x_N': (n+1,)}
    assert all(d[k].shape == shape for k, shape in shapes.items())
    assert d['contact_long_table'].ndim == 2 and d['contact_long_table'].shape[1] == 15
    assert np.array_equal(d['tick'], np.arange(n+1)) and d['tick'].dtype.kind in 'iu'
    assert close(d['time_s'], d['tick']*dt, 1e-9)
    assert np.array_equal(d['tick_time_s'], d['tick']*dt)
    finite = all(np.isfinite(v).all() for v in d.values())
    assert finite
    opt = c['compiled_model_options']
    assert opt['nu'] == opt['neq'] == opt['ntendon'] == 0
    names = opt['geom_names']; plane, box = names.index('bench_support'), names.index('phone24_collision_aabb')
    assert opt['geom_condim'][plane] == opt['geom_condim'][box] == 3
    assert opt['option']['cone'] == 0 and opt['option']['integrator'] == 3 and opt['option']['solver'] == 2
    assert close(opt['option']['gravity'], [0., 0., -9.81]) and opt['option']['timestep'] == dt
    assert opt['option']['density'] == opt['option']['viscosity'] == 0
    assert all(x == 0 for x in opt['body_gravcomp'])
    assert opt['body_mass'] == [0., mass]
    for k in ['geom_friction', 'geom_priority', 'geom_solmix', 'geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap']:
        assert opt[k][plane] == opt[k][box], k
    assert opt['geom_friction'][box] == [mu, .005, .0001]
    expect_options = binding['nominal_compiled_options']['option'].copy()
    expect_options['timestep'] = dt
    assert opt['option'] == expect_options
    center = np.array(opt['body_ipos'][1], dtype=np.float64)
    assert close(center, binding['COM_body_m'])
    half = np.array(binding['nominal_box_half_m'], dtype=np.float64)
    full = 2*half
    expected_I = np.array([mass*math.fsum(float(full[j])**2 for j in range(3) if j != i)/12 for i in range(3)])
    assert close(opt['body_inertia'][1], expected_I, 1e-15)
    q, v = d['qpos'], d['qvel']
    qnorm = np.linalg.norm(q[:, 3:], axis=1)
    assert np.all(qnorm > 0)
    offset = qvec(q[:, 3:], center)
    world_omega = qvec(q[:, 3:], v[:, 3:])
    com = q[:, :3]+offset
    com_v = v[:, :3]+np.cross(world_omega, offset)
    iquat = np.array(opt['body_iquat'][1], dtype=np.float64)
    inverse_iq = iquat*np.array([1., -1., -1., -1.])
    omega_i = qvec(inverse_iq, v[:, 3:])
    T = .5*mass*np.sum(com_v**2, axis=1)+.5*np.sum(expected_I*omega_i**2, axis=1)
    U = mass*9.81*com[:, 2]
    E = T+U
    errs = {'COM_position_maxabs_m': float(np.max(np.abs(com-d['com_world_m']))),
            'COM_velocity_maxabs_m_s': float(np.max(np.abs(com_v-d['com_velocity_world_m_s']))),
            'world_omega_maxabs_rad_s': float(np.max(np.abs(world_omega-d['angular_velocity_world_rad_s']))),
            'physical_UT_maxabs_J': float(np.max(np.abs(np.column_stack((U,T))-d['physical_energy_U_T_J']))),
            'raw_quaternion_norm_maxabs_from_one': float(np.max(np.abs(qnorm-1)))}
    assert close(com, d['com_world_m']) and close(com_v, d['com_velocity_world_m_s'])
    assert close(world_omega, d['angular_velocity_world_rad_s']) and close(np.column_stack((U,T)), d['physical_energy_U_T_J'])
    clearance = .1 if test == 'drop' else .0005
    assert close(com[0], [0,0,clearance+half[1]], 1e-14)
    assert close(q[0,3:], [math.sqrt(.5),math.sqrt(.5),0,0], 1e-14) and np.array_equal(v[0], np.zeros(6))
    cl = d['contact_long_table']
    assert np.array_equal(cl[:, [0,2,3,4]], np.trunc(cl[:, [0,2,3,4]]))
    ticks = cl[:,0].astype(np.int64)
    assert np.all((ticks >= 0) & (ticks <= n))
    assert np.all(np.isin(cl[:,3:5], [plane, box])) and np.all(cl[:,3] != cl[:,4])
    assert close(cl[:,1], ticks*dt, 1e-9)
    counts = np.bincount(ticks, minlength=n+1)
    assert np.array_equal(counts, d['contact_count'])
    offsets = np.r_[0, np.cumsum(counts)]
    assert np.array_equal(cl[:,2], np.arange(len(cl))-np.repeat(offsets[:-1], counts))
    minimum = np.full(n+1, np.inf)
    np.minimum.at(minimum, ticks, cl[:,5])
    penetration = np.maximum(0., -minimum)
    assert np.array_equal(penetration, d['penetration_m'])
    first_contact = float(np.flatnonzero(counts > 0)[0]*dt) if np.any(counts > 0) else None
    expected_force = force_schedule(test, n, dt)
    assert np.array_equal(expected_force, d['applied_com_force_world_x_N'])
    impulse = float(math.fsum(expected_force[1:])*dt)
    expected_impulse = .4 if test == 'force_push_release' else 0.
    force_limit = 16*np.finfo(expected_force.dtype).eps*max(1.,abs(expected_impulse))
    force_ok = abs(impulse-expected_impulse) <= force_limit
    work = np.r_[0., np.cumsum(expected_force[1:]*np.diff(com[:,0]))]
    rest = np.arange(n+1) >= round(2.5/dt)
    rest_v, rest_w = com_v[rest], world_omega[rest]
    m = {'completed_steps': n, 'state_rows': n+1, 'contact_rows': len(cl),
         'max_penetration_m': float(penetration.max()), 'rest_max_penetration_m': float(penetration[rest].max()),
         'rest_linear_speed_rms_m_s': rms_vec(rest_v), 'rest_angular_speed_rms_rad_s': rms_vec(rest_w),
         'rest_linear_speed_max_m_s': float(np.linalg.norm(rest_v,axis=1).max()),
         'rest_angular_speed_max_rad_s': float(np.linalg.norm(rest_w,axis=1).max()),
         'first_generated_contact_time_s': first_contact,
         'rest_contact_observed': bool(np.any(counts[rest] > 0)),
         'empty_generated_contact_state_rows': int(np.count_nonzero(counts == 0)),
         'force_impulse_Ns': impulse, 'expected_force_impulse_Ns': expected_impulse,
         'force_impulse_abs_error_Ns': abs(impulse-expected_impulse), 'force_impulse_roundoff_limit_Ns': force_limit,
         'active_force_intervals': int(np.count_nonzero(expected_force[1:])),
         'initial_energy_J': float(E[0]), 'final_energy_J': float(E[-1]), 'max_energy_J': float(E.max()),
         'external_work_J': float(work[-1]), 'final_energy_minus_initial_minus_work_J': float(E[-1]-E[0]-work[-1]),
         'minimum_kinetic_energy_J': float(T.min()), 'native_engine_vs_physical_energy_maxabs_J': float(np.max(np.abs(d['energy_potential_kinetic_J']-np.column_stack((U,T))))),
         'kinematic_arithmetic_checks': errs, 'passive_energy_windows': passive_energy(E,dt,test)}
    warnings_ok = all(int(x) == 0 for x in c['metrics']['warning_counts']) and not c['metrics']['warning_history']
    g3 = m['max_penetration_m'] <= .002 and m['rest_max_penetration_m'] <= .0005
    g4 = (m['rest_linear_speed_rms_m_s'] <= .01 and m['rest_linear_speed_max_m_s'] <= .02
          and m['rest_angular_speed_rms_rad_s'] <= .1 and m['rest_angular_speed_max_rad_s'] <= .2)
    gates = {'G1_identity_and_coverage': 'PASS',
             'G2_finite_and_solver': status(finite and np.all(qnorm > 0) and T.min() >= -1e-12 and warnings_ok and c['metrics']['error'] is None),
             'G3_penetration': status(g3), 'G4_rest_motion': status(g4),
             'G5_energy': status(all(w['status']=='PASS' for w in m['passive_energy_windows'])),
             'G6_contacts_and_input': status(m['rest_contact_observed'] and force_ok),
             'G8_CPU_Warp_parity': 'NOT_RUN'}
    producer_checks = {
        'depth': close(m['max_penetration_m'],c['metrics']['max_penetration_m']),
        'rest_depth': close(m['rest_max_penetration_m'],c['metrics']['rest_max_penetration_m']),
        'rest_v_rms': close(m['rest_linear_speed_rms_m_s'],c['metrics']['rest_linear_speed_rms_m_s']),
        'rest_w_rms': close(m['rest_angular_speed_rms_rad_s'],c['metrics']['rest_angular_speed_rms_rad_s']),
        'rest_v_max': close(m['rest_linear_speed_max_m_s'],c['metrics']['rest_linear_speed_max_m_s']),
        'rest_w_max': close(m['rest_angular_speed_max_rad_s'],c['metrics']['rest_angular_speed_max_rad_s']),
        'initial_final_max_energy': close([m['initial_energy_J'],m['final_energy_J'],m['max_energy_J']], [c['metrics'][k] for k in ['initial_energy_J','final_energy_J','max_energy_J']]),
        'known_work': close(m['external_work_J'],c['metrics']['known_external_work_J']),
        'input_impulse': close(impulse,c['metrics']['force_impulse_N_s']),
        'completed': c['metrics']['completed_steps']==c['metrics']['expected_steps']==n}
    assert all(producer_checks.values()), (c['case'],producer_checks)
    # Retain full arrays in memory for paired comparisons; raw originals remain immutable.
    data = {'qpos':q, 'com':com, 'com_v':com_v, 'omega':world_omega, 'first_contact':first_contact}
    row = {k:c[k] for k in ['case','test','mass_kg','sliding_friction','dt_s']}
    row.update(trajectory=trace_pin, model_xml_sha256=c['model_xml_sha256'],metrics=m,gates=gates,
               warning_counts=c['metrics']['warning_counts'],warning_history=c['metrics']['warning_history'],
               producer_metric_checks=producer_checks)
    return row, data


def main():
    if (OUT/'report.json').exists() or (OUT/'failure.json').exists():
        raise FileExistsError('Fresh saved-assessment output required; no automatic overwrite/retry')
    fixture = focused_closed_form()
    p, r, binding = read(PROTOCOL), read(PRODUCER), read(BINDING)
    assert pin(PROTOCOL)['sha256']==EXPECTED_PROTOCOL and r['source']['sha256']==EXPECTED_SOURCE
    binding_pins = [check_pin(x) for x in binding['pins']]
    assert binding['integration_steps_executed_at_binding']==0
    assert datetime.datetime.fromisoformat(binding['created_UTC']) < datetime.datetime.fromisoformat(r['created_UTC'])
    assert r['schema']=='phone_free_object_cpu_bench:1'
    for k in ['protocol','source','asset','visual_mesh']:
        check_pin(r[k])
    assert pathlib.Path(r['protocol']['path']).samefile(PROTOCOL)
    assert r['case_count']==len(r['cases'])==54 and r['step_count']==97200
    assert r['extra_forward_calls']==97200 and r['initial_state_forward_calls']==54
    expected=set(itertools.product([.1,.2,.3],[.3,.6,.9],['drop','table_settling','force_push_release'],[.0025,.00125]))
    assert {(c['mass_kg'],c['sliding_friction'],c['test'],c['dt_s']) for c in r['cases']}==expected
    assert all(c['mass_kg']==.2 and c['sliding_friction']==.6 for c in r['cases'][:6])
    rows, data = [], {}
    for c in r['cases']:
        row, d = evaluate_case(c,p,binding)
        rows.append(row); data[(c['mass_kg'],c['sliding_friction'],c['test'],c['dt_s'])]=d
    print(json.dumps({'status':'ALL54_RAW_RECOMPUTED',
        'max_penetration_m':max(x['metrics']['max_penetration_m'] for x in rows),
        'maximum_rest_v_RMS_m_s':max(x['metrics']['rest_linear_speed_rms_m_s'] for x in rows),
        'maximum_rest_omega_RMS_rad_s':max(x['metrics']['rest_angular_speed_rms_rad_s'] for x in rows)},allow_nan=False),flush=True)
    pairs=[]
    limits=p['engineering_gates_predeclared_not_measured']['G7_timestep_pair']
    for mass,mu,test in itertools.product([.1,.2,.3],[.3,.6,.9],['drop','table_settling','force_push_release']):
        a,b=data[(mass,mu,test,.0025)],data[(mass,mu,test,.00125)]
        assert len(b['com'])==2*len(a['com'])-1
        v={'final_COM_position_difference_m':float(np.linalg.norm(a['com'][-1]-b['com'][-1])),
           'final_orientation_difference_deg':float(np.degrees(quat_angle(a['qpos'][-1,3:],b['qpos'][-1,3:]))),
           'final_COM_velocity_vector_difference_m_s':float(np.linalg.norm(a['com_v'][-1]-b['com_v'][-1])),
           'final_angular_velocity_vector_difference_rad_s':float(np.linalg.norm(a['omega'][-1]-b['omega'][-1])),
           'shared_tick_COM_trajectory_RMS_difference_m':rms_vec(a['com']-b['com'][::2]),
           'first_contact_time_difference_s':abs(a['first_contact']-b['first_contact']) if a['first_contact'] is not None and b['first_contact'] is not None else None}
        checks={k:(v[k] is not None and v[k]<=limits[k]) for k in v}
        ps=status(all(checks.values()))
        pairs.append({'mass_kg':mass,'sliding_friction':mu,'test':test,'metrics':v,'checks':checks,'status':ps})
        for row in rows:
            if (row['mass_kg'],row['sliding_friction'],row['test'])==(mass,mu,test):
                row['gates']['G7_timestep_pair']=ps
    geometry=read(BASE/'asset_stage1/aabb_geometry_audit_v1/report.json')
    geometry_raw=read(BASE/'asset_stage1/geometry_report.json')
    assert geometry['predeclared_engineering_limits_m']==p['geometry_error_report']['predeclared_engineering_limits_m']
    old_geo_protocol_pin=geometry['source_pins']['protocol']
    # Its source pin predates final protocol prose/metadata. Preserve mismatch,
    # while checking the actually consumed three geometry thresholds separately.
    geo_current_pin_matches=pin(old_geo_protocol_pin['path'])['sha256']==old_geo_protocol_pin['sha256']
    corner_max=max(geometry_raw['collision_proxy']['eight_box_corners_to_original_triangle_surface_m'])
    geo_metrics={'maximum_scanned_vertex_outside_box':geometry['maximum_original_vertex_outside_box_m'],
                 'sampled_box_to_scan_p95':geometry['sampled_box_faces_to_original_triangles']['p95_absolute_m'],
                 'sampled_box_to_scan_max':geometry['sampled_box_faces_to_original_triangles']['maximum_m']}
    geo_checks={k:geo_metrics[k]<=p['geometry_error_report']['predeclared_engineering_limits_m'][k] for k in geo_metrics}
    assert corner_max==geo_metrics['sampled_box_to_scan_max']
    gate_counts={g:dict(collections.Counter(x['gates'][g] for x in rows)) for g in ['G1_identity_and_coverage','G2_finite_and_solver','G3_penetration','G4_rest_motion','G5_energy','G6_contacts_and_input','G7_timestep_pair','G8_CPU_Warp_parity']}
    cpu_pass=all(all(x['gates'][g]=='PASS' for g in gate_counts if not g.startswith('G8')) for x in rows)
    audit_checks={'binding_and_direct_input_SHA_samefile':True,'complete54_cases_and97200_steps':True,
                  'all54_raw_arrays_shapes_ticks_and_initial_states':True,'all_original_contact_rows_indices_counts_distances':True,
                  'all54_independent_COM_rotation_velocity_energy_arithmetic':True,'all54_producer_scalar_recomputed_checks':True,
                  'all27_timestep_pairs_retained':True,'geometry_limits_match_final_protocol':True,
                  'historical_geometry_protocol_pin_difference_explicitly_preserved':True,'scientific_failures_and_Warp_NOT_RUN_preserved':True}
    report={'schema':'phone_physics_stage1_CPU_saved_independent:1',
            'status':'COMPLETED_SAVED_CPU_ASSESSMENT_WITH_ENGINEERING_FAILURES' if not cpu_pass else 'COMPLETED_SAVED_CPU_ASSESSMENT_CPU_GATES_PASS',
            'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'auditor':pin(pathlib.Path(__file__)),'protocol_pin':pin(PROTOCOL),'producer_report':pin(PRODUCER),
            'execution_manifest_pin':pin(BINDING),'binding_pins':binding_pins,'focused_fixture':fixture,
            'checks':audit_checks,'passed':sum(audit_checks.values()),'total':len(audit_checks),'audit_integrity_qualified':True,
            'completed_counts':{'CPU_cases':54,'CPU_integration_steps':97200,'CPU_saved_state_rows':sum(x['metrics']['state_rows'] for x in rows),
                                'CPU_contact_rows':sum(x['metrics']['contact_rows'] for x in rows),'CPU_timestep_pairs':27,
                                'CPU_extra_post_forward_calls':97200,'CPU_initial_forward_calls':54,'Warp_cases_assessed':0},
            'gate_case_counts':gate_counts,'case_results':rows,'timestep_comparisons':pairs,
            'geometry':{'report_pin':pin(BASE/'asset_stage1/aabb_geometry_audit_v1/report.json'),
                        'asset_report_pin':pin(BASE/'asset_stage1/geometry_report.json'),'metrics_m':geo_metrics,'checks':geo_checks,
                        'status':status(all(geo_checks.values())),'historical_protocol_pin':old_geo_protocol_pin,
                        'historical_protocol_current_SHA_equal':geo_current_pin_matches,'three_numeric_geometry_limits_equal_to_final_protocol':True,
                        'missing_historical_protocol_bytes_reconstructed':False,
                        'scope':'Saved geometry diagnostics delegated; no new distance/mesh computation. Historical draft pin mismatch preserved, not reported as immutable SHA success.'},
            'geometry_qualified':bool(all(geo_checks.values())),'CPU_bench_qualified':cpu_pass,
            'Warp_bench_status':'NOT_RUN','backend_parity_status':'NOT_RUN','backend_parity_qualified':False,
            'global_G1_two_backend_coverage_status':'NOT_RUN','stage1_qualified':False,
            'real_parameter_calibrated':False,'ego_retargeting_qualified':False,'hand_task_qualified':False,
            'source_and_saved_limitations':['Saved warnings/call counts and compiled options are source-backed runtime captures, not a new native API interceptor.',
                 'Native POST contact forces are diagnostic under just-ended interval inputs, not solver-PRE impulses or independent world-force reconstruction; no contact frame saved.',
                 'Empty generated contacts have observed-candidate depth0, no claim of global clearance.',
                 'Mass, COM, inertia and material priors remain unmeasured. No best-prior selection or threshold adjustment.',
                 'All inference here is saved stdlib/NumPy arithmetic; no simulator/model/NN/geometry query or backend execution.']}
    write(OUT/'all54_case_assessments.json',rows)
    write(OUT/'all27_timestep_pairs.json',pairs)
    with (OUT/'all54_cases.csv').open('x',newline='',encoding='utf-8') as f:
        keys=['case','test','mass_kg','sliding_friction','dt_s','max_penetration_m','rest_max_penetration_m',
              'rest_linear_speed_rms_m_s','rest_angular_speed_rms_rad_s','rest_linear_speed_max_m_s','rest_angular_speed_max_rad_s']+list(gate_counts)
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for x in rows:
            y={k:x[k] for k in keys if k in x};y.update({k:x['metrics'][k] for k in keys if k in x['metrics']});y.update(x['gates']);w.writerow(y)
    report['output_pins']={name:pin(OUT/name) for name in ['all54_case_assessments.json','all27_timestep_pairs.json','all54_cases.csv']}
    write(OUT/'report.json',report)
    print(json.dumps({'report':pin(OUT/'report.json'),'audit_checks':f"{report['passed']}/{report['total']}",
                      'gate_case_counts':gate_counts,'CPU_bench_qualified':cpu_pass,
                      'maximums':{k:max(x['metrics'][k] for x in rows) for k in ['max_penetration_m','rest_linear_speed_rms_m_s','rest_angular_speed_rms_rad_s']},
                      'failed_timestep_pairs':[x for x in pairs if x['status']=='FAIL']},indent=2,allow_nan=False))


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        failure={'schema':'phone_physics_stage1_CPU_saved_independent_failure:1','error':repr(e),
                 'traceback':traceback.format_exc(),'auditor':pin(pathlib.Path(__file__)),'physics_executed':False}
        if not (OUT/'failure.json').exists():
            write(OUT/'failure.json',failure)
        raise
