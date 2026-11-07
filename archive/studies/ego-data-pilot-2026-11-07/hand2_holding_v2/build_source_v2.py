"""Source-only isolated adapter construction; never imports the native backend."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py'
TARGET = ROOT/'run_cpu_support_holding_v2.py'
old = OLD.read_text(encoding='utf8')
prefix = old[:old.index('def main():')]
prefix = prefix.replace('Bounded, explicit-contact CPU holding experiments; no training or object aid.',
    'Bounded v4 support-candidate CPU holding diagnostic; no training or object aid.')
prefix = prefix.replace('import json\n', 'import json\nimport math\n')

adapter = '''SCHEMA = 'hand2_phone_support_CPU_holding_protocol:2'
PAIR_GATES = {
    'final_COM_difference_m': .002,
    'final_orientation_difference_rad': .03490658503988659,
    'require_both_duration_complete': True,
}


def finite_vector(x, size):
    return (isinstance(x, list) and len(x) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in x))


def metadata_preflight(protocol_path):
    protocol = read(protocol_path)
    if protocol['schema'] != SCHEMA:
        raise ValueError('New support-diagnostic protocol schema')
    if protocol['gates'] != GATES or protocol['dt_s'] != DTS or protocol['duration_s'] != DURATION:
        raise ValueError('Unchanged original physical budget/gates')
    if protocol['dt_pair_gates'] != PAIR_GATES:
        raise ValueError('Predeclared saved endpoint dt comparison')
    if type(protocol['CPU_run_authorized']) is not bool or protocol['training_or_GPU_authorized'] is not False:
        raise ValueError('Explicit CPU-only authorization boundary')
    if set(protocol['inputs']) != {'closure_protocol', 'closure_report', 'model_binding', 'model_mjb'}:
        raise ValueError('Exact four semantic inputs')
    for rec in [*protocol['inputs'].values(), *protocol['source_pins'].values()]:
        checked(rec)
    if protocol['source_pins']['runner'] != pin(__file__):
        raise ValueError('Current runner')
    old_runner = ROOT.parent/'hand2_holding_v1/run_cpu_holding_v1.py'
    if protocol['source_pins']['old_holding_runner'] != pin(old_runner):
        raise ValueError('Unchanged original physics loop authority')
    output = Path(protocol['output'])
    if not output.is_absolute() or output.exists() or output.parent.resolve() != ROOT.resolve():
        raise ValueError('Absolute fresh output inside holding_v2')
    closure = read(checked(protocol['inputs']['closure_report']))
    cp = read(checked(protocol['inputs']['closure_protocol']))
    if (closure.get('schema') != 'fixed_phone_joint_closure_result:4'
            or closure.get('status') != 'COMPLETED_STATIC_PHONE_FIXED_GEOMETRY_POLISH'
            or closure.get('protocol') != protocol['inputs']['closure_protocol']
            or closure.get('integration_training_GPU') != 0
            or cp.get('schema') != 'fixed_phone_joint_closure_protocol:4'
            or cp.get('training_GPU_physics_authorized') is not False):
        raise ValueError('Completed matching v4 static result, not old qualification')
    for rec in [*cp['inputs'].values(), *cp['source_pins'].values()]:
        checked(rec)
    binding = read(checked(protocol['inputs']['model_binding']))
    if (binding['model'] != protocol['inputs']['model_mjb']
            or binding['source'] != protocol['source_pins']['scene']):
        raise ValueError('Same original explicit-pair compiled model')
    results = closure['seed_results']
    if len(results) != 2 or sorted(v['seed'] for v in results) != [0, 1]:
        raise ValueError('Both original v4 seeds retained exactly once')
    variants, screening = [], []
    for seed in results:
        reasons = []
        for key in ('physical_geometry_pass', 'eligible_for_bounded_dynamic_diagnostic',
                    'proposed_ctrl_reachable_without_clamping'):
            if seed.get(key) is not True:
                reasons.append(key+' not True')
        if seed.get('unknown') != []:
            reasons.append('unknown evidence not empty')
        lp = seed.get('support_LP', {})
        if (lp.get('status') != 'OPTIMAL_PRIMARY_MOMENT_RESIDUAL'
                or lp.get('feasible_force_balance') is not True
                or lp.get('solver_status') != 0):
            reasons.append('primary force-hard LP not valid/optimal')
        if lp.get('normal_cap_N') != [3., 3.]:
            reasons.append('original two 3N normal caps required')
        if (not finite_vector(lp.get('force_balance_residual_world_N'), 3)
                or max(map(abs, lp.get('force_balance_residual_world_N', [math.inf]))) > 1e-8):
            reasons.append('force equality residual missing/outside original LP bound')
        if (not finite_vector(lp.get('normal_cap_slack_N'), 2)
                or min(lp.get('normal_cap_slack_N', [-math.inf])) < -1e-9):
            reasons.append('normal cap hard bound not verified')
        if not finite_vector(seed.get('proposed_ctrl20'), 20):
            reasons.append('finite proposed ctrl20 missing')
        if (type(seed.get('actual_COM_below_dot_m2_s2')) not in (int, float)
                or not math.isfinite(seed['actual_COM_below_dot_m2_s2'])
                or seed['actual_COM_below_dot_m2_s2'] < 0):
            reasons.append('actual contact line COM-below evidence missing')
        state = seed.get('verification_state', {})
        if not finite_vector(state.get('qpos27'), 27):
            reasons.append('complete verified qpos27 missing')
        selected = seed.get('selected_state')
        if not isinstance(selected, dict):
            reasons.append('selected state pin missing')
        else:
            checked(selected)
            if Path(selected['path']).parent.resolve() != Path(protocol['inputs']['closure_report']['path']).parent.resolve():
                raise ValueError('Selected state outside matching closure run')
        # No test of strict_support_proposal_pass or the old 1e-4Nm moment gate.
        # Residual is preserved; only actual unassisted dynamics can test holding.
        screen = {'seed': seed['seed'], 'enrolled': not reasons, 'reasons': reasons,
                  'strict_support_proposal_pass': seed.get('strict_support_proposal_pass'),
                  'moment_infinity_Nm': lp.get('moment_infinity_Nm'),
                  'selected_state': selected}
        screening.append(screen)
        if not reasons:
            variants.append({'name': 'v4_seed'+str(seed['seed']), 'seed': seed['seed'],
                'reachable': True, 'ctrl20': seed['proposed_ctrl20'],
                'candidate_state': selected, 'verified_qpos27': state['qpos27'],
                'moment_infinity_Nm': lp.get('moment_infinity_Nm'),
                'strict_support_proposal_pass': seed.get('strict_support_proposal_pass')})
    if closure.get('bounded_dynamic_diagnostic_eligible_count') != sum(v.get('eligible_for_bounded_dynamic_diagnostic') is True for v in results):
        raise ValueError('Consistent full seed enrollment count')
    return protocol, output, binding, variants, screening


def dt_pair_summary(variants, cases, endpoints, np):
    pairs = []
    for variant in variants:
        both = [v for v in cases if v['name'] == variant['name']]
        if len(both) != 2 or sorted(v['dt_s'] for v in both) != sorted(DTS):
            raise ValueError('Each enrolled seed must retain both dt results')
        e = sorted(endpoints[variant['name']], key=lambda v: v['dt_s'], reverse=True)
        quat = [np.asarray(v['final_quaternion_wxyz'], dtype=float) for v in e]
        norms = [float(np.linalg.norm(v)) for v in quat]
        if not all(math.isfinite(v) and v > 0 for v in norms):
            raise ValueError('Finite nonzero terminal quaternion')
        unit = [v/n for v,n in zip(quat,norms)]
        angle = float(2*np.arccos(np.clip(abs(unit[0]@unit[1]), 0, 1)))
        drift = float(np.linalg.norm(np.asarray(e[0]['final_COM_world_m'])-e[1]['final_COM_world_m']))
        complete = all(v['gates']['duration_complete'] for v in both)
        gates = {'both_duration_complete': complete,
                 'final_COM_difference_m': drift <= PAIR_GATES['final_COM_difference_m'],
                 'final_orientation_difference_rad': angle <= PAIR_GATES['final_orientation_difference_rad']}
        pairs.append({'name': variant['name'], 'seed': variant['seed'],
            'comparison_status': 'TWO_COMPLETE_2S_ENDPOINTS' if complete else 'INCOMPLETE_DURATION_NOT_2S_CONSISTENCY',
            'final_COM_difference_m': drift, 'final_orientation_difference_rad': angle,
            'terminal_quaternions_raw_wxyz': [v.tolist() for v in quat],
            'terminal_quaternion_norms_F64': norms,
            'orientation_observer': 'F64 endpoint normalization for pair comparison only; no state repair',
            'endpoints': e, 'gates': gates, 'dt_pair_consistency_pass': all(gates.values()),
            'both_original_case_gates_pass': all(v['nominal_holding_dynamics_pass'] for v in both)})
    return pairs


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--protocol', type=Path, required=True)
    a.add_argument('--run', action='store_true')
    args = a.parse_args()
    protocol, output, binding, variants, screening = metadata_preflight(args.protocol)
    if not args.run:
        print(json.dumps({'status': 'METADATA_VALID_NO_MODEL_OR_STEPS', 'screening': screening,
                          'variants': [v['name'] for v in variants],
                          'maximum_steps': int(sum(DURATION/dt for dt in DTS)*len(variants))}))
        return
    if protocol['CPU_run_authorized'] is not True:
        raise ValueError('Root explicit CPU run authorization required')
    output.mkdir()
    write(output/'protocol.json', protocol)
    shutil.copyfile(__file__, output/'execution_source.py')
    started = time.perf_counter()
    counts = {'MJB_reads': 0, 'MjData': 0, 'initial_forward': 0, 'integration_steps': 0,
              'contact_force_reads': 0, 'neural_training_GPU': 0}
    if not variants:
        write(output/'report.json', {'schema': 'hand2_phone_support_CPU_holding_result:2',
            'status': 'BLOCKED_NO_V4_SEED_ELIGIBLE_NO_MODEL_OR_STEPS', 'protocol': pin(args.protocol),
            'screening': screening, 'counts': counts, 'cases': [], 'dt_pair_consistency': [],
            'all_original_cases_and_dt_pairs_pass': False, 'holding_or_ego_policy_qualified': False})
        return
    phase = 'backend_import'
'''

tail = old[old.index('    try:\n        import mujoco'):]
remove = '''        with np.load(checked(protocol['inputs']['candidate_state']), allow_pickle=False) as z:
            q0 = z['qpos27'].copy()
            if np.any(z['qvel26']):
                raise ValueError('Static zero-speed start')
'''
assert tail.count(remove) == 1
tail = tail.replace(remove, '')
tail = tail.replace('        COM0 = COM(q0).copy()\n', '')
tail = tail.replace('        cases = []\n', '        cases = []\n        endpoints = {v[\'name\']: [] for v in variants}\n')
old_for = '''        for variant in variants:
            if not variant['reachable']:
                cases.append({'name': variant['name'], 'status': 'BLOCKED_CONTROL_UNREACHABLE_NO_STEPS', 'integration_steps': 0})
                continue
'''
new_for = '''        for variant in variants:
            # Each eligible seed has its own pose, COM0 and quaternion0.
            with np.load(checked(variant['candidate_state']), allow_pickle=False) as z:
                q0 = z['qpos27'].copy()
                qv0 = z['qvel26'].copy()
            if (q0.shape != (27,) or qv0.shape != (26,)
                    or not np.isfinite(q0).all() or not np.isfinite(qv0).all()
                    or np.any(qv0) or not np.array_equal(q0, np.asarray(variant['verified_qpos27'], dtype=float))):
                raise ValueError('Exact finite v4 verified pose and static zero speed')
            if (np.any(q0[:20] < qlimits[:,0]) or np.any(q0[:20] > qlimits[:,1])
                    or abs(float(q0[23:27]@q0[23:27])-1.) > 1e-12):
                raise ValueError('Original real joint ranges and valid native quaternion')
            COM0 = COM(q0).copy()
'''
assert tail.count(old_for) == 1
tail = tail.replace(old_for, new_for)
tail = tail.replace('                    contact_rows=np.asarray(contact_rows,dtype=float).reshape(-1,24),',
                    '                    initial_COM_world3=COM0.copy(),\n                    contact_rows=np.asarray(contact_rows,dtype=float).reshape(-1,24),')
tail = tail.replace("                write(case_dir/'summary.json',summary)", '''                summary.update(seed=variant['seed'], source_candidate_state=variant['candidate_state'],
                    initial_COM_world_m=COM0.tolist(), initial_phone_quaternion_wxyz=q0[23:27].tolist(),
                    predecessor_moment_infinity_Nm=variant['moment_infinity_Nm'],
                    predecessor_strict_support_proposal_pass=variant['strict_support_proposal_pass'],
                    enrollment='Force-hard LP/caps/reachable controller and geometry; moment residual explicitly allowed')
                endpoints[variant['name']].append({'dt_s':dt, 'final_COM_world_m':r['post_COM'][-1].tolist(),
                    'final_quaternion_wxyz':r['post_qpos'][-1,23:27].tolist(),
                    'actual_final_time_s':float(data.time), 'termination':terminate,
                    'finite_steps_saved':completed, 'original_case_pass':summary['nominal_holding_dynamics_pass']})
                write(case_dir/'summary.json',summary)''')
start = tail.index("        report = {'schema':'hand2_phone_nominal_CPU_holding_result:1'")
end = tail.index("        write(output/'report.json',report)", start)
tail = tail[:start] + '''        pairs = dt_pair_summary(variants, cases, endpoints, np)
        report = {'schema':'hand2_phone_support_CPU_holding_result:2',
            'status':'COMPLETED_ALL_ENROLLED_V4_SEEDS_BOTH_DT_UNASSISTED_DIAGNOSTIC',
            'protocol':pin(args.protocol),'counts':counts,'screening':screening,
            'cases':cases,'dt_pair_consistency':pairs,'elapsed_seconds':time.perf_counter()-started,
            'all_original_cases_and_dt_pairs_pass':bool(cases and all(v['nominal_holding_dynamics_pass'] for v in cases)
                and all(v['dt_pair_consistency_pass'] for v in pairs)),
            'mass_or_friction_calibrated':False,'ego_trajectory_or_policy_qualified':False,
            'scope':'All v4 force-hard/geometry-eligible seeds, both fixed dt, unchanged original2s case gates. '
                'Point-support moment residual is permitted only for this finite-mesh free-object diagnostic. '
                'No object aid, pose reset, materials/gains/solver tuning, sample selection or retries. '
                'Dt endpoint consistency is an additional saved comparison and never replaces individual case gates.'}
''' + tail[end:]
source = prefix + adapter + tail
with TARGET.open('x', encoding='utf8', newline='\n') as f:
    f.write(source)
print(TARGET)
