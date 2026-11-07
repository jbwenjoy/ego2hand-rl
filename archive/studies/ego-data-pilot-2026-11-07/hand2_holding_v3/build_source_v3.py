"""Construct isolated support-pose adapter source; no backend/data execution."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent/'hand2_holding_v2/run_cpu_support_holding_v2.py'
TARGET = ROOT/'run_cpu_support_holding_v3.py'
source = PARENT.read_text(encoding='utf8')
source = source.replace('Bounded v4 support-candidate', 'Bounded joint-support-pose candidate')
source = source.replace("SCHEMA = 'hand2_phone_support_CPU_holding_protocol:2'", "SCHEMA = 'hand2_phone_support_CPU_holding_protocol:3'")
start = source.index('def metadata_preflight(protocol_path):')
end = source.index('\n\ndef dt_pair_summary', start)
adapter = '''def metadata_preflight(protocol_path):
    protocol = read(protocol_path)
    if protocol['schema'] != SCHEMA:
        raise ValueError('New joint-support-pose diagnostic protocol schema')
    if protocol['gates'] != GATES or protocol['dt_s'] != DTS or protocol['duration_s'] != DURATION:
        raise ValueError('Unchanged original physical budget/gates')
    if protocol['dt_pair_gates'] != PAIR_GATES:
        raise ValueError('Predeclared saved endpoint dt comparison')
    if type(protocol['CPU_run_authorized']) is not bool or protocol['training_or_GPU_authorized'] is not False:
        raise ValueError('Explicit CPU-only authorization boundary')
    if set(protocol['inputs']) != {'support_pose_protocol', 'support_pose_report', 'model_binding', 'model_mjb'}:
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
        raise ValueError('Absolute fresh output inside holding_v3')
    pose = read(checked(protocol['inputs']['support_pose_report']))
    cp = read(checked(protocol['inputs']['support_pose_protocol']))
    if (pose.get('schema') != 'fixed_wrist_gravity_contact_pose_result:3'
            or pose.get('status') != 'COMPLETED_BOUNDED_STATIC_SUPPORT_POSE_EXPLORATION'
            or pose.get('protocol') != protocol['inputs']['support_pose_protocol']
            or pose.get('counts', {}).get('integration_training_GPU') != 0
            or cp.get('schema') != 'fixed_wrist_gravity_contact_pose_protocol:3'
            or cp.get('training_GPU_physics_authorized') is not False):
        raise ValueError('Completed matching joint-support-pose static result')
    for rec in [*cp['inputs'].values(), *cp['source_pins'].values()]:
        checked(rec)
    binding = read(checked(protocol['inputs']['model_binding']))
    if (binding['model'] != protocol['inputs']['model_mjb']
            or binding['source'] != protocol['source_pins']['scene']
            or cp['inputs']['compiled_binding'] != protocol['inputs']['model_binding']
            or cp['inputs']['compiled_model'] != protocol['inputs']['model_mjb']):
        raise ValueError('Exact same original explicit-pair compiled model and static input authority')
    results = pose['seed_results']
    if len(results) != 2 or sorted(v['seed'] for v in results) != [0, 1]:
        raise ValueError('Both original support-pose seeds retained exactly once')
    variants, screening = [], []
    for seed in results:
        reasons = []
        for key in ('physical_geometry_pass', 'proposed_ctrl_reachable_without_clamping'):
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
        if lp.get('mu_actual') != [binding['mu_phone_hand'], binding['mu_phone_hand']]:
            reasons.append('same actual hand sliding material required')
        if (not finite_vector(lp.get('force_balance_residual_world_N'), 3)
                or max(map(abs, lp.get('force_balance_residual_world_N', [math.inf]))) > 1e-8):
            reasons.append('force equality residual missing/outside original LP bound')
        if (not finite_vector(lp.get('normal_cap_slack_N'), 2)
                or min(lp.get('normal_cap_slack_N', [-math.inf])) < -1e-9):
            reasons.append('normal cap hard bound not verified')
        for name in ('torque_slack_lower_Nm', 'torque_slack_upper_Nm'):
            if not finite_vector(lp.get(name), 20) or min(lp.get(name, [-math.inf])) < -1e-9:
                reasons.append(name+' original hard bound not verified')
        if not finite_vector(seed.get('proposed_ctrl20'), 20):
            reasons.append('finite proposed ctrl20 missing')
        below = seed.get('selected_evaluation', {}).get('actual_below_line_dot_m2_s2')
        if type(below) not in (int, float) or not math.isfinite(below) or below < 0:
            reasons.append('actual selected contact line COM-below evidence missing')
        state = seed.get('verification_state', {})
        if not finite_vector(state.get('qpos27'), 27):
            reasons.append('complete verified qpos27 missing')
        selected = seed.get('selected_state')
        if not isinstance(selected, dict):
            reasons.append('selected state pin missing')
        else:
            checked(selected)
            if Path(selected['path']).parent.resolve() != Path(protocol['inputs']['support_pose_report']['path']).parent.resolve():
                raise ValueError('Selected state outside matching support-pose run')
        # The original static producer candidate flag includes the strict moment
        # gate. It is descriptive here and MUST NOT control dynamic enrollment.
        original_strict = seed.get('candidate_qualified_for_bounded_holding_diagnostic')
        screen = {'seed': seed['seed'], 'enrolled': not reasons, 'reasons': reasons,
                  'original_strict_moment_candidate_flag': original_strict,
                  'moment_infinity_Nm': lp.get('moment_infinity_Nm'),
                  'actual_selected_COM_below_dot_m2_s2': below, 'selected_state': selected}
        screening.append(screen)
        if not reasons:
            variants.append({'name': 'support_pose_seed'+str(seed['seed']), 'seed': seed['seed'],
                'reachable': True, 'ctrl20': seed['proposed_ctrl20'],
                'candidate_state': selected, 'verified_qpos27': state['qpos27'],
                'moment_infinity_Nm': lp.get('moment_infinity_Nm'),
                'strict_support_proposal_pass': original_strict})
    if pose.get('qualified_support_proposal_count') != sum(v.get('candidate_qualified_for_bounded_holding_diagnostic') is True for v in results):
        raise ValueError('Consistent descriptive original strict-moment count')
    return protocol, output, binding, variants, screening
'''
source = source[:start] + adapter + source[end:]
source = source.replace('hand2_phone_support_CPU_holding_result:2', 'hand2_phone_support_CPU_holding_result:3')
source = source.replace('BLOCKED_NO_V4_SEED_ELIGIBLE_NO_MODEL_OR_STEPS', 'BLOCKED_NO_SUPPORT_POSE_SEED_ELIGIBLE_NO_MODEL_OR_STEPS')
source = source.replace('COMPLETED_ALL_ENROLLED_V4_SEEDS_BOTH_DT_UNASSISTED_DIAGNOSTIC', 'COMPLETED_ALL_ENROLLED_SUPPORT_POSE_SEEDS_BOTH_DT_UNASSISTED_DIAGNOSTIC')
source = source.replace('Exact finite v4 verified pose', 'Exact finite support-pose verified pose')
source = source.replace('All v4 force-hard/geometry-eligible seeds', 'All joint-support-pose force-hard/geometry-eligible seeds')
with TARGET.open('x', encoding='utf8', newline='\n') as f:
    f.write(source)
print(TARGET)
