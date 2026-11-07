"""Full151 native-cadence, zero-residual A0; pure grading is CPU executable.

No device/Env is created on import. Run only under Root's authorized worker.
Quality failures latch in output and do not shorten the150-interval episode.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util
import json
import re
import time
import numpy as np
from .io import jsonable, sha
from .metadata import write

N_STATES = 601
N_SOLVER = 600
N_ACTIONS = 150
CLOCK_CONTRACT = {'schema': 'full151_native_clock:1', 'dtype': 'float32',
    'physics_dt_s': .005, 'physics_steps': 600, 'control_substeps': 4,
    'expected_rule': 'sequential_float32_add_from_zero', 'bitexact_expected_time': True,
    'canonical_time_rule': 'integer_physics_tick_times_0p005',
    'nominal_drift_gate_s': 5e-5, 'legacy_prefix_2us_diagnostic_only': True}
GATES = {'object_position_max_m': .02, 'directed_shaft_error_max_deg': 20.,
    'actual_joint_hard_excess_max_rad': .001, 'penetration_any_category_max_m': .003,
    'object_nonhand_candidate_contact_count_max': 0, 'external_force_max_abs': 0.,
    'clock': CLOCK_CONTRACT, 'legacy_prefix_2us_diagnostic_only_s': 2e-6,
    'target_rate_max_rad_s': 15., 'actuator_load_ratio_max': 1.01,
    'load_charge_max': .8, 'charge_decay_time_s': 2.,
    'drop_center_below_default_wrist_m': .15}
CONTACT_SOURCE = Path('D:/workspace/supermimic-rough-prefix-learning-study/vendor/measured_contacts.py')
CONTACT_SHA256 = '478c3117489bd4a02238f6299a91638e791a751352b772321ce2676b0ec4f96f'


def fp32_clock_trace():
    values = np.empty(601, np.float32); values[0] = np.float32(0.)
    dt = np.float32(.005)
    for k in range(600): values[k + 1] = np.float32(values[k] + dt)
    return values


def clock_audit(raw_times):
    raw = np.asarray(raw_times, dtype=np.float64)
    expected = fp32_clock_trace()
    if raw.shape != (601,) or not np.isfinite(raw).all():
        return {'pass': False, 'status': 'INCOMPLETE_OR_NONFINITE_CLOCK', 'contract': CLOCK_CONTRACT}
    as32 = raw.astype(np.float32)
    exact_repr = np.array_equal(as32.astype(np.float64), raw)
    bits = np.array_equal(as32.view(np.uint32), expected.view(np.uint32))
    canonical = np.arange(601) * .005
    drift = np.abs(raw - canonical)
    return {'pass': bool(exact_repr and bits and drift.max() <= 5e-5),
        'status': 'FULL601_CLOCK_CHECKED', 'contract': CLOCK_CONTRACT,
        'raw_is_exact_FP32_representation': bool(exact_repr), 'all601_expected_FP32_bits_equal': bool(bits),
        'max_abs_nominal_drift_s': float(drift.max()), 'expected_final_time_s': float(expected[-1]),
        'raw_final_time_s': float(raw[-1]), 'legacy_2us_diagnostic_pass': bool(drift.max() <= 2e-6)}


def interval_used_by_poststep(index):
    if type(index) is not int or not 0 <= index <= 600:
        raise ValueError('Sample index must be integer0..600; no601st integration step')
    return 0 if index == 0 else (index - 1) // 4


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape[-1:] != (4,) or not np.isfinite(q).all() or np.any(np.abs(np.linalg.norm(q, axis=-1) - 1) > 1e-6):
        raise ValueError('Finite unit wxyz object quaternion required')
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1] + (3, 3))


def grade(states, solver, capacity, reference, lower, upper, force_lower, force_upper, wrist_z):
    """The original13 physical-quality gates, with source-informed full151 clock."""
    if len(states) != 601 or len(solver) != 600:
        return {'status': 'INCOMPLETE_UNKNOWN', 'native_A0_pass': None, 'criteria': GATES}
    pose = np.asarray([s['object_pose_w'] for s in states]); q = np.asarray([s['joint_rad'] for s in states])
    R = rotation(pose[:, 3:]); primary = np.arange(0, 601, 4)
    pos = np.linalg.norm(pose[:, :3] - reference['object_COM_world_m'], axis=1)
    angle = np.rad2deg(np.arccos(np.clip(np.sum(R[:, :, 0] * reference['object_rotation_w'][:, :, 0], axis=1), -1, 1)))
    excess = np.maximum(np.maximum(lower - q, q - upper), 0)
    forces = np.asarray([s['actuator_force_solver_native'] for s in solver])
    if forces.shape != (600, 20) or np.any(force_lower >= 0) or np.any(force_upper <= 0):
        raise ValueError('Signed20 actual force bounds/600solver rows required')
    if not all(np.isfinite(v).all() for v in (pose, q, pos, angle, forces, excess)):
        raise ValueError('Nonfinite completed grading arrays are invalid engineering evidence')
    ratio = np.where(forces >= 0, forces / force_upper, forces / force_lower)
    charge = np.zeros(20); maximum = charge.copy()
    for dt, row in zip(np.diff([s['raw_time_s'] for s in states]), ratio):
        charge = np.clip(charge + dt * (np.minimum(row, 1) - charge / 2.), 0, 1)
        maximum = np.maximum(maximum, charge)
    indices = [0, *range(5, 598, 4)]  # U0 at startup, Uk first visible at poststep4k+1.
    actual150 = np.asarray([states[k]['ctrl_rad'] for k in indices])
    rate = float(np.abs(np.diff(actual150, axis=0)).max() / .02)
    clock = clock_audit([s['raw_time_s'] for s in states])
    checks = {'object_position_151': bool(pos[primary].max() <= .02),
        'directed_shaft_151': bool(angle[primary].max() <= 20),
        'actual_q_all601': bool(excess.max() <= .001),
        'penetration_solver600': bool(max(s['penetration_m'] for s in solver) <= .003),
        'no_nonhand_object_contacts': sum(s['object_environment_contacts'] for s in solver) == 0,
        'no_exogenous': max(s['external_force_max_abs'] for s in states) == 0,
        'all_actual_u_exact': all(s['ctrl_exact'] for s in states),
        'actual_target_rate': rate <= 15, 'clock': clock['pass'],
        'load_ratio': bool(ratio.max() <= 1.01), 'load_charge': bool(maximum.max() <= .8),
        'drop': bool(pose[:, 2].min() >= wrist_z - .15), 'capacity': capacity.get('status') == 'OK'}
    return jsonable({'status': 'COMPLETED_NATIVE_A0_QUALITY_RECORDED',
        'native_A0_pass': all(checks.values()) if capacity.get('status') == 'OK' else None,
        'gates': checks, 'criteria': GATES, 'clock_audit': clock,
        'max_object_position_error_151_m': pos[primary].max(), 'max_object_position_error_601_m': pos.max(),
        'max_shaft_angle_151_deg': angle[primary].max(), 'max_shaft_angle_601_deg': angle.max(),
        'max_actual_joint_excess_rad': excess.max(), 'max_penetration_m': max(s['penetration_m'] for s in solver),
        'max_actuator_load_ratio': ratio.max(), 'max_load_charge': maximum.max(),
        'max_actual_target_rate_rad_s': rate,
        'scope': 'New full151 Manager-cadence A0,150actions/600steps. Rawsolver contact/force phase differs from poststate. No learnedactor, training, CHORD, hardware or ego qualification.'})


def run_probe(env, reader, bundle, output, emit, deadline):
    import torch
    import warp as wp
    import mujoco
    import mujoco_warp as mjwarp
    if env.num_envs != 1 or reader is None:
        raise ValueError('One native world and its attached PenSimReader capture required')
    observer = getattr(env, '_full151_capacity_observer', None)
    if observer is None:
        raise ValueError('Install source-bound reset-aware capacity observation before startup')
    if sha(CONTACT_SOURCE) != CONTACT_SHA256:
        raise ValueError('Frozen measured-contact helper changed')
    spec = importlib.util.spec_from_file_location('_full151_measured_contacts', CONTACT_SOURCE)
    contacts_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(contacts_module)
    capture_contacts = contacts_module.capture_contacts
    output = Path(output); model = env.sim.mj_model; host = mujoco.MjData(model)
    ref = bundle['arrays']['reference']; baseline = bundle['baseline_u150']
    names = [f'robot/right_finger{f}_joint{j}' for f in range(1, 6) for j in range(1, 5)]
    jids = np.array([model.joint(n).id for n in names]); qa = model.jnt_qposadr[jids]; va = model.jnt_dofadr[jids]
    aids = np.array([np.flatnonzero((model.actuator_trnid[:, 0] == j) & (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)).item() for j in jids])
    obj = model.body('object/cube').id; oj = model.joint('object/cube_freejoint').id
    oq, ov = int(model.jnt_qposadr[oj]), int(model.jnt_dofadr[oj])
    if np.any(model.body_ipos[obj]): raise ValueError('Object COM offset must remain the bound zero offset')
    hand = {i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)', model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
    object_geoms = set(np.flatnonzero(model.geom_bodyid == obj).tolist())
    lo, hi = model.actuator_forcerange[aids, 0].copy(), model.actuator_forcerange[aids, 1].copy()
    lo = np.maximum(lo, np.where(model.jnt_actfrclimited[jids], model.jnt_actfrcrange[jids, 0], -np.inf))
    hi = np.minimum(hi, np.where(model.jnt_actfrclimited[jids], model.jnt_actfrcrange[jids, 1], np.inf))
    time_dtype = env.sim.wp_data.time.numpy().dtype
    actual_dt = env.sim.wp_model.opt.timestep.numpy().copy()
    if time_dtype != np.dtype(np.float32) or actual_dt.dtype != np.float32 or not np.array_equal(actual_dt, np.full(actual_dt.shape, np.float32(.005))):
        raise ValueError('Actual Warp time/dt must match declared FP32 .005 clock ABI')
    expected_clock = fp32_clock_trace()
    states, solver, contacts, events, transitions = [], [], [], [], []
    api_counts = {'step_attempted': 0, 'step_returned': 0,
                  'forward_attempted': 0, 'forward_returned': 0}
    error = None; capacity = {'status': 'UNKNOWN'}
    original_step, original_forward = env.sim.step, env.sim.forward
    def sample(index):
        if time.perf_counter() > deadline: raise TimeoutError('Full151 A0 finite worker deadline')
        wp.synchronize_device(env.device)
        mjwarp.get_data_into(host, model, env.sim.wp_data, world_id=0)
        if not all(np.isfinite(x).all() for x in (host.qpos, host.qvel, host.ctrl, host.actuator_force)):
            raise FloatingPointError('Nonfinite full151 native state')
        if float(np.float32(host.time)) != float(host.time) or np.float32(host.time).view(np.uint32) != expected_clock[index].view(np.uint32) or abs(float(host.time) - index * .005) > 5e-5:
            raise ValueError('Native physics clock violates declared FP32 recurrence/drift contract')
        interval = interval_used_by_poststep(index); actual = host.ctrl[aids].astype(np.float32)
        state = {'index': index, 'raw_time_s': float(host.time), 'canonical_time_s': index * .005,
            'expected_raw_FP32_time_s': float(expected_clock[index]), 'joint_rad': host.qpos[qa].copy(),
            'joint_velocity_rad_s': host.qvel[va].copy(), 'full_qpos': host.qpos.copy(), 'full_qvel': host.qvel.copy(),
            'object_pose_w': host.qpos[oq:oq+7].copy(), 'object_twist_native': host.qvel[ov:ov+6].copy(),
            'ctrl_rad': host.ctrl[aids].copy(), 'ctrl_exact': bool(np.array_equal(actual.view(np.uint32), baseline[interval].view(np.uint32))),
            'actuator_force_native': host.actuator_force[aids].copy(),
            'external_force_max_abs': float(max(np.abs(host.xfrc_applied).max(), np.abs(host.qfrc_applied).max())),
            'mocap_pos_w': host.mocap_pos.copy(), 'mocap_quat_wxyz': host.mocap_quat.copy(),
            'nacon': int(env.sim.wp_data.nacon.numpy().reshape(-1)[0]), 'nefc': int(host.nefc)}
        if not state['ctrl_exact']: raise ValueError('Native zeroresidual u differs from frozen full150 baseline')
        states.append(state)
        if index:
            measured = capture_contacts(model, host, obj, object_geoms, hand)
            solver.append({'step': index, 'prestate_index': index-1, 'poststate_index': index,
                'solver_t_s': states[-2]['raw_time_s'], 'poststate_t_s': state['raw_time_s'],
                'control_interval_used': interval, 'ctrl_solver_native': host.ctrl[aids].copy(),
                'actuator_force_solver_native': host.actuator_force[aids].copy(),
                'solver_object_COM_w': host.xipos[obj].copy(), 'solver_object_body_pose_w': np.r_[host.xpos[obj], host.xquat[obj]],
                'penetration_m': measured['penetration_m'], 'penet_category_m': measured['penetration_by_category_m'],
                'object_environment_contacts': measured['object_environment_contacts'], 'measured_hand_object': measured,
                'phase': 'just_completed_solver_preintegration_before_original_Manager_forward'})
        for i, c in enumerate(host.contact):
            force = np.zeros(6); mujoco.mj_contactForce(model, host, i, force)
            if not np.isfinite(force).all() or not np.isfinite(c.dist) or not np.isfinite(c.pos).all() or not np.isfinite(c.frame).all():
                raise FloatingPointError('Nonfinite native solved contact readback')
            contacts.append({'step': index, 'geom': list(map(int, c.geom)), 'distance_m': float(c.dist),
                'pos_w': c.pos.copy(), 'frame_rows_w': c.frame.reshape(3, 3).copy(), 'force6_contact_frame': force,
                'phase': 'startup_after_native_reset' if index == 0 else 'just_completed_preintegration_solver'})
    def step():
        if len(states) > 600: raise ValueError('No601st integration step or extra terminal action permitted')
        if time.perf_counter() > deadline: raise TimeoutError('Full151 deadline before next physical step')
        api_counts['step_attempted'] += 1
        original_step()
        api_counts['step_returned'] += 1
        sample(len(states))
    def forward():
        api_counts['forward_attempted'] += 1
        result = original_forward()
        api_counts['forward_returned'] += 1
        return result
    try:
        sample(0); initial = bundle['arrays']['init']; default = mujoco.MjData(model)
        expected_q = default.qpos.astype(np.float32)
        expected_q[qa] = initial['q_init_requested_FP32_rad']; expected_q[oq:oq+7] = initial['object_pose_init_requested_FP32_w']
        expected_v = np.zeros(model.nv, np.float32); expected_v[va] = initial['qd_init_requested_FP32_rad_s']
        expected_v[ov:ov+3] = initial['object_bodyorigin_v_requested_FP32_world_m_s']
        expected_v[ov+3:ov+6] = initial['object_angular_requested_FP32_body_rad_s']
        init_checks = {'full_qpos_FP32_exact': np.array_equal(host.qpos.astype(np.float32), expected_q),
            'joint_qd_FP32_exact': np.array_equal(host.qvel[va].astype(np.float32), expected_v[va]),
            'object_world_linear_FP32_exact': np.array_equal(host.qvel[ov:ov+3].astype(np.float32), expected_v[ov:ov+3]),
            'body_omega_API_arithmetic_atol_1e_minus6': np.allclose(host.qvel[ov+3:ov+6], expected_v[ov+3:ov+6], rtol=0, atol=1e-6),
            'mocap_default_FP32_exact': np.array_equal(host.mocap_pos.astype(np.float32), default.mocap_pos.astype(np.float32)) and np.array_equal(host.mocap_quat.astype(np.float32), default.mocap_quat.astype(np.float32)),
            'initial_ctrl_exact': states[0]['ctrl_exact'], 'no_external_forces': states[0]['external_force_max_abs'] == 0,
            'time_zero_exact': host.time == 0.}
        write(output / 'startup_initialization_checks.json', {'checks': init_checks,
              'body_omega_max_error_rad_s': np.max(np.abs(host.qvel[ov+3:ov+6]-expected_v[ov+3:ov+6])),
              'arithmetic_tolerance_not_physical_gate': True, 'clock_actual_dtype': str(time_dtype),
              'clock_actual_timestep_FP32': actual_dt, 'clock_contract': CLOCK_CONTRACT})
        np.savez_compressed(output / 'requested_initialization_and_control.npz', full_qpos_requested_FP32=expected_q,
            full_qvel_requested_FP32=expected_v, mocap_pos_requested_FP32=default.mocap_pos.astype(np.float32),
            mocap_quat_requested_FP32=default.mocap_quat.astype(np.float32), selected_planned150_FP32_rad=baseline,
            position_ctrl_ids=aids, joint_qpos_ids=qa, joint_dof_ids=va, force_lower_Nm=lo, force_upper_Nm=hi,
            expected_clock601_FP32_s=expected_clock, actual_wp_timestep_FP32_s=actual_dt)
        if not all(init_checks.values()): raise ValueError('Full151 actual reset differs from requested source initialization')
        env.sim.step = step; env.sim.forward = forward
        for k in range(150):
            phase = env.command_manager.get_term('motion').time_steps.detach().cpu().numpy().copy()
            if not np.array_equal(phase, np.array([k])): raise ValueError('Native full151 preaction phase mismatch')
            reader.refresh_frame()
            x0 = reader.state().detach().cpu().numpy().copy()
            memory = {name: value.detach().cpu().numpy().copy() for name, value in reader.memory().items()}
            obs, reward, terminated, truncated, _ = env.step(torch.zeros((1, 20), device=env.device))
            last = env._full151_controller_pending[-1]
            postphase = env.command_manager.get_term('motion').time_steps.detach().cpu().numpy().copy()
            cap = reader.reward_snapshot
            if cap is None: raise ValueError('Missing native pre-forward reward snapshot')
            physics = cap['physics'].detach().cpu().numpy().copy(); self15 = cap['contact'].detach().cpu().numpy().copy()
            if physics.shape != (1, 58) or self15.shape != (1, 15) or not np.isfinite(physics).all() or not np.isfinite(self15).all():
                raise ValueError('Native state58/selfcollision15 capture ABI differs')
            events.append({'phase': k, 'preaction_phase': phase, 'postaction_phase': postphase,
                'terminated': terminated.detach().cpu().numpy(), 'truncated': truncated.detach().cpu().numpy(),
                'reference_end': env.termination_manager.get_term('reference_end').detach().cpu().numpy().copy(),
                'reward': reward.detach().cpu().numpy(), 'actualu': last['actual'].detach().cpu().numpy(),
                'residual': last['residual'].detach().cpu().numpy()})
            transitions.append({'phase': k, 'x0': x0[0], 'x1_reward_snapshot': physics[0],
                'x1_after_Manager': reader.state().detach().cpu().numpy()[0].copy(), 'native_selfcollision15': self15[0],
                'actualu': last['actual'].detach().cpu().numpy()[0].copy(),
                'torque_reward_snapshot': cap['torque'].detach().cpu().numpy()[0].copy(), 'memory': memory,
                'native_terms': {name: value.detach().cpu().numpy().copy() for name, value in cap['terms'].items()},
                'phase_note': 'x1raw state at20ms; native self15/torque lastsolver15ms before originalManagerforward; reference reward uses preactionk'})
            if not np.array_equal(postphase, np.array([k+1])): raise ValueError('Terminal observation/command advancement mismatch')
            if bool(terminated.any()): raise FloatingPointError('Hard native instability; partial retained, noreset/retry')
            if k < 149 and bool(truncated.any()): raise ValueError('Unexpected earlytimeout; no statewrites to continue')
            if k % 10 == 9: emit(status='running', phase='full151_native_A0', control_intervals=k+1,
                physics_steps=api_counts['step_returned'], sampled_solver_rows=len(solver), episode_API_counts=dict(api_counts))
        if len(states) != 601 or len(solver) != 600 or api_counts != {
                'step_attempted': 600, 'step_returned': 600,
                'forward_attempted': 150, 'forward_returned': 150}:
            raise ValueError('Full151 native cadence/coverage differs')
        if not bool(events[-1]['truncated'][0]) or not bool(events[-1]['reference_end'][0]) or any(bool(e['reference_end'][0]) for e in events[:-1]):
            raise ValueError('Natural end must be lastphase149 timeout with terminalobs150')
        if any(not np.array_equal(s['mocap_pos_w'], states[0]['mocap_pos_w']) or not np.array_equal(s['mocap_quat_wxyz'], states[0]['mocap_quat_wxyz']) for s in states):
            raise ValueError('Fixed wrist moved during native episode')
    except BaseException as exc:
        error = exc
    finally:
        env.sim.step = original_step; env.sim.forward = original_forward
        capacity_error = None
        try:
            observer.set_stage('full151_A0_terminal_or_partial')
            observer.final_read()
        except BaseException as exc:
            capacity_error = exc
            if error is None: error = exc
        capacity = observer.snapshot()
        cc = int(env.sim.wp_data.contact.geom.shape[0]); jc = int(env.sim.wp_data.efc.force.shape[-1])
        capacity.update(final_read_after_returned_episode_step=api_counts['step_returned'],
            final_read_after_sampled_solver_row=len(solver), contact_capacity=cc, constraint_capacity=jc,
            resets_during_probe=0, flag_shape_is_world_count=False,
            maximum_recorded_nacon=None if not states else max(s['nacon'] for s in states),
            maximum_recorded_nefc=None if not states else max(s['nefc'] for s in states))
        if states and (capacity['maximum_recorded_nacon'] >= cc or capacity['maximum_recorded_nefc'] >= jc):
            capacity['status'] = 'UNKNOWN_AT_ALLOCATION_BOUNDARY'
            if error is None: error = ValueError('Capacity allocation boundary reached; not qualified')
        if capacity_error is not None: capacity['final_observation_error'] = repr(capacity_error)
        preservation_errors = []
        def retain(label, operation):
            nonlocal error
            try: return operation()
            except BaseException as exc:
                preservation_errors.append({'artifact_or_stage': label, 'error': repr(exc)})
                if error is None: error = exc
                elif hasattr(error, 'add_note'): error.add_note(label + ': ' + repr(exc))
                return None
        if states:
            retain('raw_states', lambda: np.savez_compressed(output / 'raw_states.npz',
                **{name: np.asarray([s[name] for s in states]) for name in states[0]},
                joint_lower_rad=model.jnt_range[jids, 0], joint_upper_rad=model.jnt_range[jids, 1],
                force_lower_Nm=lo, force_upper_Nm=hi, selected_planned150_FP32_rad=baseline))
        retain('solver_samples', lambda: write(output / 'solver_samples.json', solver))
        retain('control_events', lambda: write(output / 'control_events.json', events))
        retain('native_reward_transitions', lambda: write(output / 'native_reward_transitions.json', transitions))
        retain('capacity_audit', lambda: write(output / 'capacity_audit.json', capacity))
        retain('raw_contacts', lambda: (output / 'raw_contacts.jsonl').write_text(
            ''.join(json.dumps(jsonable(x), allow_nan=False)+'\n' for x in contacts), encoding='utf-8'))
        result = retain('grade', lambda: grade(states, solver, capacity, ref,
            model.jnt_range[jids, 0], model.jnt_range[jids, 1], lo, hi,
            float(states[0]['mocap_pos_w'][0, 2]) if states else 0.))
        if result is None: result = {'status': 'GRADING_INVALID_UNKNOWN', 'native_A0_pass': None, 'criteria': GATES}
        if error is not None: result.update(status='INCOMPLETE_OR_INVALID_ENGINEERING_UNKNOWN', native_A0_pass=None, error=repr(error))
        result.update(complete=error is None and len(states)==601 and len(solver)==600,
            physical_steps=api_counts['step_returned'], native_postinterval_forward_calls=api_counts['forward_returned'],
            episode_API_counts=dict(api_counts), sampled_state_rows=len(states), sampled_solver_rows=len(solver),
            counter_scope='Only this probe episode after startup readback; counts describe API attempts/returns, not device-kernel calls.',
            construction_reset_setup_API_counts='UNKNOWN_NOT_WHOLEPROCESS_OBSERVED', control_intervals=len(events),
            optimizer_updates=0, resets_after_start=0, physical_reference_qualified=False,
            native_reward_transition_count=len(transitions), selfcollision15_is_not_handobject=True,
            preservation_errors=preservation_errors,
            model_data_readback='601poststate rows sampled afterstep before150original native postintervalforwards; solverderived fields preintegration')
        retain('evaluation', lambda: write(output / 'evaluation.json', result))
    if error is not None: raise error
    return {'status': 'COMPLETED_NATIVE_A0_ENGINEERING_QUALITY_RECORDED', 'evaluation': result, 'optimizer_updates': 0}
