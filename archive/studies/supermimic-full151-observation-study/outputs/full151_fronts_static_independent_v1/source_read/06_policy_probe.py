"""New105 full151 deterministic policy probe, preserving native A0 cadence.

No Env, device, Actor or Runner is created on import. Device execution requires
Root's separate authorized evaluation entry; this module never trains.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import importlib.util
import json
import re
import time
import numpy as np
from .io import jsonable, sha
from .metadata import write
from .probe import CLOCK_CONTRACT, GATES, CONTACT_SOURCE, CONTACT_SHA256, fp32_clock_trace, interval_used_by_poststep, grade as frozen_A0_grade
from .controller import PURE_PATH, PURE_SHA256


def expected_control(action, previous_residual, baseline, lower, upper, bias, pure):
    """CPU FP32 oracle for native clipped action + EMA and frozen single C2."""
    import torch
    a=np.asarray(action,np.float32); previous=np.asarray(previous_residual,np.float32)
    b=np.asarray(bias,np.float32); nominal=np.asarray(baseline,np.float32)
    lo=np.asarray(lower,np.float32); hi=np.asarray(upper,np.float32)
    if any(v.shape!=(1,20) for v in (a,previous,b)) or any(v.shape!=(20,) for v in (nominal,lo,hi)) or not all(np.isfinite(v).all() for v in (a,previous,b,nominal,lo,hi)) or np.any(b) or np.any(lo+.02>=hi-.02):
        raise ValueError('Finite one-world20 control, nonempty C2 domain and frozen zero bias required')
    residual=np.float32(.5)*np.clip(np.float32(.5)*np.clip(a,-1.,1.),np.float32(-.6),np.float32(.6))+np.float32(.5)*previous
    proposed=nominal[None]+residual-b
    bounded=np.minimum(np.maximum(proposed,lo),hi)
    actual=pure(torch.from_numpy(bounded.copy()),torch.from_numpy(lo.copy()),torch.from_numpy(hi.copy()),.02).detach().cpu().numpy().copy()
    return {'residual':residual,'proposed':proposed,'hard_bounded':bounded,'actual':actual}


def state_digest(state):
    """Digest already-copied CPU state; performs no device transfer."""
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if value.device.type!='cpu':raise ValueError('Digest requires prior CPU snapshot')
        h.update(name.encode());h.update(str((str(value.dtype),tuple(value.shape))).encode())
        h.update(value.contiguous().reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def policy_rate(initial_ctrl, events):
    """150 twenty-ms differences, including startup control -> learned interval0."""
    initial=np.asarray(initial_ctrl,float)
    if initial.shape!=(20,) or len(events)!=150:raise ValueError('Full150 controls and one startup vector required')
    actual=np.asarray([e['actualu'] for e in events],float)
    if actual.shape not in ((150,20),(150,1,20)) or not np.isfinite(actual).all() or not np.isfinite(initial).all():raise ValueError('Malformed/nonfinite applied150 controls')
    actual=actual.reshape(150,20);delta=np.diff(np.vstack((initial,actual)),axis=0)
    return {'max_rate_rad_s':float(np.abs(delta).max()/.02),'startup_jump_max_rad':float(np.abs(delta[0]).max()),
        'startup_rate_rad_s':float(np.abs(delta[0]).max()/.02),'subsequent_updates_max_rate_rad_s':float(np.abs(delta[1:]).max()/.02),
        'rate_difference_count':150,'startup_rate_convention':'startup ctrl to learned interval0 target divided by declared20ms; not instantaneous physical derivative'}


def grade_policy(states,solver,capacity,reference,lower,upper,force_lower,force_upper,wrist_z,events):
    result=frozen_A0_grade(states,solver,capacity,reference,lower,upper,force_lower,force_upper,wrist_z)
    result.pop('native_A0_pass',None)
    if len(states)!=601 or len(solver)!=600 or len(events)!=150:
        result.update(status='INCOMPLETE_UNKNOWN',native_policy_pass=None);return result
    rate=policy_rate(states[0]['ctrl_rad'],events)
    result['gates']['actual_target_rate']=rate['max_rate_rad_s']<=GATES['target_rate_max_rad_s']
    result.update(status='COMPLETED_NATIVE_POLICY_QUALITY_RECORDED',native_policy_pass=all(result['gates'].values()) if capacity.get('status')=='OK' else None,
        max_actual_target_rate_rad_s=rate['max_rate_rad_s'],startup_and_update_rate=rate,
        scope='New full151 deterministic105 Actor under native Manager cadence. Full horizon despite quality failure. No training, normalization update or method-benefit claim.')
    return result


def _policy_obs(obs):
    import torch
    if type(obs) is not dict or not isinstance(obs.get('policy'),torch.Tensor) or obs['policy'].dtype!=torch.float32 or tuple(obs['policy'].shape)!=(1,155) or not bool(torch.isfinite(obs['policy']).all()):
        raise ValueError('Native105 Actor requires one finiteFP32 policy155 observation')
    return obs['policy']


def run_policy_probe(env, reader, actor, bundle, output, emit, deadline, *, initial_obs):
    import torch
    import warp as wp
    import mujoco
    import mujoco_warp as mjwarp
    if env.num_envs != 1 or reader is None or reader.env is not env or env.cfg.auto_reset:
        raise ValueError('One attached native world with auto_resetFalse required')
    if actor.training or any(p.requires_grad for p in actor.parameters()):
        raise ValueError('Use a frozen evaluation-mode105 Actor')
    obs = initial_obs; _policy_obs(obs)
    actor_before = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    if sha(PURE_PATH) != PURE_SHA256: raise ValueError('Frozen C2 projection changed')
    pure_spec = importlib.util.spec_from_file_location('_full151_policy_C2', PURE_PATH)
    pure = importlib.util.module_from_spec(pure_spec); pure_spec.loader.exec_module(pure)
    term = env.action_manager.get_term('joint_pos')
    if term.cfg.alpha != .5 or term.cfg.scale != .5 or term.cfg.residual_clip != (-.6,.6) or term._mask_enabled:
        raise ValueError('Native policy EMA/action semantics changed')
    lower, upper = [x[0].detach().cpu().numpy().copy() for x in term.target_position_limits]
    bias = term._entity.data.encoder_bias[:, term._target_ids].detach().cpu().numpy().copy()
    pending_start = len(env._full151_controller_pending)
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
    expected_current = baseline[0].copy()
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
        if time.perf_counter() > deadline: raise TimeoutError('Full151 policy finite worker deadline')
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
            'ctrl_rad': host.ctrl[aids].copy(), 'ctrl_exact': bool(np.array_equal(actual.view(np.uint32), expected_current.view(np.uint32))),
            'actuator_force_native': host.actuator_force[aids].copy(),
            'external_force_max_abs': float(max(np.abs(host.xfrc_applied).max(), np.abs(host.qfrc_applied).max())),
            'mocap_pos_w': host.mocap_pos.copy(), 'mocap_quat_wxyz': host.mocap_quat.copy(),
            'nacon': int(env.sim.wp_data.nacon.numpy().reshape(-1)[0]), 'nefc': int(host.nefc)}
        if not state['ctrl_exact']: raise ValueError('Native policy control differs from independent EMA/C2 prediction')
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
            expected_clock601_FP32_s=expected_clock, actual_wp_timestep_FP32_s=actual_dt,
            target_lower_rad=lower, target_upper_rad=upper, frozen_encoder_bias=bias)
        if not all(init_checks.values()): raise ValueError('Full151 actual reset differs from requested source initialization')
        env.sim.step = step; env.sim.forward = forward
        for k in range(150):
            phase = env.command_manager.get_term('motion').time_steps.detach().cpu().numpy().copy()
            if not np.array_equal(phase, np.array([k])): raise ValueError('Native full151 preaction phase mismatch')
            reader.refresh_frame()
            x0 = reader.state().detach().cpu().numpy().copy()
            memory = {name: value.detach().cpu().numpy().copy() for name, value in reader.memory().items()}
            prior_residual = term.processed_action.detach().cpu().numpy().copy()
            policy_input = _policy_obs(obs)
            with torch.no_grad():
                raw_action = actor(policy_input)
                if tuple(raw_action.shape) != (1,20) or raw_action.dtype != torch.float32 or not bool(torch.isfinite(raw_action).all()):
                    raise FloatingPointError('Malformed/nonfinite native105 Actor action')
                action = raw_action.clamp(-1,1)
            action_cpu = action.detach().cpu().numpy().copy()
            prediction = expected_control(action_cpu, prior_residual, baseline[k], lower, upper, bias, pure.target_inset)
            expected_current = prediction['actual'][0].copy()
            policy_input_cpu = policy_input.detach().cpu().numpy().copy()
            obs, reward, terminated, truncated, _ = env.step(action)
            _policy_obs(obs)
            if len(env._full151_controller_pending) != pending_start+k+1:
                raise ValueError('Expected exactly one process/cache row per learned interval')
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
                'residual': last['residual'].detach().cpu().numpy(),
                'cached': last['cached'].detach().cpu().numpy(), 'prior_residual': prior_residual,
                'action': action_cpu, 'actor_raw_action': raw_action.detach().cpu().numpy(),
                'policy_input155': policy_input_cpu, 'baseline': baseline[k].copy(),
                'independent_expected': prediction})
            same = lambda a,b: np.array_equal(np.asarray(a,np.float32).view(np.uint32),np.asarray(b,np.float32).view(np.uint32))
            if not same(events[-1]['residual'], prediction['residual']) or not same(events[-1]['actualu'], prediction['actual']) or not same(events[-1]['cached'], prediction['actual']):
                raise ValueError('Native105 policy EMA/cache/actualu differs from independent prediction')
            transitions.append({'phase': k, 'x0': x0[0], 'x1_reward_snapshot': physics[0],
                'x1_after_Manager': reader.state().detach().cpu().numpy()[0].copy(), 'native_selfcollision15': self15[0],
                'actualu': last['actual'].detach().cpu().numpy()[0].copy(),
                'torque_reward_snapshot': cap['torque'].detach().cpu().numpy()[0].copy(), 'memory': memory,
                'native_terms': {name: value.detach().cpu().numpy().copy() for name, value in cap['terms'].items()},
                'phase_note': 'x1raw state at20ms; native self15/torque lastsolver15ms before originalManagerforward; reference reward uses preactionk'})
            if not np.array_equal(postphase, np.array([k+1])): raise ValueError('Terminal observation/command advancement mismatch')
            if bool(terminated.any()): raise FloatingPointError('Hard native instability; partial retained, noreset/retry')
            if k < 149 and bool(truncated.any()): raise ValueError('Unexpected earlytimeout; no statewrites to continue')
            if k % 10 == 9: emit(status='running', phase='full151_native_policy', control_intervals=k+1,
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
        # Finish every Actor device read before the final sticky-capacity read.
        actor_evidence_error = None; actor_unchanged = False
        try:
            actor_after = {k: v.detach().cpu().clone() for k,v in actor.state_dict().items()}
            actor_unchanged = set(actor_before)==set(actor_after) and all(torch.equal(v,actor_after[k]) for k,v in actor_before.items())
            if not actor_unchanged: raise ValueError('Actor weights or normalization changed in evaluation')
        except BaseException as exc:
            actor_evidence_error = exc
            if error is None: error = exc
        capacity_error = None
        try:
            observer.set_stage('full151_policy_terminal_or_partial')
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
                force_lower_Nm=lo, force_upper_Nm=hi, selected_planned150_FP32_rad=baseline,
                learned_applied_FP32_rad=np.asarray([e['actualu'][0] for e in events],np.float32).reshape(-1,20)))
        retain('solver_samples', lambda: write(output / 'solver_samples.json', solver))
        retain('control_events', lambda: write(output / 'control_events.json', events))
        retain('native_reward_transitions', lambda: write(output / 'native_reward_transitions.json', transitions))
        retain('capacity_audit', lambda: write(output / 'capacity_audit.json', capacity))
        retain('raw_contacts', lambda: (output / 'raw_contacts.jsonl').write_text(
            ''.join(json.dumps(jsonable(x), allow_nan=False)+'\n' for x in contacts), encoding='utf-8'))
        retain('actor_state_evidence', lambda: write(output/'actor_state_evidence.json', {
            'schema':'full151_policy_actor_state_evidence:1', 'all_weights_and_normalizer_unchanged':actor_unchanged,
            'actor_state_before_sha256':state_digest(actor_before),
            'actor_state_after_sha256':state_digest(actor_after) if actor_evidence_error is None else None,
            'evidence_error':None if actor_evidence_error is None else repr(actor_evidence_error),
            'state_copy_before_final_capacity_read':True, 'norm_update_calls':0, 'optimizer_objects_created':0,
            'Actor_mode':'eval', 'action_noise_draws':0, 'frozen_C2_source_sha256':PURE_SHA256,
            'policy_observations':'Initial reset supplied once; then only env.step returned policy, no extra observation.compute/reset'}))
        result = retain('grade', lambda: grade_policy(states, solver, capacity, ref,
            model.jnt_range[jids, 0], model.jnt_range[jids, 1], lo, hi,
            float(states[0]['mocap_pos_w'][0, 2]) if states else 0., events))
        if result is None: result = {'status': 'GRADING_INVALID_UNKNOWN', 'native_policy_pass': None, 'criteria': GATES}
        if error is not None: result.update(status='INCOMPLETE_OR_INVALID_ENGINEERING_UNKNOWN', native_policy_pass=None, error=repr(error))
        result.update(complete=error is None and len(states)==601 and len(solver)==600 and len(events)==150 and len(transitions)==150,
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
    return {'status': 'COMPLETED_NATIVE_POLICY_ENGINEERING_QUALITY_RECORDED', 'evaluation': result, 'optimizer_updates': 0}
