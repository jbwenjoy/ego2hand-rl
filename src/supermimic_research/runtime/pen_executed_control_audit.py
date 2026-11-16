"""Rebuild the declared EMA/PD target and compare actual recorded GPU controls."""
import numpy as np

ACTION_CONTRACT = 'RAW_CLAMP_MINUS1_PLUS1_SCALE0P5_EMA0P5_PD_BOUNDS'


def audit_executed_control(trace, case):
    raw = trace['sampled_mean_raw']
    c = len(raw)
    n = len(trace['post_qpos'])
    phase = trace['pre_action_phase']
    ids = np.asarray(case['position_ctrl_ids'])
    if (case['recorded_executed_control_schema'] != 'pen_PD_and_native_drives:1'
            or case['action_contract'] != ACTION_CONTRACT or c < 1 or raw.shape != (c, 20) or n != 8 * c
            or phase.shape != (c,) or not np.array_equal(phase, np.arange(1, c + 1))
            or ids.shape != (20,) or len(set(ids.tolist())) != 20
            or len(case['joint_order']) != 20 or len(set(case['joint_order'])) != 20):
        raise ValueError('Explicit uninterrupted raw20/eight-substep/PD contract required')
    for key in ['initial_ema', 'encoder_bias', 'target_position_lower', 'target_position_upper']:
        if trace[key].shape != (20,):
            raise ValueError('Joint20 header shape differs: ' + key)
    ctrl = trace['post_ctrl']
    if ctrl.ndim != 2 or len(ctrl) != n or np.any(ids < 0) or np.any(ids >= ctrl.shape[1]):
        raise ValueError('Recorded control binding differs')
    if trace['reference_joint_pos'].ndim != 2 or trace['reference_joint_pos'].shape[1] != 20 or phase[-1] >= len(trace['reference_joint_pos']):
        raise ValueError('Recorded reference phase outside joint table')
    for key in ['pre_qpos', 'pre_qvel', 'post_qvel', 'post_actuator_force', 'post_qfrc_actuator']:
        if len(trace[key]) != n:
            raise ValueError('Native recorder field length differs: ' + key)
    lower, upper = trace['target_position_lower'], trace['target_position_upper']
    if np.any(lower > upper) or any(not np.isfinite(v).all() for v in trace.values()):
        raise ValueError('Nonfinite trace or invalid target bounds')
    ema = trace['initial_ema'].astype(np.float32).copy()
    targets, before_limit = [], []
    for index in range(c):
        applied = np.clip(raw[index], -1, 1).astype(np.float32)
        ema = np.float32(.5) * (np.float32(.5) * applied) + np.float32(.5) * ema
        before = trace['reference_joint_pos'][phase[index]].astype(np.float32) + ema - trace['encoder_bias'].astype(np.float32)
        before_limit.append(before)
        targets.append(np.clip(before, lower, upper))
    targets = np.asarray(targets)
    before_limit = np.asarray(before_limit)
    actual = ctrl[:, ids].reshape(c, 8, 20)
    error = np.abs(actual - targets[:, None])
    checks = dict(
        EMA_reference_bias_and_limits_match_GPU_ctrl=bool(np.all(error <= 3e-7)),
        ctrl_held_for_eight_native_substeps=bool(np.array_equal(actual, np.repeat(actual[:, :1], 8, axis=1))),
        GPU_ctrl_within_declared_target_limits=bool(np.all(actual >= lower - 1e-7) and np.all(actual <= upper + 1e-7)),
        qpos_native_continuity=bool(np.array_equal(trace['pre_qpos'][1:], trace['post_qpos'][:-1])),
        qvel_native_continuity=bool(np.array_equal(trace['pre_qvel'][1:], trace['post_qvel'][:-1])),
        actual_native_drives_finite=bool(np.isfinite(trace['post_actuator_force']).all() and np.isfinite(trace['post_qfrc_actuator']).all()))
    return dict(schema='pen_executed_PD_audit:1', checks=checks, passed=sum(checks.values()), total=len(checks),
                controls=c, native_samples=n, max_abs_GPU_ctrl_reconstruction_error_rad=float(error.max()),
                raw_clipped_components=int((np.abs(raw) > 1).sum()),
                absolute_PD_target_limited_components=int(((before_limit < lower) | (before_limit > upper)).sum()),
                max_abs_recorded_actuator_force=float(np.abs(trace['post_actuator_force']).max()),
                scope='Actual transmitted PD targets and native drive arrays are recorded. EMA/reference/bias/limit reconstruction is checked against GPU ctrl;this does not establish task success or derive a saturated-drive causal model.')
