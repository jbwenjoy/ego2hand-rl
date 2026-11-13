"""Check chronology and physical EMA carry in saved native transition arrays."""
import numpy as np
from .residual_control import residual_ema


def saved_chain_checks(a, z, dt, base, negative, positive):
    n = round(.02 / dt)
    count = len(a['k'])
    if not count or len(z['tick_ranges']) != count:
        raise ValueError('Nonempty matching transition/tick rows required')
    ranges = z['tick_ranges']
    lengths = ranges[:, 1] - ranges[:, 0]
    after = residual_ema(a['action20'], a['EMA_before20'], negative, positive)
    checks = dict(
        finite_arrays=all(np.isfinite(v).all() for source in [a, z]
                          for v in source.values() if np.issubdtype(v.dtype, np.number)),
        sequential_k=np.array_equal(a['k'], np.arange(count)),
        contiguous_tick_ranges=ranges[0, 0] == 0 and ranges[-1, 1] == len(z['post_qpos'])
            and np.array_equal(ranges[1:, 0], ranges[:-1, 1]),
        legal_interval_lengths=np.all((lengths > 0) & (lengths <= n))
            and np.all(lengths[:-1] == n),
        interval_duration=np.array_equal(a['actual_duration_s'], lengths * dt),
        complete_flags=np.array_equal(a['complete_H'], lengths == n),
        no_continuation_after_termination=not np.any(a['terminated'][:-1]),
        zero_initial_EMA=np.array_equal(a['EMA_before20'][0], np.zeros(20)),
        EMA_carry=np.array_equal(a['EMA_before20'][1:], a['EMA_after20'][:-1]),
        EMA_update=np.array_equal(after, a['EMA_after20']),
        actual_control=np.array_equal(base + after, a['actual_control20']),
        native_state_carry=np.array_equal(a['source_native_state'][1:], a['next_native_state'][:-1]),
        observed_state_carry=np.array_equal(a['source55'][1:], a['next55'][:-1]),
        warmstart_carry=np.array_equal(a['warmstart26_before'][1:], a['warmstart26_after'][:-1]),
        per_tick_pose_carry=np.array_equal(z['pre_qpos'][1:], z['post_qpos'][:-1]),
        per_tick_velocity_carry=np.array_equal(z['pre_qvel'][1:], z['post_qvel'][:-1]),
    )
    return {k: bool(v) for k, v in checks.items()}
