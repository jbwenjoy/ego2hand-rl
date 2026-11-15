"""Consecutive actual parent controls; source snapshots supply observed warmstart."""
import numpy as np

from .candidate_response_targets import packet_xy


def pack_observed_controls(case, trace, sources, qids, dofs, aids, physics,
                           world_wrench, Fn, tick_valid, *, start=8):
    n = round(.02 / case['dt'])
    K = len(trace['actions20'])
    if (len(trace['post_qpos']) != K * n or set(sources) != set(range(K))
            or world_wrench.shape != (K * n, 6) or Fn.shape != (K * n, 6)
            or np.shape(tick_valid) != (K * n,)):
        raise ValueError('Complete actual controls, per-control sources and decoded labels required')
    qa, da = case['model_layout']['object_qposadr'], case['model_layout']['object_dofadr']
    xs, ys, valid, indices = [], [], [], []
    # K-1 has no saved next source, so its engine memory target is unobserved.
    for k in range(start, K - 1):
        a, b = k * n, (k + 1) * n
        source, next_source = sources[k], sources[k + 1]
        continuity = (
            np.array_equal(source['qpos'][0], trace['pre_qpos'][a])
            and np.array_equal(source['qvel'][0], trace['pre_qvel'][a])
            and np.array_equal(next_source['qpos'][0], trace['post_qpos'][b - 1])
            and np.array_equal(next_source['qvel'][0], trace['post_qvel'][b - 1])
            and np.array_equal(trace['pre_qpos'][a + 1:b], trace['post_qpos'][a:b - 1])
            and np.array_equal(trace['pre_qvel'][a + 1:b], trace['post_qvel'][a:b - 1])
            and float(source['time'][0]) == float(trace['post_GPU_time'][a - 1])
            and float(next_source['time'][0]) == float(trace['post_GPU_time'][b - 1]))
        target = trace['post_ctrl'][a, aids]
        if not continuity or not np.array_equal(trace['post_ctrl'][a:b, aids], np.broadcast_to(target, (n, 20))):
            raise ValueError('Exact actual state/clock continuity and held PD required')
        packet = {name: trace[name][a:b] for name in ['post_qpos', 'post_qvel', 'post_ctrl']}
        # This field is a derived endpoint observation from the NEXT actual
        # source, not a new per-tick raw capture or post_qacc substitution.
        packet['post_qacc_warmstart'] = next_source['qacc_warmstart']
        x, y, _ = packet_xy(source, packet, world_wrench[a:b], Fn[a:b], qids, dofs,
                           aids, qa, da, case['dt'], physics)
        xs.append(x)
        ys.append(y)
        valid.append(bool(np.all(tick_valid[a:b]) and np.isfinite(np.r_[x, y]).all()
                          and np.linalg.norm(y[43:46]) < np.pi - .001))
        indices.append(k)
    if not xs:
        raise ValueError('No actual tail intervals at predeclared start')
    return dict(x=np.array(xs), y=np.array(ys), valid=np.array(valid), control_index=np.array(indices))
