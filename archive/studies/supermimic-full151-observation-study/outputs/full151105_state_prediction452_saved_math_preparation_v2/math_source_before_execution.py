"""Independent NumPy-only all452 native state prediction error arithmetic.

No score/goal, angle, projection, orthogonalization, state conversion or model.
Each block retains original units. There is no mixed-unit overall58 scalar.
q errors are direct differences, never wrapped. R is raw9 element difference.
"""
import numpy as np

BLOCKS = {
    'q': (0, 20, 'rad'), 'qd': (20, 40, 'rad/s'), 'p': (40, 43, 'm'),
    'R': (43, 52, 'dimensionless'), 'v': (52, 55, 'm/s'), 'omega': (55, 58, 'rad/s')}
PHASES = (0, 49, 99, 149)


def expected_identities():
    """Original full05 order: donor150 then restored76 per path, never sorted."""
    identities = []
    for path_id, path in enumerate(('A0', 'A2')):
        scope = [('donor', k, None) for k in range(150)]
        for capture, k in enumerate(PHASES):
            sid = path_id*4+capture
            scope.extend(('baseline' if offset < 3 else 'direction', k, sid*19+offset) for offset in range(19))
        for mode, k, bid in scope:
            i = len(identities)
            identities.append({'group_id': i, 'path': path, 'mode': mode, 'k': k, 'branch_id': bid,
                'source_frame': 200+k, 'terminal_observation': k+1,
                'valid_next_policy_decision': k+1 < 150, 'no_u150': True,
                'first_physical_pre_index': 4*i, 'last_physical_index': 4*i+3,
                'post20_GPU_row_index': 1808+i})
    return identities


def validate_identity_order(rows):
    expected = expected_identities()
    if len(rows) != 452:
        raise ValueError('Every452 transition must be retained')
    for i, (row, wanted) in enumerate(zip(rows, expected)):
        if any(row.get(key) != value or type(row.get(key)) is not type(value) for key, value in wanted.items()):
            raise ValueError('Original identity/order/types differ at group '+str(i))
    return True


def signed_errors(predicted, target):
    p, t = np.asarray(predicted), np.asarray(target)
    if p.shape != (452, 58) or t.shape != p.shape or p.dtype != np.float32 or t.dtype != np.float32 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError('Exact full452 finite FP32 prediction and target arrays required')
    delta = p.astype(np.float64)-t.astype(np.float64)
    return delta, abs(delta)


def group_statistics(delta, mask):
    d = np.asarray(delta); mask = np.asarray(mask)
    if d.shape != (452, 58) or d.dtype != np.float64 or not np.isfinite(d).all() or mask.dtype != bool or mask.shape != (452,) or not mask.any():
        raise ValueError('Finite full452 F64 errors and a nonempty typed subset mask required')
    x = d[mask]
    features = {'signed_mean': x.mean(axis=0).tolist(), 'mean_absolute': abs(x).mean(axis=0).tolist(),
        'RMSE': np.sqrt(np.mean(x*x, axis=0)).tolist(), 'maximum_absolute': abs(x).max(axis=0).tolist()}
    result = {'rows': len(x), 'per_feature': features, 'blocks': {}}
    for name, (start, stop, units) in BLOCKS.items():
        b = x[:, start:stop]
        value = {'start': start, 'stop_exclusive': stop, 'component_count': stop-start, 'units': units,
            'component_RMSE': float(np.sqrt(np.mean(b*b))), 'mean_absolute': float(abs(b).mean()),
            'maximum_absolute': float(abs(b).max())}
        if name in ('p', 'v', 'omega'):
            value['vector_RMS'] = float(np.sqrt(np.mean(np.sum(b*b, axis=-1))))
        elif name == 'R':
            value['Frobenius_RMS'] = float(np.sqrt(np.mean(np.sum(b*b, axis=-1))))
        result['blocks'][name] = value
    return result


def all_statistics(predicted, target, identities):
    validate_identity_order(identities)
    delta, absolute = signed_errors(predicted, target)
    result = {'all452': group_statistics(delta, np.ones(452, dtype=bool))}
    for path in ('A0', 'A2'):
        for mode in ('donor', 'baseline', 'direction'):
            mask = np.asarray([row['path'] == path and row['mode'] == mode for row in identities], dtype=bool)
            result[path+'/'+mode] = group_statistics(delta, mask)
    counts = {k: v['rows'] for k, v in result.items()}
    expected = {'all452': 452, 'A0/donor': 150, 'A0/baseline': 12, 'A0/direction': 64,
                'A2/donor': 150, 'A2/baseline': 12, 'A2/direction': 64}
    if counts != expected:
        raise ValueError('Fixed all-cohort/pathmode row coverage changed')
    return {'signed_error452x58_F64': delta, 'absolute_error452x58_F64': absolute,
            'statistics': result, 'feature_units': [unit for start, stop, unit in BLOCKS.values() for _ in range(stop-start)]}


def held_controls_exact(raw_substep_u, expected_u):
    a, b = np.asarray(raw_substep_u), np.asarray(expected_u)
    return (a.shape == (452, 4, 20) and b.shape == (452, 20) and a.dtype == b.dtype == np.float32
            and np.isfinite(a).all() and np.isfinite(b).all()
            and np.array_equal(a.view('u4'), np.broadcast_to(b[:, None, :], a.shape).copy().view('u4')))
