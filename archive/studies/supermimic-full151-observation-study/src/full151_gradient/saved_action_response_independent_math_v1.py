"""Independent saved-only response algebra. No model, native method or data IO.

All19 visible prestates must match before a fixed-visible-state interpretation.
Baseline means are F64 statistical references, never valid poses/actions. There
is no accuracy floor, stable mask, score, AD, mixed58 norm or Jacobian claim.
"""
import numpy as np

JOINTS = (0, 4, 8, 12)
EPS = (.002, .001)
BLOCKS = {'q': (0, 20, 'rad', 'rad/rad'), 'qd': (20, 40, 'rad/s', '(rad/s)/rad'),
          'p': (40, 43, 'm', 'm/rad'), 'R': (43, 52, 'dimensionless', '1/rad'),
          'v': (52, 55, 'm/s', '(m/s)/rad'), 'omega': (55, 58, 'rad/s', '(rad/s)/rad')}


def f32(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError('Every original finite FP32 saved row is mandatory')
    return a


def visible_state_guard(pre, snapshot):
    pre, snapshot = f32(pre, (8, 19, 58)), f32(snapshot, (8, 58))
    wanted = np.broadcast_to(snapshot[:, None, :], pre.shape).copy()
    qbits = pre[..., :40].view('u4') == wanted[..., :40].view('u4')
    numeric = pre[..., 40:] == wanted[..., 40:]
    rawbits = pre.view('u4') == wanted.view('u4')
    return {'q_qd_exact_bits': qbits, 'other18_exact_numeric': numeric,
            'all58_raw_bit_equal_mask': rawbits, 'same_visible58_all_passed': bool(qbits.all() and numeric.all())}


def control_guard(u, u0, archived):
    u, u0 = f32(u, (8, 19, 20)), f32(u0, (8, 20))
    for name in ('plus', 'minus'):
        f32(archived[name], (8, 4, 2, 20))
    for name in ('d_eff', 'center_offset'):
        a = np.asarray(archived[name])
        if a.shape != (8, 4, 2, 20) or a.dtype != np.float64 or not np.isfinite(a).all():
            raise ValueError('Original archived F64 directions/offsets required')
    if np.asarray(archived['clipped']).shape != (8, 4, 2) or archived['clipped'].dtype != bool:
        raise ValueError('Original archived clipping bool required')
    baseline_bits = u[:, :3].view('u4') == np.broadcast_to(u0[:, None], (8, 3, 20)).copy().view('u4')
    nonj = np.ones((8, 16, 20), dtype=bool); endpoint_matches = np.ones((8, 4, 2, 2), dtype=bool)
    legacy_ok = True; positive_order = np.ones((8, 2, 4), dtype=bool)
    for s in range(8):
        for ei, eps in enumerate(EPS):
            for ji, joint in enumerate(JOINTS):
                offset = 3+ei*8+ji*2; minus, plus = u[s, offset], u[s, offset+1]
                keep = np.arange(20) != joint
                nonj[s, offset-3, keep] = minus.view('u4')[keep] == u0[s].view('u4')[keep]
                nonj[s, offset-2, keep] = plus.view('u4')[keep] == u0[s].view('u4')[keep]
                endpoint_matches[s, ji, ei, 0] = np.array_equal(minus.view('u4'), archived['minus'][s, ji, ei].view('u4'))
                endpoint_matches[s, ji, ei, 1] = np.array_equal(plus.view('u4'), archived['plus'][s, ji, ei].view('u4'))
                delta = plus.astype('f8')-minus.astype('f8')
                center = (plus.astype('f8')+minus.astype('f8'))*.5-u0[s].astype('f8')
                unit = np.eye(20, dtype='f4')[joint]
                requested_plus = u0[s]+np.float32(eps)*unit
                requested_minus = u0[s]-np.float32(eps)*unit
                clipped = not (np.array_equal(plus, requested_plus) and np.array_equal(minus, requested_minus))
                legacy_ok &= np.array_equal(delta/(2*eps), archived['d_eff'][s, ji, ei])
                legacy_ok &= np.array_equal(center, archived['center_offset'][s, ji, ei])
                legacy_ok &= bool(archived['clipped'][s, ji, ei]) == clipped
                positive_order[s, ei, ji] = delta[joint] >= 0 and float(plus[joint]) >= float(u0[s, joint]) >= float(minus[joint])
    return {'baseline_u0_exact_bits': baseline_bits, 'nonj_exact_bit_mask': nonj,
            'archived_endpoint_exact_matches': endpoint_matches, 'nonnegative_source_order_mask': positive_order,
            'legacy_fields_exact': bool(legacy_ok),
            'all_control_guards_passed': bool(baseline_bits.all() and nonj.all() and endpoint_matches.all() and positive_order.all() and legacy_ok)}


def noise3(raw):
    x = raw.astype('f8'); mean = x.mean(axis=1); centered = x-mean[:, None]
    return {'mean_F64': mean, 'range_F64': x.max(1)-x.min(1),
            'population_variance_F64': (centered*centered).mean(1),
            'maximum_absolute_centered_F64': abs(centered).max(1)}


def block_description(values, secant=False, defined=None):
    a = np.asarray(values)
    if a.dtype != np.float64 or a.shape[-1] != 58:
        raise ValueError('F64 separate-coordinate errors required')
    flat = a.reshape(-1, 58)
    if defined is None:
        if not np.isfinite(flat).all():
            raise ValueError('Unexpected undefined response')
        valid = np.ones(len(flat), dtype=bool)
    else:
        valid = np.asarray(defined)
        if valid.dtype != bool or valid.shape != (len(flat),) or not np.isfinite(flat[valid]).all() or not np.isnan(flat[~valid]).all():
            raise ValueError('Undefined secants must remain NaN plus explicit mask')
    selected = flat[valid]; result = {'rows': len(flat), 'defined_rows': int(valid.sum()), 'undefined_rows': int((~valid).sum()),
        'per_feature': {'RMS': [None]*58 if not len(selected) else np.sqrt((selected*selected).mean(0)).tolist(),
            'mean_absolute': [None]*58 if not len(selected) else abs(selected).mean(0).tolist(),
            'maximum_absolute': [None]*58 if not len(selected) else abs(selected).max(0).tolist()}, 'blocks': {}}
    for name, (lo, hi, unit, slopeunit) in BLOCKS.items():
        b = selected[:, lo:hi]
        result['blocks'][name] = {'units': slopeunit if secant else unit,
            'component_RMS': None if not len(b) else float(np.sqrt((b*b).mean())),
            'mean_absolute': None if not len(b) else float(abs(b).mean()),
            'maximum_absolute': None if not len(b) else float(abs(b).max())}
        if name in ('p', 'v', 'omega', 'R'):
            result['blocks'][name]['Frobenius_RMS' if name == 'R' else 'vector_RMS'] = None if not len(b) else float(np.sqrt((b*b).sum(-1).mean()))
    return result


def decompose(pre, physical, predicted, u, snapshot_x, u0, archived):
    pre = f32(pre, (8, 19, 58)); physical = f32(physical, (8, 19, 58)); predicted = f32(predicted, (8, 19, 58))
    u = f32(u, (8, 19, 20)); u0 = f32(u0, (8, 20))
    visible, control = visible_state_guard(pre, snapshot_x), control_guard(u, u0, archived)
    result = {'visible_guards': visible, 'control_guards': control,
              'fixed_visible_state_interpretation_allowed': visible['same_visible58_all_passed'] and control['all_control_guards_passed']}
    if not result['fixed_visible_state_interpretation_allowed']:
        result['status'] = 'BLOCKED_FIXED_VISIBLE_STATE_OR_CONTROL_MAPPING_NO_REPAIR'
        return result
    yn, pn, un = noise3(physical[:, :3]), noise3(predicted[:, :3]), noise3(u[:, :3])
    ybar, pbar, ubar = yn['mean_F64'], pn['mean_F64'], un['mean_F64']
    bias = pbar-ybar
    response_phys = physical[:, 3:].astype('f8')-ybar[:, None]
    response_pred = predicted[:, 3:].astype('f8')-pbar[:, None]
    residual = response_pred-response_phys
    endpoint_error = predicted[:, 3:].astype('f8')-physical[:, 3:].astype('f8')
    arrays = {'baseline_physical_mean_F64': ybar, 'baseline_predicted_mean_F64': pbar, 'baseline_control_mean_F64': ubar,
        'baseline_bias_F64': bias, 'physical_response128_F64': response_phys.reshape(128, 58),
        'predicted_response128_F64': response_pred.reshape(128, 58), 'response_residual128_F64': residual.reshape(128, 58),
        'endpoint_error128_F64': endpoint_error.reshape(128, 58),
        'bias_identity_closure128_F64': (endpoint_error-(bias[:, None]+residual)).reshape(128, 58),
        'actual_du_from_baseline_mean128_rad': (u[:, 3:].astype('f8')-ubar[:, None]).reshape(128, 20),
        'actual_du_from_original_u0_128_rad': (u[:, 3:].astype('f8')-u0[:, None].astype('f8')).reshape(128, 20)}
    pairs = []; actual_delta = []; halfstep = []; numer_y = []; numer_p = []; denom = []; hplus = []; hminus = []; centers = []; legacy_d = []; clips = []
    for s in range(8):
        for ei, eps in enumerate(EPS):
            for ji, joint in enumerate(JOINTS):
                off = 3+ei*8+ji*2; minus, plus = u[s, off].astype('f8'), u[s, off+1].astype('f8')
                du = plus-minus; actual_delta.append(du); halfstep.append(du/2); denom.append(du[joint])
                numer_y.append(physical[s, off+1].astype('f8')-physical[s, off].astype('f8'))
                numer_p.append(predicted[s, off+1].astype('f8')-predicted[s, off].astype('f8'))
                hplus.append(plus[joint]-float(u0[s, joint])); hminus.append(float(u0[s, joint])-minus[joint])
                centers.append((plus+minus)/2-u0[s].astype('f8')); legacy_d.append(archived['d_eff'][s, ji, ei]); clips.append(archived['clipped'][s, ji, ei])
                pairs.append({'pair_id': len(pairs), 'snapshot_id': s, 'joint_ordinal': ji, 'joint_index': joint,
                    'epsilon_index': ei, 'requested_epsilon_rad': eps, 'minus_branch_id': s*19+off, 'plus_branch_id': s*19+off+1})
    denom = np.asarray(denom, dtype='f8'); defined = denom != 0
    ynum, pnum = np.asarray(numer_y), np.asarray(numer_p)
    sy, sp = np.full((64, 58), np.nan), np.full((64, 58), np.nan)
    np.divide(ynum, denom[:, None], out=sy, where=defined[:, None]); np.divide(pnum, denom[:, None], out=sp, where=defined[:, None])
    arrays.update(actual_delta_u64_rad=np.asarray(actual_delta), actual_halfstep64_rad=np.asarray(halfstep),
        actual_denominator64_rad=denom, secant_defined64=defined, physical_numerator64_F64=ynum, predicted_numerator64_F64=pnum,
        physical_secant64_F64=sy, predicted_secant64_F64=sp, secant_residual64_F64=sp-sy,
        hplus64_rad=np.asarray(hplus), hminus64_rad=np.asarray(hminus), hplus_minus_hminus64_rad=np.asarray(hplus)-np.asarray(hminus),
        midpoint_offset64_rad=np.asarray(centers), legacy_d_eff64_dimensionless=np.asarray(legacy_d), legacy_clipped64=np.asarray(clips, dtype=bool))
    summaries = {'baseline_bias': block_description(bias), 'physical_response': block_description(arrays['physical_response128_F64']),
        'predicted_response': block_description(arrays['predicted_response128_F64']), 'response_residual': block_description(arrays['response_residual128_F64']),
        'physical_secant': block_description(sy, True, defined), 'predicted_secant': block_description(sp, True, defined),
        'secant_residual': block_description(sp-sy, True, defined)}
    result.update(status='COMPLETED_SAVED_ONLY_RESPONSE_ALGEBRA_NOT_JACOBIAN', arrays=arrays,
        baseline_noise={'physical': yn, 'predicted': pn, 'control': un}, pair_identities=pairs, summaries=summaries)
    return result
