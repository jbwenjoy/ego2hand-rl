"""Independent saved-array algebra for the preregistered same-prestate audit.

Only NumPy; no production converter, model, geometry or framework imports.
Native contacts are generated-contact witnesses, never an exhaustive gap oracle.
"""
from __future__ import annotations
import numpy as np

THRESHOLD_M = .003
NORM_M = .001
COUNTS = {'solver_pre': 1808, 'post_endpoints': 452, 'converted': 2260,
          'first_anchors': 160, 'donor': 1200, 'baseline': 96,
          'perturbation': 512, 'factory': 1, 'proxy_batches': 15}


def finite(a, shape_tail=None):
    a = np.asarray(a)
    if a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('Missing/nonfinite/nonnumeric saved array')
    if shape_tail is not None and a.shape[-len(shape_tail):] != tuple(shape_tail):
        raise ValueError('Saved shape mismatch')
    return a


def arithmetic_bound(converted, reference):
    a, b = finite(converted), finite(reference)
    if a.shape != b.shape:
        raise ValueError('Comparison shape mismatch')
    scale = np.maximum(1., np.maximum(np.abs(a.astype('f8')), np.abs(b.astype('f8'))))
    error = np.abs(a.astype('f8')-b.astype('f8'))
    return error <= 8*np.finfo('f4').eps*scale, error


def state58_comparison(converted, reference):
    """q20/qd20 must be saved FP32 bits; other18 use prereg 8eps32 bound."""
    a, b = finite(converted, (58,)), finite(reference, (58,))
    if a.dtype != np.float32 or b.dtype != np.float32 or a.shape != b.shape:
        raise ValueError('Native58 comparison requires same-shaped FP32 arrays')
    exact = np.array_equal(a[..., :40].view('u4'), b[..., :40].view('u4'))
    ok, error = arithmetic_bound(a[..., 40:], b[..., 40:])
    return {'q_qd_exact_bits': bool(exact), 'other18_within_arithmetic': bool(ok.all()),
            'other18_max_abs': float(error.max(initial=0.))}


def proxy_from_saved_sdf(sdf, sample_geom_ids):
    """Reproduce frozen FP32 P arithmetic; no SDF or surface query is made."""
    d, ids = finite(sdf), np.asarray(sample_geom_ids)
    if d.dtype != np.float32 or d.ndim != 2 or ids.shape != (d.shape[1],) or ids.dtype.kind not in 'iu':
        raise ValueError('Require FP32 rows x points and native integer sample geom IDs')
    if d.shape[1] == 0 or (ids < 0).any():
        raise ValueError('Empty/unknown proxy sample catalog')
    minimum = d.min(axis=1)
    excess = np.maximum(-d-np.float32(THRESHOLD_M), np.float32(0)).max(axis=1)/np.float32(NORM_M)
    P = excess*excess
    geoms = np.unique(ids)
    pergeom = np.stack([d[:, ids == g].min(axis=1) for g in geoms], axis=1)
    return {'min_sdf_m': minimum, 'sample_depth_m': np.maximum(-minimum, np.float32(0)),
            'P': P, 'BP': -np.float32(.1)*P, 'P_positive': P > 0,
            'source_F32_min_below_allowance': minimum < -np.float32(THRESHOLD_M),
            'source_F32_min_equal_allowance': minimum == -np.float32(THRESHOLD_M),
            'geom_ids': geoms, 'pergeom_min_sdf_m': pergeom,
            'geom_sample_counts': np.asarray([(ids == g).sum() for g in geoms], dtype='i8')}


def native_ho_witnesses(geom_pairs, distance_m, worldid, hand_ids, object_ids, valid_native_geom_ids):
    """WORLD0 exact generated HO slice, including zero-force witnesses.

    Exact deepest ties are all retained. Secondary all-violating pairs overlap
    primary labels and must never be summed as exclusive causal categories.
    """
    pairs, d, world = np.asarray(geom_pairs), finite(distance_m), np.asarray(worldid)
    if pairs.shape != (len(d), 2) or pairs.dtype.kind not in 'iu' or world.shape != d.shape or world.dtype.kind not in 'iu':
        raise ValueError('Malformed contact/geom/world inventory')
    if (world != 0).any() or not np.isin(pairs, list(valid_native_geom_ids)).all():
        raise ValueError('Unknown geom/world is not an empty contact set')
    hand, obj = set(hand_ids), set(object_ids)
    if not hand or not obj or hand & obj:
        raise ValueError('Exact disjoint hand/object geom catalogs required')
    first = np.isin(pairs[:, 0], list(hand)) & np.isin(pairs[:, 1], list(obj))
    second = np.isin(pairs[:, 1], list(hand)) & np.isin(pairs[:, 0], list(obj))
    indices = np.flatnonzero(first | second)
    depth = np.maximum(-d[indices], 0.)
    maximum = float(depth.max(initial=0.))
    tied = indices[depth == maximum] if maximum > 0 else np.empty(0, dtype='i8')
    violating = indices[depth > THRESHOLD_M]
    def handgeom(i):
        return int(pairs[i, 0] if first[i] else pairs[i, 1])
    def pairkey(i):
        return tuple(sorted(map(int, pairs[i])))
    return {'observed_generated_HO_depth_m': maximum, 'generated_HO_contact_count': len(indices),
            'HO_contact_indices': indices, 'native_gt_3mm': maximum > THRESHOLD_M,
            'deepest_exact_tie_contact_indices': tied,
            'deepest_exact_tie_hand_geom_ids': np.asarray([handgeom(i) for i in tied], dtype='i8'),
            'all_violating_contact_indices_secondary': violating,
            'all_violating_pairs_secondary': sorted({pairkey(i) for i in violating}),
            'all_violating_hand_geom_ids_secondary': np.asarray([handgeom(i) for i in violating], dtype='i8'),
            'empty_means': 'No generated HO witness; not an exhaustive nonpenetration certificate'}


def miss_labels(hand_geom_ids, sampled_geom_ids, pergeom_min_sdf_m):
    """Return overlapping evidence labels for exact ties or secondary pairs."""
    samples = np.asarray(sampled_geom_ids)
    mins = finite(pergeom_min_sdf_m)
    if samples.shape != mins.shape or len(np.unique(samples)) != len(samples):
        raise ValueError('Per-geom minimum catalog must be one-to-one')
    labels = []
    for g in hand_geom_ids:
        match = np.flatnonzero(samples == g)
        if len(match) == 0:
            label = 'GEOM_SET_COVERAGE_GAP'
        elif mins[match[0]] >= -np.float32(THRESHOLD_M):
            label = 'SAME_POSE_PROXY_UNDERSENSITIVITY_CANDIDATE'
        else:
            label = 'PROXY_GEOM_RECORDED_BELOW_ALLOWANCE'
        labels.append({'native_hand_geom_id': int(g), 'label': label})
    return labels


def support_match_from_saved(current, target):
    """Secondary frozen scoring arithmetic on captured (...,hand,body,basis)."""
    c, t = finite(current), finite(target)
    if c.shape != t.shape or c.ndim < 3:
        raise ValueError('Support tensor shape mismatch')
    commanded, live = t > 1e-3, c > 1e-3
    low, high = np.maximum(.9*t-c, 0), np.maximum(c-1.1*t, 0)
    score = np.where(commanded & live, np.exp(-(low*low+high*high)/.1), 0)
    bodies = commanded.any(axis=-1)
    body_score = score.sum(axis=-1)/np.maximum(commanded.sum(axis=-1), 1)
    hands = bodies.any(axis=-1)
    hand_score = body_score.sum(axis=-1)/np.maximum(bodies.sum(axis=-1), 1)
    return hand_score.sum(axis=-1)/np.maximum(hands.sum(axis=-1), 1)


def confusion(native_positive, proxy_positive):
    n, p = np.asarray(native_positive), np.asarray(proxy_positive)
    if n.dtype != bool or p.dtype != bool or n.shape != p.shape:
        raise ValueError('Boolean all-row masks required')
    return {'both_observed': int((n & p).sum()), 'native_only_observed': int((n & ~p).sum()),
            'proxy_only_vs_generated': int((~n & p).sum()), 'neither_observed': int((~n & ~p).sum())}
