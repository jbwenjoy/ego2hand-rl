"""CPU-only, outcome-blind directions for a symmetric interior subspace.

These directions are with respect to absolute executed joint-position targets.
They do not describe the whole feasible tangent cone at active C2 bounds.
No control target is clipped or repaired. No framework or simulator is imported.
"""
from __future__ import annotations

import numpy as np


def prepare_interior_directions(centers, lower, upper, *, seed, count=8,
                                epsilons=(.002, .001), guard_ulps=8):
    """Return deterministic directions and exact FP32 intervention targets.

    Draw once in (state, direction, all20) order; zero non-free coordinates,
    then normalize in FP64. A coordinate is free only when both boundary
    distances are at least max(epsilons) plus a declared FP32 roundoff guard.
    A zero-dimensional state remains explicitly NOT_RUN; it is not resampled.
    """
    centers = np.asarray(centers)
    lower, upper = np.asarray(lower), np.asarray(upper)
    if centers.dtype != np.float32 or centers.ndim != 2 or centers.shape[1] != 20:
        raise ValueError('centers must be explicit FP32 states x 20')
    if lower.dtype != np.float32 or upper.dtype != np.float32:
        raise ValueError('bounds must be explicit native FP32 inner bounds')
    if lower.shape != (20,) or upper.shape != (20,):
        raise ValueError('bounds must have 20 coordinates')
    if not all(np.isfinite(x).all() for x in (centers, lower, upper)):
        raise ValueError('nonfinite control domain')
    if not np.all(lower < upper) or not np.all((centers >= lower) & (centers <= upper)):
        raise ValueError('empty domain or nominal target outside it; never repair')
    if type(seed) is not int or seed < 0 or type(count) is not int or count <= 0:
        raise ValueError('explicit nonnegative seed and positive direction count required')
    if type(guard_ulps) is not int or guard_ulps < 1:
        raise ValueError('positive predeclared roundoff guard required')
    eps = np.asarray(epsilons, dtype=np.float64)
    if eps.ndim != 1 or not len(eps) or not np.isfinite(eps).all() or not np.all(eps > 0):
        raise ValueError('positive finite epsilon sequence required')
    if len(set(eps.tolist())) != len(eps):
        raise ValueError('duplicate epsilon')
    c = centers.astype(np.float64)
    lo, hi = lower.astype(np.float64), upper.astype(np.float64)
    scale = np.maximum.reduce((np.ones_like(c), abs(c),
                               np.broadcast_to(abs(lo), c.shape),
                               np.broadcast_to(abs(hi), c.shape)))
    guard = guard_ulps * np.finfo(np.float32).eps * scale
    slack = np.minimum(c - lo, hi - c)
    free = slack >= eps.max() + guard
    # A local Generator leaves legacy global NumPy RNG state untouched.
    normal = np.random.default_rng(seed).standard_normal((len(c), count, 20))
    vectors = normal * free[:, None, :]
    norms = np.linalg.norm(vectors, axis=-1)
    nonzero = norms > 0
    vectors = np.divide(vectors, norms[..., None], out=np.zeros_like(vectors),
                        where=nonzero[..., None])
    if not np.array_equal(nonzero, np.broadcast_to(free.any(-1, keepdims=True), norms.shape)):
        raise ValueError('degenerate random direction; no quality-dependent redraw')
    delta = eps[None, None, :, None] * vectors[:, :, None, :]
    plus = (c[:, None, None, :] + delta).astype(np.float32)
    minus = (c[:, None, None, :] - delta).astype(np.float32)
    crossing = (plus < lower) | (plus > upper) | (minus < lower) | (minus > upper)
    valid = nonzero[:, :, None] & ~crossing.any(-1)
    expected_valid = np.broadcast_to(nonzero[:, :, None], valid.shape)
    if not np.array_equal(valid, expected_valid):
        raise ValueError('predeclared guard failed domain safety; no target repair')
    effective = (plus.astype(np.float64) - minus.astype(np.float64)) / (2 * eps[None, None, :, None])
    return dict(centers=centers.copy(), lower=lower.copy(), upper=upper.copy(),
                epsilons=eps.copy(), guard=guard, slack=slack, free_mask=free,
                standard_normals=normal, vectors=vectors, plus=plus, minus=minus,
                crossing_mask=crossing, valid_pair=valid, effective_vectors=effective)
