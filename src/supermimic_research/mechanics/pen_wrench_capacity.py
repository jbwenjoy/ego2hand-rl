"""CPU directional capacity of a set of pyramidal hand/object contacts."""
from itertools import combinations, product

import numpy as np

from .multiclip_wrench_screen import contact_rays


def balanced_directions():
    """72 unit BODY probes: signed coordinate axes, then all signed axis pairs.

    Wrench coordinates are [Fx, Fy, Fz, Mx/L, My/L, Mz/L]. Each probe has
    an antipode, and the second moment is I/6. No random seed is needed.
    """
    axes = np.eye(6)
    directions = [*axes, *(-axes)]
    for i, j in combinations(range(6), 2):
        for a, b in product((-1., 1.), repeat=2):
            directions.append((a * axes[i] + b * axes[j]) / np.sqrt(2.))
    return np.array(directions)


def _proper_rotation(matrix):
    return (np.isfinite(matrix).all() and
            np.allclose(matrix.T @ matrix, np.eye(3), rtol=0., atol=1e-5) and
            np.linalg.det(matrix) > 0.)


def body_wrench_rays(contacts, signs, *, com_world, world_from_body, length_scale_m):
    """Return a (K,6) array of [BODY force, BODY torque about COM / L].

    Supply only the desired hand/object contact set. Each contact needs MuJoCo
    fields pos, frame (world axes in rows), dim (1, 3, 4, or 6), and friction.
    sign is +1 when geom2 is the object and -1 when geom1 is the object. Each
    ray carries one newton of *normal* force; direct torsional/rolling moments
    and contact-position lever arms are both included before division by L.
    """
    center = np.asarray(com_world, dtype=np.float64)
    rot = np.asarray(world_from_body, dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('A finite world COM position is required')
    if rot.shape != (3, 3) or not _proper_rotation(rot):
        raise ValueError('world_from_body must be a proper rotation matrix')
    if not np.isscalar(length_scale_m) or not np.isfinite(length_scale_m) or length_scale_m <= 0:
        raise ValueError('length_scale_m must be a positive finite length in metres')

    rays = []
    for contact, sign in zip(contacts, signs, strict=True):
        if sign not in (-1, 1) or contact.dim not in (1, 3, 4, 6):
            raise ValueError('Invalid geom sign or MuJoCo contact dimension')
        pos = np.asarray(contact.pos)
        mu = np.asarray(contact.friction)
        frame = np.asarray(contact.frame).reshape(3, 3)
        if pos.shape != (3,) or not np.isfinite(pos).all() or not _proper_rotation(frame):
            raise ValueError('Invalid contact position or frame')
        if (mu.ndim != 1 or len(mu) < contact.dim - 1 or
                not np.isfinite(mu[:contact.dim - 1]).all() or np.any(mu[:contact.dim - 1] < 0)):
            raise ValueError('Invalid active contact friction coefficients')
        rays.extend(contact_rays(contact, sign, center))

    w = np.asarray(rays, dtype=np.float64).reshape(-1, 6)
    return np.column_stack((w[:, :3] @ rot, (w[:, 3:] @ rot) / length_scale_m))


def directional_support(rays, directions=None):
    """Support of the convex hull of zero and the rays along each probe.

    All nonnegative ray coefficients together sum to at most one. This sets a
    *shared* one-newton normal-force budget for the entire grasp. Repeated
    contact entries cannot multiply capacity. Non-unit user directions retain
    their magnitude, as required by positive homogeneity of the support.
    """
    w = np.asarray(rays, dtype=np.float64)
    d = balanced_directions() if directions is None else np.asarray(directions, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 6 or not np.isfinite(w).all():
        raise ValueError('Expected finite (K,6) wrench rays')
    if d.ndim != 2 or d.shape[1] != 6 or not len(d) or not np.isfinite(d).all():
        raise ValueError('Expected finite (N,6) probe directions, N > 0')
    return np.max(d @ w.T, axis=1, initial=0.)
