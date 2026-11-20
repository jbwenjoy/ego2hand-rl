"""Object-frame contact capacity and measured-wrench containment per native tick."""
from types import SimpleNamespace

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .pen_contact_groups import normal_group
from .pen_wrench_capacity import balanced_directions, body_wrench_rays, directional_support


def packet_capacity(z, model, layout, geom_friction, *, length_scale_m=0.10):
    """Process an eight-tick OwnedProbeRecorder packet on CPU.

    Use hand/object contacts at nonpositive gap and same-tick decoded local
    loads. Pre-step quaternion and moving COM define the frame for each tick.
    Geometry counts even when the normal load is zero. Containment compares
    the measured BODY [F,M_COM/L] per newton of total hand normal load with
    the capacity on 72 fixed directions. Only ticks over 1e-8 N are tested.
    The skipped positive-gap diagnostic is the largest individual |Fn| in N.
    Explicit geom pairs need their recorded pair-specific friction metadata.
    """
    if model.npair or int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        raise ValueError('Expected the pen model with pyramidal cone and no explicit pairs')
    body, qa = layout['object_body'], layout['object_qposadr']
    raw = np.asarray(z['raw_contact_tick_plus23'], dtype=np.float64)
    qpos = np.asarray(z['pre_qpos'], dtype=np.float64)
    com = np.asarray(z['pre_object_COM'], dtype=np.float64)
    if (raw.ndim != 2 or raw.shape[1] != 24 or qpos.shape != (8, model.nq) or
            com.shape != (8, 3) or not all(np.isfinite(v).all() for v in (raw, qpos, com))):
        raise ValueError('Expected a finite eight-tick pen recorder packet')
    if not (0 <= body < model.nbody and 0 <= qa <= model.nq-7):
        raise ValueError('Invalid object body or free-joint position address')
    indices = raw[:, :3].astype(np.int64)
    if (not np.array_equal(indices, raw[:, :3]) or np.any(indices < 0) or
            np.any(indices[:, 0] >= 8) or np.any(indices[:, 1:] >= model.ngeom)):
        raise ValueError('Invalid tick or geometry index')
    friction = np.asarray(geom_friction, dtype=np.float64)
    if friction.shape == (1, model.ngeom, 3):
        friction = friction[0]
    if (friction.shape != (model.ngeom, 3) or not np.isfinite(friction).all() or
            np.any(friction < 0)):
        raise ValueError('Expected finite nonnegative per-geom pen friction')

    q = qpos[:, qa+np.array([4,5,6,3])]
    if np.max(np.abs(np.linalg.norm(q, axis=1)-1.)) > 1e-5:
        raise ValueError('Object quaternions must have unit norm')
    rotations = Rotation.from_quat(q).as_matrix()
    directions = balanced_directions()
    capacity = np.empty((8, len(directions)))
    normal = np.zeros(8)
    mask = np.zeros(8, dtype=np.uint8)
    world_wrench = np.zeros((8, 6))
    margin_load = np.zeros(8)
    contacts = [[] for _ in range(8)]
    signs = [[] for _ in range(8)]

    for row, (tick, g, h) in zip(raw, indices):
        on_g, on_h = model.geom_bodyid[[g,h]] == body
        if on_g == on_h:
            continue
        group = normal_group(model.geom(h if on_g else g).name or '')
        if group < 0:
            continue
        if row[5] > 0:
            margin_load[tick] = max(margin_load[tick], abs(row[18]))
            continue
        if row[18] < 0 or row[3] not in (1., 3., 4., 6.):
            raise ValueError('Invalid active normal load or contact dimension')
        priority_g, priority_h = model.geom_priority[[g,h]]
        if priority_g == priority_h:
            mu = np.maximum(friction[g], friction[h])
        else:
            mu = friction[g if priority_g > priority_h else h]
        frame = row[9:18].reshape(3, 3)
        sign = 1 if on_h else -1
        contacts[tick].append(SimpleNamespace(pos=row[6:9], frame=frame,
                                              dim=int(row[3]), friction=mu[[0,0,1,2,2]]))
        signs[tick].append(sign)
        mask[tick] |= 1 << group
        force = sign * frame.T @ row[18:21]
        world_wrench[tick,:3] += force
        world_wrench[tick,3:] += sign * frame.T @ row[21:24] + np.cross(row[6:9]-com[tick], force)
        normal[tick] += row[18]

    for tick in range(8):
        rays = body_wrench_rays(contacts[tick], signs[tick], com_world=com[tick],
                                world_from_body=rotations[tick], length_scale_m=length_scale_m)
        capacity[tick] = directional_support(rays, directions)
    body_wrench = (world_wrench.reshape(8,2,3) @ rotations).reshape(8,6)
    tested = normal > 1e-8
    normalized = np.zeros_like(body_wrench)
    normalized[tested] = body_wrench[tested] / normal[tested,None]
    normalized[:,3:] /= length_scale_m
    excess = np.zeros(8)
    excess[tested] = np.maximum(0., (normalized[tested] @ directions.T-capacity[tested]).max(axis=1))
    return dict(capacity=capacity, mask=mask, normal_load_N=normal, containment_tested=tested,
                wrench_world_com=world_wrench, wrench_body_com=body_wrench,
                normalized_wrench_body=normalized, containment_excess=excess,
                maximum_containment_excess=float(excess.max()), positive_gap_max_load_by_tick_N=margin_load,
                positive_gap_skipped_max_load_N=float(margin_load.max()))
