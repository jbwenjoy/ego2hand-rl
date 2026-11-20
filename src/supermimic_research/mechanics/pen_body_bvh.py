"""Geometry-only refit for an exported three-piece rigid pen's body BVH."""
import copy

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _rotation(quat):
    return Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()


def _solid_halfwidth(model, geom, axes):
    """Analytic support radius of the centered solid along each query axis."""
    size = model.geom_size[geom]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        return np.abs(axes) @ size
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return size[0] * np.linalg.norm(axes[:, :2], axis=1) + size[1] * np.abs(axes[:, 2])
    raise ValueError('Only the pen shaft cylinder and its box ends are supported')


def copy_refit_pen_bvh(model, object_body):
    """Return (independent MjModel copy, proof) using current geom and inertial frames.

    No call to mj_setConst is needed. Invoke after the physical fields have
    been finalized. Refit only the pen body's tree; all other body/mesh BVH
    nodes and all source-model fields remain unchanged. The input must have
    valid per-geom boxes, since the CPU leaf-vs-leaf stage also uses them.
    """
    b = model.body(object_body).id if isinstance(object_body, str) else int(object_body)
    if not 0 < b < model.nbody:
        raise ValueError('Invalid object body')
    adr, count = int(model.body_bvhadr[b]), int(model.body_bvhnum[b])
    if adr < 0 or count != 5:
        raise ValueError('Expected the three-piece pen body BVH')
    ids = model.bvh_nodeid[adr:adr+count]
    leaves = [int(g) for g in ids if g >= 0]
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    cylinder = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    if (len(set(leaves)) != 3 or any(model.geom_bodyid[g] != b for g in leaves)
            or sorted(model.geom_type[leaves]) != sorted([box, box, cylinder])):
        raise ValueError('Expected one shaft cylinder and two end boxes on a single body')

    old = model.bvh_aabb[adr:adr+count].copy()
    new = np.zeros_like(old)
    descendants = {}
    ib = _rotation(model.body_iquat[b]).T
    for node in np.argsort(-model.bvh_depth[adr:adr+count]):
        geom = int(ids[node])
        children = model.bvh_child[adr+node]
        if geom >= 0:
            if np.any(children >= 0):
                raise ValueError('Invalid BVH leaf')
            R = _rotation(model.geom_quat[geom])
            bounds = model.geom_aabb[geom]
            local_radius = _solid_halfwidth(model, geom, np.eye(3))
            if np.any(np.abs(bounds[:3]) + local_radius - bounds[3:] > 1e-8):
                raise ValueError('Per-geom AABB fails analytic solid enclosure')
            # A rigid transform maps the *box* halfwidth to abs(R) @ halfwidth.
            c = ib @ (model.geom_pos[geom] + R @ bounds[:3] - model.body_ipos[b])
            r = np.abs(ib @ R) @ np.maximum(bounds[3:], np.abs(bounds[:3]) + local_radius)
            descendants[node] = [geom]
        else:
            if not all(child in descendants for child in children):
                raise ValueError('Invalid BVH child indexing or depth')
            lo = np.min(new[children, :3] - new[children, 3:], axis=0)
            hi = np.max(new[children, :3] + new[children, 3:], axis=0)
            c, r = (hi + lo)/2, (hi - lo)/2
            descendants[node] = descendants[children[0]] + descendants[children[1]]
        new[node] = np.r_[c, r]
    if sorted(descendants[0]) != sorted(leaves):
        raise ValueError('Pen root does not span its three parts')

    # Audit against analytic extrema of each *physical primitive* rather than
    # rechecking the boxes used for construction. Check both sides of each
    # node along all three inertial-frame axes, for all its descendant solids.
    proof = []
    for node, geoms in sorted(descendants.items()):
        old_excess, new_excess = 0., 0.
        for geom in geoms:
            pos = ib @ (model.geom_pos[geom] - model.body_ipos[b])
            support = _solid_halfwidth(model, geom, ib @ _rotation(model.geom_quat[geom]))
            for is_new, bounds in ((False, old[node]), (True, new[node])):
                over = np.max(np.abs(pos - bounds[:3]) + support - bounds[3:])
                if is_new:
                    new_excess = max(new_excess, float(over))
                else:
                    old_excess = max(old_excess, float(over))
        if new_excess > 1e-10:
            raise ValueError('Refitted node fails analytic solid enclosure')
        proof.append(dict(node=int(node), descendant_geom_ids=geoms, old_excess_m=old_excess,
                          new_excess_m=new_excess))

    repaired = copy.copy(model)
    if np.shares_memory(repaired.bvh_aabb, model.bvh_aabb):
        raise RuntimeError('MjModel copy unexpectedly shares BVH storage')
    repaired.bvh_aabb[adr:adr+count] = new
    return repaired, dict(body=int(b), address=adr, rows=count, node_geoms=ids.tolist(),
                          old=old.tolist(), refit=new.tolist(), checked_nodes=proof,
                          inertial_quaternion_wxyz=model.body_iquat[b].tolist())
