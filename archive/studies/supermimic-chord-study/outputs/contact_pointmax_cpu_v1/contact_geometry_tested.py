"""Interpretable differentiable hand-surface / analytic-pen contact proxy.

Inference consumes only state58: hand q and object pose in the palm frame.
Collision assets and fixed surface samples are buffers; no simulator state,
future contact, mount geometry or self-collision label is read at inference.
Soft contact is geometric proximity, not an estimated physical contact force.
"""
from __future__ import annotations

import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _quat_matrix(q):
    w, x, y, z = np.asarray(q)/np.linalg.norm(q)
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def box_sdf_normal(points, halfsize):
    """Exact box SDF and outward normal, piecewise differentiable."""
    halfsize = torch.as_tensor(halfsize, dtype=points.dtype, device=points.device)
    offset = points.abs()-halfsize
    outside = offset.clamp_min(0)
    length = torch.linalg.vector_norm(outside, dim=-1)
    maximum, face = offset.max(-1)
    sdf = length+maximum.clamp_max(0)
    sign = torch.where(points >= 0, 1., -1.)
    exterior = outside/length.clamp_min(1e-12)[..., None]*sign
    interior = F.one_hot(face, 3).to(points.dtype)*sign
    return sdf, torch.where((length > 1e-12)[..., None], exterior, interior)


def cylinder_sdf_normal(points, radius, halfheight):
    """Exact capped-cylinder SDF, axis local z; outward normal."""
    radial = torch.linalg.vector_norm(points[..., :2], dim=-1)
    dr, dz = radial-radius, points[..., 2].abs()-halfheight
    outside_r, outside_z = dr.clamp_min(0), dz.clamp_min(0)
    length = torch.sqrt(outside_r.square()+outside_z.square()+1e-30)
    sdf = length+torch.maximum(dr, dz).clamp_max(0)
    radial_direction = points[..., :2]/radial.clamp_min(1e-12)[..., None]
    radial_direction = torch.where((radial > 1e-12)[..., None], radial_direction,
                                   points.new_tensor([1., 0.]))
    nr = torch.cat((radial_direction, torch.zeros_like(radial[..., None])), -1)
    nz = torch.zeros_like(points)
    nz = torch.cat((nz[..., :2], torch.where(points[..., 2:3] >= 0, 1., -1.)), -1)
    exterior = outside_r[..., None]/length[..., None]*nr+outside_z[..., None]/length[..., None]*nz
    interior = torch.where((dr >= dz)[..., None], nr, nz)
    return sdf, torch.where((length > 1e-12)[..., None], exterior, interior)


def _coaxial_exterior_layout(model, objects, *, tolerance_m=1e-8):
    """Reject layouts where the specialized external-patch construction is invalid."""
    cylinder = next(index for index, geom in enumerate(objects) if int(model.geom_type[geom]) == 5)
    geom = objects[cylinder]
    center = np.asarray(model.geom_pos[geom])
    axis = _quat_matrix(model.geom_quat[geom])[:, 2]
    radius, halfheight = model.geom_size[geom, :2]
    if radius <= 0 or halfheight <= 0:
        raise ValueError("positive capped-cylinder dimensions required")
    specs, signs, records = [], [], []
    for component, box in enumerate(objects):
        if component == cylinder:
            continue
        rotation, size = _quat_matrix(model.geom_quat[box]), model.geom_size[box]
        alignment = rotation.T@axis
        axial = int(np.argmax(np.abs(alignment)))
        if abs(abs(alignment[axial])-1) > 1e-10 or np.max(np.abs(np.delete(alignment, axial))) > 1e-10:
            raise ValueError("pen exterior patches require box and cylinder axial alignment")
        displacement = model.geom_pos[box]-center
        along = float(displacement@axis)
        if np.linalg.norm(displacement-along*axis) > tolerance_m:
            raise ValueError("pen exterior patches require coaxial box centers")
        if abs(abs(along)-float(size[axial])-float(halfheight)) > tolerance_m:
            raise ValueError("pen exterior patches require exact cylinder/box seam abutment, no gap or overlap")
        transverse = np.delete(size, axial)
        if np.min(transverse) < radius-tolerance_m:
            raise ValueError("box cross section must contain the cylinder disk")
        side = 1 if along > 0 else -1
        inner_sign = -side*(1 if alignment[axial] > 0 else -1)
        specs.append((component, axial, inner_sign))
        signs.append(side)
        records.append({"box_component": component, "axial_local_axis": axial, "inner_face_sign": inner_sign,
                        "center_coordinate_along_cylinder_m": along, "transverse_halfsize_m": transverse.tolist()})
    if sorted(signs) != [-1, 1]:
        raise ValueError("one marker box is required at each cylinder end")
    return cylinder, specs, {"validated_specialization": "coaxial cylinder + two exactly abutting boxes whose cross sections contain the circular disk",
                             "validation_position_tolerance_m": tolerance_m, "cylinder_component": cylinder,
                             "cylinder_axis_body": axis.tolist(), "cylinder_radius_m": float(radius),
                             "cylinder_halfheight_m": float(halfheight), "box_patches": records}


def coaxial_exterior_sdf_normal(points_o, cylinder_position, cylinder_rotation, radius, halfheight, boxes):
    """True distance to visible exterior patches, signed by union membership.

    The cylinder contributes only its lateral surface (including circular rims).
    Each box contributes six rectangle faces, with the shared cylinder disk
    removed from its inner face. A projection inside that hole moves to the
    circle rim. This fixes hidden endcaps falsely appearing as zero SDF inside
    the rigid union. Geometry must first pass _coaxial_exterior_layout.
    """
    local = torch.einsum("ij,...pj->...pi", cylinder_rotation.T, points_o-cylinder_position)
    radial = torch.linalg.vector_norm(local[..., :2], dim=-1)
    direction = local[..., :2]/radial.clamp_min(1e-12)[..., None]
    direction = torch.where((radial > 1e-12)[..., None], direction, local.new_tensor([1., 0.]))
    side_local = torch.cat((radius*direction, local[..., 2:3].clamp(min=-halfheight, max=halfheight)), -1)
    side_normal = torch.cat((direction, torch.zeros_like(radial[..., None])), -1)
    candidates = [torch.einsum("ij,...pj->...pi", cylinder_rotation, side_local)+cylinder_position]
    fallback = [torch.einsum("ij,...pj->...pi", cylinder_rotation, side_normal)]
    # A shared internal seam must not become an exterior slit solely because
    # quaternion roundoff makes each closed-primitive test miss by a few ulps.
    # This changes membership only at machine precision, not patch distances.
    membership_tolerance = 8*torch.finfo(points_o.dtype).eps*(halfheight+radius)
    inside = (radial <= radius+membership_tolerance) & (local[..., 2].abs() <= halfheight+membership_tolerance)
    for position, rotation, size, axial_axis, inner_sign in boxes:
        box_local = torch.einsum("ij,...pj->...pi", rotation.T, points_o-position)
        inside = inside | (box_local.abs() <= size+membership_tolerance).all(-1)
        clamped = torch.maximum(torch.minimum(box_local, size), -size)
        for face_axis in range(3):
            for sign in (-1, 1):
                closest = torch.stack([torch.ones_like(box_local[..., axis])*sign*size[axis]
                                       if axis == face_axis else clamped[..., axis] for axis in range(3)], -1)
                if face_axis == axial_axis and sign == inner_sign:
                    transverse = [axis for axis in range(3) if axis != axial_axis]
                    xy = closest[..., transverse]
                    length = torch.linalg.vector_norm(xy, dim=-1)
                    unit = xy/length.clamp_min(1e-12)[..., None]
                    unit = torch.where((length > 1e-12)[..., None], unit, xy.new_tensor([1., 0.]))
                    exposed = torch.where((length < radius)[..., None], radius*unit, xy)
                    closest = torch.stack([closest[..., axis] if axis == axial_axis else
                                           exposed[..., transverse.index(axis)] for axis in range(3)], -1)
                candidates.append(torch.einsum("ij,...pj->...pi", rotation, closest)+position)
                fallback.append((rotation[:, face_axis]*sign).expand_as(points_o))
    surface = torch.stack(candidates, -2)
    square = (points_o[..., None, :]-surface).square().sum(-1)
    distance2, selected = square.min(-1)
    closest = torch.gather(surface, -2, selected[..., None, None].expand(selected.shape+(1, 3))).squeeze(-2)
    unsigned = torch.sqrt(distance2.clamp_min(1e-30))
    orientation = torch.where(inside, -1., 1.)
    outward = orientation[..., None]*(points_o-closest)/unsigned[..., None]
    alternatives = torch.stack(fallback, -2)
    boundary_normal = torch.gather(alternatives, -2, selected[..., None, None].expand(selected.shape+(1, 3))).squeeze(-2)
    outward = torch.where((unsigned > 1e-12)[..., None], outward, boundary_normal)
    return unsigned*orientation, outward


def weighted_wrench_support(points_com_o, inward_normals_o, weights, basis, radius, *, mu=.1, edges=8):
    """Same unit-ray/max/basis/COM convention as CHORD, with soft weights.

    Binary weights recover the observed-contact support kernel. Continuous
    weights are a geometric proxy; they must not be interpreted as force N.
    Algebra avoids materializing the points × rays × basis tensor.
    """
    n = F.normalize(inward_normals_o, dim=-1, eps=1e-12)
    sign = torch.where(n[..., 2] >= 0, 1., -1.)
    a = -1/(sign+n[..., 2])
    b = n[..., 0]*n[..., 1]*a
    t1 = F.normalize(torch.stack((1+sign*n[..., 0].square()*a, sign*b, -sign*n[..., 0]), -1), dim=-1)
    t2 = F.normalize(torch.stack((b, sign+n[..., 1].square()*a, -n[..., 1]), -1), dim=-1)

    def project(direction):
        return (torch.einsum("...pj,nj->...pn", direction, basis[:, :3])
                +torch.einsum("...pj,nj->...pn", torch.linalg.cross(points_com_o, direction, dim=-1), basis[:, 3:])/radius)

    normal, tangent1, tangent2 = project(n), project(t1), project(t2)
    tangent_best = None
    for edge in range(edges):
        angle = 2*math.pi*edge/edges
        value = math.cos(angle)*tangent1+math.sin(angle)*tangent2
        tangent_best = value if tangent_best is None else torch.maximum(tangent_best, value)
    support = torch.maximum(normal, (normal+mu*tangent_best)/math.sqrt(1+mu*mu))
    # Keep the winning index for backward. Recomputing an amax equality mask
    # under whole-graph Inductor produced zero winner counts and NaN gradients.
    # Forward remains the exact point maximum; ties use the first winner's
    # valid subgradient instead of distributing it equally among tied points.
    return (support*weights[..., None]).max(dim=-2).values.clamp_min(0)


class HandKinematics(nn.Module):
    """Palm-local FK with hinge anchors and fixed links; native-buffer parity tested."""
    def __init__(self, model, joint_names):
        super().__init__()
        root = model.body("robot/right_palm_link").id
        self.body_ids = [root]+[index for index in range(model.nbody) if model.body(index).name.startswith("robot/right_finger")]
        lookup = {body: index for index, body in enumerate(self.body_ids)}
        joints = {model.joint("robot/"+name).id: index for index, name in enumerate(joint_names)}
        self.names = [model.body(body).name for body in self.body_ids]
        self.entries = []
        for index, body in enumerate(self.body_ids[1:], 1):
            if int(model.body_parentid[body]) not in lookup:
                raise ValueError("hand collision FK requires a connected palm/finger subtree")
            count, jid = int(model.body_jntnum[body]), int(model.body_jntadr[body])
            if count > 1 or (count and int(model.jnt_type[jid]) != 3):
                raise ValueError("only one hinge per hand body is supported")
            self.entries.append((lookup[int(model.body_parentid[body])], joints[jid] if count else -1))
            for name, value in (("offset", model.body_pos[body]), ("rotation", _quat_matrix(model.body_quat[body])),
                                ("axis", model.jnt_axis[jid] if count else [0., 0, 1]),
                                ("anchor", model.jnt_pos[jid] if count else [0., 0, 0])):
                self.register_buffer(f"{name}_{index}", torch.tensor(np.asarray(value), dtype=torch.float64))
            self.register_buffer(f"qref_{index}", torch.tensor(float(model.qpos0[model.jnt_qposadr[jid]]) if count else 0., dtype=torch.float64))

    def forward(self, q):
        shape = q.shape[:-1]
        positions = [q.new_zeros(shape+(3,))]
        rotations = [torch.eye(3, dtype=q.dtype, device=q.device).expand(shape+(3, 3))]
        for index, (parent, joint) in enumerate(self.entries, 1):
            off, rot, axis, anchor = [getattr(self, f"{name}_{index}") for name in ("offset", "rotation", "axis", "anchor")]
            angle = q[..., joint]-getattr(self, f"qref_{index}") if joint >= 0 else q.new_zeros(shape)
            x, y, z = axis.unbind()
            zero = x*0
            skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
            identity = torch.eye(3, dtype=q.dtype, device=q.device)
            rr = identity+angle.sin()[..., None, None]*skew+(1-angle.cos())[..., None, None]*(skew@skew)
            rp, pp = rotations[parent], positions[parent]
            local = off+torch.einsum("ij,...j->...i", rot, anchor-torch.einsum("...ij,j->...i", rr, anchor))
            positions.append(pp+torch.einsum("...ij,...j->...i", rp, local))
            rotations.append(rp@rot@rr)
        return torch.stack(positions, -2), torch.stack(rotations, -3)


class ContactGeometryProxy(nn.Module):
    def __init__(self, model, joint_names, basis, radius, *, points_per_geom=64, sample_seed=20711, dtype=torch.float32):
        super().__init__()
        self.fk = HandKinematics(model, joint_names)
        self.radius = float(radius)
        if self.radius <= 0 or points_per_geom < 1:
            raise ValueError("positive physical radius and surface sample count required")
        self.register_buffer("basis", torch.as_tensor(np.asarray(basis), dtype=torch.float64))
        object_body = model.body("object/cube").id
        self.register_buffer("com_offset", torch.tensor(model.body_ipos[object_body].copy(), dtype=torch.float64))
        objects = [index for index in range(model.ngeom) if int(model.geom_bodyid[index]) == object_body]
        self.object_types = []
        for local, geom in enumerate(objects):
            kind = int(model.geom_type[geom])
            if kind not in (5, 6):
                raise ValueError("expected the released pen cylinder and marker boxes")
            self.object_types.append(kind)
            for name, value in (("object_position", model.geom_pos[geom]), ("object_rotation", _quat_matrix(model.geom_quat[geom])),
                                ("object_size", model.geom_size[geom])):
                self.register_buffer(f"{name}_{local}", torch.tensor(np.asarray(value), dtype=torch.float64))
        self.object_count = len(objects)
        if self.object_types.count(5) != 1 or self.object_types.count(6) != 2:
            raise ValueError("this study proxy requires the pen union: one cylinder plus two marker boxes")
        self.exterior_cylinder, self.exterior_box_specs, exterior_layout = _coaxial_exterior_layout(model, objects)
        lower, upper = [], []
        for geom in objects:
            rotation, size = _quat_matrix(model.geom_quat[geom]), model.geom_size[geom]
            extent = (np.abs(rotation)@size if int(model.geom_type[geom]) == 6 else
                      size[0]*np.sqrt(rotation[:, 0]**2+rotation[:, 1]**2)+size[1]*np.abs(rotation[:, 2]))
            lower.append(model.geom_pos[geom]-extent)
            upper.append(model.geom_pos[geom]+extent)
        pen_lower, pen_upper = np.min(lower, 0), np.max(upper, 0)
        if not np.isclose((pen_upper-pen_lower)[0], .250, atol=1e-6):
            raise ValueError("expected the 250 mm pen along object x")
        body_lookup = {body: index for index, body in enumerate(self.fk.body_ids)}
        samples, bodies, groups, geom_indices, records = [], [], [], [], []
        for geom in range(model.ngeom):
            name, body = model.geom(geom).name, int(model.geom_bodyid[geom])
            if body not in body_lookup or not (name.startswith("robot/right_finger") or name == "robot/right_palm_collision"):
                continue
            if not any((int(model.geom_contype[geom]) & int(model.geom_conaffinity[obj])) or
                       (int(model.geom_contype[obj]) & int(model.geom_conaffinity[geom])) for obj in objects):
                continue
            rng = np.random.default_rng(sample_seed+geom)
            kind, size = int(model.geom_type[geom]), np.asarray(model.geom_size[geom])
            mesh_digest = None
            if kind == 3:
                radius_cap, half = size[:2]
                normals = rng.normal(size=(points_per_geom, 3))
                normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
                points = radius_cap*normals
                points[:, 2] += np.where(normals[:, 2] >= 0, half, -half)
                cylinder = rng.uniform(size=points_per_geom) < half/(half+radius_cap)
                angles = rng.uniform(0, 2*np.pi, cylinder.sum())
                points[cylinder] = np.stack((radius_cap*np.cos(angles), radius_cap*np.sin(angles), rng.uniform(-half, half, cylinder.sum())), -1)
            elif kind == 7:
                from scipy.spatial import ConvexHull
                mesh = int(model.geom_dataid[geom])
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                vertices = model.mesh_vert[start:start+count].astype(float)
                hull = ConvexHull(vertices)
                triangles = vertices[hull.simplices]
                area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=-1)
                selected = triangles[rng.choice(len(triangles), points_per_geom, p=area/area.sum())]
                uniform = rng.uniform(size=(points_per_geom, 2))
                root = np.sqrt(uniform[:, :1])
                points = ((1-root)*selected[:, 0]+root*(1-uniform[:, 1:2])*selected[:, 1]+root*uniform[:, 1:2]*selected[:, 2])
                mesh_digest = hashlib.sha256(vertices.tobytes()+hull.simplices.tobytes()).hexdigest()
            else:
                raise ValueError(f"unexpected active hand collision geom type {kind}: {name}")
            body_points = points@_quat_matrix(model.geom_quat[geom]).T+model.geom_pos[geom]
            group = next((finger-1 for finger in range(1, 6) if f"right_finger{finger}_" in name), 5)
            samples.append(body_points)
            bodies.extend([body_lookup[body]]*points_per_geom)
            groups.extend([group]*points_per_geom)
            geom_indices.extend([geom]*points_per_geom)
            records.append({"id": geom, "name": name, "type": kind, "body": model.body(body).name,
                            "position": model.geom_pos[geom].tolist(), "quaternion_wxyz": model.geom_quat[geom].tolist(),
                            "size": size.tolist(), "points": points_per_geom,
                            "compiled_convex_mesh_sha256": mesh_digest, "sample_body_points_sha256": hashlib.sha256(body_points.tobytes()).hexdigest()})
        self.register_buffer("sample_points_body", torch.tensor(np.concatenate(samples), dtype=torch.float64))
        self.register_buffer("sample_body_ids", torch.tensor(bodies, dtype=torch.long))
        self.register_buffer("sample_group_ids", torch.tensor(groups, dtype=torch.long))
        self.register_buffer("sample_geom_ids", torch.tensor(geom_indices, dtype=torch.long))
        self.to(dtype=dtype)
        self.metadata = {"input": "state58 q20 and object palm-frame p3/R9 only; velocities unused for static geometry",
                         "collision_sampling": "area-weighted compiled mesh convex hull / analytic capsule surface; fixed samples",
                         "sample_seed": sample_seed, "points_per_geom": points_per_geom, "sample_count": len(bodies),
                         "buffer_dtype": str(dtype),
                         "sampling_points_sha256": hashlib.sha256(self.sample_points_body.numpy().tobytes()).hexdigest(),
                         "wrench_basis_input_sha256": hashlib.sha256(np.ascontiguousarray(basis).tobytes()).hexdigest(),
                         "wrench_basis_count": len(basis), "physical_com_radius_m": self.radius,
                         "pen_union_aabb_body_m": {"lower": pen_lower.tolist(), "upper": pen_upper.tolist(),
                                                    "extent": (pen_upper-pen_lower).tolist()},
                         "signed_distance_semantics": "nearest TRUE exterior patch with union-membership sign; hidden primitive endcaps excluded",
                         "exterior_surface_layout": exterior_layout,
                         "membership_numeric_guard": "8*dtype_epsilon*(cylinder_halfheight+radius), closed union membership only; no patch-distance or task-threshold change",
                         "internal_precision": "autocast disabled inside geometry forward; configured float32/float64 buffers only, input cast preserves autograd",
                         "exterior_repair_boundary": "fixes internal seam geometry only; does not qualify force/contact prediction or dynamics gradients",
                         "hand_collision_geoms": records,
                         "excluded": "mount, visual geoms, contype/conaffinity-disabled geoms and non-hand bodies",
                         "object_geoms": [{"id": geom, "name": model.geom(geom).name, "type": int(model.geom_type[geom]),
                                           "size": model.geom_size[geom].tolist(), "position": model.geom_pos[geom].tolist(),
                                           "quaternion_wxyz": model.geom_quat[geom].tolist()} for geom in objects],
                         "support": "fixed unit 6D basis, normalized unit force cones, COM moment, radius division exactly once",
                         "point_max_backward": "saved argmax index; first winning sample at exact ties, not equal gradient sharing; forward point maximum unchanged",
                         "normal": "negative analytic object-SDF outward normal, force-on-object inward",
                         "soft_weight": "exp(-.5*(relu(SDF)/bandwidth)^2); geometric proximity, not force or a calibrated probability"}

    def hand_points_palm(self, q):
        p, r = self.fk(q)
        return p[..., self.sample_body_ids, :]+torch.einsum("...pij,pj->...pi", r[..., self.sample_body_ids, :, :], self.sample_points_body)

    def pen_sdf_normal(self, points_o):
        cylinder = self.exterior_cylinder
        position, rotation, size = [getattr(self, f"{name}_{cylinder}") for name in ("object_position", "object_rotation", "object_size")]
        boxes = [(getattr(self, f"object_position_{component}"), getattr(self, f"object_rotation_{component}"),
                  getattr(self, f"object_size_{component}"), axis, sign) for component, axis, sign in self.exterior_box_specs]
        return coaxial_exterior_sdf_normal(points_o, position, rotation, size[0], size[1], boxes)

    def forward(self, state, bandwidth_m, *, return_geometry=False):
        if self.sample_points_body.dtype not in (torch.float32, torch.float64):
            raise ValueError("contact geometry requires float32/float64 buffers, never low-precision AMP geometry")
        # A learned WM can run under AMP, but millimetre surface predicates and
        # the epsilon-sized closed-solid guard must retain physical precision.
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self._forward_full_precision(state.to(dtype=self.sample_points_body.dtype), bandwidth_m,
                                                return_geometry=return_geometry)

    def _forward_full_precision(self, state, bandwidth_m, *, return_geometry=False):
        if state.shape[-1] != 58 or bandwidth_m <= 0:
            raise ValueError("state58 and positive geometric bandwidth required")
        points_palm = self.hand_points_palm(state[..., :20])
        rotation = state[..., 43:52].reshape(state.shape[:-1]+(3, 3))
        points_o = torch.einsum("...ji,...pj->...pi", rotation, points_palm-state[..., None, 40:43])
        distance, outward = self.pen_sdf_normal(points_o)
        projected = points_o-distance[..., None]*outward
        weights = torch.exp(-.5*(distance.clamp_min(0)/bandwidth_m).square())
        support = weighted_wrench_support(projected-self.com_offset, -outward, weights, self.basis, self.radius)
        presence = torch.stack([weights[..., self.sample_group_ids == group].amax(-1) for group in range(6)], -1)
        result = {"support": support, "presence": presence, "minimum_signed_distance_m": distance.amin(-1)}
        if return_geometry:
            result.update(projected_contact_com_o_m=projected-self.com_offset, inward_normal_o=-outward,
                          soft_contact_weight=weights, sampled_signed_distance_m=distance)
        return result
