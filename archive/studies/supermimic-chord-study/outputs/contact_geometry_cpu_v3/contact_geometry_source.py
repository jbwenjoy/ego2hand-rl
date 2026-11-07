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
    return (support*weights[..., None]).amax(-2).clamp_min(0)


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
                         "hand_collision_geoms": records,
                         "excluded": "mount, visual geoms, contype/conaffinity-disabled geoms and non-hand bodies",
                         "object_geoms": [{"id": geom, "name": model.geom(geom).name, "type": int(model.geom_type[geom]),
                                           "size": model.geom_size[geom].tolist(), "position": model.geom_pos[geom].tolist(),
                                           "quaternion_wxyz": model.geom_quat[geom].tolist()} for geom in objects],
                         "support": "fixed unit 6D basis, normalized unit force cones, COM moment, radius division exactly once",
                         "normal": "negative analytic object-SDF outward normal, force-on-object inward",
                         "soft_weight": "exp(-.5*(relu(SDF)/bandwidth)^2); geometric proximity, not force or a calibrated probability"}

    def hand_points_palm(self, q):
        p, r = self.fk(q)
        return p[..., self.sample_body_ids, :]+torch.einsum("...pij,pj->...pi", r[..., self.sample_body_ids, :, :], self.sample_points_body)

    def pen_sdf_normal(self, points_o):
        distances, normals = [], []
        for geom, kind in enumerate(self.object_types):
            position, rotation, size = [getattr(self, f"{name}_{geom}") for name in ("object_position", "object_rotation", "object_size")]
            local = torch.einsum("ij,...pj->...pi", rotation.T, points_o-position)
            sd, normal = cylinder_sdf_normal(local, size[0], size[1]) if kind == 5 else box_sdf_normal(local, size)
            distances.append(sd)
            normals.append(torch.einsum("ij,...pj->...pi", rotation, normal))
        distance, selected = torch.stack(distances, -1).min(-1)
        normal_table = torch.stack(normals, -2)
        selected_normal = torch.gather(normal_table, -2, selected[..., None, None].expand(selected.shape+(1, 3))).squeeze(-2)
        return distance, selected_normal

    def forward(self, state, bandwidth_m, *, return_geometry=False):
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
