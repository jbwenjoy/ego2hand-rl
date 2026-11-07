"""Fixed material points on enabled robot collision geometry, geometry only.

These points encode prospective contact hypotheses. Their position residual is
not a signed gap, contact label, force prediction, or grasp certificate. Native
filtered collision must independently confirm any optimized contact.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import mujoco
import numpy as np


@dataclass(frozen=True)
class MaterialAnchor:
    finger: int
    geom_id: int
    geom_name: str
    body_id: int
    point_geom_local: np.ndarray
    support_normal_geom_local: np.ndarray
    construction: str
    vertex_index: int | None
    baseline_distance_to_hypothesis_m: float


class MaterialAnchorGeometry:
    def __init__(self, native_geometry):
        self.native = native_geometry
        self.robot = native_geometry.robot
        self.model, self.data = native_geometry.model, native_geometry.data
        self._vertices = {}

    def _surface_point(self, geom, target_local, direction_local):
        """Exposed material point toward the hypothesized object surface.

        A linear maximum over ALL compiled mesh vertices is a point on the
        collision convex hull. It does not require a gap query or a proxy mesh.
        """
        m = self.model
        kind, size = int(m.geom_type[geom]), m.geom_size[geom]
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(m.geom_dataid[geom])
            if mesh not in self._vertices:
                start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                vertices = np.array(m.mesh_vert[start:start + count], dtype=np.float64)
                self._vertices[mesh] = vertices
            vertices = self._vertices[mesh]
            # Exact first argmax is frozen ONCE; no derivative through selection.
            index = int(np.argmax(vertices @ direction_local))
            return vertices[index].copy(), "compiled_mesh_extreme_support_vertex", index
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            point = np.where(direction_local > 0, size, -size)
            point[direction_local == 0] = np.clip(target_local[direction_local == 0],
                                                  -size[direction_local == 0], size[direction_local == 0])
            return point, "box_boundary_point", None
        if kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return size[0] * direction_local, "sphere_support_point", None
        if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            z = np.sign(direction_local[2]) * size[1] if direction_local[2] != 0 else np.clip(target_local[2], -size[1], size[1])
            return np.array([0., 0., z]) + size[0] * direction_local, "capsule_support_point", None
        raise ValueError(f"Unsupported enabled hand boundary type {kind}; do not substitute a proxy")

    def select(self, q, finger, object_surface_point_w, object_outward_normal_w):
        if finger not in range(1, 6):
            raise ValueError("Expected ordinal robot finger1..5")
        target = np.asarray(object_surface_point_w, dtype=np.float64)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Expected finite world surface point")
        normal = np.asarray(object_outward_normal_w, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all() or abs(np.linalg.norm(normal) - 1) > 1e-8:
            raise ValueError("Expected unit object outward world normal")
        self.robot.forward(q)
        candidates = []
        pattern = rf"robot/right_finger{finger}_(link[1-4]|tip_sensor)_col"
        for geom in self.native.hand_ids:
            geom = int(geom)
            name = self.model.geom(geom).name
            if not re.fullmatch(pattern, name):
                continue
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            local = rotation.T @ (target - self.data.geom_xpos[geom])
            support_local = rotation.T @ (-normal)
            point, construction, index = self._surface_point(geom, local, support_local)
            world = self.data.geom_xpos[geom] + rotation @ point
            distance = float(np.linalg.norm(world - target))
            candidates.append(MaterialAnchor(finger, geom, name,
                              int(self.model.geom_bodyid[geom]), point, support_local,
                              construction, index, distance))
        if not candidates:
            raise ValueError("No enabled named geometry for requested finger")
        return min(candidates, key=lambda value: value.baseline_distance_to_hypothesis_m)

    def position_and_jacobian(self, q, anchor):
        point, jacp, _, _ = self.oriented_features(q, anchor)
        return point, jacp

    def oriented_features(self, q, anchor):
        self.robot.forward(q)
        rotation = self.data.geom_xmat[anchor.geom_id].reshape(3, 3)
        point = self.data.geom_xpos[anchor.geom_id] + rotation @ anchor.point_geom_local
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, point, anchor.body_id)
        normal = rotation @ anchor.support_normal_geom_local
        skew = np.array([[0., -normal[2], normal[1]],
                         [normal[2], 0., -normal[0]],
                         [-normal[1], normal[0], 0.]])
        return point, jacp[:, self.robot.va], normal, -skew @ jacr[:, self.robot.va]

    def inventory(self):
        return [{"id": int(i), "name": self.model.geom(int(i)).name,
                 "type": str(mujoco.mjtGeom(int(self.model.geom_type[i]))),
                 "body_id": int(self.model.geom_bodyid[i])}
                for i in self.native.hand_ids]
