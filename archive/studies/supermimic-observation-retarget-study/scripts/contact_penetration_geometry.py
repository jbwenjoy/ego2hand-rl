"""Geometry-only hand/object distances and rigid-point derivatives for IK.

No physics, known-contact labels, attraction, source robot labels or forces.
Negative pair distances alone enter the candidate penetration residual.
"""
from pathlib import Path
import re

import mujoco
import numpy as np

from retarget_palmroot_cpu import RobotKinematics


ALLOWANCE_M = .0005
NORMALIZATION_M = .001


class ContactGeometry:
    def __init__(self, model_path, context_path):
        self.robot = RobotKinematics(model_path)
        self.model, self.data = self.robot.model, self.robot.data
        m = self.model
        with np.load(context_path, allow_pickle=False) as context:
            for name in ("geom_size", "geom_pos"):
                value = context[name]
                if value.shape[1:] != getattr(m, name).shape or not np.array_equal(value, np.broadcast_to(value[:1], value.shape)):
                    raise ValueError("Expected shared fixed geometry")
                getattr(m, name)[:] = value[0]
        self.query_options = {"original_ccd_tolerance": float(m.opt.ccd_tolerance),
                              "original_ccd_iterations": int(m.opt.ccd_iterations),
                              "ccd_tolerance": 1e-10, "ccd_iterations": 100,
                              "scope": "CPU geometric-distance numerical refinement only; no saved model or dynamics changed"}
        m.opt.ccd_tolerance = self.query_options["ccd_tolerance"]
        m.opt.ccd_iterations = self.query_options["ccd_iterations"]
        self.names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        hand = [i for i, name in enumerate(self.names) if re.fullmatch(r"robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)", name)]
        obj = [m.geom(name).id for name in ("object/cube", "object/A_bottom", "object/A_top")]
        assert len(hand) == 26
        self.pairs = np.array([(h, o) for h in hand for o in obj], dtype=int)
        self.object_qa = int(m.joint("object/cube_freejoint").qposadr[0])
        self.last_q = None
        self.cached = None
        self.negative_witness_max_error_m = 0.

    def set_object_pose(self, pose):
        pose = np.asarray(pose)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > 1e-6:
            raise ValueError("Expected finite SI/wxyz object pose")
        self.data.qpos[self.object_qa:self.object_qa + 7] = pose
        self.robot.last_q = None
        self.last_q = None

    def distance_and_jacobian(self, q):
        if self.last_q is not None and np.array_equal(q, self.last_q):
            return self.cached
        self.robot.forward(q)  # Kinematics and comPos; no forward/step.
        m, d = self.model, self.data
        distances = np.empty(len(self.pairs))
        jac = np.zeros((len(self.pairs), 20))
        for index, (hand, obj) in enumerate(self.pairs):
            segment = np.zeros(6)
            distance = mujoco.mj_geomDistance(m, d, int(hand), int(obj), 1., segment)
            distances[index] = distance
            if distance < -ALLOWANCE_M:
                # fromto = signed distance * outward geom1->geom2 normal.
                difference = segment[3:] - segment[:3]
                error = abs(np.linalg.norm(difference) - abs(distance))
                self.negative_witness_max_error_m = max(self.negative_witness_max_error_m, error)
                if error > 1e-6:
                    raise ValueError("Inconsistent penetrating-pair witness")
                normal = difference / distance
                jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
                mujoco.mj_jac(m, d, jp, jr, segment[:3], int(m.geom_bodyid[hand]))
                jac[index] = -normal @ jp[:, self.robot.va]
        if not np.isfinite(distances).all() or not np.isfinite(jac).all():
            raise ValueError("Nonfinite geometric distance/derivative")
        self.last_q = np.array(q, copy=True)
        self.cached = distances, jac
        return self.cached

    def penetration_residual(self, q):
        distance, derivative = self.distance_and_jacobian(q)
        active = distance < -ALLOWANCE_M
        value = np.maximum(-distance - ALLOWANCE_M, 0.) / NORMALIZATION_M
        jac = -derivative / NORMALIZATION_M
        jac[~active] = 0
        return value, jac
