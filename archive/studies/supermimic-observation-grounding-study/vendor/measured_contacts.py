"""Frozen measurement helpers: read just-solved Warp contacts via host readback; never reference labels."""
import numpy as np
import mujoco


def quat_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norms < 1e-9):
        raise ValueError("invalid quaternion")
    w, x, y, z = np.moveaxis(q / norms, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def world_to_object(points_w, normals_w, object_pose_w):
    pose = np.asarray(object_pose_w)
    rt = np.swapaxes(quat_matrix(pose[..., 3:]), -1, -2)
    return (np.einsum("...ij,...kj->...ki", rt, np.asarray(points_w)-pose[..., None, :3]),
            np.einsum("...ij,...kj->...ki", rt, normals_w))


def capture_contacts(model, data, object_body, object_geoms, robot_geoms):
    """Normal is force-on-object: +frame[0] for object geom2, negated for geom1.

    MuJoCo convention: https://mujoco.readthedocs.io/en/stable/computation/index.html
    Contact force acts from geom1 toward geom2; contact frame rows are world axes.
    Capture geometric/solved-contact evidence, not reference contact targets.
    """
    pose = np.r_[data.xpos[object_body], data.xquat[object_body]]
    points, normals, forces, torques, pairs, active, friction = [], [], [], [], [], [], []
    penetration = 0.0
    category_depth = {"hand_object":0.,"hand_self":0.,"other":0.}
    object_environment_contacts = 0
    for i in range(data.ncon):
        c = data.contact[i]
        penetration = max(penetration, -float(c.dist), 0.0)
        # MuJoCo 3.11's legacy geom1/geom2 fields are not aliases of geom.
        # get_data_into populates the current geom[2] field, so read that pair.
        g1, g2 = (int(g) for g in c.geom)
        has_object = g1 in object_geoms or g2 in object_geoms
        has_robot = g1 in robot_geoms or g2 in robot_geoms
        category = "hand_object" if has_object and has_robot else "hand_self" if g1 in robot_geoms and g2 in robot_geoms else "other"
        category_depth[category] = max(category_depth[category],max(0.,-float(c.dist)))
        if has_object and not has_robot:
            object_environment_contacts += 1
        if not (has_object and has_robot):
            continue
        sign = 1.0 if g2 in object_geoms else -1.0
        frame = np.asarray(c.frame).reshape(3, 3)
        force6 = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force6)
        points.append(np.asarray(c.pos).copy())
        normals.append(sign * frame[0])
        forces.append(sign * frame.T @ force6[:3])
        torques.append(sign * frame.T @ force6[3:])
        pairs.append([g1, g2])
        active.append(bool(c.efc_address >= 0 and force6[0] > 1e-6))
        friction.append(np.asarray(c.friction).copy())
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    points_o, normals_o = world_to_object(points, normals, pose)
    return {"pos_o_m": points_o, "normal_o": normals_o,
            "force_w_N": np.asarray(forces, dtype=float).reshape(-1, 3),
            "torque_at_contact_w_Nm": np.asarray(torques, dtype=float).reshape(-1, 3),
            "geom_pair": np.asarray(pairs, dtype=np.int32).reshape(-1, 2),
            "active": np.asarray(active, dtype=bool),
            "friction": np.asarray(friction, dtype=float).reshape(-1, 5),
            "penetration_m": penetration, "penetration_by_category_m":category_depth,
            "object_environment_contacts": object_environment_contacts}
