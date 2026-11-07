"""SI units, right-handed coordinates, quaternion order wxyz throughout."""
import numpy as np


def quat_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norms < 1e-9):
        raise ValueError("invalid quaternion")
    w, x, y, z = np.moveaxis(q / norms, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3,3))


def rotation_error(q, ref):
    q, ref = np.asarray(q), np.asarray(ref)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    ref = ref / np.linalg.norm(ref, axis=-1, keepdims=True)
    return 2*np.arccos(np.clip(np.abs((q*ref).sum(-1)), 0, 1))


def world_to_object(points_w, normals_w, object_pose_w):
    pose = np.asarray(object_pose_w)
    rt = np.swapaxes(quat_matrix(pose[..., 3:]), -1, -2)
    return (np.einsum("...ij,...kj->...ki", rt, np.asarray(points_w)-pose[..., None, :3]),
            np.einsum("...ij,...kj->...ki", rt, normals_w))
