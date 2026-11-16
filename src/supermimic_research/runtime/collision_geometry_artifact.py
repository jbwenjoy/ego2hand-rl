"""Read only physical geometry inputs from mixed scene/contact evidence NPZ."""
import numpy as np


def load_collision_geometry(path):
    with np.load(path, allow_pickle=False) as packet:
        fields = {k: packet[k].copy() for k in ['qpos', 'geom_xpos', 'geom_xmat']}
    q, p, r = (fields[k] for k in ['qpos', 'geom_xpos', 'geom_xmat'])
    if q.ndim != 2 or p.ndim != 3 or p.shape[-1] != 3 or r.shape != p.shape[:2] + (3, 3) or q.shape[0] != p.shape[0]:
        raise ValueError('Matching world-owned qpos and geometry transforms required')
    if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in fields.values()):
        raise ValueError('Finite FP32 geometry inputs required')
    return fields
