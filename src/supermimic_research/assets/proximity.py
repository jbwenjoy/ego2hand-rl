"""Cached unsigned point/triangle queries with conservative candidate pruning.

A nearest mesh vertex supplies an upper bound. A triangle closer than that
bound must have its centroid within upper_bound+maximum_triangle_radius.
The KD tree only prunes candidates; exact triangle projection sets the result.
Float arithmetic is not an interval certificate or a signed solid query.
"""
import numpy as np
from scipy.spatial import cKDTree


class SurfaceQuery:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, float).copy()
        self.faces = np.asarray(faces, np.int64).copy()
        if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or
                self.faces.ndim != 2 or self.faces.shape[1] != 3 or
                not len(self.faces) or not np.isfinite(self.vertices).all() or
                self.faces.min() < 0 or self.faces.max() >= len(self.vertices)):
            raise ValueError('Finite nonempty triangle surface required')
        self.triangles = self.vertices[self.faces]
        self.centers = self.triangles.mean(1)
        self.maximum_radius = float(np.linalg.norm(self.triangles-self.centers[:, None], axis=2).max())
        # Only vertices referenced by faces can supply a valid surface bound.
        self.vertex_tree = cKDTree(self.vertices[np.unique(self.faces)])
        self.center_tree = cKDTree(self.centers)

    def nearest(self, points, *, chunk=256):
        import trimesh
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or chunk < 1:
            raise ValueError('Finite point matrix and positive chunk required')
        closest, distances, ids = [], [], []
        candidates_checked = 0
        for start in range(0, len(points), chunk):
            batch = points[start:start+chunk]
            upper, _ = self.vertex_tree.query(batch, workers=1)
            # Small outward padding avoids pruning a near-boundary float tie.
            radius = upper+self.maximum_radius+1e-10
            candidates = self.center_tree.query_ball_point(batch, radius, workers=1)
            counts = np.array([len(c) for c in candidates])
            if not np.all(counts > 0):
                raise RuntimeError('Empty conservative candidate set')
            triangle_ids = np.concatenate(candidates)
            repeated = np.repeat(batch, counts, axis=0)
            projected = trimesh.triangles.closest_point(self.triangles[triangle_ids], repeated)
            squared = np.sum((projected-repeated)**2, axis=1)
            offset = 0
            for count in counts:
                selected = offset+int(np.argmin(squared[offset:offset+count]))
                closest.append(projected[selected]); distances.append(np.sqrt(squared[selected])); ids.append(triangle_ids[selected])
                offset += count
            candidates_checked += len(triangle_ids)
        return np.array(closest), np.array(distances), np.array(ids), candidates_checked
