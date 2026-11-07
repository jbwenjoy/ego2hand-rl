"""Pruned nearest-surface queries against brute force, including unused vertices."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from supermimic_research.assets.proximity import SurfaceQuery


class SurfaceQueryTests(unittest.TestCase):
    def test_agrees_with_bruteforce_for_random_transformed_nonconvex_surface(self):
        import trimesh
        # Two disconnected boxes: a whole-object convex hull would fill the gap.
        mesh = trimesh.util.concatenate([trimesh.creation.box(transform=trimesh.transformations.translation_matrix([-1, 0, 0])),
                                         trimesh.creation.box(transform=trimesh.transformations.translation_matrix([1, 0, 0]))])
        points = np.r_[np.random.default_rng(20261108).normal(size=(73, 3)), [[0, 0, 0], [-.5, 0, 0]]]
        query = SurfaceQuery(mesh.vertices, mesh.faces)
        closest, d, ids, _ = query.nearest(points, chunk=7)
        brute, expected, _ = trimesh.proximity.closest_point_naive(mesh, points)
        np.testing.assert_allclose(d, expected, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(closest-points, axis=1), d, atol=1e-14)
        self.assertAlmostEqual(d[-2], .5)
        self.assertEqual(d[-1], 0.)
        shift = [.2, -.7, .4]
        moved = SurfaceQuery(mesh.vertices+shift, mesh.faces).nearest(points+shift)
        np.testing.assert_allclose(moved[1], d, atol=1e-14)

    def test_unused_vertex_cannot_supply_surface_upper_bound(self):
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 5.]])
        query = SurfaceQuery(vertices, [[0, 1, 2]])
        self.assertAlmostEqual(query.nearest([[0, 0, 5]])[1][0], 5.)
        with self.assertRaises(ValueError):
            query.nearest([[float('nan'), 0, 0]])


if __name__ == '__main__':
    unittest.main()
