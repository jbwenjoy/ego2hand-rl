import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.ego.gravity_transport import gravity_transport_diagnostic


def poses(rotations):
    out = np.repeat(np.eye(4)[None], len(rotations), axis=0)
    out[:, :3, :3] = rotations
    return out


class GravityTransportTests(unittest.TestCase):
    def test_rotation_about_gravity_has_zero_directional_change(self):
        human = poses(Rotation.from_rotvec([[0, 0, 0], [0, 0, .2], [0, 0, -.3]]).as_matrix())
        value = gravity_transport_diagnostic(human, np.eye(4), [0, 0, -9.81], [0, 0, -9.81])
        np.testing.assert_allclose(value['conditional_mapped_gravity'], np.tile([0, 0, -9.81], (3, 1)), atol=1e-12)
        self.assertLess(value['metrics']['minimum_RMS_unit_gravity_change_from_initial'], 1e-12)
        self.assertFalse(value['source_gravity_verified'])

    def test_noncommuting_rotations_give_tight_positive_lower_bound(self):
        human = poses(Rotation.from_rotvec([[0, 0, 0], [.2, 0, 0], [0, .3, 0], [0, 0, .1]]).as_matrix())
        result = gravity_transport_diagnostic(human, np.eye(4), [0, 0, -9.81], [0, 0, -9.81])
        metric = result['metrics']
        self.assertGreater(metric['minimum_RMS_unit_gravity_change_from_initial'], 0)
        self.assertAlmostEqual(metric['minimum_RMS_unit_gravity_change_from_initial'], metric['independent_SVD_minimum'], places=13)
        self.assertAlmostEqual(metric['minimum_RMS_unit_gravity_change_from_initial'], metric['minimizing_direction_RMS_change'], places=13)
        u = np.random.default_rng(742).normal(size=(1000, 3))
        u /= np.linalg.norm(u, axis=-1, keepdims=True)
        D = result['source_to_robot_rotation'] - result['source_to_robot_rotation'][:1]
        sampled = np.sqrt(np.mean(np.sum(np.einsum('nij,kj->nki', D, u) ** 2, axis=-1), axis=0))
        self.assertGreaterEqual(sampled.min() + 1e-12, metric['minimum_RMS_unit_gravity_change_from_initial'])

    def test_proper_world_coordinate_change_preserves_diagnostic(self):
        human = poses(Rotation.from_rotvec([[0, 0, 0], [.2, .1, -.1]]).as_matrix())
        source_change = np.eye(4)
        source_change[:3, :3] = Rotation.from_rotvec([.1, -.3, .7]).as_matrix()
        g = np.array([0, 0, -9.81])
        original = gravity_transport_diagnostic(human, np.eye(4), g, g)
        changed = gravity_transport_diagnostic(source_change @ human, np.eye(4), source_change[:3, :3] @ g, g)
        np.testing.assert_allclose(original['conditional_mapped_gravity'], changed['conditional_mapped_gravity'], atol=1e-12)
        self.assertAlmostEqual(original['metrics']['minimum_RMS_unit_gravity_change_from_initial'], changed['metrics']['minimum_RMS_unit_gravity_change_from_initial'], places=8)

    def test_reflection_and_zero_gravity_rejected(self):
        human = poses(np.tile(np.eye(3), (2, 1, 1)))
        reflected = human.copy()
        reflected[:, 0, 0] = -1
        with self.assertRaises(ValueError):
            gravity_transport_diagnostic(reflected, np.eye(4), [0, 0, -9.81], [0, 0, -9.81])
        with self.assertRaises(ValueError):
            gravity_transport_diagnostic(human, np.eye(4), [0, 0, 0], [0, 0, -9.81])


if __name__ == '__main__':
    unittest.main()
