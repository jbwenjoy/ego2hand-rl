import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from supermimic_research.mechanics.pen_elliptic_capacity import elliptic_support
from supermimic_research.mechanics.pen_wrench_capacity import body_wrench_rays, directional_support


def make(dim, frame, mu):
    return SimpleNamespace(dim=dim, frame=frame, pos=np.array([.06, -.02, .04]), friction=np.array(mu))


class TestEllipticCapacity(unittest.TestCase):
    def setUp(self):
        self.directions = np.random.default_rng(381).normal(size=(100, 6))
        self.com = np.array([.01, .02, -.03])
        self.rot = Rotation.from_rotvec([.1, .2, -.3]).as_matrix()
        self.length = .1

    def evaluate(self, contacts, signs):
        return elliptic_support(contacts, signs, self.com, self.rot, self.length, self.directions)

    def test_signed_virtual_work_and_native_pyramid_inclusion(self):
        frame = Rotation.from_rotvec([.3, -.1, .2]).as_matrix()
        contacts = [make(dim, frame, [.3, .6, .002, .001, .004]) for dim in (1, 3, 4, 6)]
        signs = [1, -1, 1, -1]
        values = [np.zeros(len(self.directions))]
        # Independent probe pullback: angular probe at COM generates a
        # translational probe at the point via omega x (point - COM).
        trans = self.directions[:, :3] @ self.rot.T
        ang = self.directions[:, 3:] @ self.rot.T / self.length
        for c, sign in zip(contacts, signs):
            point = trans + np.cross(ang, c.pos - self.com)
            local = sign * np.c_[point @ c.frame.T, ang @ c.frame.T]
            values.append(local[:, 0] + np.linalg.norm(local[:, 1:c.dim] * c.friction[:c.dim-1], axis=1))
        actual = self.evaluate(contacts, signs)
        np.testing.assert_allclose(actual, np.max(values, axis=0), atol=1e-12)
        pyramid = directional_support(body_wrench_rays(contacts, signs, com_world=self.com,
            world_from_body=self.rot, length_scale_m=self.length), self.directions)
        self.assertTrue(np.all(actual >= pyramid - 1e-12))
        self.assertTrue(np.any(actual > pyramid + .01))
        np.testing.assert_allclose(actual, self.evaluate(contacts*2, signs*2))
        np.testing.assert_array_equal(self.evaluate([], []), np.zeros(len(self.directions)))

    def test_isotropic_tangent_frame_gauge(self):
        frame = Rotation.from_rotvec([.1, -.3, .24]).as_matrix()
        gauge = Rotation.from_rotvec([.78, 0, 0]).as_matrix()
        for dim in (1, 3, 4, 6):
            c = make(dim, frame, [.7, .7, .002, .001, .001])
            rotated = make(dim, gauge @ frame, c.friction)
            np.testing.assert_allclose(self.evaluate([c], [-1]), self.evaluate([rotated], [-1]), atol=1e-12)
        anisotropic = make(3, frame, [.1, .8, 0., 0., 0.])
        self.assertGreater(np.max(abs(self.evaluate([anisotropic], [1]) -
            self.evaluate([make(3, gauge @ frame, anisotropic.friction)], [1]))), .01)


if __name__ == '__main__':
    unittest.main()
