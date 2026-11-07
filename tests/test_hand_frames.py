"""Analytic frame/handedness and coordinate-equivariance contract checks."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from supermimic_research.ego.hand_frames import pose_matrix, palm_frame, transform_points, reconstruct, numpy_skin


def profile():
    points = np.arange(63).reshape(21, 3).astype(float)
    return dict(joint_rotation_axes=[[0., 0., 1.]]*22,
        joint_rest_positions=[[10., 0., 0.]]*22, joint_frame_index=list(range(22)),
        joint_parent=[-1]*22, joint_first_child=[-1]*22, joint_next_sibling=[-1]*22,
        landmark_rest_positions=points.tolist(), landmark_rest_bone_indices=[[0, 0, 0]]*21,
        landmark_rest_bone_weights=[[1., 0., 0.]]*21,
        mesh_vertices=points.tolist(), mesh_triangles=[[0, 1, 2]],
        dense_bone_weights=np.eye(17)[[0]*21].tolist())


class HandFrameTests(unittest.TestCase):
    def test_source_pose_is_not_silently_normalized(self):
        with self.assertRaises(ValueError):
            pose_matrix(dict(translation_xyz=[0, 0, 0], quaternion_wxyz=[2, 0, 0, 0]))
        T = pose_matrix(dict(translation_xyz=[.1, .2, .3], quaternion_wxyz=[0, 0, 0, 1]))
        np.testing.assert_array_equal(transform_points(T, [[1, 0, 0]]), [[-.9, .2, .3]])

    def test_palm_convention_is_proper_equivariant_and_has_translation(self):
        bases = np.array([[1, 2, 0], [1, 1, 0], [1, -1, 0], [1, -2, 0]], float)
        P = palm_frame(np.zeros(3), bases)
        np.testing.assert_allclose(P[:3, 3], [1, 0, 0])
        np.testing.assert_allclose(P[:3, :3], np.eye(3))
        T = pose_matrix(dict(translation_xyz=[.2, -.3, .5], quaternion_wxyz=[.5, .5, .5, .5]))
        Q = palm_frame(transform_points(T, np.zeros(3)), transform_points(T, bases))
        np.testing.assert_allclose(Q, T@P, atol=1e-14)
        mirror = np.diag([-1, 1, 1])
        self.assertAlmostEqual(np.linalg.det(palm_frame(np.zeros(3), bases@mirror)[:3, :3]), 1.)
        with self.assertRaises(ValueError):
            palm_frame(np.zeros(3), np.zeros((4, 3)))

    def test_mm_once_right_reflection_and_winding(self):
        q = np.zeros((1, 22)); T = np.eye(4)[None]
        left = reconstruct(profile(), q, T, 'left')
        right = reconstruct(profile(), q, T, 'right')
        expected = np.array(profile()['mesh_vertices'])*.001
        np.testing.assert_allclose(left['vertices'][0], expected, atol=1e-8)
        np.testing.assert_array_equal(right['vertices'], left['vertices']*[-1, 1, 1])
        np.testing.assert_array_equal(right['faces'], left['faces'][..., ::-1])
        T[0, :3, 3] = [.2, -.1, .3]
        world = reconstruct(profile(), q, T, 'right')
        np.testing.assert_allclose(world['vertices'][0], expected*[-1, 1, 1]+T[0, :3, 3], atol=5e-8)

    def test_pivot_chain_numpy_and_unused_parameter_contract(self):
        p = profile(); p['dense_bone_weights'] = np.eye(17)[[2]*21].tolist()
        q = np.zeros((1, 22)); q[0, 0] = np.pi/2
        T = np.eye(4)[None]
        output = reconstruct(p, q, T, 'left')
        expected = np.array(p['mesh_vertices'])*.001
        expected = (expected-[.01, 0, 0])@np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T+[.01, 0, 0]
        np.testing.assert_allclose(output['vertices'][0], expected, atol=1e-8)
        np.testing.assert_allclose(output['vertices'], numpy_skin(p, q, T, 'left'), atol=1e-8)
        q[:, 20:] = [1.7, -2.4]
        np.testing.assert_array_equal(output['vertices'], reconstruct(p, q, T, 'left')['vertices'])


if __name__ == '__main__':
    unittest.main()
