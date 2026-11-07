import unittest
from types import SimpleNamespace
import numpy as np
import torch

from chord_study.contact_geometry import (ContactGeometryProxy, _coaxial_exterior_layout, _quat_matrix, box_sdf_normal,
                                          coaxial_exterior_sdf_normal, cylinder_sdf_normal, weighted_wrench_support)
from chord_study.contacts import sample_basis, wrench_support


class AnalyticContactGeometry(unittest.TestCase):
    def test_capped_cylinder_faces_rim_and_inside(self):
        points = torch.tensor([[.03, 0, 0], [0, 0, .14], [.05, 0, .14], [0, 0, 0]], dtype=torch.float64)
        sdf, normals = cylinder_sdf_normal(points, .02, .10)
        torch.testing.assert_close(sdf, torch.tensor([.01, .04, .05, -.02], dtype=torch.float64))
        torch.testing.assert_close(normals[:3], torch.tensor([[1., 0, 0], [0, 0, 1], [.6, 0, .8]], dtype=torch.float64))
        self.assertTrue(torch.isfinite(normals).all())

    def test_box_corner_and_inside_normal(self):
        points = torch.tensor([[.13, .14, 0], [0, 0, .09]], dtype=torch.float64)
        sdf, normals = box_sdf_normal(points, [.1, .1, .1])
        torch.testing.assert_close(sdf, torch.tensor([.05, -.01], dtype=torch.float64))
        torch.testing.assert_close(normals, torch.tensor([[.6, .8, 0], [0, 0, 1]], dtype=torch.float64))

    def test_weighted_support_binary_parity_and_zero_ghost(self):
        points = torch.tensor([[.1, .02, 0], [-.1, 0, .01]], dtype=torch.float64, requires_grad=True)
        normals = torch.tensor([[-1., 0, 0], [1., 0, 0]], dtype=torch.float64)
        basis = sample_basis(64, seed=19, dtype=torch.float64)
        weights = torch.tensor([1., 0.], dtype=torch.float64)
        observed = wrench_support(points, normals, weights, basis, .12)
        soft = weighted_wrench_support(points, normals, weights, basis, .12)
        torch.testing.assert_close(soft, observed, atol=1e-12, rtol=1e-12)
        self.assertEqual(float(weighted_wrench_support(points, normals, weights*0, basis, .12).sum().detach()), 0.)
        soft.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_saved_point_max_unique_winner_matches_legacy_value_and_gradient(self):
        points = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        normals = torch.tensor([[0., 0, 1], [0., 0, 1]], dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([.25, .75], dtype=torch.float64, requires_grad=True)
        basis = torch.tensor([[0., 0, 1, 0, 0, 0]], dtype=torch.float64)
        actual = weighted_wrench_support(points, normals, weights, basis, .12)
        per_point = torch.stack([weighted_wrench_support(points[i:i+1], normals[i:i+1], weights[i:i+1], basis, .12)
                                 for i in range(2)])
        legacy = per_point.amax(0)
        torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
        gradients = torch.autograd.grad(actual.sum(), (points, normals, weights), retain_graph=True)
        old_gradients = torch.autograd.grad(legacy.sum(), (points, normals, weights))
        for gradient, old in zip(gradients, old_gradients):
            self.assertTrue(torch.isfinite(gradient).all())
            torch.testing.assert_close(gradient, old, atol=0, rtol=0)
        torch.testing.assert_close(gradients[2], torch.tensor([0., 1.], dtype=torch.float64))

    def test_saved_point_max_tie_keeps_forward_and_uses_first_valid_subgradient(self):
        points = torch.zeros(2, 3, dtype=torch.float64)
        normals = torch.tensor([[0., 0, 1], [0., 0, 1]], dtype=torch.float64)
        weights = torch.tensor([.5, .5], dtype=torch.float64, requires_grad=True)
        basis = torch.tensor([[0., 0, 1, 0, 0, 0]], dtype=torch.float64)
        actual = weighted_wrench_support(points, normals, weights, basis, .12)
        per_point = torch.stack([weighted_wrench_support(points[i:i+1], normals[i:i+1], weights[i:i+1], basis, .12)
                                 for i in range(2)])
        legacy = per_point.amax(0)
        torch.testing.assert_close(actual, legacy, atol=0, rtol=0)
        gradient = torch.autograd.grad(actual.sum(), weights, retain_graph=True)[0]
        old_gradient = torch.autograd.grad(legacy.sum(), weights)[0]
        torch.testing.assert_close(gradient, torch.tensor([1., 0.], dtype=torch.float64))
        torch.testing.assert_close(old_gradient, torch.tensor([.5, .5], dtype=torch.float64))
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertTrue((gradient >= 0).all())
        self.assertEqual(float(gradient.sum()), 1.)

    def test_sdf_directional_gradient_matches_closed_form(self):
        point = torch.tensor([[.05, .02, .15]], dtype=torch.float64, requires_grad=True)
        sdf, normal = cylinder_sdf_normal(point, .02, .10)
        gradient = torch.autograd.grad(sdf.sum(), point)[0]
        torch.testing.assert_close(gradient, normal, atol=1e-10, rtol=1e-10)

    def test_state_only_batch_float32_and_active_mount_exclusion(self):
        import mujoco
        fingers = []
        names = []
        for finger in range(1, 6):
            chain = ""
            for joint in range(1, 5):
                name = f"right_finger{finger}_joint{joint}"
                names.append(name)
                chain += (f'<body name="robot/right_finger{finger}_link{joint}" pos="0 0 -.02">'
                          f'<joint name="robot/{name}" type="hinge" axis="0 1 0"/>'
                          f'<geom name="robot/right_finger{finger}_link{joint}_col" type="capsule" size=".006 .01"/>')
            fingers.append(chain+"</body>"*4)
        xml = ('<mujoco><worldbody><body name="robot/right_palm_link">'
               '<geom name="robot/right_palm_collision" type="capsule" size=".02 .02"/>'
               +"".join(fingers)+'<body name="robot/right_mount"><geom name="robot/right_mount_collision" type="box" size=".1 .1 .1"/></body>'
               '</body><body name="object/cube">'
               '<geom type="cylinder" size=".00745 .089" quat=".7071067811865476 0 .7071067811865476 0"/>'
               '<geom type="box" size=".018 .009 .009" pos=".107 0 0"/>'
               '<geom type="box" size=".018 .009 .009" pos="-.107 0 0"/>'
               '</body></worldbody></mujoco>')
        model = mujoco.MjModel.from_xml_string(xml)
        proxy = ContactGeometryProxy(model, names, sample_basis(8).numpy(), .13, points_per_geom=2)
        states = torch.zeros(2, 58)
        states[:, 43:52] = torch.eye(3).reshape(9)
        states[:, 40] = torch.tensor([.1, .2])
        batch = proxy(states, .001)["support"]
        individual = torch.cat([proxy(state[None], .001)["support"] for state in states])
        torch.testing.assert_close(batch, individual)
        self.assertEqual(batch.dtype, torch.float32)
        self.assertFalse(any("mount" in geom["name"] for geom in proxy.metadata["hand_collision_geoms"]))
        direct_input = states.clone().requires_grad_(True)
        direct = proxy(direct_input, .001)["support"]
        direct_gradient = torch.autograd.grad(direct.sum(), direct_input)[0]
        amp_input = states.clone().requires_grad_(True)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            amp = proxy(amp_input, .001)["support"]
        amp_gradient = torch.autograd.grad(amp.sum(), amp_input)[0]
        self.assertEqual(amp.dtype, torch.float32)
        torch.testing.assert_close(amp, direct, atol=0, rtol=0)
        torch.testing.assert_close(amp_gradient, direct_gradient, atol=0, rtol=0)
        proxy.to(dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "float32/float64"):
            proxy(states, .001)


class TrueExteriorPenUnion(unittest.TestCase):
    def evaluate(self, points):
        dtype = points.dtype
        rotation = torch.tensor([[0., 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=dtype)
        boxes = [(torch.tensor([-.107, 0, 0], dtype=dtype), torch.eye(3, dtype=dtype),
                  torch.tensor([.018, .009, .009], dtype=dtype), 0, 1),
                 (torch.tensor([.107, 0, 0], dtype=dtype), torch.eye(3, dtype=dtype),
                  torch.tensor([.018, .009, .009], dtype=dtype), 0, -1)]
        return coaxial_exterior_sdf_normal(points, torch.zeros(3, dtype=dtype), rotation,
                                            torch.tensor(.00745, dtype=dtype), torch.tensor(.089, dtype=dtype), boxes)

    def test_hidden_seam_center_and_plus_minus_one_mm(self):
        points = torch.tensor([[-.089, 0, 0], [-.088, 0, 0], [-.090, 0, 0], [.089, 0, 0]], dtype=torch.float64)
        sdf, normal = self.evaluate(points)
        expected = torch.tensor([-.00745, -.00745, -np.sqrt(.00745**2+.001**2), -.00745], dtype=torch.float64)
        torch.testing.assert_close(sdf, expected, atol=1e-12, rtol=1e-12)
        self.assertLess(abs(float(normal[0, 0])), 1e-12)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(normal[0])), 1.)
        prior = ((-sdf-.003).clamp_min(0)/.001).square()
        self.assertGreater(float(prior[0]), 19.)

    def test_rotated_model_roundoff_cannot_open_an_internal_seam_slit(self):
        for dtype in (torch.float32, torch.float64):
            cylinder = torch.tensor(_quat_matrix([np.sqrt(.5), 0, -np.sqrt(.5), 0]), dtype=dtype)
            marker = torch.tensor(_quat_matrix([np.sqrt(.5), 0, np.sqrt(.5), 0]), dtype=dtype)
            size = torch.tensor([.009, .009, .018], dtype=dtype)
            boxes = [(torch.tensor([-.107, 0, 0], dtype=dtype), marker, size, 2, 1),
                     (torch.tensor([.107, 0, 0], dtype=dtype), marker, size, 2, -1)]
            sdf, _ = coaxial_exterior_sdf_normal(torch.tensor([[-.089, 0, 0]], dtype=dtype),
                                                torch.zeros(3, dtype=dtype), cylinder,
                                                torch.tensor(.00745, dtype=dtype), torch.tensor(.089, dtype=dtype), boxes)
            self.assertLess(float(sdf[0]), -.0074)
            self.assertAlmostEqual(float(sdf[0]), -.00745, places=7)

    def test_exposed_step_face_and_external_box_corner(self):
        points = torch.tensor([[-.089, .0082, 0], [-.0888, .0082, 0], [-.107, .012, .013]], dtype=torch.float64)
        sdf, normal = self.evaluate(points)
        torch.testing.assert_close(sdf, torch.tensor([0., .0002, .005], dtype=torch.float64), atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(normal, torch.tensor([[1., 0, 0], [1, 0, 0], [0, .6, .8]], dtype=torch.float64), atol=1e-10, rtol=1e-10)

    def test_true_exterior_gradient_and_central_difference(self):
        points = torch.tensor([[-.0902, .0023, .0011], [-.0888, .0082, .001],
                               [-.107, .012, .013], [0, .006, .001]], dtype=torch.float64, requires_grad=True)
        sdf, normal = self.evaluate(points)
        gradient = torch.autograd.grad(sdf.sum(), points)[0]
        torch.testing.assert_close(gradient, normal, atol=1e-10, rtol=1e-10)
        finite = torch.zeros_like(points)
        for axis in range(3):
            delta = torch.zeros_like(points)
            delta[:, axis] = 1e-7
            finite[:, axis] = (self.evaluate(points.detach()+delta)[0]-self.evaluate(points.detach()-delta)[0])/2e-7
        torch.testing.assert_close(gradient, finite, atol=2e-7, rtol=1e-5)

    def test_specialization_rejects_gap_tilt_and_uncovered_disk(self):
        def model():
            return SimpleNamespace(geom_type=np.array([5, 6, 6]),
                                   geom_pos=np.array([[0., 0, 0], [-.107, 0, 0], [.107, 0, 0]]),
                                   geom_quat=np.array([[np.sqrt(.5), 0, np.sqrt(.5), 0], [1., 0, 0, 0], [1., 0, 0, 0]]),
                                   geom_size=np.array([[.00745, .089, 0], [.018, .009, .009], [.018, .009, .009]]))
        _coaxial_exterior_layout(model(), [0, 1, 2])
        gap = model(); gap.geom_pos[1, 0] -= .001
        with self.assertRaisesRegex(ValueError, "seam abutment"):
            _coaxial_exterior_layout(gap, [0, 1, 2])
        tilt = model(); tilt.geom_quat[1] = [np.cos(.05), 0, np.sin(.05), 0]
        with self.assertRaisesRegex(ValueError, "axial alignment"):
            _coaxial_exterior_layout(tilt, [0, 1, 2])
        narrow = model(); narrow.geom_size[1, 1] = .007
        with self.assertRaisesRegex(ValueError, "contain the cylinder disk"):
            _coaxial_exterior_layout(narrow, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
