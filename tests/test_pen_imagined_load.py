import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from supermimic_research.mechanics.pen_imagined_load import pen_load_step
from supermimic_research.mechanics.pen_load_forecast import LoadForecast, features, rotate_load
from supermimic_research.mechanics.transition_records import state_rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from check_pen_wrench_oracle import rigid


class TestPenImaginedLoad(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(715)
        self.batch = rng.normal(size=(2, 55))
        rotation = Rotation.from_rotvec(rng.normal(size=(2, 3))).as_matrix()
        self.batch[:, 43:46] = rotation[:, :, 0]
        self.batch[:, 46:49] = rotation[:, :, 1]
        self.batch[0, 52:55] = 0  # Exercise the zero-midpoint rotation derivative.
        self.pd = torch.tensor(rng.normal(size=(2, 20)))
        self.history = torch.tensor(rng.normal(size=(2, 12)))
        self.mass = 0.2
        self.inertia = np.array([[.045, .01, .006], [.01, .023, 0], [.006, 0, .06]])
        self.gravity = np.array([0.1, -0.2, -9.81])

    def net(self, has_history):
        rng = np.random.default_rng(114)
        x = torch.tensor(rng.normal(size=(20, 87 if has_history else 75)))
        y = torch.tensor(rng.normal(size=(20, 52)))
        torch.manual_seed(114)
        return LoadForecast(x, y).double().requires_grad_(False)

    def step(self, x, u, h, net):
        return pen_load_step(x, u, h, net, self.mass, self.inertia, self.gravity)

    def test_matches_saved_numpy_rigid_and_history_transport(self):
        for use_history in (True, False):
            h = self.history if use_history else None
            net = self.net(use_history)
            xs, histories, loads = self.step(torch.tensor(self.batch), self.pd, h, net)
            for i, state in enumerate(self.batch):
                sample = features(state, self.pd[i].numpy(), h[i].numpy() if use_history else None)
                estimate = net(torch.tensor(sample)).numpy()
                R = state_rotation(state)
                obj = rigid(state, R @ estimate[:3], R @ estimate[3:6],
                            self.mass, self.inertia, self.gravity)
                truth = np.r_[state[:40] + estimate[12:], obj]
                history = rotate_load(estimate[:12], state_rotation(truth).T @ R)
                np.testing.assert_allclose(xs[i].numpy(), truth, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(histories[i].numpy(), history, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(loads[i].numpy(), estimate[:12], rtol=1e-11, atol=1e-12)

    def test_two_step_gradients_against_central_difference(self):
        net = self.net(True)
        x = torch.tensor(self.batch[0], requires_grad=True)
        u = self.pd[0].clone().requires_grad_()
        u_next = self.pd[1].clone().requires_grad_()
        h = self.history[0].clone().requires_grad_()

        def rollout(x, u, h, u_next):
            next_x, next_h, load = self.step(x, u, h, net)
            last_x, _, last_load = self.step(next_x, u_next, next_h, net)
            return torch.cat([last_x, load, last_load])

        self.assertTrue(torch.autograd.gradcheck(rollout, (x, u, h, u_next), eps=1e-6,
                                                 atol=1e-6, rtol=1e-4, fast_mode=True))
        # Check that the first action can change a later pose through the model.
        derivative, = torch.autograd.grad(rollout(x, u, h, u_next)[40:49].sum(), u)
        self.assertGreater(derivative.norm().item(), 1e-9)


if __name__ == '__main__':
    unittest.main()
