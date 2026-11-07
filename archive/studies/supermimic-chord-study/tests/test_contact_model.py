import unittest

import numpy as np
import torch

from chord_study.contact_model import (CONTACT_GROUPS, ContactSurrogate, calibration_metrics,
                                       frozen_support_objective, grouped_split, pack_palm_state,
                                       qualify_local_proxy, secant_response_loss)


class ContactModelContracts(unittest.TestCase):
    def test_complete_seed_trajectories_are_disjoint(self):
        groups = np.repeat(np.arange(20), 7)
        split = grouped_split(groups, validation_groups=3, test_groups=4)
        self.assertTrue(np.all(sum(mask.astype(int) for mask in split.values()) == 1))
        memberships = [set(groups[mask]) for mask in split.values()]
        self.assertFalse(memberships[0] & memberships[1] or memberships[1] & memberships[2] or memberships[0] & memberships[2])

    def test_local_angular_velocity_and_palm_axes(self):
        # Rz90 maps local angular +x to world +y, then palm Rz90 maps it back +x.
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        state = pack_palm_state(np.zeros(20), np.zeros(20), np.array([1., 1, 0]), rotation,
                                np.array([0., 2, 0]), np.array([3., 0, 0]), np.array([1., 0, 0]), rotation)
        self.assertEqual(state.shape, (58,))
        np.testing.assert_allclose(state[40:43], [1, 0, 0])
        np.testing.assert_allclose(state[43:52], np.eye(3).reshape(-1))
        np.testing.assert_allclose(state[52:55], [2, 0, 0])
        np.testing.assert_allclose(state[55:58], [3, 0, 0])

    def test_physical_contact_heads_and_fixed_target_gradient(self):
        model = ContactSurrogate(4, support_dim=8, hidden_dim=16)
        state, action, context = torch.zeros(3, 58), torch.randn(3, 20, requires_grad=True), torch.zeros(3, 4)
        support, logits = model(state, action, context)
        self.assertEqual(logits.shape, (3, len(CONTACT_GROUPS)))
        self.assertEqual(len(CONTACT_GROUPS), 6)
        target = torch.ones(8, requires_grad=True)
        frozen_support_objective(support, target).sum().backward()
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(action.grad).all())
        self.assertGreater(float(action.grad.abs().sum()), 0)

    def test_calibration_has_explicit_empty_bins(self):
        metrics = calibration_metrics(torch.zeros(2, 6), np.zeros((2, 6)))
        self.assertEqual(metrics["brier"], .25)
        self.assertEqual(metrics["ece"], .5)
        self.assertEqual(sum(row["count"] for row in metrics["probability_bins"]), 12)

    def test_secant_supervises_flat_response_and_detaches_physics(self):
        plus = torch.tensor([[.3, .5]], requires_grad=True)
        minus = torch.tensor([[.2, .4]], requires_grad=True)
        truth = torch.tensor([[.7, .7]], requires_grad=True)
        loss = secant_response_loss(plus, minus, truth, truth, torch.tensor([.01]), torch.ones(2))
        loss.backward()
        self.assertGreater(float(plus.grad.sum()), 0)
        self.assertLess(float(minus.grad.sum()), 0)
        self.assertIsNone(truth.grad)

    def test_no_contact_or_gradient_coverage_never_qualifies(self):
        metrics = {"support_rmse_known_no_contact": 0., "support_max_known_no_contact": 0., "known_no_contact_samples": 10}
        probe = {"states": 12, "improved_fraction": 1., "sign_agreement_epsilon_stable": 1.,
                 "pearson_epsilon_stable": 1., "epsilon_stable_nonflat_direction_count": 0}
        self.assertFalse(qualify_local_proxy(metrics, probe)["qualified_as_local_candidate"])


if __name__ == "__main__":
    unittest.main()
