import unittest
import torch
from supermimic_research.runtime.gpu_contact_proposals import proposal_score


class FullWrenchProposalTests(unittest.TestCase):
    def test_force_match_does_not_hide_missing_couple(self):
        config = dict(force_scale_N=.03, torque_scale_Nm=.0015, hand_accel_weight=.05,
                      hand_accel_scale_rad_s2=50., joint_regularization=.01, joint_bound_rad=.1,
                      object_penetration_m=.002, self_penetration_m=.0002,
                      collision_scale_m=.0005, self_collision_scale_m=.0001, nonhand_contact_penalty=1000.)
        target = torch.tensor([0., 0., .3, 0., -.008, 0.])
        actual = target.repeat(2, 1)
        actual[1, 4] = 0
        metrics = torch.zeros(2, 5)
        metrics[:, 4] = 1
        cost = proposal_score(actual, torch.zeros(2, 20), metrics, target,
                              torch.zeros(20), torch.zeros(2, 20), config)
        self.assertEqual(float(cost[0]), 0)
        self.assertGreater(float(cost[1]), 20)

    def test_capacity_failure_cannot_be_selected_as_perfect_match(self):
        config = dict(force_scale_N=.03, torque_scale_Nm=.0015, hand_accel_weight=.05,
                      hand_accel_scale_rad_s2=50., joint_regularization=.01, joint_bound_rad=.1,
                      object_penetration_m=.002, self_penetration_m=.0002,
                      collision_scale_m=.0005, self_collision_scale_m=.0001, nonhand_contact_penalty=1000.)
        cost = proposal_score(torch.zeros(1, 6), torch.zeros(1, 20), torch.zeros(1, 5),
                              torch.zeros(6), torch.zeros(20), torch.zeros(1, 20), config)
        self.assertTrue(torch.isinf(cost).all())
