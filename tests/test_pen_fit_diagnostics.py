import unittest
import numpy as np
import torch
from supermimic_research.learning.pen_fit_diagnostics import constant_velocity, oracle_rollout, source_coverage, actual_action_coverage


class MemoryModel:
    def prediction(self, x):
        pred = torch.zeros((len(x), 90), dtype=torch.float64)
        pred[:, 40] = x[:, 55]
        pred[:, 52:78] = x[:, 55:81] + 1
        return pred


class FailureDiagnosisTests(unittest.TestCase):
    def test_actual_pose_coverage_requires_every_repeat_and_full_roll(self):
        x = np.zeros((171, 102))
        x[:, 43] = x[:, 47] = 1
        data = dict(x=x, y=np.zeros((171, 90)), geometry=np.zeros((171, 3)), valid=np.ones(171, bool),
            parent_id=np.zeros(171, int), repeat=np.repeat(np.arange(3), 57), candidate=np.tile(np.arange(57), 3))
        data['y'][1, 40] = .011
        data['valid'][2 + 114] = False
        data['geometry'][3 + 57, 0] = .003
        data['y'][4, 43] = .20  # Axial roll fails full SO3 while shaft stays fixed.
        parent = dict(parent_id=0, k=0, role='CHECK', mode='STATE_ONLY', dt=.005)
        case = dict(planning=[dict(k=0, phase=1, selected=1)])
        ref = dict(COM_world_m=np.zeros((2, 3)), quaternion_wxyz=np.tile([1., 0, 0, 0], (2, 1)))
        gates = dict(max_penetration_m=.002, max_self_penetration_m=.0002, max_joint_limit_excursion_rad=.001,
                     COM_max_m=.01, angle_max_rad=np.pi / 18)
        result = actual_action_coverage(data, parent, case, ref, gates)
        self.assertEqual(result['joint_pose_candidates'], 53)
        self.assertTrue(result['selected_quality'])
        self.assertFalse(result['selected_joint_pose'])

    def test_constant_velocity_world_rotation_and_no_clipping(self):
        s = torch.zeros((1, 55), dtype=torch.float64)
        s[:, 43] = 1
        s[:, 47] = 1
        s[:, 20] = 3
        s[:, 49] = .2
        s[:, 54] = np.pi / 2
        out = constant_velocity(s, 1.)
        self.assertAlmostEqual(float(out[0, 0]), 3.)
        self.assertAlmostEqual(float(out[0, 40]), .2)
        np.testing.assert_allclose(out[0, 43:49], [0, 1, 0, -1, 0, 0], atol=1e-12)

    def test_oracle_memory_intervention_is_explicit(self):
        x = np.zeros((3, 102))
        x[:, 43] = 1
        x[:, 47] = 1
        x[:, -1] = .005
        data = dict(x=x)
        ids = np.array([[0, 1, 2]])
        s, w, _ = oracle_rollout(MemoryModel(), data, ids, 'FULL', 'cpu')
        self.assertEqual(s[0, -1, 40], 3.)
        self.assertEqual(w[0, -1, 0], 3.)
        s, w, _ = oracle_rollout(MemoryModel(), data, ids, 'OBSERVED_WARM', 'cpu')
        self.assertEqual(s[0, -1, 40], 0.)
        self.assertEqual(w[0, -1, 0], 1.)
        s, _, _ = oracle_rollout(MemoryModel(), data, ids, 'OBSERVED_STATE', 'cpu')
        self.assertEqual(s[0, -1, 40], 2.)
        with self.assertRaises(ValueError):
            oracle_rollout(MemoryModel(), data, ids, 'UNKNOWN', 'cpu')

    def test_source_nearest_neighbor_excludes_PD_and_different_dt(self):
        train = dict(x=np.zeros((2, 102)), parent_id=np.array([1, 2]), phase=np.array([1, 2]), valid=np.ones(2, bool))
        train['x'][0, -1] = .005
        train['x'][1, -1] = .0025
        train['x'][0, 0] = 1
        train['x'][0, 81] = 999
        check = dict(x=np.zeros((1, 102)), parent_id=np.array([3]), phase=np.array([8]), valid=np.ones(1, bool))
        check['x'][0, -1] = .005
        report = source_coverage(train, check, dict(x_scale=np.ones(102)))
        self.assertEqual(report['CHECK_parents'][0]['nearest_TRAIN_parent'], 1)
        self.assertAlmostEqual(report['CHECK_parents'][0]['nearest_source_RMS'], 1 / 9)
