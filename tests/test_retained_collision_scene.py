import unittest
import numpy as np
import torch
from supermimic_research.runtime.retained_collision_scene import audit_boundary, derive_qpos


def fixture():
    tensors = {}
    for name in ['encoder_bias', 'default_joint_pos']:
        tensors[(('item', 'robot'), ('attr', name))] = torch.zeros(32, 20)
    tensors[(('attr', 'qids'),)] = torch.arange(20)
    tensors[(('attr', 'mount'),)] = torch.tensor([0., 0., 0., 1., 0., 0., 0.]).repeat(32, 1)
    tensors[(('attr', 'policy_WT'),)] = torch.eye(3).repeat(32, 1, 1)
    state = {'native': {'python': {'tensors': tensors}, 'native': {'data': {'arrays': {'qpos': np.zeros((32, 27), dtype=np.float32)}}}}}
    policy = torch.zeros(2, 32, 161); policy[..., 92:112] = .2
    policy[..., 135:141] = torch.tensor([1., 0., 0., 0., 1., 0.])
    memory = dict(obs=policy.clone(), terminal_next_policy=policy.clone(),
        terminated=torch.zeros(2, 32, dtype=torch.bool), truncated=torch.zeros(2, 32, dtype=torch.bool),
        contact_source_C=torch.zeros(2, 32, 3), contact_source_R=torch.eye(3).repeat(2, 32, 1, 1),
        contact_clock=torch.zeros(2, 32, 2), next_time=torch.zeros(2, 32),
        phase_source=torch.ones(2, 32, dtype=torch.long), phase_next=torch.ones(2, 32, dtype=torch.long))
    return state, memory


class RetainedSceneTests(unittest.TestCase):
    def test_consistent_scene_derived_without_parent_mutation(self):
        state, memory = fixture(); q, audit = derive_qpos(state, memory, 1, 31, 20)
        self.assertTrue(audit['admitted_for_derived_query'])
        np.testing.assert_array_equal(q[31, :20], np.full(20, .2, dtype=np.float32))
        np.testing.assert_array_equal(q[31, 23:27], [1, 0, 0, 0])
        self.assertFalse(state['native']['native']['data']['arrays']['qpos'].any())

    def test_reset_and_clock_mismatch_not_silently_used(self):
        s, m = fixture(); m['terminated'][0, 31] = True
        with self.assertRaises(ValueError): derive_qpos(s, m, 1, 31, 20)
        m['terminated'].zero_(); m['next_time'][0, 31] = 1
        self.assertFalse(audit_boundary(s, m, 1, 31)['admitted_for_derived_query'])
        with self.assertRaises(ValueError): derive_qpos(s, m, 1, 31, 20)

    def test_nonzero_encoder_bias_rejected(self):
        s, m = fixture(); s['native']['python']['tensors'][(('item', 'robot'), ('attr', 'encoder_bias'))][0, 0] = .1
        with self.assertRaises(ValueError): derive_qpos(s, m, 1, 31, 20)

    def test_pose_mismatch_and_rotation_defect_rejected(self):
        s, m = fixture(); m['contact_source_C'][1, 31, 0] = .01
        self.assertFalse(audit_boundary(s, m, 1, 31)['admitted_for_derived_query'])
        m['contact_source_C'].zero_(); m['contact_source_R'][1, 31, 0, 0] = 0
        with self.assertRaises(ValueError): derive_qpos(s, m, 1, 31, 20)


if __name__ == '__main__': unittest.main()
