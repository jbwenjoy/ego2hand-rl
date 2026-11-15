import unittest

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from supermimic_research.learning.pen_diagnostics import window_indices, recursive_prediction, decode_pen_torch
from supermimic_research.mechanics.transition_records import state_rotation


class RecursiveStub:
    def prediction(self, x):
        y = torch.zeros((len(x),90), dtype=torch.float64, device=x.device)
        y[:,0] = .1*x[:,0] + x[:,81]
        y[:,52:78] = x[:,55:81] + 1
        return y


class PenDiagnosticTests(unittest.TestCase):
    def test_recursive_state_warm_and_recorded_controls(self):
        state = torch.zeros((1,55), dtype=torch.float64); state[:,0] = 1
        state[:,43] = 1; state[:,47] = 1
        controls = torch.zeros((1,2,20), dtype=torch.float64); controls[:,:,0] = .2
        states, predictions, warm = recursive_prediction(RecursiveStub(), state, torch.zeros((1,26), dtype=torch.float64), controls, torch.full((1,1), .005))
        np.testing.assert_allclose(states[0,:,0], [1,1.3,1.63], atol=1e-7)
        np.testing.assert_array_equal(predictions[0,:,52], [1,2])
        np.testing.assert_array_equal(warm, np.full((1,26),2))

    def test_world_left_rotation_and_fn6_suffix(self):
        R = Rotation.from_rotvec([.2,.3,-.1]).as_matrix()
        s = np.zeros(55); s[43:46] = R[:,0]; s[46:49] = R[:,1]
        p = np.zeros(90); p[43:46] = [.1,-.2,.05]; p[84:] = np.arange(6)
        actual = decode_pen_torch(torch.tensor(s), torch.tensor(p)).numpy()
        np.testing.assert_allclose(state_rotation(actual), Rotation.from_rotvec(p[43:46]).as_matrix()@R, atol=1e-14)

    def test_windows_never_cross_episode_gap_invalid_or_dt(self):
        data = dict(case_id=np.array([0,0,0,1,1,1,1]), control_index=np.array([0,1,2,0,1,3,4]),
                    valid=np.array([True,True,False,True,True,True,True]), x=np.zeros((7,102)))
        data['x'][:,-1] = .005; data['x'][6,-1] = .0025
        np.testing.assert_array_equal(window_indices(data,2), [[0,1],[3,4]])
        self.assertEqual(window_indices(data,5).shape, (0,5))
        with self.assertRaises(ValueError): window_indices(data,0)
