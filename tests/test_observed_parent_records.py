import unittest
import copy
import numpy as np
from supermimic_research.mechanics.observed_parent_records import pack_observed_controls


class ObservedParentTests(unittest.TestCase):
    def fixture(self):
        n, K = 4, 12
        q = np.zeros((n * K + 1, 27))
        q[:, 3] = 1
        q[:, 0] = np.arange(len(q)) * .001
        v = np.zeros((len(q), 26))
        v[:, 0] = .2
        z = dict(actions20=np.zeros((K, 20)), pre_qpos=q[:-1], post_qpos=q[1:],
                 pre_qvel=v[:-1], post_qvel=v[1:], post_ctrl=np.zeros((n * K, 20)),
                 post_GPU_time=np.arange(1, n * K + 1) * .005,
                 post_qacc=np.full((n * K, 26), -999.))
        sources = {k: dict(qpos=q[k * n][None], qvel=v[k * n][None],
                           qacc_warmstart=np.full((1, 26), float(k)), time=np.array([k * n * .005])) for k in range(K)}
        case = dict(dt=.005, model_layout=dict(object_qposadr=0, object_dofadr=0))
        physics = dict(mass=.03, inertia_BODY=np.eye(3).tolist(), gravity=[0, 0, -9.81])
        args = (case, z, sources, np.arange(7, 27), np.arange(6, 26), np.arange(20),
                physics, np.zeros((n * K, 6)), np.zeros((n * K, 6)), np.ones(n * K, bool))
        return args

    def test_real_source_memory_and_missing_last_endpoint(self):
        data = pack_observed_controls(*self.fixture())
        np.testing.assert_array_equal(data['control_index'], [8, 9, 10])
        np.testing.assert_array_equal(data['x'][:, 55:81], np.broadcast_to(np.arange(8, 11)[:, None], (3, 26)))
        np.testing.assert_array_equal(data['y'][:, 52:78], np.broadcast_to(np.arange(9, 12)[:, None], (3, 26)))
        self.assertTrue(data['valid'].all())

    def test_no_silent_source_or_control_discontinuity(self):
        args = copy.deepcopy(self.fixture())
        args[2][9]['qvel'][0, 0] += .01
        with self.assertRaises(ValueError):
            pack_observed_controls(*args)
        args = self.fixture()
        args[1]['post_ctrl'][33, 0] = .01
        with self.assertRaises(ValueError):
            pack_observed_controls(*args)

    def test_quality_gap_remains_masked(self):
        args = self.fixture()
        args[-1][36] = False
        np.testing.assert_array_equal(pack_observed_controls(*args)['valid'], [True, False, True])
