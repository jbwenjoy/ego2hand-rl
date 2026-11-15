import unittest
import numpy as np
from supermimic_research.mechanics.candidate_response_targets import packet_xy


class CandidateTargetTests(unittest.TestCase):
    def test_actual_next_warmstart_is_not_replaced_by_post_acceleration(self):
        q=np.r_[np.zeros(23),1.,0,0,0];v=np.zeros(26)
        source=dict(qpos=q[None],qvel=v[None],qacc_warmstart=np.ones((1,26))*3)
        packet=dict(post_qpos=q[None],post_qvel=v[None],post_ctrl=np.ones((1,20))*.2,
            post_qacc=np.ones((1,26))*99,post_qacc_warmstart=np.ones((1,26))*7)
        physics=dict(mass=.03,inertia_BODY=np.eye(3)*.001,gravity=[0,0,-9.81])
        x,y,h=packet_xy(source,packet,np.zeros((1,6)),np.ones((1,6)),np.arange(20),np.arange(20),np.arange(20),20,20,.005,physics)
        self.assertEqual(x.shape,(102,));self.assertEqual(y.shape,(90,));np.testing.assert_array_equal(x[55:81],3)
        np.testing.assert_array_equal(y[52:78],7);np.testing.assert_array_equal(y[84:],1)
        np.testing.assert_allclose(h[46:49],[0,0,.03*9.81*.02]);np.testing.assert_array_equal(h[49:52],0)
