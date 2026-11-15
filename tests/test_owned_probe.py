import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.runtime.pen_owned_probe_recorder import perturbed_actor_raw,source_frame_gap


class OwnedProbeTests(unittest.TestCase):
    def test_PD_perturbation_holds_at_noninitial_EMA(self):
        base=np.linspace(-.3,.4,20,dtype=np.float32);old=np.linspace(.2,-.1,20);direction=np.arange(1,21,dtype=float);direction/=np.linalg.norm(direction)
        for eps in [.002,.005]:
            raw=perturbed_actor_raw(base,direction,eps,-1)
            np.testing.assert_allclose((.5*old+.25*raw)-(.5*old+.25*base),-eps*direction,atol=1e-8,rtol=0)

    def test_probe_that_clips_raw_is_rejected(self):
        direction=np.zeros(20);direction[0]=1
        with self.assertRaisesRegex(ValueError,'clip'):perturbed_actor_raw(np.ones(20),direction,.002,1)

    def test_cached_forward_pose_gap_is_reported_without_overwriting_source(self):
        R=Rotation.from_euler('z',.2);q=np.r_[.1,.2,.3,R.as_quat()[[3,0,1,2]],np.zeros(20)];before=q.copy()
        result=source_frame_gap(q,0,np.eye(3),np.array([.1,.2,.29]))
        self.assertAlmostEqual(result['cached_rotation_gap_rad'],.2);self.assertAlmostEqual(result['cached_COM_gap_m'],.01)
        np.testing.assert_array_equal(q,before);np.testing.assert_allclose(result['qpos_R'],R.as_matrix(),atol=1e-14)


if __name__=='__main__':unittest.main()
