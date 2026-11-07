"""Candidate cone units and COM-axis rotation stay explicit."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.contact_envelopes import unit_normal_envelope,rotate_wrenches


class ContactEnvelopeTests(unittest.TestCase):
    def test_unit_normal_not_unit_euclidean_force_and_single_moment_arm(self):
        p=np.array([[.02,0,0.]]);n=np.array([[0,0,1.]])
        W,ids=unit_normal_envelope(p,n,mu=.7,edges=4)
        np.testing.assert_allclose(W[2],1);self.assertGreater(np.linalg.norm(W[:3,1]),1)
        np.testing.assert_allclose(W[3:].T,np.cross(p,W[:3].T),atol=1e-15);self.assertEqual(len(ids),5)

    def test_rotation_preserves_virtual_power_and_original_tangent_phase(self):
        W,_=unit_normal_envelope([[.02,0,0]],[[0,0,1.]])
        R=np.array([[0,-1.,0],[1,0,0],[0,0,1]]);rotated=rotate_wrenches(W,R)
        dual=np.array([1.,2,3,4,5,6]);world_dual=np.r_[R@dual[:3],R@dual[3:]]
        np.testing.assert_allclose(W.T@dual,rotated.T@world_dual,atol=1e-15)


if __name__=='__main__':unittest.main()
