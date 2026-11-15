import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.mean_reference_wrench import mean_reference_wrench


class MeanWrenchTests(unittest.TestCase):
    def test_constant_acceleration_impulse_and_tail_mask(self):
        t=np.arange(10)*.02;a=np.array([.3,-.2,.4]);p=.5*t[:,None]**2*a
        q=np.tile([1.,0,0,0],(10,1));g=np.array([0,0,-9.81])
        r=mean_reference_wrench(p,q,.03,np.diag([1e-6,2e-4,2e-4]),g)
        np.testing.assert_allclose(r['wrench_world'][:6,:3],np.tile(.03*(a-g),(6,1)),atol=1e-13)
        np.testing.assert_allclose(r['wrench_world'][:6,3:],0,atol=1e-13)
        np.testing.assert_array_equal(r['valid_interval'],[1]*6+[0]*3)

    def test_constant_world_spin_has_correct_finite_angular_momentum_torque(self):
        t=np.arange(10)*.02;omega=np.array([.5,.8,1.1]);R=Rotation.from_rotvec(t[:,None]*omega).as_matrix()
        q=Rotation.from_matrix(R).as_quat()[:,[3,0,1,2]];I=np.diag([.00001,.0002,.0003])
        r=mean_reference_wrench(np.zeros((10,3)),q,.03,I,[0,0,-9.81])
        momentum=np.einsum('nij,j->ni',R[:7]@I@R[:7].transpose(0,2,1),omega)
        np.testing.assert_allclose(r['wrench_world'][:6,3:],np.diff(momentum,axis=0)/.02,atol=1e-13)
