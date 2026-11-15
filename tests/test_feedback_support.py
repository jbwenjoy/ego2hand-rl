import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.feedback_support import feedback_support_target


class FeedbackSupportTests(unittest.TestCase):
    def test_zero_error_reproduces_feedforward_exactly(self):
        nominal=torch.tensor([.1,.2,.3,.004,.005,.006],dtype=torch.float64)
        p=torch.tensor([.3,-.2,.7],dtype=torch.float64);v=2*p
        r=feedback_support_target(nominal,.03,p,v,p,v)
        self.assertTrue(torch.equal(r['target_world'],nominal))
        self.assertTrue(torch.equal(r['correction_force_world'],torch.zeros(3,dtype=torch.float64)))
        self.assertTrue(torch.equal(nominal,torch.tensor([.1,.2,.3,.004,.005,.006],dtype=torch.float64)))

    def test_restoring_force_units_mass_scaling_and_unmodified_torque(self):
        nominal=torch.tensor([.1,-.2,.3,.004,.005,.006],dtype=torch.float64)
        p=torch.tensor([.01,-.02,0.],dtype=torch.float64);v=torch.tensor([0.,0.,.1],dtype=torch.float64);zero=torch.zeros(3,dtype=torch.float64)
        a=feedback_support_target(nominal,.03,p,v,zero,zero)
        b=feedback_support_target(nominal,.06,p,v,zero,zero)
        torch.testing.assert_close(a['correction_force_world'],torch.tensor([-.03,.06,-.06],dtype=torch.float64))
        torch.testing.assert_close(b['correction_force_world'],2*a['correction_force_world'])
        self.assertTrue(torch.equal(a['target_world'][3:],nominal[3:]))
        self.assertLess(float(torch.dot(a['correction_force_world'],100*p+20*v)),0.)

    def test_world_rotation_covariance_and_translation_invariance(self):
        R=torch.tensor(Rotation.from_rotvec([.4,-.3,.8]).as_matrix(),dtype=torch.float64)
        n=torch.tensor([.1,-.2,.3,.004,-.005,.006],dtype=torch.float64)
        p=torch.tensor([.3,-.2,.7],dtype=torch.float64);v=.2*p;rp=p+.01;rv=v-.04
        a=feedback_support_target(n,.03,p,v,rp,rv)
        nr=torch.cat([R@n[:3],R@n[3:]]);shift=torch.tensor([3.,-2.,1.],dtype=torch.float64)
        b=feedback_support_target(nr,.03,R@p+shift,R@v,R@rp+shift,R@rv)
        torch.testing.assert_close(b['target_world'],torch.cat([R@a['target_world'][:3],R@a['target_world'][3:]]),atol=1e-14,rtol=1e-13)
