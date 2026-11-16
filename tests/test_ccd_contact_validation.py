import unittest
import numpy as np
from supermimic_research.mechanics.ccd_contact_validation import evaluate


LIMITS=dict(boundary_m=1e-6,distance_error_m=5e-6,normal_angle_deg=5.,midpoint_error_m=1e-4)


def fixture():
    c=dict(dist=np.array([-.001],np.float32),pos=np.array([[-.0005,0,0]],np.float32),frame=np.eye(3,dtype=np.float32)[None],worldid=np.array([31]),geom=np.array([[1,55]]),dim=np.array([3]))
    return c,np.array([0.,0.,0.,-.001,0.,0.])


class CCDContactValidationTests(unittest.TestCase):
    def test_valid_signed_normal_and_witness(self):
        c,w=fixture();r=evaluate(c,[0]*32,-.001,w,[1,55],LIMITS)
        self.assertEqual(r['passed'],r['total'])

    def test_reversed_normal_cannot_hide_with_absolute_cosine(self):
        c,w=fixture();c['frame'][0]=np.diag([-1,-1,1]);r=evaluate(c,[0]*32,-.001,w,[1,55],LIMITS)
        self.assertFalse(r['checks']['normal']);self.assertEqual(r['metrics']['normal_angle_deg'],180.)

    def test_finite_empty_overlap_is_failure(self):
        c,w=fixture();c={k:v[:0] for k,v in c.items()};r=evaluate(c,[0]*32,-.001,w,[1,55],LIMITS)
        self.assertTrue(r['checks']['finite']);self.assertFalse(r['checks']['overlap_has_contact'])

    def test_boundary_does_not_hide_overflow_or_bad_frame(self):
        c,w=fixture();c['frame'][0]*=2;r=evaluate(c,[256]+[0]*31,-.25e-6,w,[1,55],LIMITS)
        self.assertEqual(r['boundary_scope'],'BOUNDARY');self.assertFalse(r['checks']['no_overflow']);self.assertFalse(r['checks']['frame'])
