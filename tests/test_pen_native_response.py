import unittest
import numpy as np
from supermimic_research.learning.pen_native_response import compare_native_response


class NativeResponseTests(unittest.TestCase):
    def test_known_linear_response_and_nonprobabilistic_spread(self):
        d=np.array([2.,-3.,.5]); scales=np.array([2.,4.,.1]); epsilon=.002
        plus=np.tile(1+epsilon*d,(3,1));minus=np.tile(1-epsilon*d,(3,1))
        metric,fd=compare_native_response(plus,minus,d,scales,epsilon)
        self.assertLess(metric['relative_L2_error'],1e-12)
        self.assertAlmostEqual(metric['cosine'],1)
        self.assertTrue(metric['contrast_exceeds_observed_spread'])
        np.testing.assert_allclose(fd,d/scales,atol=1e-12)
        plus[1]+=1
        metric,_=compare_native_response(plus,minus,d,scales,epsilon)
        self.assertFalse(metric['contrast_exceeds_observed_spread'])
        with self.assertRaises(ValueError): compare_native_response(plus,minus,d,scales,0)
