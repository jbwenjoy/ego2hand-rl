import unittest

import numpy as np

from supermimic_research.mechanics.pen_ray_reference_v1 import evaluate


class RayReferenceTests(unittest.TestCase):
    def fixture(self,smooth=0.): return np.array([[1.,1.,-1.,smooth,-1.,-2.-smooth]])

    def test_analytic_quadratic_and_active_set_crossing(self):
        row=np.array([[-1.,1.,1.,0.]])
        np.testing.assert_array_equal(evaluate(0.,0,0,row,self.fixture())['point'],[0.,-2.,2.])
        np.testing.assert_array_equal(evaluate(1.,0,0,row,self.fixture())['point'],[-1.,0.,1.])
        np.testing.assert_array_equal(evaluate(1.,1,0,row,self.fixture())['point'],[-1.,0.,2.])
        np.testing.assert_array_equal(evaluate(4.,0,0,row,self.fixture(3.))['point'],[-8.5,0.,1.])
        loss=np.array([[-1.,1.,1.,.25]])
        np.testing.assert_array_equal(evaluate(0.,0,1,loss,self.fixture())['point'],[0.,-1.25,1.])
        np.testing.assert_array_equal(evaluate(1.,0,1,loss,self.fixture())['point'],[-.71875,0.,2.])

    def test_derivative_by_difference_away_from_kinks_and_tiny_shift(self):
        rows=np.array([[-1.,1.,1.,.25],[.4,-.3,2.,0.]])
        for alpha in [-.4,.2,.9,1.1,1.6]:
            h=1e-5; r=evaluate(alpha,0,1,rows,self.fixture()); lo=evaluate(alpha-h,0,1,rows,self.fixture())['point']; hi=evaluate(alpha+h,0,1,rows,self.fixture())['point']
            self.assertAlmostEqual((hi[0]-lo[0])/(2*h),r['point'][1],places=8)
            self.assertAlmostEqual((hi[1]-lo[1])/(2*h),r['point'][2],places=8)
        rows=np.array([[-1e8,1.,1.,0.]])
        # The shifted difference is retained even when x rounds back to jaref.
        self.assertAlmostEqual(evaluate(1e-16,1,0,rows,np.zeros((1,6)))['point'][0],-1e-8,places=20)

    def test_invalid_or_unsupported_ray_is_explicit(self):
        with self.assertRaises(ValueError): evaluate(0.,0,1,np.array([[-1.,1.,0.,.25]]),self.fixture())
        with self.assertRaises(ValueError): evaluate(float('nan'),0,0,np.empty((0,4)),self.fixture())
        with self.assertRaises(ValueError): evaluate(0.,2,0,np.zeros((1,4)),self.fixture())


if __name__=='__main__': unittest.main()
