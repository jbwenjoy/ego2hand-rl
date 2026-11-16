import unittest

import numpy as np
import warp as wp

from supermimic_research.mechanics.pen_ray_reference_v1 import friction_point
from supermimic_research.runtime.pen_friction_cost_probe_v1 import probe_friction_cost


class FrictionCostProbeTests(unittest.TestCase):
    def run_probe(self,rows):
        wp.init(); arrays=[wp.array(np.asarray(v,np.float32),dtype=float,device='cpu') for v in np.asarray(rows).T]
        before=[x.numpy().copy() for x in arrays]; n=len(rows)
        out=[wp.zeros((n,3),device='cpu'),wp.zeros((n,3),device='cpu'),wp.zeros(n,dtype=int,device='cpu')]
        wp.launch(probe_friction_cost,dim=n,inputs=arrays,outputs=out,device='cpu')
        for x,b in zip(arrays,before): np.testing.assert_array_equal(x.numpy(),b)
        return [x.numpy() for x in out],np.asarray(rows,np.float32)

    def test_large_linear_cost_subtraction_and_unchanged_derivatives(self):
        (old,stable,handled),rows=self.run_probe([[512.,-.001,1.,.03,1.],[-512.,.001,1.,.03,1.],[-1.,.2,1.,10.,.5]])
        np.testing.assert_array_equal(old[:,1:],stable[:,1:]); np.testing.assert_array_equal(handled,[1,1,1])
        for i in [0,1]:
            j,jv,d,loss,a=rows[i].astype(float); expected=friction_point(j,a*jv,d,loss)[0]
            self.assertLess(abs(float(stable[i,0])-expected),.01*abs(float(old[i,0])-expected))

    def test_cross_regime_fallback_is_explicit_and_bit_exact(self):
        (old,stable,handled),_=self.run_probe([[-1.,4.,1.,.25,1.],[1.,-4.,1.,.25,1.]])
        np.testing.assert_array_equal(old,stable); np.testing.assert_array_equal(handled,[0,0])


if __name__=='__main__': unittest.main()
