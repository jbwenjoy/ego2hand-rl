from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import warp as wp
from mujoco_warp._src import solver
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.pen_stable_friction_factory_v3 import generate,load_generated,verify_kernel_owner,NEW_FACTORY
from supermimic_research.runtime.pen_stable_friction_binding_v1 import FactoryBinding


class StableFrictionV3Tests(unittest.TestCase):
    def test_cpu_precision_boundary_fallback_and_factory_binding(self):
        wp.init()
        rows=np.asarray([[512.,-.001,1.,.03,1.],[-512.,.001,1.,.03,1.],[-1.,4.,1.,.25,1.],
                         [-1.,.2,1.,10.,.5],[1.,.1,0.,1.,1.],[1.,.1,-1.,1.,1.]],np.float32)
        with tempfile.TemporaryDirectory() as folder:
            manifest=generate(file_pin(solver.__file__),Path(folder)/'factory.py'); module=load_generated(manifest)
            self.addCleanup(sys.modules.pop,module.__name__,None)
            arrays=[wp.array(rows[:,i].copy(),dtype=float,device='cpu') for i in range(5)]; n=len(rows)
            outputs=[wp.zeros((n,3),device='cpu'),wp.zeros((n,3),device='cpu'),wp.zeros((n,3,3),device='cpu'),wp.zeros((n,3,3),device='cpu')]
            wp.launch(module.evaluate_cost_helpers,dim=n,inputs=arrays,outputs=outputs,device='cpu')
            old,new,old3,new3=[a.numpy() for a in outputs]
            self.assertEqual(old[:,1:].tobytes(),new[:,1:].tobytes()); self.assertEqual(old3[:,:,1:].tobytes(),new3[:,:,1:].tobytes())
            for i in [2,4,5]:
                self.assertEqual(old[i].tobytes(),new[i].tobytes()); self.assertEqual(old3[i].tobytes(),new3[i].tobytes())
            for i in [0,1]:
                expected=float(rows[i,4])*float(rows[i,1])*float(rows[i,3])*(1 if i==0 else -1)
                self.assertLess(abs(float(new[i,0])-expected),abs(expected)*1e-7)
            with FactoryBinding(manifest,synchronize=lambda:None):
                kernel=solver._linesearch_iterative_kernel(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
                self.assertTrue(verify_kernel_owner(module,kernel))


if __name__=='__main__': unittest.main()
