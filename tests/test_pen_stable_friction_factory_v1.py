import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
import warp as wp
from mujoco_warp._src import solver

from supermimic_research.artifacts import file_pin
from supermimic_research.mechanics.pen_ray_reference_v1 import friction_point
from supermimic_research.runtime.pen_stable_friction_factory_v1 import (
    generate,load_generated,validate_generated,verify_kernel_owner,NEW_FACTORY)


class StableFrictionFactoryTests(unittest.TestCase):
    def fixture(self,folder):
        manifest=generate(file_pin(solver.__file__),Path(folder)/'factory.py')
        module=load_generated(manifest)
        self.addCleanup(sys.modules.pop,module.__name__,None)
        return module,manifest

    def test_ast_contract_rejects_other_arithmetic_and_old_factory_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            module,manifest=self.fixture(folder)
            self.assertTrue(all(manifest['source_proof'].values()))
            original=Path(solver.__file__).read_text(); text=Path(manifest['generated']['path']).read_text()
            for tampered in [text.replace('gtol = wp.max(', 'gtol = 10.0 + wp.max('),text.replace('return _stable_cost(', 'return wp.vec3(0.0, 0.0, 0.0) + _stable_cost(',1)]:
                with self.assertRaises(ValueError): validate_generated(original,tampered)
            args=(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
            original_factory=solver._linesearch_iterative_kernel
            old=original_factory(*args); candidate=getattr(module,NEW_FACTORY)(*args)
            self.assertTrue(verify_kernel_owner(module,candidate))
            self.assertIsNot(old,candidate)
            with self.assertRaisesRegex(ValueError,'cached'): verify_kernel_owner(module,old)
            self.assertIs(solver._linesearch_iterative_kernel,original_factory)

    def test_distinct_paths_have_separate_python_warp_and_factory_owners(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            ma,pa=self.fixture(a); mb,pb=self.fixture(b)
            self.assertNotEqual(pa['generated']['sha256'],pb['generated']['sha256'])
            self.assertIsNot(ma,mb); self.assertIsNot(ma._KERNEL_CACHE,mb._KERNEL_CACHE)
            args=(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
            ka=getattr(ma,NEW_FACTORY)(*args); kb=getattr(mb,NEW_FACTORY)(*args)
            self.assertIsNot(ka,kb)
            self.assertTrue(verify_kernel_owner(ma,ka)); self.assertTrue(verify_kernel_owner(mb,kb))

    def test_actual_cpu_helpers_distinct_alphas_and_unchanged_branches(self):
        wp.init()
        # first four are friction, followed by equality and unilateral rows.
        rows=np.asarray([[512.,-.001,1.,.03],[-512.,.001,1.,.03],[-1.,.2,1.,10.],[-1.,4.,1.,.25],
                         [-1.,.2,1.,0.],[-1.,4.,1.,0.],[1.,-4.,1.,0.]],np.float32)
        n=len(rows); alphas=np.tile(np.asarray([1.,.5,1.5],np.float32),(n,1))
        ne=np.asarray([0,0,0,0,n,0,0],np.int32); nf=np.asarray([n,n,n,n,0,0,0],np.int32)
        def array(a,dtype=float): return wp.array(a,dtype=dtype,device='cpu')
        with tempfile.TemporaryDirectory() as folder:
            module,_=self.fixture(folder)
            inputs=[array(np.arange(n,dtype=np.int32),int),array(ne,int),array(nf,int)]+[array(rows[:,i].copy()) for i in range(4)]+[array(alphas)]
            before=[a.numpy().tobytes() for a in inputs]
            outputs=[wp.zeros((n,3),device='cpu'),wp.zeros((n,3),device='cpu'),wp.zeros((n,3,3),device='cpu'),wp.zeros((n,3,3),device='cpu')]
            wp.launch(module.evaluate_declared_rows,dim=n,inputs=inputs,outputs=outputs,device='cpu')
            old,new,old3,new3=[a.numpy() for a in outputs]
            self.assertEqual(before,[a.numpy().tobytes() for a in inputs])
            self.assertEqual(old[:,1:].tobytes(),new[:,1:].tobytes())
            self.assertEqual(old3[:,:,1:].tobytes(),new3[:,:,1:].tobytes())
            self.assertEqual(old[3:].tobytes(),new[3:].tobytes())
            self.assertEqual(old3[3:].tobytes(),new3[3:].tobytes())
            for i in range(3):
                for k in range(3):
                    jar,jv,d,f=rows[i].astype(float)
                    expected=friction_point(jar,float(alphas[i,k])*jv,d,f)[0]
                    self.assertLess(abs(float(new3[i,k,0])-expected),max(1e-10,abs(expected)*2e-6))
                    if i<2: self.assertLess(abs(float(new3[i,k,0])-expected),.01*abs(float(old3[i,k,0])-expected))
            self.assertEqual(new[:,0].tobytes(),new3[:,0,0].tobytes())

    def test_actual_cpu_factory_launch_preserves_nonfriction_and_done_paths(self):
        wp.init()
        def array(a,dtype=float): return wp.array(np.asarray(a),dtype=dtype,device='cpu')
        def zeros(shape,dtype=float): return wp.zeros(shape,dtype=dtype,device='cpu')
        with tempfile.TemporaryDirectory() as folder:
            module,_=self.fixture(folder)
            args=(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
            kernels=[solver._linesearch_iterative_kernel(*args),getattr(module,NEW_FACTORY)(*args)]
            for smooth0,done,ne in [(0.,False,0),(3.,False,0),(0.,True,0),(0.,False,1)]:
                results=[]
                for kernel in kernels:
                    j=np.zeros((1,1,26),np.float32); j[0,0,0]=1.
                    s=np.zeros((1,26),np.float32); s[0,0]=1.
                    smooth=np.zeros_like(s); smooth[0,0]=smooth0
                    jar=array([[-1.]]); jv=zeros((1,1)); quad=zeros((1,1),wp.vec3)
                    outputs=[array(-s),array(-s),jar,jv,quad,zeros(1),zeros(1),zeros(1,bool)]
                    inputs=[26,array([1e-12]),array([1e-8]),array([1.]),array([.5]),array([ne],int),array([0],int),array([1],int),array(smooth),
                        zeros((1,),solver.types.vec5),array([0],int),array([[-1]],int),array([[0]],int),array([[0]],int),
                        array([[0]],int),array([[0]],int),zeros((1,1,26),int),array(j),array([[1.]]),array([[0.]]),1,array([0],int),
                        array([False],bool),jar,array(s),array([1.]),array(s),jv,quad,array([done],bool)]
                    wp.launch_tiled(kernel,dim=1,inputs=inputs,outputs=outputs,block_dim=32,device='cpu')
                    results.append([a.numpy().tobytes() for a in outputs])
                self.assertEqual(*results)


if __name__=='__main__': unittest.main()
