import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import warp as wp
from mujoco_warp._src import solver

from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.pen_ray_trace_factory_v1 import (
    GENERATED,INT_FIELDS,RAY_FIELDS,ROW_FIELDS,VECTOR_FIELDS,BRACKET_FIELDS,generate,validate_generated)


class RayFactoryTests(unittest.TestCase):
    def load(self,folder):
        path=Path(folder)/'factory.py'; manifest=generate(file_pin(solver.__file__),path)
        name='pen_ray_trace_cpu_fixture_v1'
        if name in sys.modules: raise ValueError('Fresh fixture namespace required')
        spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module
        try: spec.loader.exec_module(module)
        except BaseException: sys.modules.pop(name,None); raise
        self.addCleanup(sys.modules.pop,name,None)
        return module,manifest

    def test_generated_physical_ast_exact_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            module,manifest=self.load(folder); text=Path(manifest['generated']['path']).read_text()
            self.assertTrue(all(manifest['source_proof'].values()))
            for name in manifest['aliased_original_globals']:
                self.assertIs(getattr(module,name),getattr(solver,name))
            with self.assertRaisesRegex(ValueError,'canonical'):
                validate_generated(Path(solver.__file__).read_text(),text.replace('gtol = wp.max(', 'gtol = 10.0 + wp.max('))

    def test_actual_cpu_kernels_preserve_outputs_and_record_both_acceptance_paths(self):
        wp.init()
        with tempfile.TemporaryDirectory() as folder:
            module,_=self.load(folder)
            original=solver._linesearch_iterative_kernel(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
            observed=getattr(module,GENERATED)(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
            def array(a,dtype=float): return wp.array(np.asarray(a),dtype=dtype,device='cpu')
            def zeros(shape,dtype=float): return wp.zeros(shape,dtype=dtype,device='cpu')
            for smooth0,already_done in [(0.,False),(3.,False),(0.,True)]:
                results=[]
                for instrumented,kernel in [(False,original),(True,observed)]:
                    j=np.zeros((1,1,26),np.float32); j[0,0,0]=1.
                    s=np.zeros((1,26),np.float32); s[0,0]=1.
                    smooth=np.zeros_like(s); smooth[0,0]=smooth0
                    qacc=-s; ma=-s; jar=array([[-1.]]); jv=zeros((1,1)); quad=zeros((1,1),wp.vec3)
                    outputs=[array(qacc),array(ma),jar,jv,quad,zeros(1),zeros(1),zeros(1,bool)]
                    inputs=[26,array([1e-12]),array([1e-8]),array([1.]),array([.5]),array([0],int),array([0],int),array([1],int),array(smooth),
                        zeros((1,),solver.types.vec5),array([0],int),array([[-1]],int),array([[0]],int),array([[0]],int),
                        array([[0]],int),array([[0]],int),zeros((1,1,26),int),array(j),array([[1.]]),array([[0.]]),1,array([0],int),
                        array([False],bool),jar,array(s),array([1.]),array(s),jv,quad,array([already_done],bool)]
                    diagnostics=None
                    if instrumented:
                        g=np.zeros((1,28),np.float32); g[0,0]=-2.-smooth0
                        inputs += [array([0],int),array(g),array([1.]),array([(2.+smooth0)**2]),array([2.+smooth0])]
                        diagnostics=[zeros((50,1,len(RAY_FIELDS))),zeros((50,1,26,len(VECTOR_FIELDS))),zeros((50,1,1,len(ROW_FIELDS))),
                            zeros((50,1,50,len(BRACKET_FIELDS))),zeros((50,1,len(INT_FIELDS)),int),zeros((1,),int)]
                    wp.launch_tiled(kernel,dim=1,inputs=inputs,outputs=outputs+(diagnostics or []),block_dim=32,device='cpu')
                    results.append([v.numpy() for v in outputs])
                    if instrumented:
                        integers=diagnostics[4].numpy(); ray=diagnostics[0].numpy()
                        self.assertEqual(diagnostics[5].numpy()[0],0)
                        self.assertEqual(integers[0,0,INT_FIELDS.index('valid')],int(not already_done))
                        self.assertEqual(integers[0,0,INT_FIELDS.index('final_recorded')],int(not already_done))
                        self.assertEqual(integers[1:,:,INT_FIELDS.index('valid')].sum(),0)
                        if not already_done:
                            self.assertEqual(ray[0,0,RAY_FIELDS.index('p0_grad0')],-2.-smooth0)
                            self.assertEqual(ray[0,0,RAY_FIELDS.index('p0_hessian0')],2.)
                            np.testing.assert_array_equal(diagnostics[2].numpy()[0,0,0],[-1.,1.,1.,0.])
                            np.testing.assert_array_equal(diagnostics[1].numpy()[0,0,0],[1.,1.,-1.,smooth0,-1.,-2.-smooth0])
                            np.testing.assert_array_equal(diagnostics[1].numpy()[0,0,1:],np.zeros((25,len(VECTOR_FIELDS))))
                            self.assertEqual(ray[0,0,RAY_FIELDS.index('accepted_alpha')],results[-1][6][0])
                            self.assertEqual(integers[0,0,INT_FIELDS.index('initial_converged')],int(smooth0==0.))
                            if smooth0==3.: self.assertGreater(integers[0,0,INT_FIELDS.index('ls_used')],0)
                for b,o in zip(*results): np.testing.assert_array_equal(b,o)
                if smooth0==0. and not already_done:
                    np.testing.assert_array_equal(results[0][0],np.zeros((1,26),np.float32))
                    self.assertEqual(results[0][5][0],1.)


if __name__=='__main__': unittest.main()
