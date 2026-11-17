import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import warp as wp

from mujoco_warp._src import solver
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.pen_stable_friction_factory_v4 import generate,load_generated,NEW_FACTORY,validate_generated,verify_kernel_owner
from supermimic_research.runtime.pen_stable_friction_binding_v2 import FactoryBinding

_base=importlib.import_module('test_pen_stable_friction_factory_v1')


class GuardedFactoryTests(_base.StableFrictionFactoryTests):
    def fixture(self,folder):
        manifest=generate(file_pin(solver.__file__),Path(folder)/'factory.py'); module=load_generated(manifest)
        self.addCleanup(sys.modules.pop,module.__name__,None)
        return module,manifest

    def test_ast_contract_rejects_other_arithmetic_and_old_factory_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            module,manifest=self.fixture(folder); args=(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
            old=solver._linesearch_iterative_kernel(*args); new=getattr(module,NEW_FACTORY)(*args)
            self.assertTrue(verify_kernel_owner(module,new)); self.assertIsNot(old,new)
            with self.assertRaisesRegex(ValueError,'cached'): verify_kernel_owner(module,old)
            p=Path(manifest['generated']['path']); p.write_text(p.read_text().replace('gtol = wp.max(', 'gtol = 10.0 + wp.max('))
            with self.assertRaisesRegex(ValueError,'Exact unchanged'): validate_generated(manifest['installed_solver'],manifest['installed_cache'],p)

    def test_guarded_binding_rejects_nested_and_restores_on_error(self):
        old=solver._linesearch_iterative_kernel; stopping=solver._solve_done
        with tempfile.TemporaryDirectory() as folder:
            _,manifest=self.fixture(folder); binding=FactoryBinding(manifest,synchronize=lambda:None)
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with binding:
                    solver._linesearch_iterative_kernel(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
                    self.assertTrue(binding.verify()); self.assertIs(solver._solve_done,stopping)
                    with self.assertRaisesRegex(ValueError,'Exclusive'):
                        with FactoryBinding(manifest,synchronize=lambda:None): pass
                    raise RuntimeError('intentional')
            self.assertIs(solver._linesearch_iterative_kernel,old); self.assertTrue(binding.audit['bindings_restored'])

    def test_actual_gpu_saved_rows_distinct_alpha_derivatives_and_fallback(self):
        if os.environ.get('SUPERMIMIC_GPU_TEST')!='1': self.skipTest('Explicit bounded CUDA qualification only')
        root=Path(__file__).resolve().parents[1]
        q=json.loads((root/'runs/pen_stable_friction_backend_qualification_v3/report.json').read_text())
        rows=np.load(verify_file_pin(q['inputs']['fixed_inputs']))['inputs']; n=len(rows)
        alphas=(rows[:,4,None]*np.asarray([1.,.5,1.5],np.float32)).astype(np.float32)
        with tempfile.TemporaryDirectory() as folder:
            module,_=self.fixture(folder); device=wp.get_device('cuda:0')
            def array(a,dtype=float): return wp.array(a,dtype=dtype,device=device)
            inputs=[array(np.arange(n,dtype=np.int32),int),array(np.zeros(n,np.int32),int),array(np.full(n,n,np.int32),int)]+[array(rows[:,i].copy()) for i in range(4)]+[array(alphas)]
            outputs=[wp.zeros((n,3),device=device),wp.zeros((n,3),device=device),wp.zeros((n,3,3),device=device),wp.zeros((n,3,3),device=device)]
            wp.launch(module.evaluate_declared_rows,dim=n,inputs=inputs,outputs=outputs,device=device)
            old,new,old3,new3=[a.numpy() for a in outputs]
            self.assertEqual(old[:,1:].tobytes(),new[:,1:].tobytes()); self.assertEqual(old3[:,:,1:].tobytes(),new3[:,:,1:].tobytes())
            j,v,d,f=rows[:,:4].astype(float).T; r=f/d
            for k in range(3):
                x=j+alphas[:,k].astype(float)*v
                h=((-r<j)&(j<r)&(-r<x)&(x<r))|((j<=-r)&(x<=-r))|((j>=r)&(x>=r))
                self.assertEqual(old3[~h,k].tobytes(),new3[~h,k].tobytes(),f'fallback point{k}')
                if k==0: self.assertEqual(old[~h].tobytes(),new[~h].tobytes())
