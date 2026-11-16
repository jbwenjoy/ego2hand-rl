import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp
from mujoco_warp._src import solver

from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.pen_ray_bindings_v1 import RayBindings
from supermimic_research.runtime.pen_ray_post_kernels_v1 import copy_post_iteration,POST_VECTOR_FIELDS,POST_FLOAT_FIELDS,POST_INT_FIELDS
from supermimic_research.runtime.pen_paired_ray_observer_v1 import PairedRayObserver,disjoint,launch_layout


class PairedRayTests(unittest.TestCase):
    def test_real_cpu_post_phase_indices_done_flags_and_no_source_writes(self):
        wp.init()
        array=lambda a,dtype=float:wp.array(np.asarray(a),dtype=dtype,device='cpu')
        zeros=lambda shape,dtype=float:wp.zeros(shape,dtype=dtype,device='cpu')
        for n in [1,50,0,51]:
            vectors=[array(np.arange(28 if i==0 else 26,dtype=np.float32)[None]+100*i) for i in range(5)]
            floats=[array([i+.25]) for i in range(6)]
            state=[array([3],int),array([2],int),array([n==50],bool),array([True],bool),array([False],bool)]
            inputs=[array([n],int)]+vectors+floats+state
            before=[v.numpy().copy() for v in inputs]
            outputs=[zeros((50,1,26,len(POST_VECTOR_FIELDS))),zeros((50,1,len(POST_FLOAT_FIELDS))),zeros((50,1,len(POST_INT_FIELDS)),int),zeros(1,int)]
            wp.launch(copy_post_iteration,dim=(1,26),inputs=inputs,outputs=outputs,device='cpu')
            for a,b in zip(inputs,before): np.testing.assert_array_equal(a.numpy(),b)
            if n in [1,50]:
                np.testing.assert_array_equal(outputs[0].numpy()[n-1,0],np.stack([v.numpy()[0,:26] for v in vectors],axis=1))
                np.testing.assert_array_equal(outputs[1].numpy()[n-1,0],np.arange(6)+.25)
                np.testing.assert_array_equal(outputs[2].numpy()[n-1,0],[1,n,3,2,int(n==50),1,0])
                self.assertEqual(outputs[2].numpy()[:,:,0].sum(),1); self.assertEqual(outputs[3].numpy()[0],0)
            else:
                self.assertEqual(outputs[2].numpy().sum(),0); self.assertEqual(outputs[3].numpy()[0],1)

    def owner(self):
        return SimpleNamespace(solver=solver,installed_solver_pin=file_pin(solver.__file__),model=object(),data=object())

    def test_original_callback_bindings_restore_on_error_and_reject_nested_scope(self):
        original={n:getattr(solver,n) for n in ['_solve','_solver_iteration','_linesearch_iterative']}
        owner=self.owner()
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with RayBindings(owner) as hook:
                with self.assertRaises(ValueError):
                    with RayBindings(owner): pass
                with self.assertRaises(ValueError): hook.require(object(),owner.data)
                raise RuntimeError('deliberate')
        self.assertTrue(hook.audit['bindings_restored'])
        for n,f in original.items(): self.assertIs(getattr(solver,n),f)

    def test_binding_tamper_is_rejected_and_all_originals_restored(self):
        original={n:getattr(solver,n) for n in ['_solve','_solver_iteration','_linesearch_iterative']}
        with self.assertRaisesRegex(ValueError,'binding changed'):
            with RayBindings(self.owner()) as hook: solver._linesearch_iterative=lambda:None
        self.assertFalse(hook.audit['exit_bindings_verified']); self.assertTrue(hook.audit['bindings_restored'])
        for n,f in original.items(): self.assertIs(getattr(solver,n),f)

    def test_real_launch_layout_and_pointer_alias_guard(self):
        layout=launch_layout(file_pin(solver.__file__))
        self.assertEqual(layout['outputs'][0],'d.qacc'); self.assertEqual(layout['inputs'][0],'m.nv')
        self.assertEqual(layout['block_dim'],'m.block_dim.linesearch_iterative')
        wp.init(); a=wp.zeros(4,device='cpu'); b=wp.zeros(4,device='cpu'); empty=wp.zeros(0,device='cpu')
        disjoint([a,empty],[b])
        with self.assertRaisesRegex(ValueError,'aliases'): disjoint([a],[a])

    def test_failed_prerequisite_precedes_superclass_or_gpu_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'bad.json'; path.write_text(json.dumps(dict(passed=0,total=8,checks={'bad':False})))
            with patch('supermimic_research.runtime.pen_solve_exit_observer_v2.SolveExitGraphObserverV2.__init__',side_effect=AssertionError('No simulator allocation')):
                with self.assertRaisesRegex(ValueError,'CPU factory qualification'):
                    PairedRayObserver(None,factory_qualification=path)


if __name__=='__main__': unittest.main()
