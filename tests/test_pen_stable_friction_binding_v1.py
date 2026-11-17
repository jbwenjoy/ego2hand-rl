from pathlib import Path
import sys
import tempfile
import unittest

from mujoco_warp._src import solver
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.pen_stable_friction_factory_v1 import generate,load_generated
from supermimic_research.runtime.pen_stable_friction_binding_v1 import FactoryBinding
from supermimic_research.runtime.pen_stable_friction_profile_v1 import execution_identity


class StableFrictionBindingTests(unittest.TestCase):
    def fixture(self,folder):
        manifest=generate(file_pin(solver.__file__),Path(folder)/'factory.py')
        self.addCleanup(sys.modules.pop,load_generated(manifest).__name__,None)
        return manifest

    def test_exclusive_binding_cleanup_and_original_stopping(self):
        original=solver._linesearch_iterative_kernel; stopping=solver._solve_done; sync=[]
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.fixture(folder)
            binding=FactoryBinding(manifest,synchronize=lambda:sync.append(True))
            with self.assertRaisesRegex(RuntimeError,'intentional'):
                with binding:
                    self.assertIs(solver._solve_done,stopping)
                    solver._linesearch_iterative_kernel(50,solver.types.ConeType.PYRAMIDAL,True,False,True)
                    with self.assertRaises(ValueError): solver._linesearch_iterative_kernel(50,solver.types.ConeType.ELLIPTIC,True,False,True)
                    with self.assertRaisesRegex(ValueError,'Exclusive'):
                        with FactoryBinding(manifest,synchronize=lambda:None): pass
                    raise RuntimeError('intentional')
            self.assertIs(solver._linesearch_iterative_kernel,original)
            self.assertIs(solver._solve_done,stopping)
            self.assertEqual(sync,[True]); self.assertTrue(binding.audit['bindings_restored']); self.assertEqual(len(binding.audit['calls']),1)

    def test_dependency_tampering_detected_and_factory_restored(self):
        original=solver._linesearch_iterative_kernel; stopping=solver._solve_done
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.fixture(folder)
            with self.assertRaisesRegex(ValueError,'stopping changed'):
                with FactoryBinding(manifest,synchronize=lambda:None):
                    solver._solve_done=object()
            self.assertIs(solver._linesearch_iterative_kernel,original)
            # Only this test mutates stopping; the binding never owns that field.
            solver._solve_done=stopping

    def test_distinct_explicit_identities_and_manifest_pins(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=self.fixture(folder)
            a=execution_identity('INSTALLED_ORIGINAL_OR','a'*64,manifest)
            b=execution_identity('STABLE_SAME_REGIME_FRICTION_OR','a'*64,manifest)
            self.assertNotEqual(a['backend_id'],b['backend_id'])
            self.assertEqual(b['identity']['stopping'],'INSTALLED_ORIGINAL_OR')
            self.assertEqual(b['identity']['domain']['worlds'],1)
            with self.assertRaises(ValueError): execution_identity('DROP_IMPROVEMENT_ONLY','a'*64,manifest)
            Path(manifest['generated']['path']).write_text('tampered')
            with self.assertRaises(ValueError): execution_identity('STABLE_SAME_REGIME_FRICTION_OR','a'*64,manifest)


if __name__=='__main__': unittest.main()
