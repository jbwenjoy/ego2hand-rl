from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mujoco_warp._src import solver
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.pen_stable_friction_factory_v1 import generate,load_generated
from supermimic_research.runtime.pen_stable_friction_profile_v1 import execution_identity,CostProfile,ALGORITHMS


class StableFrictionProfileTests(unittest.TestCase):
    def test_both_profiles_preserve_class_and_restore_after_exception(self):
        original=solver._linesearch_iterative_kernel; stopping=solver._solve_done
        methods={n:getattr(BackendSupportPenRuntime,n) for n in ['metadata','validate_resources','close']}
        with tempfile.TemporaryDirectory() as folder:
            manifest=generate(file_pin(solver.__file__),Path(folder)/'factory.py')
            self.addCleanup(sys.modules.pop,load_generated(manifest).__name__,None)
            base=SimpleNamespace(contract=dict(backend_id='a'*64),runtime_set=set())
            for algorithm in ALGORITHMS:
                contract=execution_identity(algorithm,base.contract['backend_id'],manifest)
                profile=CostProfile(contract,manifest)
                with patch('supermimic_research.runtime.explicit_collision_backend.active_backend',return_value=base),patch('warp.synchronize'):
                    with self.assertRaisesRegex(RuntimeError,'intentional'):
                        with profile:
                            self.assertTrue(profile.verify()); self.assertIs(solver._solve_done,stopping)
                            with self.assertRaisesRegex(ValueError,'W32'): profile.create_runtime(None,32)
                            with self.assertRaisesRegex(ValueError,'Exclusive'):
                                with CostProfile(contract,manifest): pass
                            raise RuntimeError('intentional')
                self.assertTrue(profile.audit['bindings_restored'])
                self.assertIs(solver._linesearch_iterative_kernel,original)
                self.assertIs(solver._solve_done,stopping)
                for n,f in methods.items(): self.assertIs(getattr(BackendSupportPenRuntime,n),f)


if __name__=='__main__': unittest.main()
