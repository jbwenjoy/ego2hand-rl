from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mujoco_warp._src import solver
from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.backend_pen_runtime import BackendSupportPenRuntime
from supermimic_research.runtime.pen_stable_friction_factory_v4 import generate,load_generated
from supermimic_research.runtime.pen_stable_friction_profile_v2 import execution_identity,CostProfile,ALGORITHMS


class GuardedProfileTests(unittest.TestCase):
    def test_both_profiles_restore_and_reject_unqualified_W32(self):
        original=solver._linesearch_iterative_kernel; stopping=solver._solve_done
        methods={n:getattr(BackendSupportPenRuntime,n) for n in ['metadata','validate_resources','close']}
        with tempfile.TemporaryDirectory() as folder:
            manifest=generate(file_pin(solver.__file__),Path(folder)/'factory.py'); module=load_generated(manifest)
            self.addCleanup(sys.modules.pop,module.__name__,None)
            base=SimpleNamespace(contract=dict(backend_id='a'*64),runtime_set=set())
            for algorithm in ALGORITHMS:
                profile=CostProfile(execution_identity(algorithm,base.contract['backend_id'],manifest),manifest)
                with patch('supermimic_research.runtime.explicit_collision_backend.active_backend',return_value=base),patch('warp.synchronize'):
                    with self.assertRaisesRegex(RuntimeError,'intentional'):
                        with profile:
                            self.assertTrue(profile.verify()); self.assertIs(solver._solve_done,stopping)
                            with self.assertRaisesRegex(ValueError,'W32'): profile.create_runtime(None,32)
                            raise RuntimeError('intentional')
                self.assertTrue(profile.audit['bindings_restored']); self.assertIs(solver._linesearch_iterative_kernel,original)
                for n,f in methods.items(): self.assertIs(getattr(BackendSupportPenRuntime,n),f)


if __name__=='__main__': unittest.main()
