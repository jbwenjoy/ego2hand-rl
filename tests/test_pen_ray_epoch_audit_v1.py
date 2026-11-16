from pathlib import Path
import unittest

from mujoco_warp._src import solver

from supermimic_research.mechanics.pen_ray_epoch_audit_v1 import source_facts


class RaySourceFactsTests(unittest.TestCase):
    def test_real_source_order_and_flag_semantics(self):
        facts=source_facts(Path(solver.__file__).read_text())
        self.assertTrue(all(facts['checks'].values()),facts)

    def test_detect_changed_noise_flag_and_acceptance(self):
        text=Path(solver.__file__).read_text()
        changed=text.replace('ctx_ls_exhausted_out[worldid] = wp.abs(alpha) < noise_floor',
                             'ctx_ls_exhausted_out[worldid] = True')
        self.assertFalse(source_facts(changed)['checks']['exhausted_is_step_noise_test'])
        changed=text.replace('initial_converged = wp.abs(lo_in[1]) < gtol and lo_in[0] < 0.0',
                             'initial_converged = wp.abs(lo_in[1]) < gtol')
        self.assertFalse(source_facts(changed)['checks']['initial_accept_requires_negative_delta'])


if __name__=='__main__': unittest.main()
