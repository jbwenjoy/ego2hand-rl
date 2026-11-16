"""Actual CPU kernel decisions plus source/binding guards, no native dynamics."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import warp as wp
from mujoco_warp._src import solver

from supermimic_research.artifacts import file_pin
from supermimic_research.runtime.pen_termination_rule import (
    implementation_identity, kernel_binding, qualify_sources)
from supermimic_research.runtime.pen_termination_rule_kernels import candidate_solve_done


class TerminationRuleTests(unittest.TestCase):
    def setUp(self):
        self.pin = file_pin(solver.__file__)

    def test_exact_ast_and_distinct_unqualified_identity(self):
        proof = qualify_sources(self.pin)
        self.assertTrue(proof['single_expression_change'])
        result = implementation_identity('2acb590a1c11a704257aaa1beeb53a5d572c5bdbc67b7555d2298641854bd162', self.pin)
        self.assertNotEqual(result['backend_id'], result['identity']['collision_backend_id'])
        self.assertFalse(result['physically_qualified'])
        self.assertFalse(result['old_source_transfer_qualified'])

    def test_reject_second_ast_change(self):
        path = Path(qualify_sources(self.pin)['candidate']['path'])
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / 'changed.py'
            changed.write_text(path.read_text().replace('solver_niter_out[worldid] += 1', 'solver_niter_out[worldid] += 2'))
            with self.assertRaisesRegex(ValueError, 'more than'):
                qualify_sources(self.pin, changed)

    def test_binding_cleanup_and_nested_rejection(self):
        original = solver._solve_done
        events = []
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            with kernel_binding(solver, self.pin, candidate_solve_done, synchronize=lambda: events.append('sync')) as audit:
                self.assertIs(solver._solve_done, candidate_solve_done)
                with self.assertRaises(ValueError):
                    with kernel_binding(solver, self.pin, candidate_solve_done, synchronize=lambda: None):
                        pass
                raise RuntimeError('deliberate')
        self.assertIs(solver._solve_done, original)
        self.assertEqual(events, ['sync'])
        self.assertTrue(audit['bindings_restored'])

    def test_synchronization_error_still_restores(self):
        original = solver._solve_done
        def fail():
            raise RuntimeError('sync failure')
        with self.assertRaisesRegex(RuntimeError, 'sync failure'):
            with kernel_binding(solver, self.pin, candidate_solve_done, synchronize=fail):
                pass
        self.assertIs(solver._solve_done, original)

    def test_cpu_kernels_cover_or_branches_cap_and_already_done(self):
        wp.init()
        # Scale is nv26 * meaninertia0.5 =13. Both tolerance layouts are exercised.
        for tolerance in [np.array([1e-6], np.float32), np.full(9, 1e-6, np.float32)]:
            grad = np.array([1, 1e-14, 1, 1, 1, 1, 1, 1, 1], np.float32)
            decrement = np.array([1, 1, 1e-8, 1, 1, 1, 1e-8, 1, 1], np.float32)
            improvement = np.array([0, 1, 1, 1, 0, 0, 0, -1, 1], np.float32)
            initial_done = np.array([False] * 5 + [True, False, False, False])
            initial_niter = np.array([1, 1, 1, 49, 49, 7, 1, 1, 1], np.int32)
            outputs = []
            for kernel in [solver._solve_done, candidate_solve_done]:
                niter = wp.array(initial_niter, dtype=int, device='cpu')
                done = wp.array(initial_done, dtype=bool, device='cpu')
                nsolving = wp.array(np.array([8], np.int32), dtype=int, device='cpu')
                inputs = [26, wp.array(tolerance, dtype=float, device='cpu'), 50,
                          wp.array(np.array([.5], np.float32), dtype=float, device='cpu'),
                          wp.array(grad, dtype=float, device='cpu'),
                          wp.array(decrement, dtype=float, device='cpu'),
                          wp.array(improvement, dtype=float, device='cpu'), done]
                wp.launch(kernel, dim=9, inputs=inputs, outputs=[niter, nsolving, done], device='cpu')
                outputs.append((niter.numpy(), nsolving.numpy(), done.numpy()))
            np.testing.assert_array_equal(outputs[0][0], [2, 2, 2, 50, 50, 7, 2, 2, 2])
            np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
            np.testing.assert_array_equal(outputs[0][2], [True] * 8 + [False])
            np.testing.assert_array_equal(outputs[1][2], [False, True, True, True, True, True, True, False, False])
            np.testing.assert_array_equal(outputs[0][1], [1])
            np.testing.assert_array_equal(outputs[1][1], [3])

    def test_cpu_strict_tolerance_boundary(self):
        wp.init()
        for kernel in [solver._solve_done, candidate_solve_done]:
            array = lambda a, dtype: wp.array(np.array(a), dtype=dtype, device='cpu')
            done = array([False], bool); niter = array([1], int); active = array([1], int)
            wp.launch(kernel, dim=1, inputs=[1, array([1.], float), 50, array([1.], float),
                      array([1.], float), array([2.], float), array([1.], float), done],
                      outputs=[niter, active, done], device='cpu')
            self.assertFalse(done.numpy()[0])
            self.assertEqual(niter.numpy()[0], 2)
            self.assertEqual(active.numpy()[0], 1)


if __name__ == '__main__':
    unittest.main()
