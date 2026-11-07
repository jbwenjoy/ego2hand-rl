"""Batched20 guarded hinge rotations versus the frozen phase1 FK on CPU."""
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

import torch
import test_geometry_inference as inference_fixtures
from supermimic_research.runtime import geometry as current


class BatchedHingeTests(unittest.TestCase):
    measurements = []

    @classmethod
    def setUpClass(cls):
        inference_fixtures.GeometryInferenceTests.setUpClass()
        cls.fixture = inference_fixtures.GeometryInferenceTests
        # Read the committed reference, not a local ignored benchmark run.
        # A shallow checkout may need this one preceding performance commit.
        try:
            source = subprocess.run(['git', 'show',
                '1e55b54:src/supermimic_research/runtime/geometry.py'],
                cwd=inference_fixtures.ROOT, check=True, capture_output=True).stdout
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise unittest.SkipTest('Requires reference commit 1e55b54 in local Git history') from error
        name = 'supermimic_research.runtime._phase1_geometry_test'
        cls.phase1 = types.ModuleType(name)
        cls.phase1.__package__ = 'supermimic_research.runtime'
        sys.modules[name] = cls.phase1
        exec(compile(source, 'git:1e55b54/runtime/geometry.py', 'exec'), cls.phase1.__dict__)

    def provider(self, module):
        return module.PhoneGeometryProvider(self.fixture.metadata_path,
            geometry_arrays_path=self.fixture.geometry_arrays_path)

    def test_scalar_batch_and_extra_leading_dimension_forward_and_AD(self):
        for dtype in (torch.float64, torch.float32):
            base = torch.tensor(self.fixture.states, dtype=dtype, device='cpu')
            for shape_name, source in [('scalar',base[0]), ('batch2',base), ('batch2x1',base[:,None,:])]:
                with self.subTest(dtype=dtype, shape=shape_name):
                    old = self.provider(self.phase1); new = self.provider(current)
                    a = source.clone().requires_grad_(True)
                    b = source.clone().requires_grad_(True)
                    expected = old.poses(a)
                    with patch.object(current._exp_module, 'so3_exp_torch', wraps=current._exp_module.so3_exp_torch) as guarded_exp:
                        observed = new.poses(b)
                        self.assertEqual(guarded_exp.call_count, 1)
                        self.assertEqual(guarded_exp.call_args.args[0].shape, source.shape[:-1]+(20,3))
                    self.assertEqual(set(expected), set(observed))
                    tolerance = 1e-12 if dtype == torch.float64 else 1e-6
                    max_error = 0.
                    for key in expected:
                        torch.testing.assert_close(expected[key], observed[key], rtol=tolerance, atol=tolerance)
                        max_error = max(max_error, float((expected[key]-observed[key]).abs().max().detach()))
                    def scalar(v):
                        return (v['body_position_world_m'].square().sum()
                            + v['geom_position_world_m'].square().sum()
                            + .1*v['pair_relative_rotation_BODY9'].square().sum())
                    ga, = torch.autograd.grad(scalar(expected), a)
                    gb, = torch.autograd.grad(scalar(observed), b)
                    self.assertTrue(torch.isfinite(ga).all() and torch.isfinite(gb).all())
                    torch.testing.assert_close(ga, gb, rtol=tolerance, atol=tolerance)
                    self.measurements.append(dict(dtype=str(dtype), shape=list(source.shape),
                        public_Rodrigues_calls_phase1=20, public_Rodrigues_calls_batch=1,
                        output_max_abs_error=max_error,
                        input_AD_max_abs_error=float((ga-gb).abs().max()),
                        input_AD_calls=2))

    def test_structural_nonfinite_rotation_and_overflow_guards(self):
        provider = self.provider(current)
        state = torch.tensor(self.fixture.states[0], dtype=torch.float64, device='cpu')
        nan = state.clone(); nan[0] = float('nan')
        improper = state.clone(); improper[43] += 1.
        overflow = state.clone(); overflow[0] = 1e200
        for name, value, message in [('wrong_width',state[:-1],'state58'),
                ('integer_dtype',state.to(torch.int64),'state58'),
                ('nonfinite',nan,'state58'), ('improper_SO3',improper,'SO3'),
                ('finite_rotvec_squared_overflow',overflow,'Rotation magnitude overflow')]:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError,message): provider.poses(value)
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
