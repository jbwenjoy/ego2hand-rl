import unittest

import numpy as np
import torch

from supermimic_research.mechanics.pen_generator_capacity import (
    GENERATOR_SCHEMA, GeneratorCapacityHead, load_generator_capacity_head, generator_support,
    pair_bound_violations)
from supermimic_research.mechanics.pen_wrench_capacity import balanced_directions, directional_support


class TestGeneratorCapacity(unittest.TestCase):
    def test_support_matches_cpu_geometry_for_batched_generators(self):
        rng=np.random.default_rng(283)
        points=rng.normal(size=(2,12,6))
        probes=balanced_directions()
        pred=generator_support(torch.tensor(points),torch.tensor(probes))
        oracle=np.stack([directional_support(g,probes) for g in points])
        np.testing.assert_allclose(pred.numpy(),oracle,atol=1e-12,rtol=1e-12)
        upper,lower=pair_bound_violations(pred)
        self.assertLess(float(torch.maximum(upper,lower).max()),1e-12)
        # Eight native substep supports averaged in BODY are still subadditive.
        upper,lower=pair_bound_violations(pred.mean(0))
        self.assertLess(float(torch.maximum(upper,lower).max()),1e-12)

    def test_signed_pair_upper_and_antipodal_lower_bounds(self):
        caps=torch.ones(72,dtype=torch.float64)
        # Probe 15 is (+e0 +e1)/sqrt(2); axis 7 is -e1.
        np.testing.assert_array_equal(balanced_directions()[15,:2],np.ones(2)/np.sqrt(2))
        caps[15]=3.
        upper,lower=pair_bound_violations(caps)
        self.assertAlmostEqual(float(upper[3]),3-np.sqrt(2))
        self.assertEqual(float(lower[3]),0.)
        caps[0]=3.;caps[7]=.5;caps[15]=.2
        upper,lower=pair_bound_violations(caps)
        self.assertEqual(float(upper[3]),0.)
        self.assertAlmostEqual(float(lower[3]),2.5/np.sqrt(2)-.2)

    def test_checkpoint_roundtrip_and_pd_derivative(self):
        torch.manual_seed(2821)
        model=GeneratorCapacityHead()
        x=torch.randn(24,75); y=torch.rand(24,72)
        model.set_training_stats(x,y)
        state=torch.randn(55);pd=torch.randn(20)
        ck=dict(schema=GENERATOR_SCHEMA,use_history=False,state_dict=model.state_dict())
        restored=load_generator_capacity_head(ck)
        torch.testing.assert_close(restored(state,pd),model(state,pd),rtol=0,atol=0)
        restored=restored.double().requires_grad_(False)
        u=pd.double().requires_grad_(True)
        fn=lambda action:restored(state.double(),action)
        self.assertTrue(torch.autograd.gradcheck(fn,(u,),fast_mode=True))
        grad,=torch.autograd.grad(fn(u).sum(),u)
        self.assertGreater(float(grad.norm()),1e-8)


if __name__=='__main__':
    unittest.main()
