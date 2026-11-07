"""CPU numerical contracts for C2's input projection; no simulation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from target_inset import target_inset, validate_inset_domain


class Checks(unittest.TestCase):
    def setUp(self):
        self.lower = torch.tensor([[-.4, 0., -.2]], dtype=torch.float32)
        self.upper = torch.tensor([[.5, 1.5, .3]], dtype=torch.float32)

    def test_matches_independent_numpy_for_all_doses(self):
        g = torch.Generator().manual_seed(4701)
        u = torch.randn((400, 3), generator=g)
        for margin in (0., .005, .020):
            validate_inset_domain(self.lower, self.upper, margin)
            actual = target_inset(u, self.lower, self.upper, margin)
            expected = np.clip(u.numpy(), self.lower.numpy() + np.float32(margin), self.upper.numpy() - np.float32(margin))
            np.testing.assert_array_equal(actual.numpy(), expected)

    def test_zero_identity_on_original_executed_targets(self):
        p = Path(r"D:\workspace\supermimic-chord-study\outputs\replication_eval_a1l_seed1_primary\rollout.npz")
        with np.load(p, allow_pickle=False) as z:
            u = torch.from_numpy(z["joint_target_rad"].astype(np.float32))
            lo = torch.from_numpy(z["joint_lower_rad"].astype(np.float32))
            hi = torch.from_numpy(z["joint_upper_rad"].astype(np.float32))
        self.assertTrue(torch.equal(target_inset(u, lo, hi, 0.), u))

    def test_no_input_mutation_or_rng_consumption(self):
        u = torch.tensor([[-.8, .9, .6]])
        saved = [x.clone() for x in (u, self.lower, self.upper)]
        rng = torch.get_rng_state().clone()
        result = target_inset(u, self.lower, self.upper, .02)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip((u, self.lower, self.upper), saved)))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertNotEqual(result.data_ptr(), u.data_ptr())

    def test_idempotent_projection(self):
        u = torch.tensor([[-.8, .9, .6]])
        for margin in (0., .005, .020):
            once = target_inset(u, self.lower, self.upper, margin)
            self.assertTrue(torch.equal(target_inset(once, self.lower, self.upper, margin), once))

    def test_interior_and_saturated_gradients(self):
        u = torch.tensor([[-.8, .9, .6]], requires_grad=True)
        target_inset(u, self.lower, self.upper, .02).sum().backward()
        self.assertTrue(torch.equal(u.grad, torch.tensor([[0., 1., 0.]])))

    def test_autocast_does_not_reduce_output_precision(self):
        u = torch.tensor([[-.399, .9, .299]], dtype=torch.float32)
        reference = target_inset(u, self.lower, self.upper, .005)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = target_inset(u, self.lower, self.upper, .005)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output, reference))

    def test_invalid_domain_rejected(self):
        for margin in (-.001, float("nan"), float("inf"), .5):
            with self.assertRaises(ValueError):
                validate_inset_domain(self.lower, self.upper, margin)
        with self.assertRaises(ValueError):
            validate_inset_domain(torch.tensor([float("nan")]), torch.ones(1), 0.)

    def test_equal_endpoint_domain_is_valid(self):
        lo, hi = torch.tensor([0.]), torch.tensor([.04])
        validate_inset_domain(lo, hi, .02)
        self.assertTrue(torch.equal(target_inset(torch.tensor([.1]), lo, hi, .02), torch.tensor([.02])))


if __name__ == "__main__":
    outcome = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    output = ROOT / "outputs/target_inset_math_cpu_v1"
    output.mkdir(parents=True, exist_ok=True)
    module = ROOT / "src/target_inset.py"
    record = {"status": "passed" if outcome.wasSuccessful() else "failed", "tests_run": outcome.testsRun,
              "failures": len(outcome.failures), "errors": len(outcome.errors),
              "module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
              "cuda_initialized": torch.cuda.is_initialized(),
              "qualification": "CPU target math only; no physics or real/imag training qualification"}
    (output / "checks.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))
    sys.exit(0 if outcome.wasSuccessful() else 1)
