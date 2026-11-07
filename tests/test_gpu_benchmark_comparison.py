"""Saved comparison boundaries; no Torch/NN/GPU or artificial optimizer."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_gpu_benchmarks import array_metrics, protocol_comparison, TOLERANCES
from unittest.mock import patch


class ComparisonTests(unittest.TestCase):
    def test_analytic_norm_and_zero_reference(self):
        m = array_metrics(np.array([3., 4.]), np.array([3., 4.000001]), category="gradient")
        self.assertAlmostEqual(m["reference_L2"], 5.)
        self.assertAlmostEqual(m["relative_L2"], 1e-6 / 5., places=14)
        z = array_metrics(np.zeros(2), np.array([1e-4, 0.]), category="gradient")
        self.assertIsNone(z["relative_L2"])
        self.assertFalse(z["pass"])

    def test_dtype_shape_nan_and_normalizer_bits_rejected(self):
        self.assertFalse(array_metrics(np.ones(2, np.float32), np.ones(2), category="state")["pass"])
        self.assertFalse(array_metrics(np.ones(2), np.ones((1, 2)), category="state")["pass"])
        self.assertFalse(array_metrics(np.array([np.nan]), np.array([np.nan]), category="gradient")["pass"])
        m = array_metrics(np.array([0.]), np.array([-0.]), category="state", exact=True)
        self.assertEqual(m["max_abs"], 0.)
        self.assertFalse(m["pass"])

    def test_fixed_envelope_detects_large_gradient(self):
        self.assertEqual(TOLERANCES["gradient"]["relative_L2_max"], 1e-5)
        self.assertTrue(array_metrics(np.ones(4), np.ones(4), category="gradient")["pass"])
        self.assertFalse(array_metrics(np.ones(4), np.ones(4) + .01, category="gradient")["pass"])

    def test_micro_flag_does_not_allow_changed_rows_or_budget(self):
        base = dict(schema="x", status="x", device="x", B=8, H=2, NN_dtype="float32", physical_dtype="float64", AMP=False, TF32=False, WM_sources={}, private_paired_updates=6, private_Adam_steps=30, real_bank_resident=True, real_bank_scope="same", microbatch=2, parent_checkpoint={"bytes": 1, "sha256": "p"}, tool={"bytes": 1, "sha256": "t"}, rows=[dict(index=0, phase="measured", row_ids=[1, 2])])
        other = base | {"microbatch": 8}
        with patch("compare_gpu_benchmarks.validate_report", side_effect=lambda _: {"training": {"microbatch": 2}}):
            self.assertFalse(protocol_comparison(base, other)["microbatch_permission"])
            self.assertTrue(all(protocol_comparison(base, other, allow_microbatch_change=True).values()))
            changed = other | {"rows": [dict(index=0, phase="measured", row_ids=[2, 1])], "private_Adam_steps": 35}
            checks = protocol_comparison(base, changed, allow_microbatch_change=True)
            self.assertFalse(checks["same_indices_phases_row_IDs"])
            self.assertFalse(checks["same_private_Adam_steps"])


if __name__ == "__main__":
    unittest.main()
