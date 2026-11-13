from types import SimpleNamespace
import unittest
from supermimic_research.runtime.numerical_options import validate_numerical_options, set_model_numerical_options


class NumericalOptionsTests(unittest.TestCase):
    def test_physical_keys_rejected_before_any_option_changes(self):
        host = SimpleNamespace(opt=SimpleNamespace(iterations=10))
        with self.assertRaisesRegex(ValueError, 'Only iteration'):
            set_model_numerical_options(host, None, {'iterations': 50, 'gravity': -1.})
        self.assertEqual(host.opt.iterations, 10)

    def test_bad_limits_never_coerce_into_legal_settings(self):
        for value in [True, 1.5, 0, -1, float('nan')]:
            with self.assertRaises(ValueError):
                validate_numerical_options({'iterations': value})

    def test_tight_explicit_values_preserve_requested_precision(self):
        result = validate_numerical_options({'iterations': 50., 'tolerance': 1e-12, 'ls_tolerance': 1e-8})
        self.assertIsInstance(result['iterations'], int)
        self.assertEqual(result['tolerance'], 1e-12)
        self.assertEqual(result['ls_tolerance'], 1e-8)
