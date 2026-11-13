from dataclasses import dataclass
import unittest
import numpy as np
import warp as wp
from supermimic_research.runtime.gpu_model_snapshot import capture_model, restore_model_arrays


@dataclass
class ModelFixture:
    mass: object
    nested: tuple
    count: object = np.int64(2)
    callback: object = None


class ModelSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        wp.init()

    def model(self):
        return ModelFixture(wp.array(np.array([1., 2.], dtype=np.float32), device='cpu'),
                            (wp.array(np.array([3.], dtype=np.float32), device='cpu'),))

    def test_nested_arrays_and_numpy_scalar_survive_exact_restore(self):
        original, restored = self.model(), self.model()
        values, scalars, hashes = capture_model(original)
        self.assertEqual(set(values), {'mass', 'nested.0'})
        self.assertEqual(scalars['count'], 2)
        restored.mass.fill_(5.)
        restore_model_arrays(restored, values, scalars)
        self.assertEqual(capture_model(restored)[2], hashes)

    def test_one_incompatible_late_field_prevents_all_mutation(self):
        model = self.model()
        values, scalars, hashes = capture_model(model)
        values['mass'][:] = 99
        values['nested.0'] = np.zeros(2, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'layout mismatch'):
            restore_model_arrays(model, values, scalars)
        self.assertEqual(capture_model(model)[2], hashes)
