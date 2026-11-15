import unittest
import numpy as np
import torch
from supermimic_research.learning.candidate_response_evaluation import candidate_pairs, evaluate_pairs
from supermimic_research.learning.pen_native_response import vector_comparison
from supermimic_research.learning.pen_response_evaluation import pooled_error


def fixture():
    r = np.repeat(np.arange(3), 57)
    a = np.tile(np.arange(57), 3)
    x = np.zeros((171, 102))
    x[:, -1] = .005
    for repeat in range(3):
        for axis in range(20):
            x[repeat * 57 + axis + 1, 81 + axis] = .03
            x[repeat * 57 + axis + 21, 81 + axis] = -.01
    # Asymmetric clipping: the true midpoint is .01, not the zero candidate.
    return dict(x=x, y=np.zeros((171, 90)), valid=np.ones(171, bool),
                parent_id=np.zeros(171, int), candidate=a, repeat=r, phase=np.ones(171, int))


class Cubic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('y_scale', torch.ones(90, dtype=torch.float64))

    def prediction(self, x):
        value = x[:, 81:101].double().pow(3).sum(-1, keepdim=True)
        return value.expand(-1, 90)


class CandidateEvaluationTests(unittest.TestCase):
    def test_zero_native_response_keeps_absolute_model_error(self):
        zero = vector_comparison([2., 0.], [0., 0.])
        report = pooled_error([zero])
        self.assertIsNone(report['pooled_relative_L2_error'])
        self.assertEqual(report['pooled_absolute_L2_error'], 2.)
        # Same correct handling when reading the older summary schema.
        zero.pop('absolute_L2_error')
        self.assertEqual(pooled_error([zero])['pooled_absolute_L2_error'], 2.)

    def test_native_cubic_secant_differs_from_midpoint_AD(self):
        data = fixture()
        data['y'][:] = np.sum(data['x'][:, 81:101]**3, axis=1)[:, None]
        pairs = candidate_pairs(data)
        self.assertEqual(len(pairs), 20)
        self.assertTrue(all(p['eligible'] for p in pairs))
        self.assertAlmostEqual(pairs[0]['midpoint'][81], .01)
        report, _ = evaluate_pairs(Cubic(), data, pairs, 'cpu')
        row = report['rows'][0]
        self.assertLess(row['secant']['object12']['relative_L2_error'], 1e-6)
        self.assertAlmostEqual(row['AD']['object12']['relative_L2_error'], 4 / 7, places=5)
        self.assertTrue(row['AD']['object12']['contrast_exceeds_observed_spread'])

    def test_masks_source_memory_and_control_repeats_are_required(self):
        for column, expected in [(55, 'exact_source_warm_dt'), (81, 'exact_repeated_PD')]:
            data = fixture()
            data['x'][58, column] += .001
            pairs = candidate_pairs(data)
            self.assertFalse(pairs[0]['eligible'])
            self.assertFalse(pairs[0]['checks'][expected])
        data = fixture()
        data['valid'][1] = False
        self.assertFalse(candidate_pairs(data)[0]['eligible'])

    def test_zero_clipped_span_and_missing_repeat_remain_failures(self):
        data = fixture()
        data['x'][[1, 58, 115], 81] = -.01
        self.assertFalse(candidate_pairs(data)[0]['checks']['nonzero_PD_span'])
        data['candidate'][0] = 1
        with self.assertRaises(ValueError):
            candidate_pairs(data)
