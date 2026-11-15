import unittest
import numpy as np
from supermimic_research.learning.pen_native_response import repeated_contrast_spread


class ContrastSpreadTests(unittest.TestCase):
    def test_shared_baseline_shift_preserves_signal_but_noisy_constant_effect_does_not_pass(self):
        p=np.array([[2.,0],[2.,0],[2.,0]]);m=np.zeros_like(p);scale=np.array([2.,1.])
        a=repeated_contrast_spread(p,m,scale);b=repeated_contrast_spread(p+10,m+10,scale)
        self.assertEqual(a,b);self.assertTrue(a['contrast_exceeds_observed_spread']);self.assertEqual(a['normalized_contrast_L2'],1)
        noisy=np.array([[-2.,0],[0,0],[2.,0]])
        self.assertFalse(repeated_contrast_spread(noisy,np.zeros_like(noisy),scale)['contrast_exceeds_observed_spread'])

    def test_degenerate_scale_and_single_response_are_rejected(self):
        for p,m,s in [(np.ones((1,2)),np.zeros((1,2)),[1,1]),(np.ones((3,2)),np.zeros((3,2)),[1,0]),
                      (np.full((3,2),np.nan),np.zeros((3,2)),[1,1])]:
            with self.assertRaises(ValueError):repeated_contrast_spread(p,m,s)
