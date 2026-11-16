import unittest
import numpy as np
from supermimic_research.learning.pen_native_suggestions import direction_from_raw_gradient,proposal_raw,profiles,summarize_improvement


class SuggestionTests(unittest.TestCase):
    def test_exact_direction_and_raw_to_pd_radius(self):
        g=np.arange(1,21,dtype=float);u,n=direction_from_raw_gradient(g);self.assertAlmostEqual(np.linalg.norm(u),1.);self.assertAlmostEqual(n,np.linalg.norm(g))
        raw,ok=proposal_raw(np.zeros(20),u,.005,1);self.assertTrue(ok);np.testing.assert_allclose(.25*raw,.005*u,atol=2e-10)

    def test_clipped_proposal_is_retained_unclamped(self):
        u=np.zeros(20);u[0]=1;raw,ok=proposal_raw(np.ones(20),u,.005,1);self.assertFalse(ok);self.assertGreater(raw[0],1.)

    def test_zero_gradient_has_no_substitute(self):
        u,n=direction_from_raw_gradient(np.zeros(20));self.assertIsNone(u);self.assertEqual(n,0)
        with self.assertRaises(ValueError):direction_from_raw_gradient(np.full(20,np.nan))

    def test_every_declared_profile_and_reverse_control_remains(self):
        p=profiles(['A','B','C']);self.assertEqual(len(p),50);self.assertEqual(len({v['name'] for v in p}),50)
        for name in ['A','B','C']:
            self.assertEqual(sum(v['model']==name for v in p),16)

    def test_improvement_requires_original_center_and_all_physics(self):
        v=summarize_improvement([1,1],[1.1,1.1],[.9,.9],epsilon=.005,gradient_norm=5.,physical_valid=True,original_base_valid=True)
        self.assertTrue(v['improvement_exceeds_observed_repeat_envelope']);self.assertTrue(v['ascent_beats_reverse_exceeds_observed_repeat_envelope']);self.assertAlmostEqual(v['native_FD_per_RAW'],5.)
        for physical,base in [(False,True),(True,False)]:
            q=summarize_improvement([1,1],[1.1,1.1],[.9,.9],epsilon=.005,gradient_norm=5.,physical_valid=physical,original_base_valid=base)
            self.assertFalse(q['eligible']);self.assertFalse(q['improvement_exceeds_observed_repeat_envelope']);self.assertGreater(q['positive_gain'],0)

    def test_repeat_envelope_and_negative_gains_are_not_silently_dropped(self):
        v=summarize_improvement([1,1],[.8,.9],[1.1,1.1],epsilon=.002,gradient_norm=1.,physical_valid=True,original_base_valid=True)
        self.assertTrue(v['descent_exceeds_observed_repeat_envelope']);self.assertLess(v['positive_gain'],0)
        noisy=summarize_improvement([.8,1.2],[1,1.2],[1,1],epsilon=.002,gradient_norm=1.,physical_valid=True,original_base_valid=True)
        self.assertFalse(noisy['improvement_exceeds_observed_repeat_envelope'])


if __name__=='__main__':unittest.main()
