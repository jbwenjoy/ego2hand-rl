import unittest
import numpy as np
from supermimic_research.learning.pen_native_screening import proposal,random_direction,choose


class ScreeningTests(unittest.TestCase):
    def test_zero_nan_and_nonfinite_prediction_retain_original_clipped_RAW(self):
        raw=np.zeros(20,np.float32);raw[0]=1.4
        for direction,finite in [(np.zeros(20),True),(np.full(20,np.nan),True),(np.ones(20),False)]:
            p=proposal(raw,direction,prediction_finite=finite)
            self.assertTrue(p['duplicate_BASE']);np.testing.assert_array_equal(p['raw'],raw)
        direction=np.zeros(20);direction[1]=1;p=proposal(raw,direction)
        self.assertFalse(p['duplicate_BASE']);self.assertEqual(p['raw'][0],raw[0]);self.assertAlmostEqual(float(p['raw'][1]),.008)

    def test_new_clipping_is_duplicate_without_projection_or_smaller_radius(self):
        raw=np.zeros(20,np.float32);raw[0]=.999;direction=np.zeros(20);direction[0]=1
        p=proposal(raw,direction);self.assertEqual(p['reason'],'CHANGED_COORDINATE_CLIPS');np.testing.assert_array_equal(p['raw'],raw)
        p=proposal(np.zeros(20,np.float32),np.ones(20));self.assertAlmostEqual(np.linalg.norm(p['actual_raw_delta']),.008,places=8)

    def test_random_index_is_reproducible_and_does_not_change_global_rng(self):
        before=np.random.get_state();a=random_direction(9901,1,0,7);b=random_direction(9901,1,0,7);after=np.random.get_state()
        np.testing.assert_array_equal(a,b);self.assertFalse(np.array_equal(a,random_direction(9901,1,0,8)))
        self.assertTrue(all(np.array_equal(x,y) for x,y in zip(before,after)))

    def branch(self,**kw):return dict(dict(physical_valid=True,terminated=False,truncated=False,reward=1.),**kw)

    def test_invalid_baseline_cannot_be_rescued_and_terminal_can(self):
        bad=self.branch(physical_valid=False);good=self.branch(reward=2.)
        self.assertEqual(choose(bad,good)['reason'],'BASE_PHYSICAL_INVALID');self.assertTrue(choose(bad,None)['stop'])
        self.assertEqual(choose(self.branch(),self.branch(physical_valid=False,reward=None))['selected'],'BASE')
        fail=self.branch(terminated=True);out=choose(fail,good);self.assertTrue(out['accepted']);self.assertFalse(out['stop'])

    def test_rejected_candidate_restores_baseline_failure_or_completion_semantics(self):
        base=self.branch(terminated=True)
        for candidate in [self.branch(reward=1.00009),self.branch(reward=2.,physical_valid=False),self.branch(reward=2.,terminated=True)]:
            out=choose(base,candidate);self.assertEqual(out['selected'],'BASE');self.assertTrue(out['selected_failure'])
        complete=self.branch(truncated=True,reward=2.);out=choose(self.branch(),complete);self.assertTrue(out['selected_completion']);self.assertFalse(out['selected_failure'])

    def test_bad_finite_and_terminal_contracts_are_rejected(self):
        with self.assertRaises(ValueError):choose(self.branch(),None)
        with self.assertRaises(ValueError):choose(self.branch(reward=float('nan')),self.branch())
        with self.assertRaises(ValueError):choose(self.branch(),self.branch(terminated=True,truncated=True))


if __name__=='__main__':unittest.main()
