import unittest
from supermimic_research.learning.backend_control_study import study_budget,require_fresh_common,require_final_training


class BackendStudyTests(unittest.TestCase):
    def test_shared_calibration_counted_once(self):
        c=dict(study='FRESH_BACKEND_LONG_CONTROL',num_envs=32,rollout_controls=64,iterations=2000,checkpoint_interval=100,arms=['RESIDUAL','SUPPORT'],calibration_controls=8,repetitions=3,case_order=['INITIAL','RESIDUAL','SUPPORT'],video_repetition=0)
        b=study_budget(c);self.assertEqual(b['cohort_native_before_evaluation'],2*2000*32*64*8+8*32*8)
        c['iterations']=200
        with self.assertRaises(ValueError):study_budget(c)

    def test_legacy_report_ineligible(self):
        with self.assertRaises(ValueError):require_final_training(dict(schema='pen_geometry_support_long_control_training:2'),{},'backend',{})

    def test_trained_common_cannot_seed_as_fresh(self):
        s=dict(schema='pen_geometry_support_PPO_training:4',iteration=1,optimizer_steps=32,training_transitions=2048,calibration_transitions=256,native=dict(worlds=32))
        with self.assertRaises(ValueError):require_fresh_common(s,{})
