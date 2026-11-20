"""Pair identity, time boundary, and tied-score checks for contact forecasts."""
import unittest
import numpy as np
from supermimic_research.mechanics.pen_switch_event import DT,first_event,metrics,source_features


class TestPenSwitchEvents(unittest.TestCase):
    def test_any_event_type_is_first_internal_transition(self):
        self.assertEqual(first_event([1,1,1,0,0,1,1,0]), (2,3))
        self.assertEqual(first_event([0,0,1,1,1,1,0,0]), (1,2))
        self.assertEqual(first_event([1]*8), (0,0))

    def test_source_score_does_not_see_unobserved_eighth_interval(self):
        def prediction(g,vel):
            return source_features(dict(gap_m=g,normal_speed_m_s=vel),0,1)[1:3]
        self.assertEqual(prediction(3*DT,-1.),(1,3))
        self.assertEqual(prediction(-3*DT,1.),(2,4))
        self.assertEqual(prediction(7.5*DT,-1.),(0,0))

    def test_tied_ap_is_prevalence(self):
        y=np.array([True,False,True]);zero=np.zeros(3)
        self.assertAlmostEqual(metrics(y,zero.astype(bool),zero,zero,zero)['average_precision'],2/3)
