import unittest
import numpy as np
from supermimic_research.learning.pen_interventions import intervention_profiles, first_raw_action


class PenInterventionTests(unittest.TestCase):
    def test_execution_ema_maps_first_raw_probe_to_requested_PD_target(self):
        directions, rows = intervention_profiles(); delta = np.linspace(-.08,.08,20).astype(np.float32)
        self.assertEqual(len(rows),27); self.assertEqual(len({r['name'] for r in rows}),27)
        np.testing.assert_allclose(directions@directions.T, np.eye(2), atol=1e-15)
        for row in rows:
            direction = np.zeros(20) if row['direction'] < 0 else directions[row['direction']]
            raw = first_raw_action(delta,direction,row['epsilon'],row['sign'])
            actual = .5*delta + .5*np.clip(.5*raw,-.6,.6)
            np.testing.assert_allclose(actual,delta+row['sign']*row['epsilon']*direction,atol=1e-8)
        with self.assertRaises(ValueError): first_raw_action(np.ones(20),directions[0],.002,1)
