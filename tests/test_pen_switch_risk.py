import unittest

import mujoco
import numpy as np

from supermimic_research.mechanics.pen_switch_risk import PenSwitchRisk


class TestSwitchRisk(unittest.TestCase):
    def setUp(self):
        # Offset the moving fingertip so angular velocity contributes to the gap.
        xml = """<mujoco><worldbody>
          <body><freejoint/><geom name="robot/tip" pos="0 .01 0" size=".01"/></body>
          <body name="pen" pos=".021 .01 0"><freejoint/>
            <geom name="pen_surface" size=".01"/>
            <geom name="decoration" size=".012" contype="0" conaffinity="0"/>
          </body></worldbody></mujoco>"""
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.risk = PenSwitchRisk(self.m, 'pen')

    def evaluate(self):
        return self.risk.evaluate(self.d.qpos, self.d.qvel,
                                  self.d.mocap_pos, self.d.mocap_quat)

    def test_onset_release_and_angular_velocity(self):
        self.d.qvel[0] = .11
        self.d.qvel[5] = 1.  # omega × (0, .01, 0) subtracts .01 m/s
        onset = self.evaluate()
        self.assertEqual(onset['candidate_pairs'], 1)
        self.assertEqual(onset['closest_event']['event'], 'onset')
        self.assertAlmostEqual(onset['closest_event']['normal_speed_m_s'], -.10, places=9)
        self.assertAlmostEqual(onset['closest_event']['crossing_time_s'], .01, places=9)
        self.d.qpos[7] = .019
        deeper = self.evaluate()
        self.assertFalse(deeper['predicted_crossing'])
        self.assertGreater(onset['score'], deeper['score'])
        self.d.qvel[:] *= -1
        release = self.evaluate()
        self.assertEqual(release['closest_event']['event'], 'release')
        self.assertAlmostEqual(release['closest_event']['crossing_time_s'], .01, places=9)

    def test_rigid_translation_cutoff_and_exact_touch(self):
        self.d.qvel[0] = self.d.qvel[6] = .2
        self.assertAlmostEqual(self.evaluate()['score'], 0.)
        self.d.qpos[7] = .09
        far = self.evaluate()
        self.assertEqual(far['censored_pairs'], 1)
        self.assertEqual(far['pairs'], [])
        self.d.qpos[7] = .02
        self.d.qvel[0] = 0.
        touch = self.evaluate()
        self.assertEqual(touch['pairs'][0]['derivative_method'], 'directional_distance')
        self.assertAlmostEqual(touch['pairs'][0]['normal_speed_m_s'], .2, places=8)
        self.assertEqual(touch['closest_event']['event'], 'release')


if __name__ == '__main__':
    unittest.main()
