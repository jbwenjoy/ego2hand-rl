from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from supermimic_research.runtime.controlled_evaluation import summarize_trace


class ControlledEvaluationTests(unittest.TestCase):
    def fixture(self):
        z = dict(post_qpos=np.zeros((2,27)),post_COM=np.zeros((2,3)),
            post_quaternion_wxyz=np.tile([1.,0,0,0],(2,1)),pre_penetration_m=np.zeros(2),
            pre_self_penetration_m=np.zeros(2),post_joint_limit_excursion_rad=np.zeros(2),
            pre_object_nonhand_normal_N=np.zeros(2),pre_hand_normal_N=np.ones(2),
            post_COM_velocity=np.zeros((2,3)),post_omega_world=np.zeros((2,3)),
            post_qfrc_applied=np.zeros((2,26)),post_xfrc_applied=np.zeros((2,3,6)),overflow=np.zeros(2,dtype=bool))
        ref=dict(time_s=np.array([0.,.04]),COM_world_m=np.zeros((2,3)),quaternion_wxyz=np.tile([1.,0,0,0],(2,1)))
        c=dict(COM_RMS_m=.005,COM_max_m=.01,angle_RMS_rad=np.pi/36,angle_max_rad=np.pi/18,
            max_penetration_m=.002,max_self_penetration_m=.0002,max_joint_limit_excursion_rad=.001,
            max_object_nonhand_normal_force_N=.1)
        return z,ref,c

    def test_incomplete_duration_does_not_qualify_despite_perfect_short_prefix(self):
        z,r,c=self.fixture()
        self.assertFalse(summarize_trace(z,r,.02,False,c)['qualified'])
        self.assertTrue(summarize_trace(z,r,.02,True,c)['qualified'])

    def test_full_pen_roll_error_cannot_hide_in_perfect_shaft_direction(self):
        z,r,c=self.fixture()
        z['post_quaternion_wxyz'][:]=[np.cos(.5),np.sin(.5),0,0]
        result=summarize_trace(z,r,.02,True,c)
        self.assertAlmostEqual(result['metrics']['angle_RMS_rad'],1.)
        self.assertFalse(result['qualified'])

    def test_force_aid_and_capacity_overflow_reject_geometrically_perfect_run(self):
        for field in ['post_qfrc_applied','post_xfrc_applied','overflow']:
            with self.subTest(field=field):
                z,r,c=self.fixture();z[field].flat[-1]=1
                self.assertFalse(summarize_trace(z,r,.02,True,c)['qualified'])


if __name__=='__main__':unittest.main()
