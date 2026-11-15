import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.pen_replay_fidelity import compare_response


class ReplayFidelityTests(unittest.TestCase):
    def setup_case(self):
        q=np.zeros((2,27));q[:,3]=1;v=np.zeros((2,26));ctrl=np.zeros((2,20))
        actual=dict(post_qpos=q,post_qvel=v,post_ctrl=ctrl,post_GPU_time=np.array([.005,.01]))
        gates=dict(hand_position_rad=.002,object_position_m=.0005,object_rotation_rad=.01,hand_velocity_rad_s=.2,
            object_velocity_m_s=.01,object_omega_rad_s=.2,control_rad=2e-7,EFC_force_N=1e-4,EFC_torque_Nm=1e-5)
        return actual,gates

    def test_identical_states_do_not_hide_force_failure(self):
        a,g=self.setup_case();p=copy.deepcopy(a);w=np.zeros((2,6))
        _,checks=compare_response(p,a,w,w,np.arange(7,27),np.arange(6,26),0,0,g)
        self.assertTrue(all(checks.values()))
        wrong=w.copy();wrong[0,0]=2e-4
        _,checks=compare_response(p,a,wrong,w,np.arange(7,27),np.arange(6,26),0,0,g)
        self.assertFalse(checks['EFC_force_N']);self.assertTrue(checks['object_position_m'])

    def test_object_first_indices_and_world_angular_velocity(self):
        a,g=self.setup_case();p=copy.deepcopy(a);p['post_qpos'][:,0]=.0006
        xyzw=Rotation.from_rotvec([0,0,np.pi/2]).as_quat();p['post_qpos'][:,3:7]=xyzw[[3,0,1,2]]
        a['post_qvel'][:,3]=1;p['post_qvel'][:,3]=1
        w=np.zeros((2,6));metrics,checks=compare_response(p,a,w,w,np.arange(7,27),np.arange(6,26),0,0,g)
        self.assertFalse(checks['object_position_m']);self.assertTrue(checks['hand_position_rad'])
        self.assertAlmostEqual(metrics['object_omega_rad_s'],np.sqrt(2));self.assertFalse(checks['object_omega_rad_s'])
