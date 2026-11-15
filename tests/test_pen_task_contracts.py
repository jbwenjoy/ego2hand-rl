import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.pen_task_contracts import pose_components,first_exceed,rebase_fixed_wrist,normal_loads,candidate_pose_coverage


def wxyz(R):return R.as_quat()[...,[3,0,1,2]]


class PenTaskContractTests(unittest.TestCase):
    def test_axial_roll_is_distinct_from_shaft_angle_in_reference_frame(self):
        ref=Rotation.from_euler('zy',[.7,.3]);actual=ref*Rotation.from_euler('x',np.pi/2)
        m=pose_components(wxyz(actual)[None],wxyz(ref)[None])
        self.assertAlmostEqual(m['full_angle_rad'][0],np.pi/2)
        self.assertAlmostEqual(m['shaft_angle_rad'][0],0.,places=7)
        self.assertAlmostEqual(m['reference_axis_twist_rad'][0],np.pi/2)

    def test_pi_swing_does_not_invent_a_twist_measurement(self):
        m=pose_components(wxyz(Rotation.from_euler('y',np.pi))[None],wxyz(Rotation.identity())[None])
        self.assertFalse(m['twist_defined'][0]);self.assertTrue(np.isnan(m['reference_axis_twist_rad'][0]))

    def test_strict_gate_and_actual_clock_are_separate(self):
        v=[.01,.010001];r=first_exceed(v,.01,[.005,.010],[.00500001,.01000001])
        self.assertEqual(r['index'],1);self.assertEqual(r['nominal_time_s'],.01)
        self.assertEqual(r['actual_GPU_time_s'],.01000001)
        self.assertIsNone(first_exceed([.01],.01,[0],[0]))

    def test_moving_recorded_wrist_cancels_before_mount_rebase(self):
        W=Rotation.from_euler('z',[[.1],[.7],[1.2]]);wp=np.array([[1,0,0],[2,3,0],[4,5,6.]])
        local=np.tile([.1,.2,.3],(3,1));L=Rotation.from_euler('y',.4)
        wrist=np.c_[wp,wxyz(W)];obj=np.c_[wp+W.apply(local),wxyz(W*L)]
        M=Rotation.from_euler('x',.8);mount=np.r_[[.4,.5,.6],wxyz(M)]
        rebased=rebase_fixed_wrist(wrist,obj,mount)
        np.testing.assert_allclose(rebased[:,:3],mount[:3]+M.apply(local),atol=1e-14)
        np.testing.assert_allclose(rebased[:,3:],np.tile(wxyz(M*L),(3,1)),atol=1e-14)

    def test_contact_ownership_counts_finger_palm_and_nonhand_once(self):
        raw=np.zeros((4,24));raw[:,:3]=[[0,0,1],[0,2,0],[0,0,3],[0,1,2]];raw[:,18]=[2,3,4,9]
        names=['object/shaft','robot/right_finger2_tip_col','robot/right_palm_col','floor']
        m=SimpleNamespace(ngeom=4,geom_bodyid=np.array([1,2,2,0]),geom=lambda g:SimpleNamespace(name=names[g]))
        fn,other=normal_loads(dict(post_qpos=np.zeros((1,27)),raw_contact_tick_plus23=raw),m,1)
        np.testing.assert_array_equal(fn,[[0,2,0,0,0,3]]);np.testing.assert_array_equal(other,[4])

    def test_future_endpoint_is_not_mislabeled_as_first_control(self):
        q=np.zeros((57,8,27));q[:,:,3]=1;q[:,4:,0]=.02
        ref=dict(COM_world_m=np.zeros((3,3)),quaternion_wxyz=np.tile([1.,0,0,0],(3,1)))
        first,all_steps,_,_=candidate_pose_coverage(dict(Q=q,score_feasible=np.ones(57,bool)),ref,0,4,.01,np.pi/18,0)
        self.assertTrue(first.all());self.assertFalse(all_steps.any())
