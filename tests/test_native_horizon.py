import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.native_tracking import NativeTrackingPlanner,TrackingResidual


class NativeHorizonTests(unittest.TestCase):
    def test_six_step_lookahead_and_end_truncation_keep_legal_clock(self):
        cfg=dict(active_joints=list(range(12)),horizon=6,FD_action=.1,ridge=.1,max_action_step=.5,line_search=[1,.5,.25,.1],COM_scale_m=.005,angle_scale_rad=.1,velocity_scale_m_s=.05,omega_scale_rad_s=.5,joint_weight=.02,contact_weight=.1,other_force_weight=.05,action_weight=.002,smooth_weight=.01)
        reference=dict(COM_world_m=np.zeros((101,3)),T_world_from_BODY=np.tile(np.eye(4),(101,1,1)),COM_velocity_world_m_s=np.zeros((101,3)),angular_velocity_world_rad_s=np.zeros((101,3)))
        calls=[]
        class Oracle:
            def rollout(self,snapshot,action,horizon):
                calls.append((snapshot['k'],horizon));s=np.zeros(55);s[43:49]=[1,0,0,0,1,0];contact=dict(normal_by_finger_N=np.ones(5)*.1,other_normal_N=0.,table_normal_N=0.,max_native_penetration_m=0.,max_self_penetration_m=0.)
                return dict(states55=np.tile(s,(horizon,1)),contacts=[contact]*horizon,native_states=np.zeros((horizon,1)))
        oracle=Oracle();cost=TrackingResidual(reference,np.zeros(20),np.tile([-2,2],(20,1)),cfg,'REFERENCE');planner=NativeTrackingPlanner(oracle,cost,cfg)
        for k,expected in [(0,6),(98,2),(99,1)]:
            action,record=planner.choose(dict(k=k));self.assertEqual(record['horizon'],expected);self.assertEqual(record['oracle_calls'],30);self.assertEqual(record['proposal_residuals'].shape,(30,40*expected+40));np.testing.assert_array_equal(action,0)
        self.assertTrue(all(k+h<=100 for k,h in calls))


if __name__=='__main__':unittest.main()
