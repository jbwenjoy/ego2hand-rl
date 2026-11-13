import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.native_tracking import TrackingResidual,NativeTrackingPlanner


class NativeTrackingTests(unittest.TestCase):
    def setUp(self):
        self.config=dict(active_joints=list(range(12)),horizon=3,FD_action=.1,ridge=.1,max_action_step=.5,line_search=[1,.5,.25,.1],COM_scale_m=.005,angle_scale_rad=.08726646,velocity_scale_m_s=.05,omega_scale_rad_s=.5,joint_weight=.02,contact_weight=.1,other_force_weight=.05,action_weight=.002,smooth_weight=.01)
        T=np.tile(np.eye(4),(101,1,1));self.reference=dict(COM_world_m=np.tile([.002,0,0],(101,1)),T_world_from_BODY=T,COM_velocity_world_m_s=np.zeros((101,3)),angular_velocity_world_rad_s=np.zeros((101,3)))
        self.contact=dict(normal_by_finger_N=np.ones(5)*.1,other_normal_N=0.,table_normal_N=0.,max_native_penetration_m=0.,max_self_penetration_m=0.)
    def test_perfect_tracking_contact_and_zero_action_has_zero_residual(self):
        s=np.zeros(55);s[40]=.002;s[43:49]=[1,0,0,0,1,0];p=dict(states55=np.tile(s,(3,1)),contacts=[self.contact]*3)
        objective=TrackingResidual(self.reference,np.zeros(20),np.tile([-2,2],(20,1)),self.config,'REFERENCE')
        np.testing.assert_array_equal(objective.residual(p,0,np.zeros(20),np.zeros(20)),0)
    def test_box_native_search_improves_known_local_problem_and_restores(self):
        contact=self.contact
        class Oracle:
            def rollout(self,snapshot,action,horizon):
                s=np.zeros(55);s[40]=.01*action[0];s[43:49]=[1,0,0,0,1,0]
                return dict(states55=np.tile(s,(horizon,1)),contacts=[contact]*horizon,native_states=np.zeros((horizon,1)))
        objective=TrackingResidual(self.reference,np.zeros(20),np.tile([-2,2],(20,1)),self.config,'REFERENCE');planner=NativeTrackingPlanner(Oracle(),objective,self.config);state=planner.state_dict()
        action,record=planner.choose(dict(k=0));self.assertLess(record['selected_cost'],record['incumbent_cost']);self.assertGreater(action[0],0);self.assertTrue(np.all(abs(action)<=1));np.testing.assert_array_equal(action[12:],0)
        planner.restore(state);repeat,second=planner.choose(dict(k=0));np.testing.assert_array_equal(action,repeat);np.testing.assert_array_equal(record['proposal_costs'],second['proposal_costs'])


if __name__=='__main__':unittest.main()
