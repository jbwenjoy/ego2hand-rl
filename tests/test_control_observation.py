import copy,sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.control_observation import actor_observation


class ControlObservationTests(unittest.TestCase):
    def setUp(self):
        self.state=np.zeros(55);self.state[43:49]=[1,0,0,0,1,0];T=np.tile(np.eye(4),(101,1,1));self.ref=dict(T_world_from_BODY=T,COM_world_m=np.tile([.01,0,0],(101,1)),COM_velocity_world_m_s=np.zeros((101,3)),angular_velocity_world_rad_s=np.zeros((101,3)))
    def test_end_clock_masks_physical_EMA_and_relative_target(self):
        ema=np.ones(20)*.02;prev=np.ones(20)*.5;x=actor_observation(self.state,ema,prev,99,self.ref)
        self.assertEqual(x.shape,(144,));np.testing.assert_array_equal(x[55:75],ema);np.testing.assert_array_equal(x[75:95],prev);self.assertEqual(x[95],.99);np.testing.assert_array_equal(x[-3:],[1,0,0]);np.testing.assert_allclose(x[96:99],[.01,0,0])
    def test_relative_future_components_invariant_to_world_rigid_transform(self):
        a=actor_observation(self.state,np.zeros(20),np.zeros(20),7,self.ref);Q=Rotation.from_rotvec([.2,-.4,.3]).as_matrix();t=np.array([.1,.2,.3]);s=self.state.copy();s[40:43]=Q@s[40:43]+t;s[43:49]=np.r_[Q[:,0],Q[:,1]]
        ref=copy.deepcopy(self.ref);ref['COM_world_m']=ref['COM_world_m']@Q.T+t;ref['T_world_from_BODY'][:,:3,:3]=Q
        b=actor_observation(s,np.zeros(20),np.zeros(20),7,ref);np.testing.assert_allclose(a[96:],b[96:],atol=1e-15)


if __name__=='__main__':unittest.main()
