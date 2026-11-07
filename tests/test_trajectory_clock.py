"""No epoch precision loss, silent time warp, or BODY/COM velocity confusion."""
from pathlib import Path
import sys,unittest
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.ego.trajectory_clock import resample_body_poses


class TrajectoryClockTests(unittest.TestCase):
    def test_nonuniform_source_clock_and_offset_COM_velocity(self):
        offset=np.array([0,17_000_000,50_000_000,63_000_000],dtype=np.int64);T=np.tile(np.eye(4),(4,1,1));T[:,0,3]=offset*1e-9
        T[:,:3,:3]=Rotation.from_rotvec(np.c_[np.zeros(4),np.zeros(4),offset*1e-9*.6]).as_matrix();COM=np.array([.02,.01,0.])
        result=resample_body_poses(offset+54_050_174_863_789,T,COM);self.assertEqual(result['metadata']['excluded_tail_ns'],3_000_000)
        np.testing.assert_allclose(result['T_world_from_BODY'][:,0,3],[0,.02,.04,.06],atol=1e-15)
        np.testing.assert_allclose(result['angular_velocity_world_rad_s'],np.tile([0,0,.6],(4,1)),atol=1e-14)
        arm=result['COM_world_m']-result['T_world_from_BODY'][:,:3,3]
        np.testing.assert_allclose(result['COM_velocity_world_m_s'],[1,0,0]+np.cross([0,0,.6],arm),atol=1e-14)

    def test_large_epoch_keeps_nanosecond_grid_exact(self):
        t0=1_700_000_000_000_000_000;offset=np.array([0,7,20],dtype=np.int64);T=np.tile(np.eye(4),(3,1,1));T[:,0,3]=offset*1e-9
        result=resample_body_poses(t0+offset,T,[0,0,0],period_ns=5)
        np.testing.assert_array_equal(result['timestamps_ns']-t0,[0,5,10,15,20]);np.testing.assert_allclose(result['T_world_from_BODY'][:,0,3],np.arange(5)*5e-9,atol=1e-22)

    def test_bad_clock_or_rotation_is_not_repaired(self):
        T=np.tile(np.eye(4),(3,1,1))
        with self.assertRaises(ValueError):resample_body_poses(np.array([0,2,2],dtype=np.int64),T,[0,0,0])
        T[1,0,0]=2
        with self.assertRaises(ValueError):resample_body_poses(np.array([0,1,2],dtype=np.int64),T,[0,0,0])


if __name__=='__main__':unittest.main()
