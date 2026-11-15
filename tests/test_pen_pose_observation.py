import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from supermimic_research.runtime.pen_pose_observation import pose9_world_in_wrist,pack_full_pose_policy,require_full_pose_contract,OBSERVATION_SCHEMA,ACTION_CONTRACT


class FullPoseObservationTests(unittest.TestCase):
    def test_full_axial_rotation_and_matrix_columns_are_visible(self):
        pose=torch.tensor([[0.,0,0,1,0,0,0],[0,0,0,np.sqrt(.5),np.sqrt(.5),0,0]],dtype=torch.float64)
        mount=torch.tensor([0.,0,0,1,0,0,0],dtype=torch.float64)
        f=pose9_world_in_wrist(pose,mount)
        torch.testing.assert_close(f[0,3:],torch.tensor([1.,0,0,0,1,0],dtype=f.dtype))
        torch.testing.assert_close(f[1,3:],torch.tensor([1.,0,0,0,0,1],dtype=f.dtype),atol=1e-14,rtol=0)
        self.assertGreater(float(torch.linalg.vector_norm(f[1]-f[0])),1.)

    def test_common_world_transform_and_quaternion_sign_scale_invariance(self):
        W=Rotation.from_euler('zy',[.3,.2]);P=Rotation.from_euler('yx',[.8,.4]);G=Rotation.from_euler('xz',[.9,.7])
        wp=np.array([1.,2,3]);pp=np.array([1.1,2.2,3.3]);gp=np.array([4.,5,6])
        pose=torch.tensor(np.r_[pp,P.as_quat()[[3,0,1,2]]]);mount=torch.tensor(np.r_[wp,W.as_quat()[[3,0,1,2]]])
        shifted=torch.tensor(np.r_[G.apply(pp)+gp,(G*P).as_quat()[[3,0,1,2]]])
        shifted_mount=torch.tensor(np.r_[G.apply(wp)+gp,(G*W).as_quat()[[3,0,1,2]]])
        f=pose9_world_in_wrist(pose,mount);torch.testing.assert_close(f,pose9_world_in_wrist(shifted,shifted_mount),atol=1e-14,rtol=0)
        pose[3:]*=-3;mount[3:]*=2
        torch.testing.assert_close(f,pose9_world_in_wrist(pose,mount),atol=1e-14,rtol=0)

    def test_invalid_orientation_does_not_become_identity(self):
        pose=torch.zeros(7);mount=torch.tensor([0.,0,0,1,0,0,0])
        self.assertFalse(bool(torch.isfinite(pose9_world_in_wrist(pose,mount)).all()))

    def test_explicit_feature_layout_and_checkpoint_contract(self):
        pieces=[torch.zeros(2,n) for n in [20,9,63,20,20,9,20]]
        self.assertEqual(pack_full_pose_policy(*pieces).shape,(2,161))
        pieces[1]=torch.zeros(2,6)
        with self.assertRaises(ValueError):pack_full_pose_policy(*pieces)
        metadata=dict(observation_schema=OBSERVATION_SCHEMA,policy_input_dim=161,action_dim=20,action_contract=ACTION_CONTRACT)
        require_full_pose_contract(metadata)
        for change in [dict(policy_input_dim=155),dict(observation_schema='legacy_cube'),dict(action_contract='unbounded')]:
            with self.assertRaises(ValueError):require_full_pose_contract({**metadata,**change})
