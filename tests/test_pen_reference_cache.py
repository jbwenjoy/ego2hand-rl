"""Reference cache parity, live suffix, clamped phases and cold-path guards."""
from types import SimpleNamespace
import unittest
import torch
from supermimic_research.runtime.pen_native_policy import NativePenFeatures
from supermimic_research.runtime.pen_reference_cache import FrozenPenReferenceCache


class Motion:
    def __init__(self,origins):
        self.origins=origins;self.time_steps=torch.tensor([0,2]);F=3
        wrist=torch.tensor([[.1,.2,.3,1,0,0,0]]).repeat(F,1)
        obj=torch.tensor([[.02,-.03,.04,1,0,0,0],[.03,.01,.02,.5,.5,.5,.5],[-.01,.02,.03,0,1,0,0]])
        self._frames=dict(joint_pos=torch.arange(F*20,dtype=torch.float32).reshape(F,20)*.01,link_pos=torch.arange(F*63,dtype=torch.float32).reshape(F,21,3)*.001,wrist_pose_w=wrist,object_pose=obj)
    @property
    def frame_ids(self):return self.time_steps.clamp(0,2)
    @property
    def joint_pos(self):return self._frames['joint_pos'][self.frame_ids]
    @property
    def link_pos(self):return self._frames['link_pos'][self.frame_ids]
    @property
    def wrist_pose_w(self):
        pose=self._frames['wrist_pose_w'][self.frame_ids];pose[:,:3]+=self.origins;return pose
    @property
    def object_pose(self):return self._frames['object_pose'][self.frame_ids]


def fixture():
    origins=torch.tensor([[0.,0,0],[1.,2.,3.]])
    qpos=torch.zeros(2,27);qpos[:,:7]=torch.tensor([[.1,.2,.4,1,0,0,0],[1.2,2.3,3.4,.5,.5,.5,.5]])
    qpos[:,7:]=torch.arange(40).reshape(2,20)*.001
    xpos=torch.zeros(2,2,3);xpos[:,0]=origins+torch.tensor([.1,.2,.3]);xquat=torch.zeros(2,2,4);xquat[:,:,0]=1
    data=SimpleNamespace(qpos=qpos,qvel=torch.arange(52).reshape(2,26)*.01,xpos=xpos,xquat=xquat)
    robot=SimpleNamespace(data=SimpleNamespace(encoder_bias=torch.ones(2,20)*.001,default_joint_pos=torch.ones(2,20)*.02,default_joint_vel=torch.ones(2,20)*.03))
    f=NativePenFeatures.__new__(NativePenFeatures);f.env=SimpleNamespace(device='cpu',num_envs=2,scene=SimpleNamespace(env_origins=origins),sim=SimpleNamespace(data=data))
    f.motion=Motion(origins);f.robot=robot;f.body=1;f.palm=0;f.qa=0;f.qids=torch.arange(7,27);f.dofs=torch.arange(6,26);f.local_joints=torch.arange(20)
    f.action=SimpleNamespace(_raw_actions=torch.arange(40).reshape(2,20)*.02)
    return f


class ReferenceCacheTests(unittest.TestCase):
    def test_mixed_phases_origins_and_reference_world_pose_match_original(self):
        f=fixture();c=FrozenPenReferenceCache(f)
        torch.testing.assert_close(c.policy(f),f.policy(),atol=2e-7,rtol=0)
        torch.testing.assert_close(c.reference_pose_world(f.motion.frame_ids),f.reference_pose_world(),atol=1e-7,rtol=0)
        self.assertEqual(c.prefix.shape,(2,3,92));self.assertTrue(all(c.validate(f).values()))

    def test_hold_and_end_clamps_use_the_command_frame_ids(self):
        f=fixture();c=FrozenPenReferenceCache(f)
        for t in [-10,-1,0,1,2,3,1000]:
            f.motion.time_steps.fill_(t);torch.testing.assert_close(c.policy(f),f.policy(),atol=2e-7,rtol=0)

    def test_current_joints_velocity_pose_and_previous_action_stay_live(self):
        f=fixture();c=FrozenPenReferenceCache(f);old=c.policy(f).clone()
        f.env.sim.data.qpos[:,0]+=.07;f.env.sim.data.qpos[:,7:]+=.1;f.env.sim.data.qvel[:,6:]+=.2;f.action._raw_actions+=.3
        actual=c.policy(f);torch.testing.assert_close(actual,f.policy(),atol=2e-7,rtol=0)
        torch.testing.assert_close(actual[:,:92],old[:,:92],atol=0,rtol=0)
        self.assertFalse(torch.equal(actual[:,92:],old[:,92:]))

    def test_mount_origins_reference_and_table_mutations_are_rejected(self):
        for kind in ['mount','origins','reference','table']:
            f=fixture();c=FrozenPenReferenceCache(f)
            if kind=='mount':f.env.sim.data.xpos[0,0,0]+=.01
            if kind=='origins':f.env.scene.env_origins[0,0]+=.01
            if kind=='reference':f.motion._frames['joint_pos'][1,0]+=.01
            if kind=='table':c.prefix[0,0,0]+=.01
            with self.assertRaisesRegex(ValueError,'Frozen reference cache input changed'):c.validate(f)

    def test_invalid_current_quaternion_propagates_the_same_NaNs(self):
        f=fixture();c=FrozenPenReferenceCache(f);f.env.sim.data.qpos[0,3:7]=0
        a=c.policy(f);b=f.policy();torch.testing.assert_close(a,b,atol=2e-7,rtol=0,equal_nan=True)
        self.assertTrue(torch.isnan(a[0,132:141]).any())

    def test_constructor_preserves_source_reference_bytes(self):
        f=fixture();before={k:v.clone() for k,v in f.motion._frames.items()};c=FrozenPenReferenceCache(f)
        for k,v in before.items():self.assertTrue(torch.equal(v,f.motion._frames[k]))
        self.assertGreater(c.metadata()['resident_bytes'],0)


if __name__=='__main__':unittest.main()
