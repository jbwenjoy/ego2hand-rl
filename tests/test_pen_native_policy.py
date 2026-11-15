import unittest
from types import SimpleNamespace
import numpy as np
import torch
from supermimic_research.runtime.pen_native_policy import orientation_errors,tracking_reward,NativePenPolicyAdapter


class NativePenPolicyTests(unittest.TestCase):
    def test_full_rotation_reward_distinguishes_invisible_shaft_roll(self):
        q=torch.tensor([[1.,0,0,0],[np.cos(np.pi/18),np.sin(np.pi/18),0,0]],dtype=torch.float32)
        angle,shaft=orientation_errors(q,torch.tensor([[1.,0,0,0]]).expand(2,-1))
        self.assertAlmostEqual(float(angle[1]),np.pi/9,places=6)
        torch.testing.assert_close(shaft,torch.zeros_like(shaft))
        cfg=dict(COM_scale_m=.005,angle_scale_rad=np.pi/36,hand_scale_rad=.02,
            COM_weight=1.,angle_weight=1.,hand_weight=.1,action_weight=.001)
        ref=torch.tensor([[0.,0,0,1,0,0,0]]).expand(2,-1)
        r,_=tracking_reward(torch.zeros(2,3),q,torch.zeros(2,20),ref,torch.zeros(2,20),torch.zeros(2,20),cfg)
        self.assertGreater(float(r[0]-r[1]),.99)

    def test_quaternion_sign_scale_and_invalid_rotation(self):
        q=torch.tensor([[np.cos(.2),np.sin(.2),0.,0.]],dtype=torch.float32)
        r=torch.tensor([[1.,0,0,0]])
        a,b=orientation_errors(q,r);c,d=orientation_errors(-3*q,4*r)
        torch.testing.assert_close(a,c);torch.testing.assert_close(b,d)
        a,_=orientation_errors(torch.zeros_like(q),r)
        self.assertFalse(bool(torch.isfinite(a).all()))

    def test_small_float32_rotation_survives_reward_boundary(self):
        q=torch.tensor([[1.,5e-6,0,0]],dtype=torch.float32)
        a,_=orientation_errors(q,torch.tensor([[1.,0,0,0]]))
        self.assertAlmostEqual(float(a[0]),1e-5,places=10)

    def test_action_cost_uses_applied_action_and_reward_is_unscaled(self):
        cfg=dict(COM_scale_m=.005,angle_scale_rad=np.pi/36,hand_scale_rad=.02,
            COM_weight=1.,angle_weight=1.,hand_weight=.1,action_weight=.001)
        ref=torch.tensor([[0.,0,0,1,0,0,0]])
        r,parts=tracking_reward(ref[:,:3],ref[:,3:],torch.zeros(1,20),ref,
            torch.zeros(1,20),torch.full((1,20),2.).clamp(-1,1),cfg)
        self.assertAlmostEqual(float(r[0]),2.099,places=6)
        self.assertEqual(float(parts['action'][0]),1.)

    def test_step_reuses_native_observation_without_second_pose_encoding(self):
        a=NativePenPolicyAdapter.__new__(NativePenPolicyAdapter)
        pose=torch.tensor([[0.,0,0,1,0,0,0]])
        policy=torch.zeros(1,161);phase=torch.tensor([1])
        a.num_envs=1;a.pending=torch.zeros(1,dtype=torch.bool);a.control_transitions=0;a.physics_integrations=0
        a.cfg={'reward':dict(COM_scale_m=.005,angle_scale_rad=np.pi/36,hand_scale_rad=.02,
            COM_weight=1.,angle_weight=1.,hand_weight=.1,action_weight=.001)}
        a.motion=SimpleNamespace(time_steps=phase,joint_pos=torch.zeros(1,20),_frames={'joint_pos':torch.zeros(369,20)})
        def reencode():raise AssertionError('Policy observation was already computed by native env.step')
        a.features=SimpleNamespace(reference_pose_world=lambda:pose,pose=lambda:pose,qids=torch.arange(7,27),
            policy=reencode,critic=lambda p:torch.cat((p,torch.zeros(1,27)),-1))
        def step(raw):
            phase.add_(1)
            return {'policy':policy},None,None,None,None
        a.env=SimpleNamespace(step=step,cfg=SimpleNamespace(decimation=8),sim=SimpleNamespace(data=SimpleNamespace(
            qpos=torch.cat((pose,torch.zeros(1,20)),-1),qvel=torch.zeros(1,26))))
        obs,_,term,trunc,info=a.step(torch.full((1,20),1.5))
        self.assertIs(obs['policy'],policy)
        self.assertEqual(obs['critic'].shape,(1,188));self.assertEqual(a.physics_integrations,8)
        self.assertFalse(bool((term|trunc).any()))
        self.assertEqual(int(info['reference_phase'][0]),1)
        torch.testing.assert_close(info['applied_raw'],torch.ones(1,20))


if __name__=='__main__':unittest.main()
