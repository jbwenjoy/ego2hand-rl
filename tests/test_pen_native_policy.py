import unittest
import numpy as np
import torch
from supermimic_research.runtime.pen_native_policy import orientation_errors,tracking_reward


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

    def test_action_cost_uses_applied_action_and_reward_is_unscaled(self):
        cfg=dict(COM_scale_m=.005,angle_scale_rad=np.pi/36,hand_scale_rad=.02,
            COM_weight=1.,angle_weight=1.,hand_weight=.1,action_weight=.001)
        ref=torch.tensor([[0.,0,0,1,0,0,0]])
        r,parts=tracking_reward(ref[:,:3],ref[:,3:],torch.zeros(1,20),ref,
            torch.zeros(1,20),torch.full((1,20),2.).clamp(-1,1),cfg)
        self.assertAlmostEqual(float(r[0]),2.099,places=6)
        self.assertEqual(float(parts['action'][0]),1.)


if __name__=='__main__':unittest.main()
