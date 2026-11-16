import unittest
import torch
import numpy as np
from supermimic_research.mechanics.pen_reward_chain import controller,observable_proxy


class PenRewardChainTests(unittest.TestCase):
    def context(self):
        return dict(EMA=torch.full((20,),.1),reference_joints=torch.full((20,),.2),encoder_bias=torch.full((20,),.03),
            lower=torch.full((20,),-1.),upper=torch.full((20,),1.),reference_pose=torch.tensor([0.,0.,0.,1.,0.,0.,0.]),
            target_world=torch.tensor([0.,0.,.3,0.,0.,0.],dtype=torch.float64),target_valid=True)

    def test_raw_scale_and_EMA_are_quarter_gain_and_bias_correct(self):
        raw=torch.full((1,20),.4,requires_grad=True);PD,ema,applied=controller(raw,self.context())
        torch.testing.assert_close(PD,torch.full_like(PD,.32));torch.testing.assert_close(ema,torch.full_like(ema,.15))
        grad=torch.autograd.grad(PD.sum(),raw)[0];torch.testing.assert_close(grad,torch.full_like(raw,.25))

    def test_outer_clamp_and_target_clamp_zero_the_corresponding_derivative(self):
        raw=torch.full((1,20),2.,requires_grad=True);PD,_,_=controller(raw,self.context())
        self.assertTrue(torch.equal(torch.autograd.grad(PD.sum(),raw)[0],torch.zeros_like(raw)))
        c=self.context();c['upper'].fill_(.1);raw=torch.zeros((1,20),requires_grad=True);PD,_,_=controller(raw,c)
        self.assertTrue(torch.equal(torch.autograd.grad(PD.sum(),raw)[0],torch.zeros_like(raw)))

    def test_raw_fd_amplitude_is_four_times_actual_PD_epsilon(self):
        c=self.context();raw=torch.zeros((1,20));direction=torch.eye(20)[:1];e=.005
        plus=controller(raw+4*e*direction,c)[0];minus=controller(raw-4*e*direction,c)[0]
        torch.testing.assert_close(plus-minus,2*e*direction,atol=4e-8,rtol=0)

    def test_observable_proxy_has_no_Fn6_or_window_geometry_derivative(self):
        x=torch.zeros((1,102),dtype=torch.float64);x[:,43]=x[:,47]=1;x[:,-1]=.0025
        y=torch.zeros((1,90),dtype=torch.float64,requires_grad=True);raw=torch.zeros((1,20),dtype=torch.float64,requires_grad=True)
        c=self.context();c['reference_joints'].zero_();cfg=dict(COM_scale_m=.005,angle_scale_rad=.08726646,hand_scale_rad=.02,
            COM_weight=1.,angle_weight=1.,hand_weight=.1,action_weight=.001)
        parts=observable_proxy(x,y,raw,c,cfg,dict(force_scale_N=.3,torque_scale_Nm=.003))
        grad=torch.autograd.grad(parts['observable_support'].sum(),y)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertTrue(torch.equal(grad[:,84:],torch.zeros_like(grad[:,84:])))
        self.assertNotIn('geometry_penalty',parts);self.assertNotIn('full_objective',parts)
