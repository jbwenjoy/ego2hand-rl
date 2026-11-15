import unittest
import numpy as np
import torch
from supermimic_research.runtime.pen_native_planning_math import candidate_offsets,candidate_targets,score_candidates


class PlanningMathTests(unittest.TestCase):
    def test_fixed_candidate_budget_and_original_ema_clamp(self):
        a=candidate_offsets();self.assertEqual(a.shape,(368,57,20));np.testing.assert_array_equal(a,candidate_offsets())
        self.assertLessEqual(abs(a).max(),.08);np.testing.assert_array_equal(a[:,0],0)
        raw=torch.tensor([[1.,-1.]]);processed=torch.tensor([[.2,-.2]])
        target,ema,intended=candidate_targets(raw,processed,torch.zeros_like(raw),torch.zeros_like(raw),torch.full_like(raw,-.1),torch.full_like(raw,.1))
        torch.testing.assert_close(ema,torch.tensor([[.35,-.35]]));torch.testing.assert_close(target,torch.tensor([[.1,-.1]]))
        torch.testing.assert_close(intended,ema)

    def test_support_changes_ranking_and_constraints_take_precedence(self):
        cfg=dict(COM_scale_m=.005,shaft_scale_deg=5.,hand_scale_rad=.02,residual_scale_rad=.02,hand_weight=.01,residual_weight=.01,
            force_scale_N=.1,torque_scale_Nm=.005,max_penetration_m=.002,max_self_penetration_m=.0002,max_joint_limit_excursion_rad=.001,max_object_nonhand_normal_force_N=.1)
        q=torch.zeros(2,1,27);q[:,:,3]=1;q[1,0,0]=.001
        v=torch.zeros(2,1,26);ctrl=torch.zeros(2,1,20);w=torch.zeros(2,1,6);w[1,0,2]=.3
        pen=torch.zeros(2,3);other=torch.zeros(2,1);overflow=torch.zeros(2,1,dtype=torch.int32);ema=torch.zeros(2,20)
        args=(q,v,ctrl,w,pen,other,overflow,ema,torch.arange(7,27),0,torch.zeros(3),torch.tensor([1.,0,0]),torch.zeros(20),torch.tensor([0.,0,.3,0,0,0]),True)
        self.assertEqual(int(score_candidates(*args,0.,cfg)['index']),0)
        self.assertEqual(int(score_candidates(*args,.1,cfg)['index']),1)
        pen[1,1]=.0003
        self.assertEqual(int(score_candidates(*args,.1,cfg)['index']),0)
        pen[0,0]=.003
        result=score_candidates(*args,.1,cfg);self.assertFalse(bool(result['feasible'].any()))
        self.assertTrue(torch.isfinite(result['violation']).all())
        pen.zero_();other[0,0]=float('nan')
        result=score_candidates(*args,.1,cfg)
        self.assertFalse(bool(result['finite'][0]));self.assertEqual(int(result['index']),1)
