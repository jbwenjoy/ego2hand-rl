import unittest
import numpy as np
import torch
from supermimic_research.learning.backend_reward_gradient import raw_reward_gradient,compare_reward_responses


class RewardGradientTests(unittest.TestCase):
    def test_chain_includes_quarter_PD_gain_and_direct_RAW_action_penalty(self):
        class Model:
            def prediction(self,x):
                y=x.new_zeros((len(x),90));y[:,0]=.2*x[:,81];return y
        x=torch.zeros((1,102),dtype=torch.float64);x[:,0]=.003;x[:,43]=x[:,47]=1;x[:,81]=.004;x[:,-1]=.0025
        raw=torch.full((1,20),.2,dtype=torch.float64)
        context=dict(EMA=torch.zeros((1,20),dtype=torch.float64),reference_joints=torch.zeros((1,20),dtype=torch.float64),
            encoder_bias=torch.zeros((1,20),dtype=torch.float64),lower=torch.full((1,20),-1.),upper=torch.full((1,20),1.),
            reference_pose=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]],dtype=torch.float64),target_world=torch.zeros((1,6),dtype=torch.float64),target_valid=torch.tensor([True]))
        cfg=dict(COM_scale_m=.005,angle_scale_rad=.08726646,hand_scale_rad=.02,COM_weight=1.,angle_weight=1.,hand_weight=.1,action_weight=.001)
        r=raw_reward_gradient(Model(),x,raw,context,cfg,dict(force_scale_N=.3,torque_scale_Nm=.003))
        q=.003+.2*.004;expected=-2*.1*q/(20*.02**2)*np.exp(-q*q/(20*.02**2))*.2*.25-.002*.2/20
        self.assertAlmostEqual(float(r['RAW'][0,0,0]),expected,places=12)
        self.assertAlmostEqual(float(r['RAW_direct_partial'][0,0,0]),-.002*.2/20,places=14)
        torch.testing.assert_close(r['RAW'],r['PD_partial']*.25+r['RAW_direct_partial'])

    def example(self):
        metadata=[dict(valid=True)]*8;groups=[];values=np.zeros((8,10));model=np.zeros((8,2));gradient={}
        for block,e in enumerate([.002,.005]):
            p=[4*block,4*block+1];m=[4*block+2,4*block+3]
            for ids,sign in [(p,1),(m,-1)]:
                values[ids,4]=sign*4*e;values[ids,5]=sign*4*e;values[ids,7]=sign*8*e;values[ids,8]=sign*8*e
                model[ids]=sign*4*e
            groups.append(dict(source_name='s',direction=0,epsilon=e,rows={'1':p,'-1':m},complete=True,qualified=True))
        groups.append(dict(source_name='missing',direction=1,epsilon=.002,rows={'1':[],'-1':[]},complete=False,qualified=False))
        gradient['s',0]=np.ones(2)
        return dict(metadata=metadata,response_groups=groups),values,model,gradient,dict(s=0)

    def test_perfect_observable_response_does_not_become_perfect_full_reward(self):
        rows,a=compare_reward_responses(*self.example())
        self.assertIsNone(rows[-1]['metrics']);self.assertEqual(len(rows),3)
        self.assertEqual(a['OBSERVABLE_NATIVE']['observable_residual']['ALL_PHYSICAL_VALID']['AD_relative_L2'],0)
        self.assertEqual(a['FULL_NATIVE']['observable_residual']['ALL_PHYSICAL_VALID']['AD_relative_L2'],.5)

    def test_missing_AD_keeps_all_finite_response_and_original_masks(self):
        args=list(self.example());args[3]={}
        _,a=compare_reward_responses(*args);v=a['FULL_NATIVE']['observable_support']['ALL_PHYSICAL_VALID']
        self.assertEqual((v['pairs'],v['AD_pairs'],v['missing_AD_pairs']),(2,0,2));self.assertIsNone(v['AD_relative_L2'])
        args=self.example();args[0]['metadata'][0]=dict(valid=False)
        with self.assertRaises(ValueError):compare_reward_responses(*args)

    def test_zero_native_response_has_no_invented_relative_ratio(self):
        args=self.example();args[1][:]=0
        _,a=compare_reward_responses(*args)
        for target in a.values():
            v=target['observable_residual']['ALL_PHYSICAL_VALID'];self.assertIsNone(v['AD_relative_L2']);self.assertEqual(v['candidate_local_AD_pairs'],0)
