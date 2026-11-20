import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from supermimic_research.mechanics.pen_capacity_forecast import (
    CapacityHead, SCHEMA, capacity_features, capacity_mismatch, load_capacity_head)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from fit_pen_capacity_forecast import episode_rows, evaluate


class TestPenCapacityForecast(unittest.TestCase):
    def test_no_current_label_leakage_and_correct_phase_origin(self):
        x=np.zeros((5,102));x[:,0]=2;x[:,81]=7
        valid=np.array([True,True,False,True,True])
        ticks=(np.arange(5)[:,None,None]*10+np.arange(8)[None,:,None])*np.ones((1,1,72))
        target=np.repeat(np.arange(6)[:,None],72,axis=1)
        elig=np.array([False,True,False,True,False])
        data=episode_rows(x,valid,ticks,np.arange(1,6),target,elig)
        self.assertEqual(data['phase'].tolist(),[1,2,4,5])
        self.assertEqual(data['present'].tolist(),[False,True,False,True])
        self.assertEqual(data['x'][:,55].tolist(),[5]*4)
        self.assertEqual(data['y'][:,0].tolist(),[3.5,13.5,33.5,43.5])
        self.assertEqual(data['previous'][:,0].tolist(),[0,3.5,0,33.5])
        self.assertEqual(data['target'][:,0].tolist(),[0,1,3,4])
        np.testing.assert_array_equal(data['x'][:,-1].numpy(),data['present'].numpy())
        m=evaluate(data['y'],data,np.array(['switch','slow','slow','turn','switch']))
        self.assertEqual(m['mismatch']['eligible']['n'],2)
        self.assertEqual(m['mismatch']['eligible']['mae'],0)
        self.assertAlmostEqual(m['mismatch']['eligible']['measured'],(12.5+30.5)/2)

    def test_previous_zero_has_presence_flag_but_missing_future_is_ignored(self):
        x=torch.zeros((2,55));pd=torch.zeros((2,20));prev=torch.zeros((2,72))
        prev[1,:]=123.
        f=capacity_features(x,pd,use_history=True,previous_capacity=prev,history_present=torch.tensor([True,False]))
        torch.testing.assert_close(f[:,:-1],torch.zeros_like(f[:,:-1]))
        self.assertEqual(f[:,-1].tolist(),[1.,0.])
        cold=capacity_features(x,pd,use_history=True)
        torch.testing.assert_close(cold[1],f[1])

    def test_checkpoint_reconstruction_and_action_gradient(self):
        torch.manual_seed(7411)
        state,pd,prev=torch.randn(8,55),torch.randn(8,20),torch.rand(8,72)
        for hist in (False,True):
            model=CapacityHead(hist)
            x=capacity_features(state,pd,use_history=hist,previous_capacity=prev if hist else None)
            model.set_training_stats(x,torch.rand(8,72))
            restored=load_capacity_head(dict(schema=SCHEMA,use_history=hist,state_dict=model.state_dict()))
            h=prev if hist else None
            torch.testing.assert_close(restored(state,pd,h),model(state,pd,h),atol=0,rtol=0)
            restored=restored.double().requires_grad_(False)
            u=pd[0].double().requires_grad_(True);s=state[0].double()
            h=prev[0].double() if hist else None
            target=torch.full((72,),2.,dtype=torch.float64)
            fn=lambda a:capacity_mismatch(restored(s,a,h),target)
            self.assertTrue(torch.autograd.gradcheck(fn,(u,),fast_mode=True))
            derivative,=torch.autograd.grad(fn(u),u)
            self.assertGreater(float(derivative.norm()),1e-7)
            self.assertTrue(bool((restored(s,u,h)>=0).all()))


if __name__=='__main__':
    unittest.main()
