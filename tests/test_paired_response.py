import copy,sys,unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.intervention_pairs import intervention_pairs
from supermimic_research.mechanics.residual_control import residual_ema
from supermimic_research.learning.paired_response import secant_loss,paired_update,paired_cache
from supermimic_research.learning.compound_response import fit_scales,CompoundResponseWM
from supermimic_research.learning.world_model import raw_tree_equal


class PairedResponseTests(unittest.TestCase):
    def test_pair_seed_boundary_and_full_native_state(self):
        raw=dict(seed=np.array([200,200,201,201]),dt=np.full(4,.0003125),k=np.full(4,2),joint=np.zeros(4,int),amplitude=np.full(4,.2),sign=np.array([-1,1,-1,1]),x=np.zeros((4,102)),source_native_state=np.zeros((4,9)))
        raw['x'][:,81]=[-.005,.005,-.005,.005];pairs=intervention_pairs(raw);np.testing.assert_allclose(pairs[:,:2],[[0,1],[2,3]])
        raw['source_native_state'][1,0]=1
        with self.assertRaises(ValueError):intervention_pairs(raw)

    def test_actual_control_denominator_handles_asymmetric_EMA(self):
        lo=residual_ema(np.array([-.2]),np.zeros(1),np.array([.01]),np.array([.05]));hi=residual_ema(np.array([.2]),np.zeros(1),np.array([.01]),np.array([.05]))
        du=torch.tensor(hi-lo,dtype=torch.float32);slope=torch.arange(87,dtype=torch.float32).reshape(1,87)/100
        target_lo=torch.tensor(lo,dtype=torch.float32).reshape(1,1)*slope;target_hi=torch.tensor(hi,dtype=torch.float32).reshape(1,1)*slope
        pred_lo=target_lo.detach().clone().requires_grad_();pred_hi=target_hi.detach().clone().requires_grad_();loss=secant_loss(pred_lo,pred_hi,target_lo,target_hi,du)
        self.assertEqual(float(loss.detach()),0.);loss.backward();self.assertTrue(torch.isfinite(pred_hi.grad).all())
        # Ignoring the asymmetric actual-u denominator yields the wrong slope.
        self.assertNotAlmostEqual(float(du),.2*.05*2*.5)

    def test_zero_coefficient_matches_absolute_gradient_with_identical_sampler_budget(self):
        rng=np.random.default_rng(19);x=rng.normal(size=(8,102));y=rng.normal(size=(8,87));y[:,-3:]=abs(y[:,-3:]);raw=dict(x=x,y=y,pairs=np.array([[0,1,0,.01],[2,3,1,.02]]))
        a=CompoundResponseWM(fit_scales(x,y),width=12);b=copy.deepcopy(a);oa=torch.optim.AdamW(a.parameters(),lr=.001);ob=torch.optim.AdamW(b.parameters(),lr=.001)
        c=paired_cache(raw,a,'cpu');absolute={k:torch.tensor(raw[k],dtype=torch.float32) for k in ['x','y']};cfg=dict(batch_size=4,pair_batch_size=2,gradient_clip_norm=10.,response_reference_rad=.025)
        ga=torch.Generator().manual_seed(5);gb=torch.Generator().manual_seed(5)
        _,idx,pidx,_=paired_update(a,oa,ga,absolute,c,cfg,coefficient=0.)
        expected=torch.randint(8,(4,),generator=gb);torch.randint(2,(2,),generator=gb);ob.zero_grad(set_to_none=True);loss=b.loss(absolute['x'][expected],absolute['y'][expected]);loss.backward();torch.nn.utils.clip_grad_norm_(b.parameters(),10.);ob.step()
        self.assertTrue(torch.equal(idx,expected));self.assertTrue(torch.equal(ga.get_state(),gb.get_state()));self.assertTrue(raw_tree_equal(a.state_dict(),b.state_dict()))


if __name__=='__main__':unittest.main()
