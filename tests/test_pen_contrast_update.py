import unittest

import numpy as np
import torch

from supermimic_research.learning.pen_contrast_update import finite_contrast_loss,update_contrast,restore_contrast_arm
from supermimic_research.learning.pen_contrast_fit import payload
from supermimic_research.learning.pen_response import PenResponseWM
from supermimic_research.learning.world_model import raw_tree_equal


class ContrastUpdateTests(unittest.TestCase):
    def test_analytic_response_units_blocks_and_parameter_gradient(self):
        epsilon=torch.tensor([.005,.02],dtype=torch.float64)
        weight=torch.tensor(1.,dtype=torch.float64,requires_grad=True)
        direction=torch.ones((2,90),dtype=torch.float64)
        plus=weight*epsilon[:,None]*direction;minus=-plus
        target=torch.full((2,90),10000.,dtype=torch.float64)
        target[:,40:52]=3;target[:,78:84]=2
        scales=torch.ones(90,dtype=torch.float64);scales[40:52]=2;scales[78:84]=3
        loss=finite_contrast_loss(plus,minus,target,epsilon,scales)
        self.assertAlmostEqual(float(loss.detach()),1+1/9)
        self.assertAlmostEqual(float(torch.autograd.grad(loss,weight)[0]),-1-2/9)

    def test_separate_sampling_and_full_next_step_restore(self):
        torch.manual_seed(29)
        cfg=dict(width=16,learning_rate=.0003,weight_decay=.00001,batch_size=4,pair_batch_size=2,
            gradient_clip_norm=1.,auxiliary_weight=.1)
        scales=dict(x_mean=np.zeros(102),x_scale=np.ones(102),y_mean=np.zeros(90),y_scale=np.ones(90))
        heads=dict(head_mean=np.zeros(90),head_scale=np.ones(90))
        physics=dict(mass=.03,inertia_BODY=np.diag([1e-6,2e-4,2e-4]),gravity=[0,0,-9.81])
        model=PenResponseWM(scales,heads,physics,width=16,coupled=False)
        arm=dict(model=model,optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],fused=False),
            sampler=torch.Generator().manual_seed(7),pair_sampler=torch.Generator().manual_seed(9),mode='DIRECT',contrast_weight=.1)
        x=torch.zeros((6,102));x[:,43]=1;x[:,47]=1;x[:,-1]=.005;x[:,81]=torch.linspace(-.03,.03,6)
        y=torch.zeros((6,90),dtype=torch.float64);xp=x[:3].clone();xm=x[:3].clone();xp[:,81]+=.005;xm[:,81]-=.005
        cache=dict(x=x,y=y,DIRECT=y,x_plus=xp,x_minus=xm,epsilon=torch.full((3,),.005,dtype=torch.float64),finite_contrast=torch.ones((3,90),dtype=torch.float64))
        expected=torch.randint(len(x),(4,),generator=torch.Generator().manual_seed(7))
        _,point_ids,_,gradient=update_contrast(arm,cache,cfg,capture_gradient=True)
        self.assertTrue(torch.equal(point_ids,expected))
        saved=payload(arm,cfg,scales,heads,physics,1,'cpu',gradient)
        clone=restore_contrast_arm(saved,'cpu')
        self.assertTrue(raw_tree_equal(arm['sampler'].get_state(),clone['sampler'].get_state()))
        self.assertTrue(raw_tree_equal(arm['pair_sampler'].get_state(),clone['pair_sampler'].get_state()))
        # Serialize rather than share optimizer moment tensors with the live arm.
        import io
        buffer=io.BytesIO();torch.save(saved,buffer);buffer.seek(0)
        clone=restore_contrast_arm(torch.load(buffer,weights_only=False),'cpu')
        a=update_contrast(arm,cache,cfg,capture_gradient=True);b=update_contrast(clone,cache,cfg,capture_gradient=True)
        self.assertTrue(raw_tree_equal(a,b))
        self.assertTrue(raw_tree_equal(arm['model'].state_dict(),clone['model'].state_dict()))
        self.assertTrue(raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()))
