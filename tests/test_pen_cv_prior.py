import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import torch
from supermimic_research.learning.pen_response import (
    PenResponseWM,kinematic_baseline,kinematic_baseline_numpy,training_head_scales,
    INPUT_SCHEMA,OUTPUT_SCHEMA,CV_HEAD_SCHEMA,model_from_arm,pen_families)
from supermimic_research.learning.pen_continuation import validate_parent
from supermimic_research.learning.pen_diagnostics import decode_pen_torch
from supermimic_research.learning.pen_fit_diagnostics import constant_velocity


def fixtures():
    scales=dict(x_mean=np.zeros(102),x_scale=np.ones(102),y_mean=np.zeros(90),y_scale=np.ones(90))
    hs=dict(head_mean=np.zeros(90),head_scale=np.ones(90))
    physics=dict(mass=.03,inertia_BODY=np.diag([1e-6,2e-4,2e-4]),gravity=[0,0,-9.81])
    x=torch.zeros((3,102));x[:,43]=x[:,47]=1;x[:,20:40]=2;x[:,49:55]=torch.tensor([1.,2,3,.1,.2,.3]);x[:,-1]=.005
    return scales,hs,physics,x


class CVPriorTests(unittest.TestCase):
    def test_world_left_H20_baseline_and_zero_direct_PD_derivative(self):
        _,_,_,x=fixtures();x[1,-1]=.0025
        baseline=kinematic_baseline(x)
        np.testing.assert_array_equal(baseline.numpy(),kinematic_baseline_numpy(x.numpy()))
        torch.testing.assert_close(baseline[0],baseline[1])
        torch.testing.assert_close(decode_pen_torch(x[:,:55].double(),baseline),constant_velocity(x[:,:55].double(),.02))
        v=torch.zeros_like(x);v[:,81:101]=1
        _,jvp=torch.autograd.functional.jvp(kinematic_baseline,x,v)
        self.assertTrue(torch.equal(jvp,torch.zeros_like(jvp)))
        v.zero_();v[:,49]=1
        _,jvp=torch.autograd.functional.jvp(kinematic_baseline,x,v)
        torch.testing.assert_close(jvp[:,40],torch.full((3,),.02,dtype=torch.float64))
        self.assertTrue(torch.equal(baseline[:,46:],torch.zeros_like(baseline[:,46:])))

    def test_train_residual_center_preserves_other_heads_and_loss_scales(self):
        scales,hs,_,x=fixtures();y=np.ones((3,90))
        fitted=training_head_scales(scales,hs,x.numpy(),y,['DIRECT','DIRECT_CV'])
        expected=(y-kinematic_baseline_numpy(x.numpy())).mean(0)
        np.testing.assert_array_equal(fitted['DIRECT_CV']['head_mean'][:46],expected[:46])
        np.testing.assert_array_equal(fitted['DIRECT_CV']['head_mean'][46:],scales['y_mean'][46:])
        np.testing.assert_array_equal(fitted['DIRECT_CV']['head_scale'],scales['y_scale'])

    def test_supervised_loss_decodes_the_prior_and_has_equal_parameter_count(self):
        scales,hs,physics,x=fixtures()
        cv=PenResponseWM(scales,hs,physics,width=16,coupled=False,pose_baseline='CONSTANT_VELOCITY')
        direct=PenResponseWM(scales,hs,physics,width=16,coupled=False)
        self.assertEqual(sum(p.numel() for p in cv.parameters()),sum(p.numel() for p in direct.parameters()))
        p=cv.prediction(x);h=cv.physical_heads(x)
        torch.testing.assert_close(p,h.double()+kinematic_baseline(x),rtol=0,atol=0)
        loss,_=cv.supervised_loss(x,p.detach(),h.detach())
        self.assertEqual(float(loss.detach()),0.)
        self.assertTrue(bool((p[:,84:]>=0).all()))

    def test_explicit_family_checkpoint_roundtrip_and_legacy_rejection(self):
        scales,hs,physics,x=fixtures();m=PenResponseWM(scales,hs,physics,width=16,coupled=False,pose_baseline='CONSTANT_VELOCITY')
        saved=dict(schema='controlled_pen_response_arm:2',mode='DIRECT_CV',input_schema=INPUT_SCHEMA,output_schema=OUTPUT_SCHEMA,
            head_schema=CV_HEAD_SCHEMA,coupled=False,pose_baseline='CONSTANT_VELOCITY',width=16,model=m.state_dict(),scales=scales,
            head_scales=hs,physics=physics,config=dict(schema=2,model_comparison='DIRECT_VS_DIRECT_CV'))
        restored=model_from_arm(saved,'cpu');torch.testing.assert_close(restored.prediction(x),m.prediction(x),rtol=0,atol=0)
        for change in [dict(schema='controlled_pen_response_arm:1'),dict(pose_baseline='ZERO'),dict(coupled=True),dict(head_schema=OUTPUT_SCHEMA)]:
            with self.assertRaises(ValueError):model_from_arm({**saved,**change},'cpu')
        with self.assertRaises(ValueError):pen_families(dict(schema=True))

    def test_cv_continuation_rejects_mixed_legacy_pair(self):
        from test_pen_continuation import fixture
        s,h,p=fixture();s['schema']='controlled_pen_response_pair_checkpoint:2';h['schema']='controlled_pen_first100_health:2'
        arm=s['arms'].pop('IMPULSE');s['logs']['DIRECT_CV']=s['logs'].pop('IMPULSE');s['arms']['DIRECT_CV']=arm
        for mode,arm in s['arms'].items():
            arm.update(schema='controlled_pen_response_arm:2',mode=mode,coupled=False,
                pose_baseline='CONSTANT_VELOCITY' if mode=='DIRECT_CV' else 'ZERO',head_schema=CV_HEAD_SCHEMA if mode=='DIRECT_CV' else OUTPUT_SCHEMA)
            arm['config'].update(schema=2,model_comparison='DIRECT_VS_DIRECT_CV')
        self.assertEqual(validate_parent(s,h,p)['schema'],2)
        changed=copy.deepcopy(s);changed['arms']['DIRECT_CV']['pose_baseline']='ZERO'
        with self.assertRaises(ValueError):validate_parent(changed,h,p)
        changed=copy.deepcopy(s);changed['arms']['DIRECT']['schema']='controlled_pen_response_arm:1'
        with self.assertRaises(ValueError):validate_parent(changed,h,p)

    def test_legacy_prediction_loss_and_parameter_gradients_remain_bitwise(self):
        from supermimic_research.learning.impulse_response import ImpulseResponseWM
        path=Path(__file__).resolve().parents[1]/'archive/pen_pre_cv_prior_v1/impulse_response.py'
        spec=importlib.util.spec_from_file_location('supermimic_research.learning._frozen_pre_CV',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        scales,hs,physics,x=fixtures();scales={k:v[:87] if k.startswith('y_') else v for k,v in scales.items()}
        hs={k:v[:87] for k,v in hs.items()}
        for coupled in [False,True]:
            a=old.ImpulseResponseWM(scales,hs,physics,width=16,coupled=coupled)
            b=ImpulseResponseWM(scales,hs,physics,width=16,coupled=coupled);b.load_state_dict(a.state_dict())
            self.assertTrue(torch.equal(a.prediction(x),b.prediction(x)))
            y=torch.ones((3,87),dtype=torch.float64);head=torch.zeros_like(y)
            la,ta=a.supervised_loss(x,y,head);lb,tb=b.supervised_loss(x,y,head)
            self.assertTrue(torch.equal(la,lb));self.assertTrue(torch.equal(ta,tb))
            ga=torch.autograd.grad(la,tuple(a.parameters()));gb=torch.autograd.grad(lb,tuple(b.parameters()))
            self.assertTrue(all(torch.equal(u,v) for u,v in zip(ga,gb)))
