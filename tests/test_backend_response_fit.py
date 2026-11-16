import copy
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import numpy as np
import torch
from supermimic_research.learning.backend_response_fit import (
    MODELS,model_spec,new_arm,saved_arm,restore_backend_arm,checkpoint_health)
from supermimic_research.learning.pen_contrast_update import update_contrast
from supermimic_research.learning.pen_response import fit_pen_scales
from supermimic_research.learning.world_model import raw_tree_equal


class BackendResponseFitTests(unittest.TestCase):
    def example(self):
        gen=np.random.default_rng(11); x=np.zeros((4,102));x[:,43]=x[:,47]=1;x[:,-1]=.0025
        x[:,81]=[.002,.002,-.002,-.002];y=gen.normal(0,.001,(4,90));y[:,84:]=abs(y[:,84:])
        scales,heads=fit_pen_scales(x,y,y.copy())
        cfg=dict(width=16,batch_size=4,pair_batch_size=2,learning_rate=.0003,weight_decay=.00001,
            gradient_clip_norm=1.,auxiliary_weight=.1,sampler_seed=14,pair_sampler_seed=15)
        context=dict(config=cfg,backend=dict(backend_id='NUMERIC_FIXTURE_ONLY',version=1),dataset=dict(fixture=True),
            physics=dict(mass=.03,inertia_BODY=np.diag([1e-6,2e-4,2e-4]),gravity=[0,0,-9.81]),
            scales=scales,head_scales=heads,response_scale=np.ones(90),pair_contract=dict(fixture=True))
        tensor=lambda v,dtype:torch.tensor(v,dtype=dtype)
        cache=dict(x=tensor(x,torch.float32),y=tensor(y,torch.float64),DIRECT=tensor(y,torch.float64),
            IMPULSE=tensor(y,torch.float64),x_plus=tensor(x[[0,1]],torch.float32),x_minus=tensor(x[[2,3]],torch.float32),
            finite_contrast=tensor((y[[0,1]]-y[[2,3]])/.004,torch.float64),epsilon=torch.full((2,),.002,dtype=torch.float64),
            response_scale=torch.ones(90,dtype=torch.float64),pair_contract=context['pair_contract'])
        return context,cache

    def test_owned_restore_and_next_update_are_exact_for_both_decoders(self):
        context,cache=self.example()
        for name in ['CONTACT_DIRECT','CONTACT_IMPULSE']:
            arm=new_arm(name,context,'cpu');cache['response_contract']=arm['response_contract']
            update_contrast(arm,cache,context['config'])
            saved=copy.deepcopy(saved_arm(arm,context,1,'cpu'));clone=restore_backend_arm(saved,context,'cpu')
            a=update_contrast(arm,cache,context['config'],capture_gradient=True)
            b=update_contrast(clone,cache,context['config'],capture_gradient=True)
            self.assertTrue(raw_tree_equal(a,b));self.assertTrue(raw_tree_equal(arm['model'].state_dict(),clone['model'].state_dict()))
            self.assertTrue(raw_tree_equal(arm['optimizer'].state_dict(),clone['optimizer'].state_dict()))

    def test_foreign_backend_dataset_types_or_old_schema_reject_before_construction(self):
        context,_=self.example();saved=saved_arm(new_arm('POINT_DIRECT',context,'cpu'),context,0,'cpu')
        for change in ['schema','backend','dataset','type','cursor']:
            bad=copy.deepcopy(saved)
            if change=='schema':bad['schema']='controlled_pen_response_arm:1'
            elif change=='backend':bad['context']['backend']['backend_id']='old'
            elif change=='dataset':bad['context']['dataset']['fixture']=False
            elif change=='type':bad['context']['backend']['version']=True
            else:bad['update']=True
            with self.subTest(change=change),patch('supermimic_research.learning.backend_response_fit.new_arm',side_effect=AssertionError('Constructed')):
                with self.assertRaises(ValueError):restore_backend_arm(bad,context,'cpu')

    def test_normalizer_or_pair_identity_change_cannot_enter_owned_resume(self):
        context,_=self.example();saved=saved_arm(new_arm('POINT_DIRECT',context,'cpu'),context,0,'cpu')
        for change in ['buffer','pairs','units']:
            bad=copy.deepcopy(saved)
            if change=='buffer':bad['model']['y_scale'][0]+=1
            elif change=='pairs':bad['pair_contract']['fixture']=False
            else:bad['response_contract']['scale_sha256']='other'
            with self.subTest(change=change),self.assertRaises(ValueError):restore_backend_arm(bad,context,'cpu')

    def test_cpu_clone_adam_storage_is_owned_and_input_payload_stays_unchanged(self):
        context,cache=self.example();arm=new_arm('CONTACT_DIRECT',context,'cpu');cache['response_contract']=arm['response_contract']
        update_contrast(arm,cache,context['config']);saved=copy.deepcopy(saved_arm(arm,context,1,'cpu'))
        before=copy.deepcopy(saved);left=restore_backend_arm(saved,context,'cpu');right=restore_backend_arm(saved,context,'cpu')
        lm=next(iter(left['optimizer'].state.values()))['exp_avg'];rm=next(iter(right['optimizer'].state.values()))['exp_avg']
        self.assertNotEqual(lm.data_ptr(),rm.data_ptr())
        update_contrast(left,cache,context['config'])
        self.assertTrue(raw_tree_equal(saved,before));self.assertTrue(raw_tree_equal(right['optimizer'].state_dict(),before['optimizer']))

    def test_all_six_checkpoint_parity_checks_leave_live_arms_unchanged(self):
        context,cache=self.example();arms={};caches={};buffers={}
        for name in MODELS:
            arm=arms[name]=new_arm(name,context,'cpu');caches[name]={**cache,'response_contract':arm['response_contract']}
            buffers[name]={k:v.detach().clone() for k,v in arm['model'].named_buffers()}
            update_contrast(arm,caches[name],context['config'])
        before=copy.deepcopy({n:(a['model'].state_dict(),a['optimizer'].state_dict(),a['sampler'].get_state(),a['pair_sampler'].get_state()) for n,a in arms.items()})
        work=dict(QA_attempted_calls=0,QA_completed_steps=0)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'ck.pt';torch.save(dict(update=1,planned_updates=2000,
                arms={n:saved_arm(a,context,1,'cpu') for n,a in arms.items()},logs={n:[[1.]] for n in arms}),path)
            checks=checkpoint_health(path,arms,context,caches,buffers,1,'cpu',work)
        self.assertTrue(all(checks.values()),[k for k,v in checks.items() if not v])
        self.assertEqual(work,dict(QA_attempted_calls=12,QA_completed_steps=12))
        after={n:(a['model'].state_dict(),a['optimizer'].state_dict(),a['sampler'].get_state(),a['pair_sampler'].get_state()) for n,a in arms.items()}
        self.assertTrue(raw_tree_equal(before,after))

    def test_fixed_matrix_budget_and_validation_schedule_cannot_drift(self):
        cfg=tomllib.loads((Path(__file__).resolve().parents[1]/'configs/pen_backend_response_fit_v1.toml').read_text())
        self.assertEqual(model_spec(cfg)['models'],MODELS)
        for change in [dict(planned_updates=100),dict(response_weight=.1),dict(validation_evaluation_updates=[100,2000]),dict(models=MODELS[:2])]:
            with self.subTest(change=change),self.assertRaises(ValueError):model_spec({**cfg,**change})
