from pathlib import Path
import tomllib
import unittest
import numpy as np
import torch
from supermimic_research.learning.pen_contact_supervision import model_config
from supermimic_research.learning.pen_response_units import response_contract
from supermimic_research.learning.pen_contrast_update import finite_contrast_loss,update_contrast,restore_contrast_arm

ROOT=Path(__file__).resolve().parents[1]


class FnResponse(unittest.TestCase):
    def test_Fn_addition_keeps_existing_value_and_gradient_coefficients(self):
        w=torch.tensor(1.,dtype=torch.float64,requires_grad=True);e=torch.tensor([.002,.005],dtype=torch.float64);p=w*e[:,None]*torch.ones((2,90),dtype=torch.float64)
        target=torch.zeros((2,90),dtype=torch.float64);target[:,40:52]=3;target[:,78:84]=2;target[:,84:90]=4;scale=torch.ones(90,dtype=torch.float64);scale[84:90]=2
        old=finite_contrast_loss(p,-p,target,e,scale);new=finite_contrast_loss(p,-p,target,e,scale,include_fn=True)
        self.assertAlmostEqual(float((new-old).detach()),9/4)
        extra=torch.autograd.grad(new-old,w)[0];self.assertAlmostEqual(float(extra),-1.5)

    def test_disabled_Fn_block_is_exactly_insensitive_to_Fn_targets(self):
        p=torch.zeros((2,90),dtype=torch.float64);e=torch.ones(2,dtype=torch.float64)*.002;t=torch.zeros_like(p);s=torch.ones(90,dtype=torch.float64)
        a=finite_contrast_loss(p,p,t,e,s);t[:,84:90]=1e8
        self.assertTrue(torch.equal(a,finite_contrast_loss(p,p,t,e,s)))

    def test_block_identity_differs_without_changing_units(self):
        a=response_contract(np.ones(90),'TRAIN_DERIVATIVE_RMS',.1,{'pinned':'same'},blocks=['object12','wrench6']);b=response_contract(np.ones(90),'TRAIN_DERIVATIVE_RMS',.1,{'pinned':'same'},blocks=['object12','wrench6','Fn6'])
        self.assertEqual(a['scale_sha256'],b['scale_sha256']);self.assertNotEqual(a,b)

    def test_wrong_block_cache_rejects_before_sampling(self):
        with self.assertRaisesRegex(ValueError,'before sampling'):update_contrast({'response_contract':{'blocks':['object12','wrench6']}},{'response_contract':{'blocks':['object12','wrench6','Fn6']}},{})

    def test_missing_explicit_study5_blocks_rejects_before_model_restore(self):
        pair=dict(rank=20,rows=949,index_sha256='x',point_bank={},pair_bank={},subset_bank={})
        with self.assertRaisesRegex(ValueError,'study5 response blocks'):restore_contrast_arm({'config':{'study_schema':5},'pair_contract':pair},'cpu')

    def test_registered_config_cannot_change_old_term_or_anchor_weights(self):
        s=tomllib.loads((ROOT/'configs/pen_fn_response_fit_v1.toml').read_text());c=model_config(s);self.assertEqual((c['schema'],c['study_schema']),(1,5))
        for changes in [{'fn_response_coefficients':[0.,.333]},{'point_anchor_weight':2.},{'auxiliary_weight':.2},{'schema':4}]:
            with self.assertRaises(ValueError):model_config({**s,**changes})
