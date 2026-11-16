from pathlib import Path
import tomllib
import unittest
import numpy as np
import torch
from supermimic_research.learning.pen_response_units import model_config,derivative_scale,response_contract
from supermimic_research.learning.pen_contrast_update import finite_contrast_loss,update_contrast,restore_contrast_arm

ROOT=Path(__file__).resolve().parents[1]


class ResponseUnits(unittest.TestCase):
    def test_RMS_and_positive_zero_response_floor_have_declared_units(self):
        target=np.zeros((2,90));target[:,40]=[3,4];point=np.ones(90)*2
        r=derivative_scale(target,point,role='TRAIN_CONTROLLED_PHYSICAL_RESPONSE')
        self.assertAlmostEqual(r['response_scale'][40],np.sqrt(12.5));self.assertEqual(r['response_scale'][0],.002)
        np.testing.assert_array_equal(r['floor'],.001*point)

    def test_validation_or_invalid_scale_cannot_supply_units(self):
        for role,point in [('VALIDATE_CONTROLLED_PHYSICAL_RESPONSE_NEVER_FIT',np.ones(90)),('TRAIN_CONTROLLED_PHYSICAL_RESPONSE',np.zeros(90))]:
            with self.assertRaises(ValueError):derivative_scale(np.zeros((2,90)),point,role=role)

    def test_config_cannot_add_forward_anchor_or_response_blocks_silently(self):
        spec=tomllib.loads((ROOT/'configs/pen_response_unit_fit_v1.toml').read_text());cfg=model_config(spec)
        self.assertEqual((cfg['schema'],cfg['study_schema']),(1,4))
        for change in [{'point_anchor_weight':2.},{'normalized_response_weights':[.1,10.]},{'response_floor_relative':.1},{'schema':3}]:
            with self.assertRaises(ValueError):model_config({**spec,**change})

    def test_wrong_response_units_reject_before_RNG_sampling(self):
        sampler=torch.Generator().manual_seed(72);before=sampler.get_state().clone()
        arm={'response_contract':{'mode':'TRAIN_DERIVATIVE_RMS'},'sampler':sampler}
        with self.assertRaisesRegex(ValueError,'before sampling'):update_contrast(arm,{'response_contract':{'mode':'POINT_Y_SCALE'}},{})
        self.assertTrue(torch.equal(before,sampler.get_state()))
        a=response_contract(np.ones(90),'POINT_Y_SCALE',.1,{'pinned':'scale'});b=response_contract(np.ones(90),'TRAIN_DERIVATIVE_RMS',.1,{'pinned':'scale'})
        self.assertNotEqual(a,b)

    def test_missing_serialized_units_reject_before_model_restore(self):
        pair=dict(rank=20,rows=949,index_sha256='x',point_bank={},pair_bank={},subset_bank={})
        with self.assertRaisesRegex(ValueError,'study4 response units'):restore_contrast_arm({'config':{'study_schema':4},'pair_contract':pair},'cpu')

    def test_normalized_loss_and_gradient_are_invariant_to_physical_unit_change(self):
        weight=torch.tensor(1.,dtype=torch.float64,requires_grad=True);e=torch.tensor([.002,.005],dtype=torch.float64)
        plus=weight*e[:,None]*torch.ones((2,90),dtype=torch.float64);target=torch.zeros((2,90),dtype=torch.float64);target[:,40:52]=3;target[:,78:84]=2
        scale=torch.ones(90,dtype=torch.float64);scale[40:52]=3;scale[78:84]=2
        loss=finite_contrast_loss(plus,-plus,target,e,scale);g=torch.autograd.grad(loss,weight,retain_graph=True)[0]
        units=torch.linspace(.01,100,90,dtype=torch.float64);converted=finite_contrast_loss(plus*units,-plus*units,target*units,e,scale*units)
        self.assertAlmostEqual(float(loss.detach()),4/9+1/4);self.assertAlmostEqual(float(g),-4/9-1/2)
        torch.testing.assert_close(converted,loss,atol=1e-12,rtol=0);torch.testing.assert_close(torch.autograd.grad(converted,weight)[0],g,atol=1e-12,rtol=0)
