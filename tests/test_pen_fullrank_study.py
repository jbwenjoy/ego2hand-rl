import copy
from pathlib import Path
import tomllib
import unittest
import numpy as np
import torch
from supermimic_research.learning.pen_fullrank_study import model_config,pair_contract,evaluation_roles
from supermimic_research.learning.pen_contrast_update import update_contrast,restore_contrast_arm

ROOT=Path(__file__).resolve().parents[1]


class FullRankStudy(unittest.TestCase):
    def test_study3_does_not_reinterpret_model_family2(self):
        spec=tomllib.loads((ROOT/'configs/pen_fullrank_fit_v1.toml').read_text());cfg=model_config(spec)
        self.assertEqual((cfg['schema'],cfg['study_schema']),(1,3))
        for changes in [{'schema':2},{'model_comparison':'DIRECT_VS_DIRECT_CV'},{'contrast_weight':.2},{'response_scale':'NEW_DERIVATIVE_SCALE'},{'validation_evaluation_update':100}]:
            with self.assertRaises(ValueError):model_config({**spec,**changes})

    def test_validation_is_not_an_early_curve_or_health_input(self):
        for update in [100,500,1000]:self.assertEqual(evaluation_roles(update),['TRAIN'])
        self.assertEqual(evaluation_roles(2000),['TRAIN','VALIDATE'])
        with self.assertRaises(ValueError):evaluation_roles(1)

    def test_pair_identity_mismatch_rejects_before_sampling_or_optimizer(self):
        arm=dict(pair_contract={'rank':4},sampler=torch.Generator().manual_seed(7),pair_sampler=torch.Generator().manual_seed(9))
        before=[arm[k].get_state().clone() for k in ['sampler','pair_sampler']]
        with self.assertRaisesRegex(ValueError,'before sampling'):update_contrast(arm,{'pair_contract':{'rank':20}}, {})
        for k,s in zip(['sampler','pair_sampler'],before):self.assertTrue(torch.equal(arm[k].get_state(),s))

    def test_contract_pins_indices_and_point_bank(self):
        p=dict(outputs={'TRAIN':{'pinned':'TRAIN'}},pairs={'pinned':'pairs'},pair_subsets={'pinned':'subsets'})
        a=pair_contract(p,4,[0,2,5]);b=pair_contract(p,4,[0,2,6]);self.assertNotEqual(a,b)
        self.assertEqual(a['point_bank'],p['outputs']['TRAIN'])
        with self.assertRaises(ValueError):pair_contract(p,0,[0])

    def test_missing_or_cross_study_cache_identity_rejects_before_model_restore(self):
        with self.assertRaisesRegex(ValueError,'serialized'):restore_contrast_arm({'config':{'study_schema':3}},'cpu')
        with self.assertRaisesRegex(ValueError,'Historical'):restore_contrast_arm({'config':{},'pair_contract':{'rank':4}},'cpu')
