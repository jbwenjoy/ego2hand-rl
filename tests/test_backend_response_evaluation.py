import unittest
import numpy as np
from supermimic_research.learning.backend_response_evaluation import group_responses,BLOCKS


class BackendEvaluationTests(unittest.TestCase):
    def example(self):
        y=np.zeros((4,90));y[:,40]=[.002,.002,-.002,-.002]
        data=dict(y=y,valid=np.ones(4,bool))
        groups=[dict(source_name='observed',direction=0,epsilon=.002,rows={'1':[0,1],'-1':[2,3]},complete=True,qualified=True),
                dict(source_name='excluded',direction=1,epsilon=.005,rows={'1':[],'-1':[]},complete=False,qualified=False)]
        derivative=np.zeros(90);derivative[40]=1
        return dict(response_groups=groups),data,{'fixture':y.copy()},{('fixture','observed',0):derivative},dict(observed=0),{
            ('observed',0):{b:dict(physical_screen=b=='object12') for b in BLOCKS}},np.ones(90)

    def test_empty_group_retains_null_metrics_and_exact_perfect_response(self):
        rows,aggregate=group_responses(*self.example());self.assertEqual(len(rows),2)
        self.assertIsNone(rows[1]['models']);self.assertFalse(rows[1]['qualified'])
        self.assertTrue(rows[0]['models']['fixture']['object12']['candidate_AD_screen'])
        a=aggregate['fixture']['object12']['ALL_PHYSICAL_VALID'];self.assertEqual(a['pairs'],1)
        self.assertEqual(a['AD_relative_L2_error'],0);self.assertEqual(a['model_finite_response_relative_L2_error'],0)

    def test_missing_baseline_never_removes_finite_response_denominator(self):
        args=list(self.example());args[3]={};args[4]={}
        rows,aggregate=group_responses(*args);a=aggregate['fixture']['object12']['ALL_PHYSICAL_VALID']
        self.assertEqual((a['pairs'],a['AD_pairs'],a['missing_AD_pairs']),(1,0,1))
        self.assertEqual(a['model_finite_response_relative_L2_error'],0);self.assertIsNone(a['AD_relative_L2_error'])
        self.assertFalse(rows[0]['models']['fixture']['object12']['candidate_AD_screen'])

    def test_zero_native_effect_has_undefined_relative_ratios(self):
        rows,aggregate=group_responses(*self.example());a=aggregate['fixture']['Fn6']['ALL_PHYSICAL_VALID']
        self.assertEqual(a['normalized_native_squared_norm'],0);self.assertIsNone(a['AD_relative_L2_error'])
        self.assertIsNone(a['zero_response_relative_L2_error'])
        self.assertIsNone(rows[0]['models']['fixture']['Fn6']['AD']['relative_L2_error'])

    def test_invalid_physics_mask_cannot_be_promoted_by_good_model(self):
        args=self.example();args[1]['valid'][0]=False
        with self.assertRaises(ValueError):group_responses(*args)
        args[0]['response_groups'][0]['qualified']=False
        rows,aggregate=group_responses(*args)
        self.assertFalse(rows[0]['models']['fixture']['object12']['candidate_AD_screen'])
        self.assertEqual(aggregate['fixture']['object12']['ALL_PHYSICAL_VALID']['pairs'],0)
