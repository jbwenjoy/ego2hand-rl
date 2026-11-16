import copy
import unittest
import numpy as np
from supermimic_research.learning.pen_fullrank_dataset import full_rank_response_data,TRAIN_ROLE,VALIDATION_ROLE


class FullRankDataset(unittest.TestCase):
    def example(self):
        rng=np.random.default_rng(991);x=np.zeros((9,102));x[:,-1]=.0025
        for offset,direction in [(0,0),(4,7)]:
            x[[offset,offset+2],81+direction]=.002;x[[offset+1,offset+3],81+direction]=-.002
        y=rng.normal(0,.1,(9,90));y[:,84:]=abs(y[:,84:]);data=dict(x=x,y=y,heads=y.copy(),valid=np.array([True]*8+[False]))
        common=dict(schema='controlled_pen_response_preparation:2',CHECK_response_rows_used=0,validation_rows_used_for_TRAIN=0,passed=1,total=1,scales_fitted=False,total_rows=9,qualified_rows=8)
        groups=[dict(source_name='s',direction=d,epsilon=.002,rows={'1':[o,o+2],'-1':[o+1,o+3]},complete=True,qualified=True) for o,d in [(0,0),(4,7)]]
        tr={**common,'role':TRAIN_ROLE,'response_groups':groups};vr={**common,'role':VALIDATION_ROLE}
        return tr,data,vr,copy.deepcopy(data),np.eye(20)

    def test_validation_and_rejected_labels_cannot_change_training_products(self):
        args=self.example();first=full_rank_response_data(*args)
        for k in ['y','heads']:
            args[3][k][:,:84]+=1e8;args[1][k][8,:84]-=1e9
        second=full_rank_response_data(*args)
        for section in ['scales','head_scales','pairs','pair_subsets']:
            for k in first[section]:np.testing.assert_array_equal(first[section][k],second[section][k])

    def test_rank_subsets_share_one_point_bank_and_preserve_signed_repeat_average(self):
        args=self.example();built=full_rank_response_data(*args)
        np.testing.assert_array_equal(built['pair_subsets']['rank4'],[0]);np.testing.assert_array_equal(built['pair_subsets']['rank20'],[0,1])
        np.testing.assert_array_equal(built['pairs']['finite_contrast'][0],(args[1]['y'][[0,2]].mean(0)-args[1]['y'][[1,3]].mean(0))/.004)

    def test_role_relabelling_and_mixed_validation_reject(self):
        for index,change in [(0,{'role':VALIDATION_ROLE}),(2,{'role':TRAIN_ROLE}),(0,{'validation_rows_used_for_TRAIN':1}),(2,{'schema':'controlled_pen_TRAIN_preparation:1'})]:
            args=self.example();args[index].update(change)
            with self.assertRaises(ValueError):full_rank_response_data(*args)

    def test_incomplete_pairs_remain_audited_but_cannot_become_eligible(self):
        args=self.example();g=args[0]['response_groups'][1];g.update(rows={'1':[4],'-1':[5,7]},complete=False,qualified=False)
        built=full_rank_response_data(*args);self.assertFalse(built['pair_audit'][1]['eligible']);self.assertEqual(len(built['pairs']['epsilon']),1)
        g['qualified']=True
        with self.assertRaises(ValueError):full_rank_response_data(*args)

    def test_actual_PD_or_row_mask_cannot_be_overridden(self):
        args=self.example();args[1]['x'][0,81]+=.01
        with self.assertRaises(ValueError):full_rank_response_data(*args)
        args=self.example();args[1]['valid'][0]=False;args[0]['qualified_rows']=7
        with self.assertRaises(ValueError):full_rank_response_data(*args)
