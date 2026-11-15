import unittest
import copy
import numpy as np
from supermimic_research.learning.pen_response_dataset import common_response_data,TRAIN_ROLE,CHECK_ROLE


class ResponseDataset(unittest.TestCase):
    def example(self):
        gen=np.random.default_rng(491);x=np.zeros((5,102));x[:,-1]=.0025;x[[0,2],81]=.002;x[[1,3],81]=-.002
        y=gen.normal(0,.1,(5,90));y[:,84:]=abs(y[:,84:]);data=dict(x=x,y=y,heads=y.copy(),valid=np.array([True,True,True,True,False]));check=copy.deepcopy(data);check['valid'][:]=True
        groups=[dict(source_name='source',direction=0,epsilon=.002,rows={'1':[0,2],'-1':[1,3]},qualified=True)]
        train=dict(role=TRAIN_ROLE,CHECK_response_rows_used=0,total_rows=5,qualified_rows=4,passed=1,total=1,response_groups=groups);check_report=dict(role=CHECK_ROLE,total_rows=5,qualified_rows=5,passed=1,total=1)
        directions=np.zeros((1,20));directions[0,0]=1
        return train,data,check_report,check,directions

    def test_CHECK_labels_and_invalid_TRAIN_rows_cannot_affect_scales(self):
        args=self.example();first=common_response_data(*args);args[3]['y'][:,:84]+=1e8;args[3]['heads'][:,:84]-=1e8;args[1]['y'][4,:84]+=1e9;args[1]['heads'][4,:84]-=1e9
        second=common_response_data(*args)
        for key in ['scales','head_scales']:
            for name in first[key]:np.testing.assert_array_equal(first[key][name],second[key][name])
        np.testing.assert_array_equal(first['pairs']['finite_contrast'],second['pairs']['finite_contrast'])

    def test_pair_targets_use_all_declared_valid_repeats(self):
        args=self.example();result=common_response_data(*args);y=args[1]['y'];expected=(y[[0,2]].mean(0)-y[[1,3]].mean(0))/.004
        np.testing.assert_array_equal(result['pairs']['finite_contrast'][0],expected)
        np.testing.assert_array_equal(result['pairs']['TRAIN_row_plus'],[[0,2]])

    def test_ROLE_relabel_or_mixed_CHECK_rejects(self):
        args=self.example();args[0]['role']=CHECK_ROLE
        with self.assertRaises(ValueError):common_response_data(*args)
        args=self.example();args[0]['CHECK_response_rows_used']=1
        with self.assertRaises(ValueError):common_response_data(*args)

    def test_group_cannot_override_row_or_actual_PD(self):
        args=self.example();args[1]['valid'][0]=False;args[0]['qualified_rows']=3
        with self.assertRaises(ValueError):common_response_data(*args)
        args=self.example();args[1]['x'][0,81]+=.01
        with self.assertRaises(ValueError):common_response_data(*args)
