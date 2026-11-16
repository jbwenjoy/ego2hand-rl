import unittest
import numpy as np
from supermimic_research.learning.reward_gradient_export import add_reward_arrays


class RewardExportTests(unittest.TestCase):
    def fixture(self):
        row=np.arange(10,dtype=float).reshape(5,2)
        base=dict(values=np.full((2,2),17.),RAW=np.ones((2,2,20)),PD_partial=np.ones((2,2,20)),
            RAW_direct_partial=np.zeros((2,2,20)),controller_diagonal=np.full((2,20),.25),computed_PD=np.zeros((2,20)))
        return row,base

    def test_all_rows_and_base_scores_survive_same_export(self):
        row,base=self.fixture();out={};add_reward_arrays(out,'fixture_',row,base)
        np.testing.assert_array_equal(out['fixture_values'],row);np.testing.assert_array_equal(out['fixture_base_values'],base['values'])
        self.assertEqual(len(out),7)

    def test_overwrite_or_misaligned_derivative_rejected_before_mutation(self):
        row,base=self.fixture();out={'fixture_values':row.copy()}
        with self.assertRaises(ValueError):add_reward_arrays(out,'fixture_',row,base)
        self.assertEqual(len(out),1);base['RAW']=np.zeros((1,2,20));out={}
        with self.assertRaises(ValueError):add_reward_arrays(out,'fixture_',row,base)
        self.assertFalse(out)
