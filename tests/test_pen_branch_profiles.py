import unittest
import numpy as np
from supermimic_research.learning.pen_branch_profiles import branch_profiles


class BranchProfileTests(unittest.TestCase):
    def test_fixed_cohort_disjoint_roles_and_balanced_signs(self):
        cfg=dict(anchors=[0,2,4,6],dt=[.005,.0025],PD_target_eps_rad=[.005,.02],
            train_direction_seed=7501,check_direction_seed=7511,train_direction_count=12,check_direction_count=4)
        directions,profiles=branch_profiles(cfg)
        self.assertEqual(len(profiles),128);self.assertEqual(len({p['name'] for p in profiles}),128)
        self.assertEqual(sum(p['role']=='TRAIN' for p in profiles),96)
        np.testing.assert_allclose(np.linalg.norm(directions['TRAIN'],axis=1),1,atol=1e-15)
        self.assertTrue(all(not np.array_equal(a,b) for a in directions['TRAIN'] for b in directions['CHECK']))
        for role in ['TRAIN','CHECK']:
            for anchor in cfg['anchors']:
                for direction in range(len(directions[role])):
                    pair=[p for p in profiles if p['role']==role and p['anchor']==anchor and p['direction']==direction]
                    self.assertEqual({p['sign'] for p in pair},{-1,1})
        cfg['check_direction_seed']=7501
        with self.assertRaises(ValueError):branch_profiles(cfg)
