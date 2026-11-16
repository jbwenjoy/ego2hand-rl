import copy
from pathlib import Path
import tomllib
import unittest
import numpy as np
from supermimic_research.learning.pen_interventions import seeded_training_profiles,validate_full_rank_cohort

ROOT=Path(__file__).resolve().parents[1]


class FullRankRoles(unittest.TestCase):
    def configs(self):
        return [tomllib.loads((ROOT/'configs'/('pen_response_fullrank_'+name+'_v1.toml')).read_text()) for name in ['train','validation']]

    def test_orthonormal_basis_has_all20_directions_and_complete_signed_repeats(self):
        for cfg in self.configs():
            d,rows=validate_full_rank_cohort(cfg)
            np.testing.assert_allclose(d@d.T,np.eye(20),atol=2e-15,rtol=0)
            self.assertEqual(np.linalg.matrix_rank(d),20);self.assertLess(abs(np.linalg.cond(d)-1),2e-15)
            self.assertEqual(len(rows),162)
            for direction in range(20):
                for epsilon in [.002,.005]:
                    r=[x for x in rows if x['direction']==direction and x['epsilon']==epsilon]
                    self.assertEqual({(x['repeat'],x['sign']) for x in r},{(0,1),(0,-1),(1,1),(1,-1)})

    def test_reserved_validation_cannot_acquire_TRAIN_identity(self):
        train,val=self.configs();self.assertTrue(set(train['noise_seeds']).isdisjoint(val['noise_seeds']))
        self.assertTrue(set(train['source_phases']).isdisjoint(val['source_phases']))
        val['role']=train['role']
        with self.assertRaisesRegex(ValueError,'split/direction/budget'):validate_full_rank_cohort(val)

    def test_old_CHECK_role_or_non_strict_restore_cannot_enter_new_cohort(self):
        train,_=self.configs()
        for changes in [{'role':'CHECK_TRAINED_ACTOR_PHYSICAL_RESPONSE_NEVER_FIT'},{'restore_mode':'READ_LOCKED_EPOCH'},{'noise_seeds':[8801,8802]},{'directions':4}]:
            bad=copy.deepcopy(train);bad.update(changes)
            with self.assertRaises(ValueError):validate_full_rank_cohort(bad)

    def test_old_gaussian4_exact_recipe_is_preserved(self):
        d,rows=seeded_training_profiles(9301,4,2)
        expected=np.random.Generator(np.random.PCG64(9301)).standard_normal((4,20));expected/=np.linalg.norm(expected,axis=1,keepdims=True)
        np.testing.assert_array_equal(d,expected);self.assertEqual(len(rows),34)
