"""Patch labels retain unknowns and reject a wrong force-normal direction."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.patch_intent import match_patches


class PatchIntentTests(unittest.TestCase):
    def test_same_position_opposite_normal_cannot_match(self):
        result=match_patches([[0,0,0]],[[1,0,0]],[1],[[0,0,0]],[[-1,0,0]],[1],[True],groups=(1,))
        self.assertFalse(result['all_groups_matched']);self.assertFalse(result['physical_success_certified'])

    def test_unknown_is_not_a_negative_or_zero_contact_label(self):
        result=match_patches(np.empty((0,3)),np.empty((0,3)),[],[[float('nan'),0,0]],[[0,0,0]],[1],[False],groups=(1,))
        self.assertIsNone(result['all_groups_matched']);self.assertIsNone(result['groups'][0]['matched'])

    def test_joint_rigid_transform_preserves_correspondence(self):
        p=np.array([[0.,0.,0.],[.01,0.,0.]]);r=p+[.001,0,0];n=np.array([[0,0,1.],[1,0,0.]])
        a=match_patches(r,n,[1,2],p,n,[1,2],[True,True],groups=(1,2))
        R=np.array([[0,-1,0],[1,0,0],[0,0,1]]);t=[.4,-.3,.2]
        b=match_patches(r@R.T+t,n@R.T,[1,2],p@R.T+t,n@R.T,[1,2],[True,True],groups=(1,2))
        self.assertTrue(a['all_groups_matched']);self.assertEqual(a['all_groups_matched'],b['all_groups_matched'])


if __name__=='__main__':unittest.main()
