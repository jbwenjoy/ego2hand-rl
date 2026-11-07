"""A hypothetical triplet must carry all declared fingers and balance moments."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.contact_selection import triplet_balance


class ContactSelectionTests(unittest.TestCase):
    def test_three_finger_minimum_can_exclude_a_force_only_solution(self):
        W=np.array([[0,0,1,0,.02,0],[0,0,1,0,-.02,0],[0,0,1,.02,0,0.]]).T
        self.assertFalse(triplet_balance(W,[0,0,1,0,0,0],[1,2,3])['success'])

    def test_all_three_minimums_and_COM_moments_are_checked(self):
        W=np.array([[0,0,1,0,.02,0],[0,0,1,0,-.02,0],[0,0,1,0,0,0.]]).T
        result=triplet_balance(W,[0,0,1,0,0,0],[1,2,3]);self.assertTrue(result['success'])
        self.assertGreaterEqual(min(result['group_normal_coefficients_N']),.1-1e-9)
        np.testing.assert_allclose(W@result['coefficients'],[0,0,1,0,0,0],atol=1e-12)


if __name__=='__main__':unittest.main()
