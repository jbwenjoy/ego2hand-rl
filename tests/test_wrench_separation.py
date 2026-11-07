"""Separate gravity force from off-axis moment, with explicit COM and units."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.wrench_separation import separating_witness


class WrenchSeparationTests(unittest.TestCase):
    def test_force_can_balance_while_offset_moment_is_separated(self):
        W=np.array([[0,0,1,0,-.02,0.]]).T;t=np.array([0.,0,1,0,0,0])
        result=separating_witness(W,t);self.assertTrue(result['verified'])
        y=np.array(result['witness_dual_to_F_N_T_Nm']);self.assertGreaterEqual(float((W.T@y)[0]),-1e-9)
        self.assertAlmostEqual(float(t@y),-1)

    def test_balanced_target_has_no_separating_witness(self):
        W=np.array([[0,0,1,0,-.02,0.],[0,0,1,0,.02,0.]]).T
        self.assertFalse(separating_witness(W,[0,0,1,0,0,0])['verified'])


if __name__=='__main__':unittest.main()
