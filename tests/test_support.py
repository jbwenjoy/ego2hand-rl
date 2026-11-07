"""Six-dimensional support and PD-limited load against analytic examples."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.support import pyramid3,static_feasibility,maximum_support


class SupportTests(unittest.TestCase):
    def test_native_condim3_rays_match_reviewed_public_decode(self):
        import mujoco
        for mu in [[.7,.7],[.2,.9]]:
            decoded=[]
            if mujoco.__version__!='3.11.0':self.skipTest('Version-specific public buffer contract reviewed for3.11.0')
            for coefficient in np.eye(4):
                out=np.zeros(4);mujoco.mju_decodePyramid(out,np.r_[coefficient,0.,0.],np.r_[mu,0.]);decoded.append(out[:3])
            np.testing.assert_array_equal(pyramid3(mu),decoded)

    def test_force_balance_cannot_substitute_for_offset_COM_moment_balance(self):
        W=np.array([[0],[0],[1],[0],[-.06],[0]],float);target=np.array([0,0,2.4525,0,0,0])
        self.assertTrue(static_feasibility(W,target,force_only=True)['success'])
        self.assertFalse(static_feasibility(W,target)['success'])
        W=np.c_[W,np.array([0,0,1,0,.06,0])]
        answer=static_feasibility(W,target);self.assertTrue(answer['success'])
        np.testing.assert_allclose(answer['witness']['ray_coefficients_N'],[1.22625,1.22625],atol=1e-12)

    def test_support_includes_actuator_and_declared_dry_friction_box(self):
        W=np.zeros((6,1));W[2]=1;loads=np.zeros((20,1));loads[0]=1
        z=np.zeros(20);lo=-np.ones(20);hi=np.ones(20)
        result=maximum_support(W,loads,z,lo,hi,np.full(20,.3))
        self.assertAlmostEqual(result['support_weight_N'],1.3)
        target=np.array([0,0,1.4,0,0,0])
        self.assertFalse(static_feasibility(W,target,loads=loads,offset=z,lower=lo,upper=hi,friction=np.full(20,.3))['success'])


if __name__=='__main__':unittest.main()
