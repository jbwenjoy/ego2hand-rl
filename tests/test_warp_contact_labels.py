from pathlib import Path
import sys,unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from supermimic_research.mechanics.warp_contact_labels import decode_local_forces


class WarpContactLabelTests(unittest.TestCase):
    def test_inactive_contact_does_not_alias_last_constraint_force(self):
        f,bad=decode_local_forces(0,[3],np.ones((1,5)),np.full((1,10),-1),np.array([123.]),1)
        np.testing.assert_array_equal(f,0);self.assertFalse(bad.any())

    def test_pyramid_uses_explicit_noncontiguous_edge_indices(self):
        force=np.zeros(8);force[[5,2,7,0]]=[1.,2.,3.,4.]
        f,bad=decode_local_forces(0,[3],np.full((1,5),.7),np.array([[5,2,7,0,-1,-1,-1,-1,-1,-1]]),force,8)
        np.testing.assert_allclose(f,[[10.,-.7,-.7,0,0,0]],rtol=0,atol=1e-15);self.assertFalse(bad.any())

    def test_elliptic_indices_and_overflow_are_explicit(self):
        f,bad=decode_local_forces(1,[3,3],np.ones((2,5)),np.array([[2,0,1],[2,0,8]]),np.array([2.,3.,4.]),3)
        np.testing.assert_array_equal(f[0],[4,2,3,0,0,0]);np.testing.assert_array_equal(f[1],0)
        np.testing.assert_array_equal(bad,[False,True])

    def test_copied_host_adhesion_residue_cannot_contaminate_constraint_labels(self):
        import mujoco
        m=mujoco.MjModel.from_xml_string('<mujoco><option cone="pyramidal"/>'
            '<worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .09">'
            '<freejoint/><geom type="sphere" size=".1" mass="1" condim="3"/>'
            '</body></worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d);self.assertEqual(d.ncon,1)
        c=d.contact[0];old=np.zeros(6);mujoco.mj_contactForce(m,d,0,old)
        address=np.arange(c.efc_address,c.efc_address+4)[None,:]
        direct,_=decode_local_forces(0,[3],np.array([c.friction]),address,d.efc_force,d.nefc)
        np.testing.assert_allclose(direct[0],old,rtol=0,atol=1e-12)
        c.adhesion=-.04
        contaminated=np.zeros(6);mujoco.mj_contactForce(m,d,0,contaminated)
        self.assertAlmostEqual(contaminated[0]-direct[0,0],.04)
        again,_=decode_local_forces(0,[3],np.array([c.friction]),address,d.efc_force,d.nefc)
        np.testing.assert_array_equal(again,direct)


if __name__=='__main__':unittest.main()
