"""Force signs and moving-COM wrench sum against native postconstraint arrays."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.compound_contacts import CompoundContacts


class CompoundContactTests(unittest.TestCase):
    def test_two_piece_one_body_sum_matches_native_external_wrench(self):
        import mujoco
        xml='''<mujoco><option gravity="0 0 -9.81"/><worldbody><geom name="table" type="plane" size="1 1 .1"/><body name="object" pos="0 0 .099"><freejoint/><inertial pos=".03 0 0" mass=".2" diaginertia=".001 .002 .002"/><geom name="a" type="sphere" pos="-.04 0 0" size=".1" density="0" condim="3"/><geom name="b" type="sphere" pos=".04 0 0" size=".1" density="0" condim="3"/></body></worldbody></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d);mujoco.mj_rnePostConstraint(m,d)
        reader=CompoundContacts(m,[1,2],m.body('object').id,{},0);s=reader.read(d,raw=True)
        self.assertEqual(s['object_contacts'],2);self.assertGreater(s['force_world_N'][2],0)
        native=d.cfrc_ext[m.body('object').id]
        np.testing.assert_allclose(np.r_[s['torque_about_COM_world_Nm'],s['force_world_N']],native,atol=1e-12)
        # The offset COM also changes force distribution between the pieces;
        # equal normal loads must not be assumed. Native rne is the independent sum.
        np.testing.assert_allclose(s['object_COM_world_m'],[.03,0,.099],atol=1e-14)
        self.assertGreater(abs(s['torque_about_COM_world_Nm'][1]),1e-5)
        self.assertEqual(s['raw23'].shape,(2,23))

    def test_reversing_declared_object_reverses_contact_force(self):
        import mujoco
        xml='''<mujoco><option gravity="0 0 0"/><worldbody><body name="a" pos="-.09 0 0"><freejoint/><geom name="ga" type="sphere" size=".1" mass=".1"/></body><body name="b" pos=".09 0 0"><freejoint/><geom name="gb" type="sphere" size=".1" mass=".1"/></body></worldbody></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d);mujoco.mj_rnePostConstraint(m,d)
        a=CompoundContacts(m,[0],m.body('a').id,{1:[1]},-1).read(d)
        b=CompoundContacts(m,[1],m.body('b').id,{1:[0]},-1).read(d)
        np.testing.assert_allclose(a['force_world_N'],-b['force_world_N'],atol=1e-12)
        self.assertLess(a['force_world_N'][0],0);self.assertGreater(b['force_world_N'][0],0)


if __name__=='__main__':unittest.main()
