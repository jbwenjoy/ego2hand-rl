"""Spin-couple signs, native generators and offset-COM support examples."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.contact_modes import local_pyramid_wrenches,contact_wrench_basis,enable_pad_spin,differences
from supermimic_research.mechanics.compound_contacts import CompoundContacts
from supermimic_research.mechanics.support import static_feasibility


class ContactModeTests(unittest.TestCase):
    def test_native_spin_couple_aggregation_and_EFC_columns_at_offset_COM(self):
        import mujoco
        xml='''<mujoco><option cone="pyramidal"/><worldbody><geom name="floor" type="plane" size="1 1 .1"/><body name="object" pos="0 0 .099" quat=".965925826 0 0 .258819045"><freejoint/><inertial pos=".03 0 0" mass=".2" diaginertia=".001 .002 .002"/><geom name="object_geom" type="sphere" size=".1" density="0"/></body></worldbody><contact><pair geom1="floor" geom2="object_geom" condim="4" friction=".7 .7 .003 .0001 .0001"/></contact></mujoco>'''
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);data.qvel[5]=1.
        mujoco.mj_forward(model,data);mujoco.mj_rnePostConstraint(model,data)
        state=CompoundContacts(model,[1],1,{},0).read(data,raw=True)
        self.assertLess(state['raw23'][0,20],0.)
        np.testing.assert_allclose(np.r_[state['torque_about_COM_world_Nm'],state['force_world_N']],data.cfrc_ext[1],atol=1e-12)
        basis=contact_wrench_basis(model,data,[1],1,[0],free_dof=0,hand_dofs=())
        self.assertEqual(basis['wrench'].shape,(6,6));self.assertLess(basis['native_wrench_error'],1e-12)

    def test_spin_opposes_rotation_and_reverses_with_declared_object_side(self):
        import mujoco
        xml='''<mujoco><option gravity="0 0 0"/><worldbody><body name="a" pos="-.09 0 0"><freejoint/><geom name="ga" type="sphere" size=".1" mass=".1"/></body><body name="b" pos=".09 0 0"><freejoint/><geom name="gb" type="sphere" size=".1" mass=".1"/></body></worldbody><contact><pair geom1="ga" geom2="gb" condim="4" friction=".7 .7 .003 .0001 .0001"/></contact></mujoco>'''
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);d.qvel[3]=1.;d.qvel[9]=-1.;mujoco.mj_forward(m,d)
        a=CompoundContacts(m,[0],m.body('a').id,{1:[1]},-1).read(d)
        b=CompoundContacts(m,[1],m.body('b').id,{1:[0]},-1).read(d)
        self.assertLess(a['torque_about_COM_world_Nm'][0],0.)
        np.testing.assert_allclose(a['torque_about_COM_world_Nm'],-b['torque_about_COM_world_Nm'],atol=1e-12)

    def test_torsion_can_balance_a_moment_that_point_forces_cannot(self):
        target=np.array([0.,0.,2.4525,0.,0.,0.]);results=[]
        for dim in [3,4]:
            columns=[]
            for sign in [1,-1]:
                frame=np.array([[0,sign,0],[0,0,1],[sign,0,0]],float);p=np.array([-.06,sign*.01,0])
                for ray in local_pyramid_wrenches(dim,[.7,.7,.003,.0001,.0001]):
                    F=frame.T@ray[:3];T=frame.T@ray[3:]+np.cross(p,F);columns.append(np.r_[F,T])
            results.append(static_feasibility(np.array(columns).T,target)['success'])
        self.assertEqual(results,[False,True])


if __name__=='__main__':unittest.main()
