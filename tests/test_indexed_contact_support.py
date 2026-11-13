from pathlib import Path
import sys,unittest
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.indexed_contact_support import friction_rays,indexed_static_support


class IndexedContactSupportTests(unittest.TestCase):
    def fixture(self):
        bodies=[]
        for i in range(20):
            x=-.039 if i==0 else .039 if i==1 else .5+i*.03
            geom=f'<geom name="pad{i}" type="sphere" size=".02" mass=".01" condim="4" friction=".7 .005 .0001"/>' if i<2 else '<inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/>'
            bodies.append(f'<body pos="{x} 0 .2"><joint name="j{i}" type="hinge" axis="0 1 0" range="-1 1" actuatorfrcrange="-1 1"/>{geom}</body>')
        actuators=''.join(f'<position joint="j{i}" kp="3" ctrlrange="-1 1" forcerange="-1 1"/>' for i in range(20))
        xml='<mujoco><compiler angle="radian"/><option cone="pyramidal"/><worldbody><body name="object" pos="0 0 .2"><freejoint/>'
        xml+='<geom name="object_col" type="sphere" size=".02" mass=".03" condim="4" friction=".7 .005 .0001"/></body>'+''.join(bodies)+'</worldbody><actuator>'+actuators+'</actuator></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        return m,d

    def test_torsional_rays_are_force_couple_columns(self):
        rays=friction_rays(4,[.7,.7,.005])
        self.assertEqual(rays.shape,(6,6));np.testing.assert_array_equal(rays[:,0],1)
        self.assertEqual(rays[4,3],.005);self.assertEqual(rays[5,3],-.005)
        np.testing.assert_array_equal(rays[:,1:].sum(axis=0),0)

    def test_object_first_layout_supports_full_wrench_and_enforces_PD_authority(self):
        m,d=self.fixture();body=m.body('object').id;parts=[m.geom('object_col').id];allowed=[m.geom(f'pad{i}').id for i in range(2)];dofs=np.arange(6,26);actuators=np.arange(20)
        result=indexed_static_support(m,d,body,parts,allowed,dofs,actuators,np.full(20,-.05),np.full(20,.05))
        self.assertTrue(result['lp']['success']);self.assertEqual(result['full_wrench_rank'],6)
        self.assertLess(np.max(abs(result['wrench_BODY']@result['lp']['witness']['ray_coefficients_N']-result['target_BODY'])),1e-8)
        constrained=indexed_static_support(m,d,body,parts,allowed,dofs,actuators,np.zeros(20),np.zeros(20))
        self.assertFalse(constrained['lp']['success'])


if __name__=='__main__':unittest.main()
