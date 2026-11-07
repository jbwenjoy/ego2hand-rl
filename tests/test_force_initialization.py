"""Native initialization contract against a small20-hinge/three-contact fixture."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.force_initialization import NativeEquilibrium


class ForceInitializationTests(unittest.TestCase):
    def fixture(self):
        import mujoco
        bodies=[]
        for i in range(20):
            if i<3:
                pos=[[-.059,0,.2],[.059,0,.2],[0,-.059,.2]][i]
                geom=f'<geom name="pad{i}" type="sphere" size=".02" mass=".01" condim="3"/>'
            else:
                pos=[.5+i*.02,0,0];geom='<inertial pos="0 0 0" mass=".01" diaginertia=".001 .001 .001"/>'
            bodies.append(f'<body pos="{pos[0]} {pos[1]} {pos[2]}"><joint name="j{i}" axis="0 1 0" range="-1 1" damping=".1" armature=".001"/>{geom}</body>')
        actuators=''.join(f'<position joint="j{i}" kp="2" ctrlrange="-1 1" forcerange="-1 1"/>' for i in range(20))
        xml='<mujoco><compiler angle="radian"/><worldbody><geom name="mug_task_table" type="plane" size="1 1 .1"/>'
        xml+='<body name="right_palm_link">'+''.join(bodies)+'</body>'
        xml+='<body name="object" pos="0 0 .2"><freejoint/><inertial pos=".005 0 0" mass=".25" diaginertia=".001 .001 .001"/><geom name="object_col" type="box" size=".04 .04 .04" density="0" condim="3"/></body></worldbody><actuator>'+actuators+'</actuator></mujoco>'
        m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
        binding=dict(mug_body_id=m.body('object').id,mug_part_geom_ids=[m.geom('object_col').id],mug_free_qpos=20,mug_free_dof=20)
        groups={i+1:[m.geom(f'pad{i}').id] for i in range(3)}
        return m,d,binding,groups

    def test_zero_decode_preserves_raw_state_and_repeated_native_reset(self):
        m,d,b,g=self.fixture();engine=NativeEquilibrium(m,d.qpos,np.zeros(20),b,g)
        q,u=engine.decode(np.zeros(46));np.testing.assert_array_equal(q,d.qpos);np.testing.assert_array_equal(u,np.zeros(20))
        a=engine.state(np.zeros(46));c=engine.state(np.zeros(46))
        self.assertEqual(a['raw23'].tobytes(),c['raw23'].tobytes())
        self.assertTrue(np.isfinite(engine.residual(np.zeros(46),c)).all());self.assertEqual(engine.counts['integration_steps'],0)

    def test_decoded_object_rotation_is_proper_and_illegal_proposal_is_rejected(self):
        m,d,b,g=self.fixture();engine=NativeEquilibrium(m,d.qpos,np.zeros(20),b,g)
        z=np.zeros(46);z[0]=.1;z[20]=.3;z[23]=.2;z[26]=.4
        q,u=engine.decode(z);self.assertAlmostEqual(q[0],.003);self.assertAlmostEqual(q[20],.0006)
        self.assertAlmostEqual(np.linalg.norm(q[23:27]),1.);self.assertAlmostEqual(u[0],.12)
        with self.assertRaises(ValueError):engine.decode(engine.hi+.001)


if __name__=='__main__':unittest.main()
