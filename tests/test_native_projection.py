"""Bounded probes and unactuated inverse forces never become applied aid."""
from pathlib import Path
from types import SimpleNamespace
import io,sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.native_projection import InversePoseProjection,ForwardForceProjection,search_projection


class NativeProjectionTests(unittest.TestCase):
    def test_one_sided_secants_do_not_escape_an_active_bound(self):
        class Engine:
            lo=np.array([0.]);hi=np.array([1.]);data=SimpleNamespace(contact=[]);counts={}
            def state(self,z):
                if z[0]<0 or z[0]>1:raise AssertionError('Out-of-bound numerical probe')
                return dict(normal_by_finger_N=np.ones(3),max_native_penetration_m=0.,max_self_penetration_m=0.,root_residual6=np.zeros(6))
            def residual(self,z,s):return z+.5
        result=search_projection(Engine(),io.StringIO(),maximum_calls=100,maximum_seconds=2.)
        self.assertLess(result['best']['z'][0],1e-6);self.assertLessEqual(result['calls'],100)

    def test_inverse_gravity_requirement_is_not_applied_to_free_object(self):
        import mujoco
        bodies=''.join(f'<body pos="{i*.1} 0 1"><joint name="j{i}" axis="0 0 1" range="-1 1" actuatorfrcrange="-10 10"/><geom name="g{i}" type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/></body>' for i in range(20))
        actuators=''.join(f'<position joint="j{i}" kp="10" ctrlrange="-1 1" forcerange="-10 10"/>' for i in range(20))
        xml=f'<mujoco><worldbody>{bodies}<geom name="mug_task_table" type="plane" size="1 1 .1" contype="0" conaffinity="0"/><body name="object" pos="0 0 2"><freejoint/><geom name="object_geom" type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/></body></worldbody><actuator>{actuators}</actuator></mujoco>'
        model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);q=data.qpos.copy();groups={i:[model.geom(f'g{i-1}').id] for i in [1,2,3]}
        binding=dict(mug_body_id=model.body('object').id,mug_part_geom_ids=[model.geom('object_geom').id],mug_free_qpos=20,mug_free_dof=20)
        inverse=InversePoseProjection(model,q,binding,groups).state(np.zeros(20));self.assertAlmostEqual(inverse['root_residual6'][2],9.81)
        forward=ForwardForceProjection(model,q,inverse['ctrl20'],binding,groups).state(np.zeros(40))
        self.assertAlmostEqual(forward['qacc26'][22],-9.81);self.assertAlmostEqual(forward['net_object_COM_wrench_world'][2],-9.81)


if __name__=='__main__':unittest.main()
