"""Native witness sign and a smooth pair Jacobian against finite differences."""
from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import ArtifactRegistry
from supermimic_research.mechanics.contact_ik import native_pair,ContactAwareIK
from supermimic_research.mechanics.compound_collision import CompoundCollisionAudit
from supermimic_research.mechanics.collision import self_pairs


class ContactIKTests(unittest.TestCase):
    def test_signed_native_witness_orientation_for_gap_and_partial_overlap(self):
        import mujoco
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" size="1 1 1"/><body pos="1.1 0 0"><freejoint/><geom type="sphere" size=".2" mass=".01"/></body></worldbody></mujoco>')
        d=mujoco.MjData(m)
        for x,expected in [(1.4,.2),(1.1,-.1)]:
            d.qpos[0]=x;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
            distance,points=native_pair(m,d,0,1,2.)
            self.assertAlmostEqual(distance,expected)
            np.testing.assert_allclose((points[1]-points[0])/distance,[1,0,0],atol=1e-12)

    def test_native_witness_point_jacobian_in_one_smooth_actual_hand_pair(self):
        import mujoco
        m=mujoco.MjModel.from_binary_path(str(ArtifactRegistry(ROOT/'configs/artifacts.toml').resolve('phone_model')))
        d=mujoco.MjData(m);mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        hand,_=self_pairs(m);part=m.geom('phone24_collision_convex').id;table=m.geom('stage1_single_table').id
        audit=CompoundCollisionAudit(m,[part],hand,table,{1:[15],2:[28],3:[41]})
        targets=d.site_xpos[[m.site(f'right_finger{i}_tip').id for i in range(1,6)]].copy()
        engine=ContactAwareIK(m,d.qpos,targets,audit);q=np.full(20,.1);engine.state(q)
        distance,points=native_pair(m,engine.data,part,15,1.)
        jac=engine.distance_jacobian(part,15,distance,points);numeric=[]
        for i in range(20):
            delta=np.zeros(20);delta[i]=1e-6;engine.state(q+delta);plus=native_pair(m,engine.data,part,15,1.)[0]
            engine.state(q-delta);minus=native_pair(m,engine.data,part,15,1.)[0];numeric.append((plus-minus)/2e-6)
        np.testing.assert_allclose(jac,numeric,atol=1e-5,rtol=1e-4)
        self.assertEqual(engine.counts['integration_steps'],0)


if __name__=='__main__':unittest.main()
