from pathlib import Path
import sys,unittest,tomllib
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.mechanics.bounded_contact_projection import BoundedContactProjection


class BoundedProjectionTests(unittest.TestCase):
    def fixture(self,upper):
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody>'
            '<body><freejoint/><geom name="object" type="sphere" size=".01" mass=".01"/></body>'
            '<body pos=".016 0 0"><joint type="slide" axis="1 0 0" range="-.001 .01"/>'
            '<geom name="pad" type="sphere" size=".01" mass=".01"/></body>'
            '</worldbody></mujoco>')
        d=mujoco.MjData(m)
        cfg=tomllib.loads((ROOT/'configs/controlled_pen_projection_v1.toml').read_text())
        cfg.update(maximum_wall_s=5.,maximum_residual_calls=200,max_nfev=30)
        return m,d,BoundedContactProjection(m,d,[7],[-.001],[upper],[0],[1],{1:[1]},cfg)

    def test_object_first_indexing_clears_contact_without_moving_object(self):
        m,d,p=self.fixture(.01);initial=d.qpos[:7].copy();result=p.solve()
        self.assertTrue(result['geometry_qualified'])
        self.assertGreater(result['q'][0],.0035)
        np.testing.assert_array_equal(d.qpos[:7],initial)
        self.assertEqual(result['counts']['integration_steps'],0)

    def test_insufficient_joint_room_is_preserved_as_failed_geometry(self):
        _,_,p=self.fixture(.001);result=p.solve()
        self.assertFalse(result['geometry_qualified'])
        self.assertLessEqual(result['q'][0],.001)
        self.assertFalse(result['checks']['object_penetration'])


if __name__=='__main__':unittest.main()
