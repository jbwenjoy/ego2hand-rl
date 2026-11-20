import unittest

import mujoco
import numpy as np

from supermimic_research.mechanics.pen_body_bvh import copy_refit_pen_bvh


class TestExportedPenBvh(unittest.TestCase):
    def setUp(self):
        xml = """<mujoco><worldbody>
        <body name="pen"><freejoint/>
        <inertial pos="0 0 0" quat=".7071067812 0 .7071067812 0"
                 mass=".05" diaginertia=".00012 .00012 .000001"/>
        <geom type="cylinder" size=".00745 .089" euler="0 90 0"/>
        <geom type="box" pos="-.107 0 0" size=".018 .009 .009"/>
        <geom type="box" pos=" .107 0 0" size=".018 .009 .009"/>
        </body>
        <body name="hand" pos=".105 .013 0"><freejoint/><geom size=".006"/></body>
        </worldbody></mujoco>"""
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.b = self.m.body('pen').id

    def contacts(self, m):
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return {tuple(map(int,c.geom)) for c in d.contact if c.dist <= 0.}

    def test_export_rotation_stales_bounds_and_refit_changes_only_copy(self):
        m = self.m
        compiled = self.contacts(m)
        self.assertTrue(compiled)
        bvh = m.bvh_aabb.copy()
        m.body_iquat[self.b] = [1.,0.,0.,0.]
        mujoco.mj_setConst(m,mujoco.MjData(m))
        np.testing.assert_array_equal(bvh, m.bvh_aabb)
        self.assertFalse(self.contacts(m))
        repaired, proof = copy_refit_pen_bvh(m, self.b)
        self.assertEqual(compiled, self.contacts(repaired))
        np.testing.assert_array_equal(bvh, m.bvh_aabb)
        self.assertGreater(proof['checked_nodes'][0]['old_excess_m'], .1)
        self.assertLess(max(r['new_excess_m'] for r in proof['checked_nodes']), 1e-12)
        for key in ('body_iquat','body_inertia','geom_aabb','geom_size','geom_pos','geom_quat','geom_rbound'):
            np.testing.assert_array_equal(getattr(repaired,key),getattr(m,key))
        self.assertFalse(np.shares_memory(repaired.qpos0,m.qpos0))
        twice,_ = copy_refit_pen_bvh(repaired, self.b)
        np.testing.assert_array_equal(twice.bvh_aabb,repaired.bvh_aabb)

    def test_inconsistent_compiled_geom_box_is_rejected(self):
        self.m.geom_aabb[0,3:] *= .5
        with self.assertRaisesRegex(ValueError,'Per-geom AABB'):
            copy_refit_pen_bvh(self.m, self.b)


if __name__ == '__main__':
    unittest.main()
