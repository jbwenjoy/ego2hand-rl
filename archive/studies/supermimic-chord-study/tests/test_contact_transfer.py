"""Regression for MuJoCo Warp host-copy contacts with stale legacy geom fields."""
import sys
from pathlib import Path
import unittest
import numpy as np
import mujoco
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from controlled_pen_rollout import capture_contacts


class CopiedContactGeometry(unittest.TestCase):
    def test_modern_geom_pair_is_authoritative_when_legacy_fields_are_stale(self):
        model=mujoco.MjModel.from_xml_string('''<mujoco><option timestep=".002"/><worldbody>
        <geom name="robot/right_palm_collision" type="plane" size="1 1 .1"/>
        <body name="object" pos="0 0 .1"><freejoint/><geom name="pen" type="sphere" size=".1" mass="1"/></body>
        </worldbody></mujoco>''')
        data=mujoco.MjData(model)
        for _ in range(500):mujoco.mj_step(model,data)
        mujoco.mj_forward(model,data)
        obj=model.body("object").id
        before=capture_contacts(model,data,obj,{1},{0})
        self.assertTrue(before["active"].any())
        # get_data_into fills geom[], while legacy geom1/geom2 can retain
        # unrelated values. Reproduce that interop state without a GPU.
        for c in data.contact:
            modern=c.geom.copy();c.geom1=0;c.geom2=0
            np.testing.assert_array_equal(c.geom,modern)
        after=capture_contacts(model,data,obj,{1},{0})
        np.testing.assert_array_equal(after["geom_pair"],before["geom_pair"])
        np.testing.assert_allclose(after["force_w_N"],before["force_w_N"])
        self.assertEqual(after["object_environment_contacts"],0)


if __name__=="__main__":unittest.main()
