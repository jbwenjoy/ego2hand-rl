import unittest
import mujoco
from supermimic_research.runtime.pen_frozen_cost_rollout_v2 import require_structure


class FrozenCostStructureTests(unittest.TestCase):
    def model(self,free_palm=False,gravity_compensation=False):
        xml='<mujoco><worldbody><body name="robot/right_palm_link">'+('<freejoint/>' if free_palm else '')+'<geom type="sphere" size=".03"/><body><joint name="finger"/><geom type="sphere" size=".01"/></body></body><body pos="0 0 .1"'+(' gravcomp="1"' if gravity_compensation else '')+'><freejoint/><geom type="sphere" size=".01"/></body></worldbody><actuator><position joint="finger"/></actuator></mujoco>'
        return mujoco.MjModel.from_xml_string(xml)

    def test_actual_mujoco_one_element_joint_count_accepts_fixed_mount(self):
        model=self.model(); self.assertEqual(model.body('robot/right_palm_link').jntnum.shape,(1,)); self.assertTrue(require_structure(model))

    def test_dynamic_palm_or_gravity_compensation_stays_rejected(self):
        for model in [self.model(free_palm=True),self.model(gravity_compensation=True)]:
            with self.assertRaises(ValueError): require_structure(model)


if __name__=='__main__': unittest.main()
