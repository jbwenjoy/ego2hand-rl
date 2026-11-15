import unittest
from supermimic_research.runtime.response_parent_split import parent_roles
from supermimic_research.mechanics.pen_contact_groups import normal_group


class ResponseParentTests(unittest.TestCase):
    def test_parent_roles_keep_all_candidates_and_repeats_together(self):
        spec=dict(TRAIN=[1,2,3,5,6,7],CHECK=[8,9,10],excluded=[0,4],repeats=3)
        roles=parent_roles(spec);self.assertEqual(len(roles),9);self.assertNotIn(0,roles);self.assertNotIn(4,roles)
        self.assertEqual([k for k,v in roles.items() if v=='CHECK'],[8,9,10])
        with self.assertRaises(ValueError):parent_roles({**spec,'TRAIN':[0,1,2,3,5,6,7]})

    def test_Fn6_accounts_for_all_fingers_and_other_robot_but_not_nonhand(self):
        self.assertEqual([normal_group(f'robot/right_finger{i}_link4_col') for i in range(1,6)],list(range(5)))
        self.assertEqual(normal_group('robot/right_palm_col'),5);self.assertEqual(normal_group('robot/other_geom'),5)
        self.assertEqual(normal_group('object/cylinder'),-1);self.assertEqual(normal_group('floor'),-1)
