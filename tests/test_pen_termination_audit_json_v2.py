import unittest

from supermimic_research.mechanics.pen_termination_audit_json_v2 import audit_json_equal


class AuditJsonTests(unittest.TestCase):
    def test_audit_list_representation_matches_tuple_without_mutating_state(self):
        typed={'support_weight_state_path':(('attr','support_weight'),)}
        saved={'support_weight_state_path':[['attr','support_weight']]}
        self.assertTrue(audit_json_equal(saved,typed))
        self.assertIsInstance(typed['support_weight_state_path'],tuple)

    def test_value_and_missing_field_still_fail(self):
        typed={'path':(('attr','support_weight'),),'value':1.}
        self.assertFalse(audit_json_equal({'path':[['attr','wrong']],'value':1.},typed))
        self.assertFalse(audit_json_equal({'path':[['attr','support_weight']]},typed))


if __name__=='__main__':
    unittest.main()
