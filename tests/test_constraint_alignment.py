import unittest
import numpy as np
from supermimic_research.mechanics.constraint_alignment import canonical_constraint_rows


class ConstraintAlignmentTests(unittest.TestCase):
    def fixture(self):
        contacts = dict(nefc=np.array([5]), worldid=np.array([0]), dim=np.array([3]),
                        geom=np.array([[1, 2]]), geomcollisionid=np.array([0]), efc_address=np.array([[1, 3, 4, 0]]))
        fields = dict(type=np.array([[6, 6, 1, 6, 6]]), id=np.array([[0, 0, 8, 0, 0]]),
                      J=np.arange(15).reshape(1, 5, 3), force=np.arange(5)[None])
        return contacts, fields

    def test_engine_row_permutation_preserves_physical_readback(self):
        contacts, fields = self.fixture()
        reference = canonical_constraint_rows(contacts, fields, 0)
        permutation = np.array([4, 2, 0, 3, 1])
        inverse = np.argsort(permutation)
        shuffled = {k: v[:, permutation].copy() for k, v in fields.items()}
        moved = {k: v.copy() for k, v in contacts.items()}
        moved['efc_address'] = inverse[moved['efc_address']]
        actual = canonical_constraint_rows(moved, shuffled, 0)
        np.testing.assert_array_equal(reference['keys'], actual['keys'])
        np.testing.assert_array_equal(reference['fields']['J'], actual['fields']['J'])
        np.testing.assert_array_equal(reference['fields']['force'], actual['fields']['force'])
        self.assertFalse(np.array_equal(reference['source_addresses'], actual['source_addresses']))

    def test_duplicate_source_addresses_never_become_valid_rows(self):
        contacts, fields = self.fixture()
        contacts['efc_address'][0, 1] = contacts['efc_address'][0, 0]
        with self.assertRaisesRegex(ValueError, 'Unique valid'):
            canonical_constraint_rows(contacts, fields, 0)
