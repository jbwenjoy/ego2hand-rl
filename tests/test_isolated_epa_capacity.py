import unittest
from supermimic_research.runtime.isolated_epa_capacity import ALLOCATION, capacity_source


class IsolatedCapacityTests(unittest.TestCase):
    def test_only_allocation_expression_changes(self):
        source = '# licensed header\n' + ALLOCATION + '\nrest_of_kernel = 24\n'
        out = capacity_source(source, 48)
        self.assertEqual(out.replace('(d.naccdmax, 48)', '(d.naccdmax, MJ_MAX_EPAHORIZON)'), source)

    def test_missing_or_multiple_sites_rejected(self):
        for source in ['', ALLOCATION + '\n' + ALLOCATION]:
            with self.assertRaises(ValueError): capacity_source(source, 48)

    def test_unregistered_capacity_rejected(self):
        for size in [24, 49, 192]:
            with self.assertRaises(ValueError): capacity_source(ALLOCATION, size)


if __name__ == '__main__': unittest.main()
