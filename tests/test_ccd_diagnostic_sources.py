import unittest
from supermimic_research.runtime.ccd_diagnostic_sources import replace_once, convex_source


class DiagnosticSourceTests(unittest.TestCase):
    def test_unique_edit_required(self):
        for source in ['', 'xx']:
            with self.assertRaises(ValueError): replace_once(source, 'x', 'y')

    def test_all_three_face_arrays_change_together(self):
        s = 'epa_horizon = wp.empty(shape=(d.naccdmax, MJ_MAX_EPAHORIZON), dtype=int)\n'
        for name, dtype in [('epa_face','int'), ('epa_pr','wp.vec3'), ('epa_norm2','float')]:
            s += f'{name} = wp.empty(shape=(d.naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype={dtype})\n'
        result, edits = convex_source(s, horizon=96, face_factor=20)
        self.assertEqual(result.count('6 + 20 * epa_iterations'), 3)
        for old, new in reversed(edits): result = result.replace(new, old)
        self.assertEqual(result, s)

    def test_unknown_scratch_sizes_rejected(self):
        with self.assertRaises(ValueError): convex_source('', horizon=256, face_factor=50)
