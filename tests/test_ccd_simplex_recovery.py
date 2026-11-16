import unittest
from supermimic_research.runtime.ccd_simplex_recovery import retry_inconsistent_tetrahedron


class SimplexRecoveryRecipeTests(unittest.TestCase):
    def test_existing_face_reconstruction_is_preserved(self):
        source = 'prefix\n    if dist[idx] > MINVAL:\n      pt.status = 12\n      return pt, GJKResult()\n\n    # fallback to closest face\n    pt.status = -1\n    return _replace_simplex3()\n'
        result, edit = retry_inconsistent_tetrahedron(source)
        self.assertTrue(result.endswith('    pt.status = -1\n    return _replace_simplex3()\n'))
        self.assertEqual(result.replace(edit['new'], edit['old']), source)

    def test_unknown_source_refuses_edit(self):
        with self.assertRaises(ValueError): retry_inconsistent_tetrahedron('other source')
