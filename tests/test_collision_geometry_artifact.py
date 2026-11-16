import tempfile
import unittest
from pathlib import Path
import numpy as np
from supermimic_research.runtime.collision_geometry_artifact import load_collision_geometry


class GeometryArtifactTests(unittest.TestCase):
    def test_mixed_evidence_keeps_only_geometry_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.npz'
            np.savez(path, qpos=np.zeros((2, 27), np.float32), geom_xpos=np.zeros((2, 3, 3), np.float32),
                geom_xmat=np.zeros((2, 3, 3, 3), np.float32), overflow=np.ones(2, np.int32), contact_worldid=np.array([1]))
            loaded = load_collision_geometry(path)
            self.assertEqual(set(loaded), {'qpos', 'geom_xpos', 'geom_xmat'})
            # New collision responses can be serialized without duplicate names.
            np.savez(Path(directory) / 'result.npz', **loaded, contact_worldid=np.array([0]), overflow=np.zeros(2, np.int32))

    def test_mismatched_worlds_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.npz'
            np.savez(path, qpos=np.zeros((1, 27), np.float32), geom_xpos=np.zeros((2, 3, 3), np.float32), geom_xmat=np.zeros((2, 3, 3, 3), np.float32))
            with self.assertRaises(ValueError): load_collision_geometry(path)

    def test_nonfinite_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.npz'
            np.savez(path, qpos=np.full((2, 27), np.nan, np.float32), geom_xpos=np.zeros((2, 3, 3), np.float32), geom_xmat=np.zeros((2, 3, 3, 3), np.float32))
            with self.assertRaises(ValueError): load_collision_geometry(path)
