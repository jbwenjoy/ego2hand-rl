"""Artifact migration boundaries; no learning/simulator dependencies."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supermimic_research.artifacts import ArtifactRegistry, file_pin


class ArtifactMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.data = self.base / "data"
        self.data.mkdir()
        (self.data / "fixture.bin").write_bytes(b"immutable fixture\x00\x01")
        self.manifest = self.base / "artifacts.toml"
        self.manifest.write_text('schema=1\n[roots]\nfixture="data"\n[relocations]\n"Z:/previous/data"="fixture"\n[artifacts.fixture]\nroot="fixture"\npath="fixture.bin"\nrole="ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE"\n', encoding="utf8")
        self.registry = ArtifactRegistry(self.manifest)

    def tearDown(self):
        self.temporary.cleanup()

    def test_explicit_relocation_preserves_bytes_and_role(self):
        pin = file_pin(self.data / "fixture.bin")
        pin["path"] = "z:\\previous\\data\\fixture.bin"
        self.assertEqual(self.registry.resolve_pin(pin), (self.data / "fixture.bin").resolve())
        self.assertEqual(self.registry.entries["fixture"].role, "ONLY_HELDOUT_TRIPOD_ACTION_RESPONSE")

    def test_changed_data_is_rejected_after_relocation(self):
        pin = file_pin(self.data / "fixture.bin")
        pin["path"] = "Z:/previous/data/fixture.bin"
        (self.data / "fixture.bin").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.registry.resolve_pin(pin)

    def test_artifact_path_cannot_escape_declared_root(self):
        self.manifest.write_text('schema=1\n[roots]\nfixture="data"\n[artifacts.bad]\nroot="fixture"\npath="../outside.bin"\nrole="TRAIN"\n', encoding="utf8")
        with self.assertRaisesRegex(ValueError, "escapes"):
            ArtifactRegistry(self.manifest)

    def test_relative_recorded_pin_is_not_resolved_against_cwd(self):
        pin = file_pin(self.data / "fixture.bin")
        pin["path"] = "fixture.bin"
        with self.assertRaisesRegex(ValueError, "absolute"):
            self.registry.resolve_pin(pin)


if __name__ == "__main__":
    unittest.main()
