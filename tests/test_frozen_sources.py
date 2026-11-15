import tempfile
import unittest
import sys
import types
from unittest.mock import patch
from pathlib import Path

from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin,frozen_import_sources


class FrozenSourceTests(unittest.TestCase):
    def test_live_change_detected_without_corrupting_historical_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'src'/'core.py';source.parent.mkdir();source.write_bytes(b'old physics\n')
            m=freeze_sources([source],root/'run'/'sources',root,'test-revision')
            verify_source_snapshot(m,live=True)
            source.write_bytes(b'new physics\n')
            verify_source_snapshot(m)
            with self.assertRaises(ValueError):verify_source_snapshot(m,live=True)
            Path(m['records'][0]['snapshot']['path']).write_bytes(b'bad archive\n')
            with self.assertRaises(ValueError):verify_source_snapshot(m)

    def test_exact_relocation_rejects_mutation_and_wrong_original_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'old.py';b=root/'frozen.py';a.write_bytes(b'original');b.write_bytes(b'original')
            p=file_pin(a);mapping=dict(schema='explicit_source_relocations:1',records=[dict(original=p,archived=file_pin(b))])
            a.write_bytes(b'changed');self.assertEqual(verify_source_pin(p,mapping),b.resolve())
            with self.assertRaises(ValueError):verify_source_pin(p)
            wrong={**p,'path':str(root/'different.py')}
            with self.assertRaises(FileNotFoundError):verify_source_pin(wrong,mapping)
            b.write_bytes(b'changed!')
            with self.assertRaises(ValueError):verify_source_pin(p,mapping)

    def test_reject_duplicate_paths_and_external_sources_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'a.py';p.write_bytes(b'x');outside=root/'outside';outside.mkdir()
            with self.assertRaises(ValueError):freeze_sources([p,p],root/'snap',root,'rev')
            with self.assertRaises(ValueError):freeze_sources([p],outside/'snap',outside,'rev')
            self.assertFalse((root/'snap').exists());self.assertFalse((outside/'snap').exists())

    def test_final_live_import_enumeration_uses_captured_namespace_and_checks_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);p=root/'src'/'child.py';p.parent.mkdir();p.write_bytes(b'physics')
            m=freeze_sources([p],root/'sources',root,'rev')
            module=types.ModuleType('frozen_test_fixture.child');module.__file__=str(p)
            with patch.dict(sys.modules,{'frozen_test_fixture.child':module}):
                self.assertEqual(frozen_import_sources(m,root/'src','frozen_test_fixture'),[m['records'][0]['snapshot']])
                q=root/'untracked.py';q.write_bytes(b'external');module.__file__=str(q)
                with self.assertRaises(ValueError):frozen_import_sources(m,root/'src','frozen_test_fixture')
                module.__file__=str(p);p.write_bytes(b'changed')
                with self.assertRaises(ValueError):frozen_import_sources(m,root/'src','frozen_test_fixture')
