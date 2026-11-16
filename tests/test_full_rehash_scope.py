from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from supermimic_research import artifacts,frozen_sources
from supermimic_research.runtime import source_validation_epoch
from supermimic_research.runtime.full_rehash_scope import FullRehashVerifierScope,implementation_pins,ORIGINAL


class RehashScopeTests(unittest.TestCase):
    def test_both_bindings_rehash_each_call_and_default_never_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'data';p.write_bytes(b'abc');pin=artifacts.file_pin(p)
            scope=FullRehashVerifierScope('SINGLE_RESOLVE_FULL_REHASH',implementation_pins())
            with scope:
                self.assertIs(artifacts.verify_file_pin,ORIGINAL)
                self.assertEqual(frozen_sources.verify_file_pin(pin),p.resolve())
                self.assertEqual(source_validation_epoch.verify_file_pin(pin),p.resolve())
                p.write_bytes(b'xyz')
                with self.assertRaises(ValueError):frozen_sources.verify_file_pin(pin)
            self.assertEqual(scope.calls,3);self.assertTrue(scope.validated);self.assertTrue(scope.audit()['bindings_restored'])

    def test_exception_restores_both_without_validating_failed_transaction(self):
        scope=FullRehashVerifierScope('ORIGINAL_FULL_REHASH',implementation_pins())
        with self.assertRaises(RuntimeError):
            with scope:raise RuntimeError('expected')
        self.assertFalse(scope.validated);self.assertTrue(scope.audit()['bindings_restored'])

    def test_nested_or_unexpected_binding_rejects_before_installing(self):
        with FullRehashVerifierScope('ORIGINAL_FULL_REHASH',implementation_pins()):
            with self.assertRaises(ValueError):
                with FullRehashVerifierScope('SINGLE_RESOLVE_FULL_REHASH',implementation_pins()):pass
        with patch.object(frozen_sources,'verify_file_pin',lambda p:None):
            with self.assertRaises(ValueError):
                with FullRehashVerifierScope('SINGLE_RESOLVE_FULL_REHASH',implementation_pins()):pass
        self.assertIs(frozen_sources.verify_file_pin,ORIGINAL);self.assertIs(source_validation_epoch.verify_file_pin,ORIGINAL)

    def test_mid_scope_binding_tamper_is_rejected_and_cleanup_restores_original(self):
        scope=FullRehashVerifierScope('SINGLE_RESOLVE_FULL_REHASH',implementation_pins())
        with self.assertRaises(ValueError):
            with scope:frozen_sources.verify_file_pin=lambda p:None
        self.assertFalse(scope.validated);self.assertIsNotNone(scope.postcheck_error)
        self.assertTrue(scope.audit()['bindings_restored'])
