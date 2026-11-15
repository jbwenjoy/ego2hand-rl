import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources
from supermimic_research.runtime.source_validation_epoch import SourceValidationEpoch,validate_state_sources


@unittest.skipUnless(os.name=='nt','Windows file-sharing contract')
class SourceEpochTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.source=self.root/'source.py';self.source.write_bytes(b'fixed source')
        self.engine=self.root/'engine.py';self.engine.write_bytes(b'fixed engine');self.manifest=freeze_sources([self.source],self.root/'snapshot',self.root,'test');self.engines=[file_pin(self.engine)]
    def tearDown(self):self.temp.cleanup()

    def test_default_still_hashes_and_rejects_changed_source(self):
        validate_state_sources(self.manifest,self.engines);self.source.write_bytes(b'changed data')
        with self.assertRaises(ValueError):validate_state_sources(self.manifest,self.engines)

    def test_active_epoch_blocks_writes_deletes_and_replacements_then_releases(self):
        replacement=self.root/'replacement.py';replacement.write_bytes(b'replacement')
        with SourceValidationEpoch(self.manifest,self.engines) as epoch:
            for pin in [self.manifest['records'][0]['observed'],self.manifest['records'][0]['snapshot'],self.engines[0]]:
                p=Path(pin['path'])
                with self.assertRaises(PermissionError):p.write_bytes(b'bad')
                with self.assertRaises(PermissionError):p.unlink()
                with self.assertRaises(PermissionError):os.replace(replacement,p)
            epoch.verify(self.manifest,self.engines)
        self.assertTrue(epoch.validated);self.source.write_bytes(b'allowed after epoch')

    def test_mutated_or_foreign_manifest_and_expired_tokens_reject(self):
        with SourceValidationEpoch(self.manifest,self.engines) as epoch:
            wrong=copy.deepcopy(self.manifest);wrong['code_revision']='another revision'
            with self.assertRaisesRegex(ValueError,'bound'):epoch.verify(wrong,self.engines)
            wrong=copy.deepcopy(self.engines);wrong[0]['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'bound'):epoch.verify(self.manifest,wrong)
        with self.assertRaisesRegex(ValueError,'expired'):epoch.verify(self.manifest,self.engines)
        with self.assertRaisesRegex(ValueError,'single-use'):epoch.__enter__()

    def test_bad_hash_releases_all_already_acquired_handles(self):
        wrong=copy.deepcopy(self.manifest);wrong['records'][0]['observed']['sha256']='0'*64;wrong['records'][0]['snapshot']['sha256']='0'*64
        with self.assertRaises(ValueError):
            with SourceValidationEpoch(wrong,self.engines):pass
        self.source.write_bytes(b'unlocked after failure');self.engine.write_bytes(b'unlocked engine')

    def test_body_exception_does_not_commit_and_releases(self):
        epoch=SourceValidationEpoch(self.manifest,self.engines)
        with self.assertRaisesRegex(RuntimeError,'body'):
            with epoch:raise RuntimeError('body')
        self.assertFalse(epoch.validated);self.source.write_bytes(b'unlocked')

    def test_each_use_checks_path_identity_without_rehashing(self):
        import supermimic_research.runtime.source_validation_epoch as module
        with patch.object(module,'verify_source_snapshot',wraps=module.verify_source_snapshot) as verifier:
            with SourceValidationEpoch(self.manifest,self.engines) as epoch:
                for _ in range(3):validate_state_sources(self.manifest,self.engines,epoch)
                self.assertEqual(verifier.call_count,1)
                with patch.object(module,'_stamp',return_value=(-1,-1,-1,-1)):
                    with self.assertRaisesRegex(ValueError,'identity'):epoch.verify(self.manifest,self.engines)
            self.assertEqual(verifier.call_count,2)
        self.assertEqual(epoch.verify_calls,3)

    def test_untyped_validation_shortcut_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'SourceValidationEpoch'):validate_state_sources(self.manifest,self.engines,object())

    def test_noncanonical_alias_is_not_silently_rebound(self):
        wrong=copy.deepcopy(self.manifest);wrong['records'][0]['observed']['path']=str(self.root)+'\\..\\'+self.root.name+'\\source.py'
        with self.assertRaisesRegex(ValueError,'canonical'):
            with SourceValidationEpoch(wrong,self.engines):pass


if __name__=='__main__':unittest.main()
