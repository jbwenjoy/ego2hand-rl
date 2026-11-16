import os
from pathlib import Path
import tempfile
import unittest
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_file_verification import verify_file_pin_single_resolve


class FullRehashVerifierTests(unittest.TestCase):
    def test_original_and_candidate_resolve_relative_dot_and_large_multichunk_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);sub=root/'sub';sub.mkdir();p=root/'large.bin';p.write_bytes(b'x'*(1024*1024)+b'last-chunk')
            pin=file_pin(p);pin['path']=str(sub/'..'/'large.bin')
            self.assertEqual(verify_file_pin(pin),verify_file_pin_single_resolve(pin))
            p.write_bytes(b'x'*(1024*1024)+b'last-chunX')
            for verifier in [verify_file_pin,verify_file_pin_single_resolve]:
                with self.assertRaises(ValueError):verifier(pin)

    def test_same_length_content_change_with_original_mtime_is_not_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'source.py';p.write_bytes(b'abc');pin=file_pin(p);stamp=p.stat()
            self.assertEqual(verify_file_pin_single_resolve(pin),p.resolve())
            p.write_bytes(b'xyz');os.utime(p,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
            for verifier in [verify_file_pin,verify_file_pin_single_resolve]:
                with self.assertRaises(ValueError):verifier(pin)

    def test_missing_size_and_digest_fail_like_original(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'source';p.write_bytes(b'payload');pin=file_pin(p)
            for bad in [{**pin,'bytes':pin['bytes']+1},{**pin,'sha256':'0'*64}]:
                for verifier in [verify_file_pin,verify_file_pin_single_resolve]:
                    with self.assertRaises(ValueError):verifier(bad)
            p.unlink()
            for verifier in [verify_file_pin,verify_file_pin_single_resolve]:
                with self.assertRaises(FileNotFoundError):verifier(pin)
