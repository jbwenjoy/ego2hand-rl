"""Full-rehash verifier with one canonical path resolution per call.

This is an explicit candidate,not the default artifact or native verifier.
No byte/hash/mtime cache or source-validation epoch is used. As with the
original verifier,concurrent filesystem mutations are not an atomic snapshot.
"""
import hashlib
from pathlib import Path


def verify_file_pin_single_resolve(pin):
    target=Path(pin['path']).resolve(strict=True)
    digest=hashlib.sha256()
    with target.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    if target.stat().st_size!=pin['bytes'] or digest.hexdigest()!=pin['sha256']:
        raise ValueError(f'Artifact bytes changed: {target}')
    return target
