"""Explicit external data paths with content-pin validation.

Legacy JSON source fields are provenance, never Python import selectors. A
caller may relocate a data artifact through an explicit path override, while
the original bytes/sha256 still have to match. No repository/study discovery.
"""
from pathlib import Path
import hashlib


def checked_data_pin(expected, *, override=None):
    path = Path(expected['path'] if override is None else override).resolve(strict=True)
    data = path.read_bytes()
    if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
        raise ValueError('External data artifact content pin changed: ' + str(path))
    return path
