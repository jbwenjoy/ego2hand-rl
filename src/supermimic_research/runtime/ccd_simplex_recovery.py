"""Isolated geometric fallback recipe, not production backend admission."""
from .ccd_diagnostic_sources import replace_once


def retry_inconsistent_tetrahedron(source):
    old = '''    if dist[idx] > MINVAL:
      pt.status = 12
      return pt, GJKResult()

    # fallback to closest face'''
    new = '''    # Isolated recovery variant: GJK selected a four-point simplex,
    # but the independent tetrahedron test disagrees. Reuse the existing
    # closest-face reconstruction instead of declaring a zero-depth boundary.
    # The next EPA initialization may still reject it; no contact is invented.
    # fallback to closest face'''
    return replace_once(source, old, new), dict(old=old, new=new,
        scope='Algorithm variant for derived-fixture evaluation only; broad collision/state/gradient admission remains required.')
