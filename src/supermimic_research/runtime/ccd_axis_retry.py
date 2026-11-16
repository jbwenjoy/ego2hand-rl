"""Bounded isolated cylinder/mesh CCD retries; no shape/contact fabrication."""
from .ccd_diagnostic_sources import replace_once


def axis_retry_source(source):
    old = '''    if dist >= gap and not is_collision_sensor:
      return

    # CCD operates on margin-inflated shapes'''
    new = '''    # Isolated cylinder/mesh recovery: only an ambiguous EPA result
    # retries. Original GJK-separated pairs and valid contacts are untouched.
    if wp.static(geomtype1 == GeomType.CYLINDER and geomtype2 == GeomType.MESH):
      if needs_epa and not is_collision_sensor and dist >= gap:
        for retry_axis in range(3):
          initial = wp.vec3()
          initial[retry_axis] = 1.0
          retry_geom1 = geom1
          retry_geom2 = geom2
          retry_geom1.index = -1
          retry_geom2.index = -1
          retry_dist, retry_count, retry_w1, retry_w2, retry_idx = ccd(
            tolerance, cutoff, gjk_iterations, epa_iterations,
            retry_geom1, retry_geom2, geomtype1, geomtype2,
            initial, wp.vec3(), epa_vert_in[ccdid], epa_vert_index_in[ccdid],
            epa_face_in[ccdid], epa_pr_in[ccdid], epa_norm2_in[ccdid],
            epa_horizon_in[ccdid], wp.static(warn_overflow), worldid, overflow_out,
          )
          wp.printf("TRACE_AXIS_RETRY world=%d axis=%d dist=%.12g count=%d\\n", worldid, retry_axis, retry_dist, retry_count)
          if retry_count > 0 and wp.isfinite(retry_dist) and retry_dist < gap:
            dist = retry_dist
            ncollision = retry_count
            w1 = retry_w1
            w2 = retry_w2
            multiccd_idx = retry_idx
            break

    if dist >= gap and not is_collision_sensor:
      return

    # CCD operates on margin-inflated shapes'''
    return replace_once(source, old, new), dict(old=old, new=new,
        scope='At most3 same-geometry CCD retries for ambiguous cylinder/mesh EPA only. Original flags preserved. Diagnostic printf counts attempts; no production admission.')
