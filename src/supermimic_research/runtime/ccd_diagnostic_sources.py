"""Explicit isolated CCD source recipes; installed bytes stay untouched."""
import re


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('One exact source site required: ' + old[:90])
    return source.replace(old, new)


def traced_gjk(source):
    edits = [
        ('  if pt.nface == pt.face.shape[0]:\n    return 0.0',
         '  if pt.nface == pt.face.shape[0]:\n    wp.printf("TRACE_ATTACH_FACE_CAPACITY nface=%d capacity=%d\\n", pt.nface, pt.face.shape[0])\n    return 0.0'),
        ('  if ret:\n    return 0.0\n\n  # ensure projection points outward',
         '  if ret:\n    wp.printf("TRACE_ATTACH_FACE_PROJECTION ret=%d nface=%d\\n", ret, pt.nface)\n    return 0.0\n\n  # ensure projection points outward'),
        ('  if result.dist > tolerance or result.dim < 2:\n    return False, result.dist, 1, result.x1, result.x2, empty, geom1, geom2',
         '  if result.dist > tolerance or result.dim < 2:\n    wp.printf("TRACE_GJK_NO_EPA dim=%d dist=%.12g tol=%.12g\\n", result.dim, result.dist, tolerance)\n    return False, result.dist, 1, result.x1, result.x2, empty, geom1, geom2'),
        ('  return True, result.dist, 1, result.x1, result.x2, result, geom1, geom2',
         '  wp.printf("TRACE_GJK_EPA dim=%d dist=%.12g tol=%.12g\\n", result.dim, result.dist, tolerance)\n  return True, result.dist, 1, result.x1, result.x2, result, geom1, geom2'),
        ('  if pt.status:\n    return result.dist, 1, result.x1, result.x2, -1',
         '  if pt.status:\n    wp.printf("TRACE_EPA_INIT_STATUS world=%d dim=%d status=%d dist=%.12g nface=%d\\n", worldid, result.dim, pt.status, result.dist, pt.nface)\n    return result.dist, 1, result.x1, result.x2, -1'),
        ('  if idx == -1:\n    return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), -1',
         '  if idx == -1:\n    wp.printf("TRACE_EPA_INVALID world=%d nface=%d capacity=%d\\n", worldid, pt.nface, pt.face.shape[0])\n    return FLOAT_MAX, 0, wp.vec3(), wp.vec3(), -1')]
    for old, new in edits: source = replace_once(source, old, new)
    return source, edits


def convex_source(source, *, horizon, face_factor=5, gjk_module=None):
    if horizon not in [24, 48, 96] or face_factor not in [5, 20, 40]:
        raise ValueError('Registered isolated scratch sizes required')
    edits = []
    old = 'epa_horizon = wp.empty(shape=(d.naccdmax, MJ_MAX_EPAHORIZON), dtype=int)'
    new = f'epa_horizon = wp.empty(shape=(d.naccdmax, {horizon}), dtype=int)'
    source = replace_once(source, old, new); edits.append((old, new))
    for field, dtype in [('epa_face', 'int'), ('epa_pr', 'wp.vec3'), ('epa_norm2', 'float')]:
        old = f'{field} = wp.empty(shape=(d.naccdmax, 6 + MJ_MAX_EPAFACES * epa_iterations), dtype={dtype})'
        new = f'{field} = wp.empty(shape=(d.naccdmax, 6 + {face_factor} * epa_iterations), dtype={dtype})'
        source = replace_once(source, old, new); edits.append((old, new))
    if gjk_module:
        if not re.fullmatch(r'[a-z][a-z0-9_]*', gjk_module): raise ValueError('Explicit isolated module identifier required')
        old = 'from mujoco_warp._src.collision_gjk import '
        new = f'from {gjk_module} import '
        if source.count(old) != 5: raise ValueError('Exactly five GJK imports required')
        source = source.replace(old, new); edits.append((old, new))
    return source, edits
