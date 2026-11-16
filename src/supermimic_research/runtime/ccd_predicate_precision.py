"""Isolated FP64 tetrahedron predicates with FP32 shapes/state/witness storage."""
from .ccd_diagnostic_sources import replace_once


CAST = '''@wp.func
def _predicate_vec3d(v: wp.vec3) -> wp.vec3d:
  return wp.vec3d(wp.float64(v[0]), wp.float64(v[1]), wp.float64(v[2]))


@wp.func
def _predicate_det3d(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float64:
  return wp.dot(_predicate_vec3d(a), wp.cross(_predicate_vec3d(b), _predicate_vec3d(c)))


'''


def predicate_source(source, *, side, simplex):
    if not side and not simplex: raise ValueError('At least one explicit predicate variant required')
    edits = []
    if simplex:
        old = '@wp.func\ndef _S3D(s1: wp.vec3, s2: wp.vec3, s3: wp.vec3, s4: wp.vec3) -> wp.vec4:'
        source = replace_once(source, old, CAST + old); edits.append((old, CAST + old))
        old = '''  C41 = -_det3(s2, s3, s4)
  C42 = _det3(s1, s3, s4)
  C43 = -_det3(s1, s2, s4)
  C44 = _det3(s1, s2, s3)'''
        new = old.replace('_det3(', '_predicate_det3d(')
        source = replace_once(source, old, new); edits.append((old, new))
        for i in range(1, 5):
            old = f'  comp{i} = _same_sign(m_det, C4{i})'
            new = f'  comp{i} = (m_det > 0.0 and C4{i} > 0.0) or (m_det < 0.0 and C4{i} < 0.0)'
            source = replace_once(source, old, new); edits.append((old, new))
        old = '    return wp.vec4(C41 / m_det, C42 / m_det, C43 / m_det, C44 / m_det)'
        new = '    return wp.vec4(wp.float32(C41 / m_det), wp.float32(C42 / m_det), wp.float32(C43 / m_det), wp.float32(C44 / m_det))'
        source = replace_once(source, old, new); edits.append((old, new))
    elif side:
        old = '@wp.func\ndef _same_side('
        source = replace_once(source, old, CAST + old); edits.append((old, CAST + old))
    if side:
        old = '''  n = wp.cross(p1 - p0, p2 - p0)
  dot1 = wp.dot(n, p3 - p0)
  dot2 = wp.dot(n, -p0)'''
        new = '''  a = _predicate_vec3d(p0)
  b = _predicate_vec3d(p1)
  c = _predicate_vec3d(p2)
  d = _predicate_vec3d(p3)
  n = wp.cross(b - a, c - a)
  dot1 = wp.dot(n, d - a)
  dot2 = wp.dot(n, -a)'''
        source = replace_once(source, old, new); edits.append((old, new))
    return source, edits
