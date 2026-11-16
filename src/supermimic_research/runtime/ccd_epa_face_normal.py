"""Isolated same-EPA-face normal for cylinder/mesh single-contact CCD."""
from .ccd_diagnostic_sources import replace_once


def retain_cylinder_mesh_face(source):
    old='''  if (geomtype1 != GeomType.BOX and geomtype1 != GeomType.MESH) or (geomtype2 != GeomType.BOX and geomtype2 != GeomType.MESH):
    idx = -1'''
    new='''  if ((geomtype1 != GeomType.BOX and geomtype1 != GeomType.MESH) or (geomtype2 != GeomType.BOX and geomtype2 != GeomType.MESH)) and not (geomtype1 == GeomType.CYLINDER and geomtype2 == GeomType.MESH):
    idx = -1'''
    return replace_once(source,old,new),dict(old=old,new=new,scope='Expose selected EPA face for cylinder/mesh;convex static multicontact gate still excludes cylinders.')


def face_frame_source(source):
    old='''    frame = make_frame(witness1[0] - witness2[0])
    if is_collision_sensor:'''
    new='''    frame = make_frame(witness1[0] - witness2[0])
    # Distance already uses this EPA face's projected origin. Use its
    # direction directly instead of subtracting two rounded WORLD witnesses.
    if wp.static(geomtype1 == GeomType.CYLINDER and geomtype2 == GeomType.MESH):
      if ccdid >= 0 and multiccd_idx >= 0:
        frame = make_frame(epa_pr_in[ccdid, multiccd_idx])
    if is_collision_sensor:'''
    return replace_once(source,old,new),dict(old=old,new=new,scope='Single-contact normal from same EPA face as signed depth;positions/materials unchanged.')
