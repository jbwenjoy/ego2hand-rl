"""Pinned saved JSON checks; compiled-vertex check uses caller's same model only."""
from pathlib import Path
import hashlib
import json

SCHEMA = 'source_region_filled_triangle_paired_surface_anchors:11'

def pin(path):
    path = Path(path).resolve(strict=True)
    b = path.read_bytes()
    return {'path': str(path), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def validate_anchor_bundle(path, expected_pin):
    import numpy as np
    record = pin(path)
    if record != expected_pin:
        raise ValueError('Exact protocol expected anchor pin')
    a = json.loads(Path(path).read_text(encoding='utf8'))
    if a['schema'] != SCHEMA:
        raise ValueError('New source-region anchor namespace')
    for p in a['input_pins'].values():
        if pin(p['path']) != p:
            raise ValueError('Anchor input changed')
    old = json.loads(Path(a['input_pins']['original_anchors']['path']).read_text(encoding='utf8'))
    if a['source_pose'] != old['source_pose']:
        raise ValueError('Original fixed source pose and remaining12q')
    if not a['checks'] or any(type(v) is not bool or v is not True for v in a['checks'].values()):
        raise ValueError('Completed producer checks, not sufficient by themselves')
    rows = a['anchors']
    if len(rows) != 2 or [r['finger'] for r in rows] != ['thumb', 'index']:
        raise ValueError('Exactly anatomical thumb/index order')
    checks = []
    for i, row in enumerate(rows):
        if row['hand_geom_id'] != [15, 28][i] or row['phone_geom_id'] != 71:
            raise ValueError('Original hand and phone geom IDs')
        if row['surfaces']['hand'] != old['anchors'][i]['surfaces']['hand']:
            raise ValueError('These two branches preserve the original actual hand triangles')
        if row['original_ego_material_target_phone_body_m'] != old['anchors'][i]['original_ego_material_target_phone_body_m']:
            raise ValueError('Keep original ego target, separately from new selected B')
        for side, gid in [('hand', [15, 28][i]), ('phone', 71)]:
            s = row['surfaces'][side]
            tri = np.asarray(s['source_triangle_vertices_geom_local_m'], dtype=np.float64)
            w = np.asarray(s['barycentric_in_source_vertex_order'], dtype=np.float64)
            point = np.asarray(s['projected_point_geom_local_m'], dtype=np.float64)
            normal = np.asarray(s['outward_normal_geom_local'], dtype=np.float64)
            ids = s['source_triangle_vertex_ids']
            if (s['geom_id'] != gid or len(ids) != 3 or len(set(ids)) != 3
                    or any(type(v) is not int or v < 0 for v in ids)
                    or type(s['source_triangle_id']) is not int or s['source_triangle_id'] < 0
                    or tri.shape != (3, 3) or w.shape != (3,) or point.shape != (3,) or normal.shape != (3,)
                    or not np.isfinite(np.r_[tri.ravel(), w, point, normal]).all()):
                raise ValueError('Finite actual triangle and source identity')
            cr = np.cross(tri[1]-tri[0], tri[2]-tri[0])
            area2 = float(np.linalg.norm(cr))
            if (area2 <= 0 or abs(float(np.linalg.norm(normal))-1) > 1e-8
                    or abs(float(w.sum())-1) > 1e-12 or float(w.min()) < -1e-12
                    or float(np.max(abs(w @ tri-point))) > 1e-12
                    or float(np.max(abs((tri-point) @ normal))) > 1e-12
                    or abs(abs(float(cr @ normal))/area2-1) > 1e-8):
                raise ValueError('Nondegenerate filled triangle/barycentric/unit plane normal')
            if side == 'phone':
                bt = np.asarray(s['source_triangle_vertices_phone_body_m'], dtype=np.float64)
                bp = np.asarray(s['projected_point_phone_body_m'], dtype=np.float64)
                bn = np.asarray(s['outward_normal_phone_body'], dtype=np.float64)
                if (bt.shape != (3, 3) or bp.shape != (3,) or bn.shape != (3,)
                        or not np.isfinite(np.r_[bt.ravel(), bp, bn]).all()
                        or np.max(abs(w @ bt-bp)) > 1e-12
                        or abs(np.linalg.norm(bn)-1) > 1e-8
                        or np.max(abs((bt-bp) @ bn)) > 1e-12):
                    raise ValueError('Phone body-coordinate triangle and true normal')
            checks.append({'finger': row['finger'], 'side': side, 'triangle_id': s['source_triangle_id'],
                           'barycentric_reconstruction_max_abs_m': float(np.max(abs(w @ tri-point)))})
    return a, {'status': 'PASS_PINNED_JSON_TRIANGLE_CONTRACT', 'anchor_pin': record,
               'triangles': checks, 'compiled_mesh_support_checked': False,
               'scope': 'No model/FK/query/NPZ payload. Global supporting normals require same compiled model at constructor.'}

def verify_compiled_anchors(model, mj, anchors, phone_body_id):
    import numpy as np
    evidence = []
    for row in anchors['anchors']:
        for side in ['hand', 'phone']:
            s = row['surfaces'][side]
            gid = s['geom_id']
            mid = int(model.geom_dataid[gid])
            if model.geom_type[gid] != int(mj.mjtGeom.mjGEOM_MESH) or mid < 0:
                raise ValueError('Original mesh collider only')
            adr, count = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
            V = np.asarray(model.mesh_vert[adr:adr+count], dtype=np.float64)
            ids = s['source_triangle_vertex_ids']
            if max(ids) >= count:
                raise ValueError('Source compiled vertex ID outside original mesh')
            tri = np.asarray(s['source_triangle_vertices_geom_local_m'], dtype=np.float64)
            point = np.asarray(s['projected_point_geom_local_m'], dtype=np.float64)
            normal = np.asarray(s['outward_normal_geom_local'], dtype=np.float64)
            if not np.array_equal(V[ids], tri):
                raise ValueError('Actual saved FP32 triangle vertices must match compiled mesh values')
            support_max = float(np.max((V-point) @ normal))
            if support_max > 1e-12:
                raise ValueError('Normal is not supporting outward for all original compiled vertices')
            rotation = np.empty(9, dtype=np.float64)
            mj.mju_quat2Mat(rotation, model.geom_quat[gid])
            rotation = rotation.reshape(3, 3)
            if side == 'phone':
                if int(model.geom_bodyid[gid]) != int(phone_body_id):
                    raise ValueError('Original phone body identity')
                p = np.asarray(model.geom_pos[gid], dtype=np.float64)
                if (np.max(abs(tri @ rotation.T+p-np.asarray(s['source_triangle_vertices_phone_body_m']))) > 1e-12
                        or np.max(abs(rotation @ point+p-np.asarray(s['projected_point_phone_body_m']))) > 1e-12
                        or np.max(abs(rotation @ normal-np.asarray(s['outward_normal_phone_body']))) > 1e-8):
                    raise ValueError('Original geom-to-phone-body transform')
            evidence.append({'finger': row['finger'], 'side': side, 'geom_id': gid,
                             'compiled_vertices_checked': count, 'support_plane_max_m': support_max})
    return {'status': 'PASS_SAME_MODEL_COMPILED_VERTICES_AND_OUTWARD_SUPPORT',
            'evidence': evidence, 'new_model_loads_FK_forward_queries': 0}
