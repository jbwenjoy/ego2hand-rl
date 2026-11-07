"""All61/all788 saved unsigned proximity descriptors; no FK or new query."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
THRESHOLDS = [.001, .002, .005]
QUANTILES = [0., .01, .05, .25, .5, .75, .95, 1.]
DIGITS = ['thumb', 'index', 'middle', 'ring', 'pinky']
BONE_NAMES = ['root_frame0', 'wrist_frame1'] + [f'{f}_chain_{c}' for f in DIGITS for c in 'ABC']
COARSE_NAMES = ['root_wrist_influenced'] + DIGITS
INPUTS = {
    'geometry': P/'ego_hand_geometry_v1/left_hand_phone_geometry61.npz',
    'proximity': P/'ego_hand_geometry_v1/proximity_candidates61.npz',
    'profile': P/'ego_hand_geometry_v1/raw_profile_and_poses.json',
    'source_report': P/'ego_hand_geometry_v1/report.json',
    'geometry_source': P/'ego_hand_geometry_v1/build_ego1856_left_geometry_v1.py',
    'official_skinning_source': P/'ego_hand_geometry_v1/vendor_umetrack_9064a2/hand_models/umetrack_skinning.py',
    'official_UmeTrack_source': P/'ego_hand_geometry_v1/vendor_umetrack_9064a2/hand_models/umetrack_hand_model.py',
    'official_canonical_digit_mapping_source': P/'retarget_geometry_v1/official_mano_mapping_source_readonly.py',
    'mapping_receipt': P/'retarget_geometry_v1/mapping_source.json'}


def pin(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, obj):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def stats(values, inside):
    a = np.asarray(values, np.float64).reshape(-1)
    inside = np.asarray(inside).reshape(-1)
    return {'vertex_observations': len(a), 'minimum_unsigned_m': float(a.min()) if len(a) else None,
            'quantile_unsigned_m': {str(q): float(np.quantile(a, q, method='linear')) if len(a) else None for q in QUANTILES},
            'count_at_or_below': {f'{int(t*1000)}mm': int((a <= t).sum()) for t in THRESHOLDS},
            'convex_envelope_inside_count': int((inside < 0).sum()),
            'near_and_convex_envelope_inside_count': {f'{int(t*1000)}mm': int(((a <= t) & (inside < 0)).sum()) for t in THRESHOLDS}}


def adjacency_from_faces(faces, size):
    adjacency = [set() for _ in range(size)]
    for a, b, c in faces:
        adjacency[a].update((int(b), int(c)))
        adjacency[b].update((int(a), int(c)))
        adjacency[c].update((int(a), int(b)))
    return adjacency


def components(mask, adjacency):
    todo = set(np.flatnonzero(mask).tolist())
    out = []
    while todo:
        start = min(todo)
        todo.remove(start)
        queue = [start]
        ids = []
        while queue:
            a = queue.pop()
            ids.append(a)
            for b in sorted(adjacency[a]):
                if b in todo:
                    todo.remove(b)
                    queue.append(b)
        out.append(np.array(sorted(ids), np.int64))
    return out


def main():
    protocol = read(ROOT/'frozen_rules_v1.json')
    if protocol['source'] != pin(__file__) or protocol['thresholds_m'] != THRESHOLDS or protocol['quantiles'] != QUANTILES:
        raise ValueError('Source or descriptive rules changed after locking')
    for k, p in INPUTS.items():
        if protocol['inputs'][k] != pin(p):
            raise ValueError('Saved source changed: '+k)
    if (ROOT/'report.json').exists():
        raise FileExistsError('No automatic rerun/overwrite')
    profile = read(INPUTS['profile'])['umetrack_profile']
    weights = np.asarray(profile['dense_bone_weights'], np.float64)
    if weights.shape != (788, 17) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('Saved dense skinning ABI')
    rowsum = weights.sum(1)
    if not np.all(rowsum > 0) or np.max(np.abs(rowsum-1)) > 1e-5:
        raise ValueError('Unexpected weight normalization, no repair')
    # Verify official frame ordering against the actual profile's canonical fingertip bones.
    if np.asarray(profile['landmark_rest_bone_indices'])[:5, 0].tolist() != [4, 7, 10, 13, 16]:
        raise ValueError('Canonical fingertip/frame mapping differs')
    if not np.array_equal(np.asarray(profile['joint_frame_index']), [255,2,3,4,255,5,6,7,255,8,9,10,255,11,12,13,255,14,15,16,255,0]):
        raise ValueError('Actual joint/frame order differs')
    tied = weights == weights.max(1, keepdims=True)
    bone = np.where(tied.sum(1) == 1, weights.argmax(1), -1).astype(np.int64)
    frame_to_coarse = np.array([0,0]+[d+1 for d in range(5) for _ in range(3)])
    coarse = np.full(788, -1, np.int64)
    for v in range(788):
        c = np.unique(frame_to_coarse[np.flatnonzero(tied[v])])
        if len(c) == 1:
            coarse[v] = c[0]
    sorted_weights = np.sort(weights, axis=1)
    assignments = []
    for v in range(788):
        assignments.append({'vertex_id':v,'dominant_bone':int(bone[v]),'bone_region':BONE_NAMES[bone[v]] if bone[v]>=0 else 'unknown_exact_tie',
                            'coarse_region':COARSE_NAMES[coarse[v]] if coarse[v]>=0 else 'unknown_cross_region_tie',
                            'all_tied_dominant_bones':np.flatnonzero(tied[v]).tolist(),
                            'dominant_weight':float(sorted_weights[v,-1]),'dominance_margin':float(sorted_weights[v,-1]-sorted_weights[v,-2]),
                            'positive_weight_bones':np.flatnonzero(weights[v]>0).tolist(),
                            'mixed_weight_boundary_descriptor':bool((weights[v]>0).sum()>1), 'raw_weights17':weights[v].tolist()})
    write(ROOT/'vertex_region_assignments788.json', assignments)
    with np.load(INPUTS['geometry'], allow_pickle=False) as z:
        ids = z['frame_ids'].copy(); vertices = z['hand_vertices_object_F64'].copy(); faces = z['hand_triangles'].copy()
    with np.load(INPUTS['proximity'], allow_pickle=False) as z:
        distance = z['hand_vertex_unsigned_scan_distance_m'].copy()
        closest = z['hand_vertex_closest_scan_point_object_m'].copy()
        tri = z['hand_vertex_closest_scan_triangle'].copy()
        inside = z['hand_vertex_convex_max_plane_signed_m'].copy()
        landmark_distance = z['landmark21_unsigned_scan_distance_m'].copy()
    if ids.dtype != np.int64 or not np.array_equal(ids, np.arange(42,103)) or vertices.shape != (61,788,3) or distance.shape != (61,788) or closest.shape != vertices.shape or inside.shape != distance.shape or tri.shape != distance.shape:
        raise ValueError('All61/all788 saved ABI')
    if faces.shape != (1544,3) or faces.dtype.kind not in 'iu' or faces.min()<0 or faces.max()>=788:
        raise ValueError('Saved mesh topology ABI')
    if not all(np.isfinite(a).all() for a in (vertices,distance,closest,inside,landmark_distance)) or (distance<0).any() or (tri<0).any():
        raise ValueError('Invalid saved values; do not filter')
    # This is a saved-coordinate coherence check, not a new point-to-mesh query.
    coordinate_distance = np.linalg.norm(vertices-closest, axis=-1)
    if np.max(abs(coordinate_distance-distance)) > 1e-12:
        raise ValueError('Saved closest-point/distance mismatch')
    regions = [('ALL', np.ones(788,bool), 'all_mesh')]
    regions += [(name,bone==b,'dominant_bone') for b,name in enumerate(BONE_NAMES)]
    regions += [('unknown_exact_tie',bone<0,'dominant_bone')]
    regions += [(name,coarse==b,'coarse_folded_dominant_bone') for b,name in enumerate(COARSE_NAMES)]
    regions += [('unknown_cross_region_tie',coarse<0,'coarse_folded_dominant_bone')]
    by_frame = []
    temporal = []
    for name,mask,scheme in regions:
        for i,frame in enumerate(ids):
            by_frame.append({'frame_id':int(frame),'region':name,'scheme':scheme,'static_vertex_count':int(mask.sum()),**stats(distance[i,mask],inside[i,mask])})
        minima = [float(distance[i,mask].min()) if mask.any() else None for i in range(61)]
        temporal.append({'region':name,'scheme':scheme,'static_vertex_count':int(mask.sum()),
                         'pooled61_vertex_observations':stats(distance[:,mask],inside[:,mask]),
                         'frames_with_any_at_or_below':{f'{int(t*1000)}mm':int((distance[:,mask].min(1)<=t).sum()) if mask.any() else 0 for t in THRESHOLDS},
                         'minimum_unsigned_by_all61_frame_m':minima})
    write(ROOT/'all61_region_statistics.json',by_frame)
    write(ROOT/'all61_region_temporal_summary.json',temporal)
    with (ROOT/'all61_region_statistics.csv').open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(by_frame[0]));w.writeheader()
        for row in by_frame:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    adjacency=adjacency_from_faces(faces,788)
    comp_rows=[]
    for i,frame in enumerate(ids):
        for threshold in THRESHOLDS:
            comp=[]
            for c in components(distance[i]<=threshold,adjacency):
                comp.append({'vertex_ids':c.tolist(),'vertex_count':len(c),'minimum_unsigned_m':float(distance[i,c].min()),
                             'coarse_region_counts':{name:int((coarse[c]==j).sum()) for j,name in enumerate(COARSE_NAMES)},
                             'unknown_cross_region_tie_count':int((coarse[c]<0).sum()),
                             'closest_scan_point_object_AABB_m':[closest[i,c].min(0).tolist(),closest[i,c].max(0).tolist()],
                             'convex_envelope_inside_count':int((inside[i,c]<0).sum())})
            comp_rows.append({'frame_id':int(frame),'threshold_m':threshold,'sampled_induced_mesh_component_count':len(comp),'components':comp})
    write(ROOT/'all61_threshold_mesh_components.json',comp_rows)
    frame42=[]
    # Preserve ALL region minima/ties and every frame42 vertex below5mm, not selected contacts.
    for name,mask,scheme in regions:
        row={'region':name,'scheme':scheme,'statistics':stats(distance[0,mask],inside[0,mask]),'minimum_tie_vertex_ids':[],'minimum_tie_scan_points_object_m':[]}
        if mask.any():
            mm=distance[0,mask].min();mi=np.flatnonzero(mask & (distance[0]==mm))
            row.update(minimum_tie_vertex_ids=mi.tolist(),minimum_tie_scan_points_object_m=closest[0,mi].tolist())
        frame42.append(row)
    write(ROOT/'frame42_all_region_minima_and_ties.json',frame42)
    near=[]
    for v in np.flatnonzero(distance[0]<=max(THRESHOLDS)):
        near.append({**assignments[int(v)],'unsigned_scan_distance_m':float(distance[0,v]),'hand_vertex_object_m':vertices[0,v].tolist(),
                     'closest_original_scan_material_point_object_m':closest[0,v].tolist(),'closest_original_scan_triangle':int(tri[0,v]),
                     'convex_envelope_max_plane_signed_m':float(inside[0,v]),'convex_envelope_inside_proxy':bool(inside[0,v]<0)})
    write(ROOT/'frame42_all_vertices_within5mm_material_points.json',near)
    # Predeclared third-region question: enumerate all non-thumb/index coarse regions at all thresholds.
    third=[]
    for frame in ids:
        rows=[r for r in by_frame if r['frame_id']==int(frame) and r['scheme']=='coarse_folded_dominant_bone']
        third.append({'frame_id':int(frame),'non_thumb_index_near_regions_by_threshold':{
            f'{int(t*1000)}mm':[r['region'] for r in rows if r['region'] not in ('thumb','index','unknown_cross_region_tie') and r['count_at_or_below'][f'{int(t*1000)}mm']>0]
            for t in THRESHOLDS},'unknown_tie_near_counts':{f'{int(t*1000)}mm':int(((coarse<0)&(distance[int(frame)-42]<=t)).sum()) for t in THRESHOLDS}})
    write(ROOT/'all61_non_thumb_index_region_inventory.json',third)
    report={'schema':'saved_ego_hand_mesh_contact_region_audit:1','status':'COMPLETED_ALL61_ALL788_UNSIGNED_PROXIMITY_NOT_CONTACT_TRUTH',
            'protocol':pin(ROOT/'frozen_rules_v1.json'),'source':pin(__file__),'input_pins':protocol['inputs'],
            'counts':{'frames':61,'vertices_per_frame':788,'vertex_observations':61*788,'faces':1544,'new_distance_queries':0,'new_FK_model_NN_physics_GPU':0},
            'region_mapping':{'bone_names':BONE_NAMES,'coarse_names':COARSE_NAMES,'dominant_bone_counts':{name:int((bone==i).sum()) for i,name in enumerate(BONE_NAMES)},
                              'unknown_exact_tie_vertices':int((bone<0).sum()),'unknown_cross_region_tie_vertices':int((coarse<0).sum()),
                              'mixed_positive_weight_vertices':int(((weights>0).sum(1)>1).sum()),'row_weight_sum_max_abs_error':float(np.max(abs(rowsum-1))),
                              'interpretation':'Dominant skinning influence, not anatomical pad/side segmentation. A/B/C are sequential finger chain frames; C carries canonical fingertip. Root/wrist-influenced is coarse base/palm evidence, not a labeled palm patch.'},
            'frame42_coarse_rows':[r for r in by_frame if r['frame_id']==42 and r['scheme']=='coarse_folded_dominant_bone'],
            'frame42_third_region_inventory':third[0],
            'frame42_tip_landmark_unsigned_distances_m':{f:float(landmark_distance[0,i]) for i,f in enumerate(DIGITS)},
            'all61_coarse_temporal_summary':[r for r in temporal if r['scheme']=='coarse_folded_dominant_bone'],
            'saved_coordinate_distance_max_abs_error_m':float(np.max(abs(coordinate_distance-distance))),
            'limits':protocol['limitations'],'outputs':{p.name:pin(p) for p in ROOT.iterdir() if p.is_file() and p.name not in ('report.json','description.md')},
            'robot_contact_target_modified':False,'force_contact_holding_or_ego_learning_qualified':False}
    if any(x in sys.modules for x in ('torch','mujoco','warp','trimesh')):raise RuntimeError('Saved statistics only')
    write(ROOT/'report.json',report)
    desc=['# 已保存人体全网格近距区域审查','',
          '全部61帧×788顶点均保留；仅统计原unsigned扫描距离，0新FK/最近点query/模型/物理。Dominant skinning仅提供可核影响区域，A/B/C不硬命名为指腹；并列unknown和所有混合权重边界均保存。','',
          '|frame42区域|顶点数|min mm|p05 mm|median mm|≤1mm|≤2mm|≤5mm|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in report['frame42_coarse_rows']:
        q=r['quantile_unsigned_m'];fmt=lambda v:'—' if v is None else f'{v*1000:.4f}'
        desc.append(f"|{r['region']}|{r['static_vertex_count']}|{fmt(r['minimum_unsigned_m'])}|{fmt(q['0.05'])}|{fmt(q['0.5'])}|{r['count_at_or_below']['1mm']}|{r['count_at_or_below']['2mm']}|{r['count_at_or_below']['5mm']}|")
    desc += ['', 'frame42非thumb/index近距区域（预定阈值，完整列出）：'+json.dumps(third[0]['non_thumb_index_near_regions_by_threshold'],ensure_ascii=False),
             '', '所有≤5mm顶点的hand点、原scan最近材料点与triangleID：`frame42_all_vertices_within5mm_material_points.json`；每区域min的所有exact ties：`frame42_all_region_minima_and_ties.json`。这些不是新接触目标，也不是可任意取代表点的有限接触patch。',
             '', '其他tip很远不能排除同指链A/B/C或掌根表面的近距。反之，一/二/五毫米以内的皮肤采样点可能来自tracking/skin/profile/scan标定误差或穿入；原scan非watertight，unsigned距离无内外符号。convex_inside是独立凸包代理，不是真实scan signed深度。网格顶点与阈值诱导连通分量不代表连续面接触面积、独立contact数或力；61帧同谱系、非独立样本。不得直接给机器人增第三接触或宣称静态支持。',
             '', '完整61帧/全部17frame+coarse区域统计、阈值顶点连通分量及所有失败范围见report.json及所列文件。旧参考和优化未改。']
    (ROOT/'description.md').write_text('\n'.join(desc)+'\n',encoding='utf8')
    print(json.dumps({'report':pin(ROOT/'report.json'),'description':pin(ROOT/'description.md'),'frame42':report['frame42_coarse_rows'],'third':third[0]},ensure_ascii=False))


if __name__=='__main__':main()
