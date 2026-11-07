"""Frozen edge-adjacent real convex-surface proposals; saved geometry only."""
from pathlib import Path
import hashlib,importlib.util,json,math
import numpy as np

H=Path(__file__).resolve().parent
P=H.parent.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,j):
    with Path(p).open('x',encoding='utf8') as f:json.dump(j,f,indent=2,allow_nan=False)
def checked(v):
    p=Path(v['path']);assert pin(p)==v,'Changed pin '+str(p);return p
def barycentric(q,t):
    a,b,c=t;e0=b-a;e1=c-a;d=q-a;g00=float(e0@e0);g01=float(e0@e1);g11=float(e1@e1);h0=float(d@e0);h1=float(d@e1)
    det=g00*g11-g01*g01
    if det<=0:raise ValueError('Degenerate actual selected triangle, not repaired')
    v=(g11*h0-g01*h1)/det;w=(g00*h1-g01*h0)/det;out=np.array([1-v-w,v,w])
    assert out.min()>=-1e-12 and abs(out.sum()-1)<=1e-12 and np.max(abs(out@t-q))<=1e-12
    return out
def edge_dist(q,t):
    a=t;b=np.roll(t,-1,axis=0);e=b-a;u=np.clip(np.sum((q-a)*e,axis=1)/np.sum(e*e,axis=1),0,1)
    return np.linalg.norm(q-a-u[:,None]*e,axis=1)
def adjacency(triangles):
    edges={}
    for i,tri in enumerate(triangles):
        for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:edges.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
    if any(len(v)!=2 for v in edges.values()):raise ValueError('Closed manifold convex triangulation required; no guessed adjacency')
    graph=[set() for _ in triangles]
    for u,v in edges.values():graph[u].add(v);graph[v].add(u)
    assert all(len(v)==3 for v in graph)
    return graph,edges

def main():
    protocol=read(H/'preregistration_v1.json');assert protocol['source']==pin(__file__)
    for rec in protocol['inputs'].values():checked(rec)
    pairs=read(checked(protocol['inputs']['paired_surface_anchors']))
    arrayspath=checked(protocol['inputs']['geometry_arrays'])
    with np.load(arrayspath,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    helperpath=checked(protocol['inputs']['filled_triangle_projection_helper']);spec=importlib.util.spec_from_file_location('frozen_filled_triangle_helper',helperpath);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    phoneR=np.asarray(pairs['source_pose']['phone_body_R_world']);results=[];full=[]
    for pair in pairs['anchors']:
        finger=pair['finger'];original=pair['surfaces']['hand'];gid=pair['hand_geom_id'];root=original['source_triangle_id']
        assert (finger,gid,root) in [('thumb',15,2332),('index',28,1876)]
        V=a[f'geom{gid}_vertices_geom_local_FP32'].astype(np.float64);T=a[f'geom{gid}_hull_triangles'];equations=a[f'geom{gid}_hull_equations_F64'];R=a[f'geom{gid}_current_world_R'];p=a[f'geom{gid}_current_world_p']
        assert original['source_triangle_vertex_ids']==T[root].tolist()
        graph,edges=adjacency(T);distance={int(root):0};frontier=[int(root)]
        for depth in [1,2]:
            nxt=[]
            for tid in sorted(frontier):
                for adjacent in sorted(graph[tid]):
                    if adjacent not in distance:distance[adjacent]=depth;nxt.append(adjacent)
            frontier=nxt
        ids=sorted(i for i,d in distance.items() if d in [1,2]);assert len(ids)>=3 and root not in ids
        world_query=np.asarray(pair['native_position_world_m']);query=np.linalg.solve(R,world_query-p)
        inward=-phoneR@np.asarray(pair['surfaces']['phone']['outward_normal_phone_body']);inward/=np.linalg.norm(inward)
        triangles=V[T[ids]];projections=helper._closest_triangle_points_origin(triangles-query)+query
        candidates=[]
        for tid,tri,point in zip(ids,triangles,projections):
            normal=equations[tid,:3]/np.linalg.norm(equations[tid,:3]);normalworld=R@normal;normalworld/=np.linalg.norm(normalworld)
            angle=float(math.acos(float(np.clip(normalworld@inward,-1,1))));pointworld=R@point+p;projectiondistance=float(np.linalg.norm(pointworld-world_query));weights=barycentric(point,tri)
            edgedist=float(edge_dist(point,tri).min());vertexdist=float(np.linalg.norm(tri-point,axis=1).min())
            label='NEAR_TRIANGLE_VERTEX' if vertexdist<=1e-6 else 'NEAR_TRIANGLE_EDGE' if edgedist<=1e-6 else 'TRIANGLE_INTERIOR'
            assert np.max((V-point)@normal)<=1e-12
            candidates.append(dict(name=f'{finger}_adjacent_tri{tid}',triangle_id=int(tid),edge_graph_distance=distance[tid],source_vertex_ids=T[tid].tolist(),triangle_vertices_geom_local_m=tri.tolist(),original_native_point_geom_local_m=query.tolist(),projected_point_geom_local_m=point.tolist(),barycentric_in_source_vertex_order=weights.tolist(),outward_normal_geom_local=normal.tolist(),outward_normal_at_original_pose_world=normalworld.tolist(),area_m2=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2),original_native_point_to_filled_triangle_distance_m=projectiondistance,normal_angle_to_fixed_phone_inward_rad=angle,normal_angle_to_fixed_phone_inward_deg=math.degrees(angle),projection_to_triangle_boundary_m=edgedist,projection_to_nearest_triangle_vertex_m=vertexdist,projection_classification=label,barycentric_near_zero_count_1e_minus8=int(np.count_nonzero(np.abs(weights)<=1e-8)),displacement_from_original_hand_anchor_geom_local_m=(point-np.asarray(original['projected_point_geom_local_m'])).tolist(),displacement_from_original_hand_anchor_m=float(np.linalg.norm(point-np.asarray(original['projected_point_geom_local_m']))),original_pose_projected_point_world_m=pointworld.tolist(),planar_group_id=int(a[f'geom{gid}_planar_triangle_group_ids'][tid]),near_coplanar_group_id=int(a[f'geom{gid}_near_coplanar_triangle_group_ids'][tid]),functional_pad_or_independent_patch_qualified=False))
        candidates.sort(key=lambda c:(c['normal_angle_to_fixed_phone_inward_rad'],c['original_native_point_to_filled_triangle_distance_m'],c['triangle_id']))
        for rank,c in enumerate(candidates):
            c['rank']=rank;c['selected_in_top3']=rank<3
            c['other_candidate_projections_within_1nm']=[q['triangle_id'] for q in candidates if q['triangle_id']!=c['triangle_id'] and np.linalg.norm(np.asarray(c['projected_point_geom_local_m'])-np.asarray(q['projected_point_geom_local_m']))<=1e-9]
        # Fixed phone point, normal, source triangle and ego target are copied exactly.
        result=dict(finger=finger,hand_geom_id=gid,original_hand_triangle=root,original_hand_anchor=original,ring1_triangle_ids=sorted(i for i,d in distance.items() if d==1),ring2_triangle_ids=sorted(i for i,d in distance.items() if d==2),universe_count=len(ids),target_phone_inward_world_at_original_pose=inward.tolist(),fixed_phone_surface=pair['surfaces']['phone'],original_ego_material_target_phone_body_m=pair['original_ego_material_target_phone_body_m'],original_human_scan_candidate_phone_body_m=pair['original_human_scan_candidate_phone_body_m'],top3=candidates[:3])
        assert result['fixed_phone_surface']==pair['surfaces']['phone']
        results.append(result);full.append(dict(finger=finger,candidates_ranked=candidates))
    save(H/'all_ring_candidates_v1.json',dict(schema='edge_adjacent_actual_triangle_rankings:9',ranking=protocol['ranking'],results=full,all_candidates_retained=True))
    output=dict(schema='predeclared_adjacent_surface_triangle_candidates:9',status='COMPLETED_SAVED_GEOMETRY_BACKUP_CANDIDATES_NOT_ASSEMBLY_OR_CONTACT_QUALIFICATION',source=pin(__file__),preregistration=pin(H/'preregistration_v1.json'),inputs=protocol['inputs'],selection_scope='Backup names only. Original anchors/source/phone material points and normals are unchanged. No candidate selected on future assembly/native results.',ranking=protocol['ranking'],near_boundary_descriptor=protocol['near_boundary_descriptor'],results=results,full_ring_rankings=pin(H/'all_ring_candidates_v1.json'),unselected_second_index_contact=pairs['unselected_native_index_contact'],counts=dict(native_model_load=0,FK_forward_steps_LP_NN_GPU=0,convex_hull_rebuilds=0,saved_filled_triangle_projection_batches=2),checks=dict(closed_edge_adjacency_not_vertex_adjacency=True,original_hand_triangles_excluded_from_backup_pool=True,all_triangle_points_reconstruct_nonnegative_barycentrics=True,phone_material_point_normal_and_ego_targets_unchanged=True,all_ring_candidates_preserved_and_top3_rule_fixed=True),functional_pad_or_patch_qualified=False,static_or_dynamic_support_qualified=False,scope='Actual triangles of the saved compiled convex hull triangulation, not arbitrary support-feature centroids. A shared edge can be a coplanar triangulation diagonal or fine-curvature tessellation boundary, not necessarily a physical CAD ridge. Normals/projections are geometric proposals at the original pose. Near-edge/vertex and coincident candidate projections remain explicit; no virtual finite patch, mirroring, scaling, model load, force solve or simulation.')
    save(H/'adjacent_candidates_v1.json',output)
    print(json.dumps({'output':pin(H/'adjacent_candidates_v1.json'),'top3':[{ 'finger':r['finger'],'ring1':r['ring1_triangle_ids'],'ring2':r['ring2_triangle_ids'],'top3':[{'triangle':c['triangle_id'],'angle_deg':c['normal_angle_to_fixed_phone_inward_deg'],'distance_um':c['original_native_point_to_filled_triangle_distance_m']*1e6,'classification':c['projection_classification']} for c in r['top3']]} for r in results]}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(H/'failure_v1.json',dict(error=repr(e),source=pin(__file__),unknown_not_repaired=True,no_automatic_retry=True));raise
