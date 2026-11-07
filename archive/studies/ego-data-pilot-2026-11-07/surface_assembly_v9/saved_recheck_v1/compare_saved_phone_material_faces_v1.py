"""Reuse cached phone triangle localizations; no new nearest-mesh query."""
from pathlib import Path
import hashlib,json,math
import numpy as np
H=Path(__file__).resolve().parent
P=H.parent.parent
pin=lambda p:dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
read=lambda p:json.loads(p.read_text())
pairpath=P/'contact_patch_v7/paired_surface_anchors_v1.json';locpath=P/'contact_patch_v7/contact_localization_v1.json';arraypath=P/'contact_patch_v7/geometry_arrays_v1.npz'
pairs=read(pairpath);loc=read(locpath)
with np.load(arraypath,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
R=np.asarray(pairs['source_pose']['phone_body_R_world']);p=np.asarray(pairs['source_pose']['phone_root_world_m']);GR=a['geom71_current_world_R'];gp=a['geom71_current_world_p'];V=a['geom71_vertices_geom_local_FP32'].astype(float);T=a['geom71_hull_triangles'];rows=[]
for pair,e in zip(pairs['anchors'],loc['ego_frame42_material_candidates']):
    assert pair['finger']==e['finger'] and pair['original_ego_material_target_phone_body_m']==e['projected_hull_material_target_object_m']
    d=e['localization'];tid=d['chosen_triangle_id'];ids=T[tid];tri=np.linalg.solve(R,(V[ids]@GR.T+gp-p).T).T
    projected=np.linalg.solve(R,GR@np.asarray(d['chosen_closest_surface_point_geom_local_m'])+gp-p);target=np.asarray(pair['original_ego_material_target_phone_body_m']);assert np.max(abs(projected-target))<=1e-12
    n=np.cross(tri[1]-tri[0],tri[2]-tri[0]);n/=np.linalg.norm(n);ng=a['geom71_hull_equations_F64'][tid,:3]
    if (R@n)@(GR@ng)<0:n=-n
    edge=tri[1]-tri[0];u=edge-n*(edge@n);u/=np.linalg.norm(u);v=np.cross(n,u);v/=np.linalg.norm(v)
    X=np.column_stack((tri[1]-tri[0],tri[2]-tri[0]));b=np.linalg.solve(X.T@X,X.T@(projected-tri[0]));bary=np.r_[1-b.sum(),b];assert bary.min()>=-1e-12 and np.max(abs(bary@tri-projected))<=1e-12
    cad=pair['surfaces']['phone'];cn=np.asarray(cad['outward_normal_phone_body']);cp=np.asarray(cad['projected_point_phone_body_m']);angle=math.atan2(np.linalg.norm(np.cross(n,cn)),float(n@cn));ego_planar=int(a['geom71_planar_triangle_group_ids'][tid]);cad_planar=cad['planar_group_id'];ego_near=int(a['geom71_near_coplanar_triangle_group_ids'][tid]);cad_near=cad['near_coplanar_group_id']
    rows.append(dict(finger=pair['finger'],original_ego_material_target_phone_body_m=target.tolist(),original_human_scan_candidate_phone_body_m=pair['original_human_scan_candidate_phone_body_m'],original_ego_triangle=dict(triangle_id=int(tid),vertex_ids=ids.tolist(),vertices_phone_body_m=tri.tolist(),projected_point_phone_body_m=projected.tolist(),barycentric_in_source_vertex_order=bary.tolist(),outward_normal_phone_body=n.tolist(),basis_t1_t2_phone_body=[u.tolist(),v.tolist()],planar_group_id=ego_planar,near_coplanar_group_id=ego_near,cached_projection_distance_m=d['closest_surface_distance_m']),fixed_CAD_phone_triangle=cad,CAD_minus_original_ego_material_m=(cp-target).tolist(),CAD_to_original_ego_distance_m=float(np.linalg.norm(cp-target)),outward_normal_difference_rad=angle,outward_normal_difference_deg=math.degrees(angle),same_triangle=int(tid)==cad['source_triangle_id'],same_planar_group=ego_planar==cad_planar,same_near_coplanar_group=ego_near==cad_near))
result=dict(schema='saved_original_ego_vs_CAD_phone_facet_comparison:9',source=pin(Path(__file__)),inputs=dict(original_pairs=pin(pairpath),cached_localization=pin(locpath),geometry_arrays=pin(arraypath)),comparisons=rows,counts=dict(new_model_FK_nearest_geometry_LP_physics_calls=0),source_targets_and_CAD_anchors_changed=False,scope='Original ego targets are unsigned human-scan proximity candidates projected to the phone convex hull, not measured contact/force labels. Reused existing cached nearest triangle IDs/points only. Different triangles/group IDs or normal angles describe this convex triangulation/material relocation, not functional material identity, force capacity or true-grasp evidence. No new nearest-surface search, model or LP.')
out=H/'original_ego_vs_CAD_phone_faces_v1.json'
with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps({'output':pin(out),'comparison':[dict(finger=r['finger'],ego_triangle=r['original_ego_triangle']['triangle_id'],CAD_triangle=r['fixed_CAD_phone_triangle']['source_triangle_id'],normal_angle_deg=r['outward_normal_difference_deg'],distance_mm=r['CAD_to_original_ego_distance_m']*1000,same_planar=r['same_planar_group']) for r in rows]}))
