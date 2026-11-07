"""Source phone-surface anchors plus explicit 1mm sensitivity; saved meshes only."""
from pathlib import Path
import copy, hashlib, importlib.util, json
import numpy as np

H=Path(__file__).resolve().parent; P=H.parent

def pin(p):
    p=Path(p).resolve(strict=True); b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,v):
    with Path(p).open('x',encoding='utf8') as f: json.dump(v,f,indent=2,allow_nan=False)
def checked(v):
    assert pin(v['path'])==v, 'Input changed '+v['path']; return Path(v['path'])
def bary(point,tri):
    E=(tri[1:]-tri[0]).T; vw=np.linalg.solve(E.T@E,E.T@(point-tri[0]))
    b=np.r_[1-vw.sum(),vw]
    assert b.min()>=-1e-10 and np.max(abs(b@tri-point))<1e-12
    return b
def boundary(point,tri):
    a=tri; e=np.roll(tri,-1,axis=0)-a
    u=np.clip(np.sum((point-a)*e,axis=1)/np.sum(e*e,axis=1),0,1)
    return dict(to_edge_m=float(np.linalg.norm(point-a-u[:,None]*e,axis=1).min()),
                to_vertex_m=float(np.linalg.norm(point-tri,axis=1).min()))

def main():
    cfg=read(H/'preregistration_v1.json'); assert cfg['source']==pin(__file__)
    inputs={k:checked(v) for k,v in cfg['inputs'].items()}
    pairs=read(inputs['original_anchors']); comparisons=read(inputs['source_face_comparison'])['comparisons']
    with np.load(inputs['geometry_arrays'],allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
    spec=importlib.util.spec_from_file_location('saved_projection',inputs['projection_helper'])
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    Rp=np.array(pairs['source_pose']['phone_body_R_world']); pp=np.array(pairs['source_pose']['phone_root_world_m'])
    Rg=a['geom71_current_world_R']; pg=a['geom71_current_world_p']
    Vg=a['geom71_vertices_geom_local_FP32'].astype(np.float64); T=a['geom71_hull_triangles']
    V=np.linalg.solve(Rp,(Vg@Rg.T+pg-pp).T).T
    Ng=a['geom71_hull_equations_F64'][:,:3]; Ng=Ng/np.linalg.norm(Ng,axis=1)[:,None]
    N=np.linalg.solve(Rp,(Ng@Rg.T).T).T; N=N/np.linalg.norm(N,axis=1)[:,None]
    source=np.array([c['original_ego_material_target_phone_body_m'] for c in comparisons])
    rows=[]; allc=[]
    for i,(old,c) in enumerate(zip(pairs['anchors'],comparisons)):
        s=c['original_ego_triangle']; tid=s['triangle_id']; target=source[i]
        assert np.max(abs(V[T[tid]]-s['vertices_phone_body_m']))<1e-12
        assert np.max(abs(target-s['projected_point_phone_body_m']))<1e-12
        assert float(N[tid]@np.array(s['outward_normal_phone_body']))>1-1e-12
        desired=source[i]-source[1-i]; desired/=np.linalg.norm(desired)
        Q=helper._closest_triangle_points_origin(V[T]-target)+target
        distance=np.linalg.norm(Q-target,axis=1)
        angle=np.arccos(np.clip(N@desired,-1,1))
        valid=np.flatnonzero(distance<=cfg['sensitivity_radius_m']+1e-12)
        ranked=sorted((int(t) for t in valid),key=lambda t:(float(angle[t]),float(distance[t]),t))
        # Full finite surface universe retained before choosing geometric angle rank0.
        candidates=[]
        for t in ranked:
            q=Q[t]; b=bary(q,V[T[t]])
            candidates.append(dict(triangle_id=t,source_vertex_ids=T[t].tolist(),
                projected_point_phone_body_m=q.tolist(),barycentric=b.tolist(),
                source_displacement_m=float(distance[t]),outward_normal_phone_body=N[t].tolist(),
                angle_to_source_pair_outward_line_deg=float(np.degrees(angle[t])),
                **boundary(q,V[T[t]])))
        allc.append(dict(finger=old['finger'],source_triangle=tid,
                         source_target_phone_body_m=target.tolist(),candidates_ranked=candidates))
        rows.append((old,c,tid,ranked[0],Q))

    outputs=[]
    for branch in cfg['branches']:
        result=copy.deepcopy(pairs)
        result.update(schema='source_region_filled_triangle_paired_surface_anchors:11',
            status='GEOMETRIC_SOURCE_MAPPING_NOT_MEASURED_CONTACT',source=pin(__file__),
            preregistration=pin(H/'preregistration_v1.json'),input_pins=cfg['inputs'],
            branch=branch,source_region_contract=cfg['source_region_contract'],
            checks=dict(actual_triangle_barycentric_reconstruction=True,
                phone_normals_outward_for_frozen_convex_mesh=True,
                original_ego_targets_unchanged=True,hand_material_points_unchanged=True),
            selection_rule=cfg['selection_rule'],new_native_model_FK_LP_physics_calls=0,
            saved_filled_triangle_projection_batches=2)
        for i,(old,c,stid,rank0,Q) in enumerate(rows):
            tid=stid if branch=='exact_source_original_hand' else rank0
            # Exact branch retains target bytes, rather than a repeat numerical projection.
            q=source[i].copy() if branch=='exact_source_original_hand' else Q[tid].copy()
            b=bary(q,V[T[tid]])
            qg=np.linalg.solve(Rg,Rp@q+pp-pg)
            rec=dict(geom_id=71,geom_name=old['surfaces']['phone']['geom_name'],
                source_triangle_id=tid,source_triangle_vertex_ids=T[tid].tolist(),
                source_triangle_vertices_geom_local_m=Vg[T[tid]].tolist(),
                source_triangle_vertices_phone_body_m=V[T[tid]].tolist(),
                projected_point_geom_local_m=qg.tolist(),projected_point_phone_body_m=q.tolist(),
                barycentric_in_source_vertex_order=b.tolist(),
                barycentric_reconstruction_max_abs_m=float(np.max(abs(b@V[T[tid]]-q))),
                outward_normal_geom_local=Ng[tid].tolist(),outward_normal_phone_body=N[tid].tolist(),
                source_triangle_area_m2=float(np.linalg.norm(np.cross(V[T[tid]][1]-V[T[tid]][0],V[T[tid]][2]-V[T[tid]][0]))/2),
                planar_group_id=int(a['geom71_planar_triangle_group_ids'][tid]),
                near_coplanar_group_id=int(a['geom71_near_coplanar_triangle_group_ids'][tid]),
                initial_projected_point_world_m=(Rp@q+pp).tolist(),
                initial_outward_normal_world=(Rp@N[tid]).tolist(),
                point_definition='Exact source ego unsigned proximity projection' if branch=='exact_source_original_hand' else 'Closest filled triangle point in explicitly algorithmic 1mm source ball, ranked by normal angle to fixed source pair line',
                **boundary(q,V[T[tid]]))
            pair=result['anchors'][i]
            # Remove stale initial geometry statements from prior CAD/native pairing.
            for k in ['initial_projected_surface_separation_world_m','initial_projected_surface_point_distance_m',
                      'initial_signed_gap_along_original_native_normal_m','initial_hand_phone_outward_normal_dot']:
                pair.pop(k,None)
            pair['surfaces']['phone']=rec
            pair['source_phone_triangle_id']=stid
            pair['old_CAD_phone_surface_retained']=old['surfaces']['phone']
            pair['projected_phone_anchor_to_original_ego_material_target_m']=float(np.linalg.norm(q-source[i]))
            pair['source_outward_normal_difference_deg']=float(np.degrees(np.arccos(np.clip(N[tid]@N[stid],-1,1))))
            pair['surface_pair_is_unique_closest_pair_qualified']=False
            assert pair['surfaces']['hand']==old['surfaces']['hand']
            assert pair['original_ego_material_target_phone_body_m']==old['original_ego_material_target_phone_body_m']
            assert float(np.max((V-q)@N[tid]))<1e-12
            assert pair['projected_phone_anchor_to_original_ego_material_target_m']<=cfg['sensitivity_radius_m']+1e-12
        result['scope']='Source ego unsigned proximity material regions. Surface normals are convex mesh geometry only, not observed forces or true human contact normals. Hand facets are fixed original actual triangles, functional-pad identity UNKNOWN. 1mm branch is an algorithmic sensitivity assumption, not a measured reconstruction confidence radius; separate named branch cannot be pooled as faithful exact-source results. No native query or simulation.'
        dest=H/(branch+'_anchors_v1.json');save(dest,result);outputs.append(pin(dest))
    save(H/'source_phone_region_catalog_v1.json',dict(schema='source_phone_surface_ball_candidates:11',
        radius_m=cfg['sensitivity_radius_m'],inputs=cfg['inputs'],all_candidates_retained=True,
        ranking=cfg['selection_rule'],results=allc,new_model_FK_LP_steps=0))
    save(H/'report_v1.json',dict(schema='source_region_mapping_build:11',source=pin(__file__),
        preregistration=pin(H/'preregistration_v1.json'),outputs=outputs,
        catalog=pin(H/'source_phone_region_catalog_v1.json'),
        branches=[dict(name=read(p['path'])['branch'],anchors=[dict(finger=r['finger'],
            source_triangle=r['source_phone_triangle_id'],selected_triangle=r['surfaces']['phone']['source_triangle_id'],
            displacement_m=r['projected_phone_anchor_to_original_ego_material_target_m'],
            source_normal_difference_deg=r['source_outward_normal_difference_deg']) for r in read(p['path'])['anchors']]) for p in outputs],
        passed=True,new_model_FK_LP_steps=0,saved_filled_triangle_projection_batches=2))
    print(json.dumps(read(H/'report_v1.json')['branches']))

if __name__=='__main__':main()
