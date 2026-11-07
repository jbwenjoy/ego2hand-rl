"""Actual filled-triangle projections from frozen saved localization only."""
from pathlib import Path
import hashlib,json
import numpy as np

H=Path(__file__).resolve().parent
P=H.parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def checked(x):
    assert pin(x['path'])==x,'Changed saved input '+x['path']
    return Path(x['path'])
def barycentric(point,tri):
    a,b,c=tri;e0=b-a;e1=c-a;q=point-a
    x00=float(e0@e0);x01=float(e0@e1);x11=float(e1@e1);y0=float(q@e0);y1=float(q@e1)
    denominator=x00*x11-x01*x01
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('Degenerate fixed selected triangle; no fallback')
    v=(x11*y0-x01*y1)/denominator;w=(x00*y1-x01*y0)/denominator
    out=np.array([1-v-w,v,w]);assert np.min(out)>=-1e-12 and abs(out.sum()-1)<=1e-12
    err=float(np.max(np.abs(out@tri-point)));assert err<=1e-12
    return out,err

def main():
    freeze=read(H/'paired_surface_anchors_source_freeze_v1.json')
    assert freeze['source']==pin(__file__)
    for x in freeze['input_pins'].values():checked(x)
    catalog=read(H/'facet_catalog_v1.json');loc=read(H/'contact_localization_v1.json')
    with np.load(H/'geometry_arrays_v1.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    verify=read(P/'native_equilibrium_v6/completed_v1/seed0_rank0_verification.json');raw=verify['native']['raw']
    phoneR=np.asarray(raw['phone_body_R_world']);phonep=np.asarray(raw['phone_root_world_m'])
    prep=read(P/'retarget_geometry_v2/fixed_anchors_and_targets.json')['preparation']
    targets={x['finger']:x for x in prep}
    selected=[]
    for native_index,finger,handgid in [(0,'thumb',15),(1,'index',28)]:
        c=loc['contacts'][native_index];assert c['original_contact_index']==native_index and c['geom1']==handgid and c['geom2']==71
        pair=dict(finger=finger,original_native_contact_index=native_index,hand_geom_id=handgid,phone_geom_id=71,native_position_world_m=c['native_position_world_m'],native_dist_m=c['dist_m'],native_normal_geom1_to_geom2_world=c['native_normal_geom1_to_geom2_world'],surfaces={})
        worldpoints=[];worldnormals=[]
        for role,gid in [('hand',handgid),('phone',71)]:
            descriptor=c['localization'][str(gid)];triangle_id=descriptor['chosen_triangle_id'];ids=a[f'geom{gid}_hull_triangles'][triangle_id]
            V=a[f'geom{gid}_vertices_geom_local_FP32'].astype(np.float64);tri=V[ids];point=np.asarray(descriptor['chosen_closest_surface_point_geom_local_m'])
            weights,error=barycentric(point,tri)
            eq=a[f'geom{gid}_hull_equations_F64'][triangle_id];normal=eq[:3]/np.linalg.norm(eq[:3])
            assert np.max((V-point)@normal)<=1e-12,'Outward supporting normal'
            R=a[f'geom{gid}_current_world_R'];p=a[f'geom{gid}_current_world_p'];wp=R@point+p;wn=R@normal;wn/=np.linalg.norm(wn)
            worldpoints.append(wp);worldnormals.append(wn)
            record=dict(geom_id=gid,geom_name=catalog['geometries'][str(gid)]['name'],source_triangle_id=int(triangle_id),source_triangle_vertex_ids=ids.tolist(),source_triangle_vertices_geom_local_m=tri.tolist(),projected_point_geom_local_m=point.tolist(),barycentric_in_source_vertex_order=weights.tolist(),barycentric_reconstruction_max_abs_m=error,outward_normal_geom_local=normal.tolist(),source_triangle_area_m2=float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2),native_position_to_this_surface_distance_m=descriptor['closest_surface_distance_m'],planar_group_id=int(a[f'geom{gid}_planar_triangle_group_ids'][triangle_id]),near_coplanar_group_id=int(a[f'geom{gid}_near_coplanar_triangle_group_ids'][triangle_id]),initial_projected_point_world_m=wp.tolist(),initial_outward_normal_world=wn.tolist(),point_definition='Closest filled-triangle projection of the original native contact record position, selected by the frozen catalog; not triangle/feature centroid')
            if role=='phone':
                bodytri=np.linalg.solve(phoneR,(tri@R.T+p-phonep).T).T
                bodypoint=np.linalg.solve(phoneR,wp-phonep)
                bn=np.cross(bodytri[1]-bodytri[0],bodytri[2]-bodytri[0]);bn/=np.linalg.norm(bn)
                if (phoneR@bn)@wn<0:bn=-bn
                assert np.max(np.abs(weights@bodytri-bodypoint))<=1e-12
                record.update(source_triangle_vertices_phone_body_m=bodytri.tolist(),projected_point_phone_body_m=bodypoint.tolist(),outward_normal_phone_body=bn.tolist(),normal_transform='Geometric cross product of the same triangle in phone-body coordinates, oriented to its saved outward world normal; no R projection/repair')
                target=np.asarray(targets[finger]['target_projected_phone_convex_object_m'])
                pair['original_ego_material_target_phone_body_m']=target.tolist()
                pair['original_human_scan_candidate_phone_body_m']=targets[finger]['human_scan_target_object_m']
                pair['projected_phone_anchor_to_original_ego_material_target_m']=float(np.linalg.norm(bodypoint-target))
                pair['original_native_record_to_material_target_m']=float(np.linalg.norm(np.asarray(c['phone_body_material_coordinate_m'])-target))
            pair['surfaces'][role]=record
        separation=worldpoints[1]-worldpoints[0];nn=np.asarray(c['native_normal_geom1_to_geom2_world'])
        pair.update(initial_projected_surface_separation_world_m=separation.tolist(),initial_projected_surface_point_distance_m=float(np.linalg.norm(separation)),initial_signed_gap_along_original_native_normal_m=float(separation@nn),initial_hand_phone_outward_normal_dot=float(worldnormals[0]@worldnormals[1]),surface_pair_is_unique_closest_pair_qualified=False,functional_pad_or_pressure_patch_qualified=False)
        selected.append(pair)
    excluded=loc['contacts'][2]
    result=dict(schema='saved_filled_triangle_paired_surface_anchors:7',status='COMPLETED_FIXED_GEOMETRIC_ANCHORS_NOT_CONTACT_OR_HOLDING_QUALIFICATION',source=pin(__file__),source_freeze=pin(H/'paired_surface_anchors_source_freeze_v1.json'),input_pins=freeze['input_pins'],source_pose=dict(seed=0,rank=0,qpos27=raw['qpos27'],phone_body_R_world=raw['phone_body_R_world'],phone_root_world_m=raw['phone_root_world_m']),selection_rule='Exactly first native thumb contact index0 and first native index-finger contact index1, as requested before this computation. Each uses the frozen closest filled triangle projection, no centroid or witness repair.',anchors=selected,unselected_native_index_contact=dict(original_native_contact_index=2,reason='Not chosen by first-index-contact rule; original evidence retained, no merge into a finite patch or assumed independent support',native_position_world_m=excluded['native_position_world_m'],separation_from_selected_index_native_record_m=loc['index_two_native_contact_position_separation_m'],retained_in_localization_pin=freeze['input_pins']['localization']),checks=dict(all_barycentric_weights_nonnegative_within_1e_minus12_roundoff=True,all_four_points_reconstruct_actual_selected_triangles=True,all_normals_outward_for_saved_convex_body=True,phone_body_material_coordinates_and_original_targets_kept=True,no_centroid_mirror_scaling_or_new_native_call=True),new_model_FK_geometry_query_LP_physics_calls=0,functional_pad_or_patch_qualified=False,static_or_dynamic_support_qualified=False,scope='Fixed geometric material anchors for a future proposal. Paired means shared original native contact source, not a certified mutually closest witness pair. The source records have ~20um penetration, so the independently projected points can overlap along the old contact normal. Phone-body normals are actual triangle normals, not the old native average normal. Functional pad identity, compliance/pressure area, forces and grasp remain UNKNOWN. Two near-duplicate original index contacts do not establish an independent patch. Original left-human/right-robot mismatch and ego material deviations remain explicit.')
    save(H/'paired_surface_anchors_v1.json',result)
    print(json.dumps({'output':pin(H/'paired_surface_anchors_v1.json'),'anchors':[dict(finger=x['finger'],hand_triangle=x['surfaces']['hand']['source_triangle_id'],phone_triangle=x['surfaces']['phone']['source_triangle_id'],ego_material_deviation_m=x['projected_phone_anchor_to_original_ego_material_target_m'],initial_surface_point_distance_m=x['initial_projected_surface_point_distance_m']) for x in selected]}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(H/'paired_surface_anchors_failure_v1.json',dict(error=repr(e),source=pin(__file__),unknown_not_repaired=True));raise
