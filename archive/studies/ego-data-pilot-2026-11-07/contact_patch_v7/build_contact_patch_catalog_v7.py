"""Saved-geometry facet catalog. No MuJoCo, FK, collision query or physics."""
from pathlib import Path
import hashlib, importlib.util, json, math, platform, time
import numpy as np
import scipy
from scipy.spatial import ConvexHull

HERE=Path(__file__).resolve().parent
P=HERE.parent
GIDS=[15,28,71]
LABELS={15:'thumb_sensor_collider',28:'index_sensor_collider',71:'phone_convex_collider'}
LEVELS={'planar':{'normal_angle_rad':1e-4,'all_vertex_plane_distance_m':1e-7},'near_coplanar':{'normal_angle_rad':math.radians(.5),'all_vertex_plane_distance_m':5e-5}}
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def checked(x):
    p=Path(x['path']);assert pin(p)==x,'Changed input '+str(p)
    return p
def save(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def basis(n):
    a=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,a);u/=np.linalg.norm(u)
    return u,np.cross(n,u)
def segment_dist(q,a,b):
    e=b-a;t=np.clip(np.sum((q-a)*e,axis=-1)/np.sum(e*e,axis=-1),0,1)
    return np.linalg.norm(q-a-t[...,None]*e,axis=-1)

def groups(vertices,hull,threshold):
    """Connected triangle groups bounded by one fixed seed plane, no chain drift."""
    tri=vertices[hull.simplices];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    area=np.linalg.norm(cross,axis=1)/2;assert np.all(area>0)
    normals=hull.equations[:,:3];normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    assigned=np.full(len(tri),-1,dtype=np.int64);output=[]
    order=sorted(range(len(tri)),key=lambda i:(-float(area[i]),tuple(sorted(map(int,hull.simplices[i])))))
    for seed in order:
        if assigned[seed]>=0:continue
        gid=len(output);n=normals[seed];offset=-float(n@tri[seed,0]);stack=[seed];members=[]
        while stack:
            k=stack.pop()
            if assigned[k]>=0:continue
            if n@normals[k]<math.cos(threshold['normal_angle_rad']):continue
            if np.max(np.abs(tri[k]@n+offset))>threshold['all_vertex_plane_distance_m']:continue
            assigned[k]=gid;members.append(k)
            stack.extend(sorted(int(x) for x in hull.neighbors[k] if x>=0 and assigned[x]<0))
        assert members
        ids=np.unique(hull.simplices[members]);points=vertices[ids];A=float(math.fsum(area[members]))
        c=np.sum(np.mean(tri[members],axis=1)*area[members,None],axis=0)/A
        u,v=basis(n);uv=np.column_stack(((points-c)@u,(points-c)@v))
        polygon=ConvexHull(uv);poly_order=polygon.vertices
        eqdist=np.max(np.abs(points@n+offset));spread=math.acos(float(np.clip(np.min(normals[members]@n),-1,1)))
        assert eqdist<=threshold['all_vertex_plane_distance_m']+1e-14 and spread<=threshold['normal_angle_rad']+1e-9
        output.append(dict(group_id=gid,triangle_ids=sorted(members),vertex_ids=ids.tolist(),triangle_count=len(members),vertex_count=len(ids),triangle_area_sum_m2=A,outward_reference_normal_geom_local=n.tolist(),reference_plane_offset_m=offset,area_weighted_centroid_geom_local_m=c.tolist(),basis_u_geom_local=u.tolist(),basis_v_geom_local=v.tolist(),polygon_vertex_ids=ids[poly_order].tolist(),polygon_uv_m=uv[poly_order].tolist(),convex_projected_polygon_area_m2=float(polygon.volume),extent_uv_m=np.ptp(uv,axis=0).tolist(),max_vertex_distance_to_reference_plane_m=float(eqdist),max_triangle_normal_angle_rad=float(spread),functional_pad_semantics='UNKNOWN',independent_force_patch_qualified=False))
    assert np.all(assigned>=0)
    return output,assigned

def locate(q,expected_outward,vertices,hull,levels,groupids,closest):
    tri=vertices[hull.simplices];pts=closest(tri-q)+q;d=np.linalg.norm(pts-q,axis=1);minimum=float(d.min())
    tied=np.flatnonzero(d<=minimum+1e-9);chosen=int(np.argmin(d));surface=pts[chosen]
    eq=hull.equations;eq=eq/np.linalg.norm(eq[:,:3],axis=1)[:,None]
    scores=vertices@expected_outward
    result=dict(input_geom_local_m=q.tolist(),expected_outward_normal_geom_local=expected_outward.tolist(),closest_surface_distance_m=minimum,chosen_triangle_id=chosen,all_triangles_within_1nm_of_minimum=tied.tolist(),chosen_closest_surface_point_geom_local_m=surface.tolist(),max_halfspace_signed_m=float(np.max(eq[:,:3]@q+eq[:,3])),normal_support_deficit_m=float(scores.max()-q@expected_outward),closest_triangle_outward_normal=eq[chosen,:3].tolist(),normal_alignment_cosine=float(eq[chosen,:3]@expected_outward),groups={})
    for level in levels:
        ids=sorted(set(int(groupids[level][i]) for i in tied));entries=[]
        for idx in ids:
            g=levels[level][idx];c=np.asarray(g['area_weighted_centroid_geom_local_m']);u=np.asarray(g['basis_u_geom_local']);v=np.asarray(g['basis_v_geom_local']);uv=np.asarray(g['polygon_uv_m'])
            qs=np.asarray([(surface-c)@u,(surface-c)@v]);edge=float(segment_dist(qs,uv,np.roll(uv,-1,axis=0)).min())
            entries.append(dict(group_id=idx,area_m2=g['triangle_area_sum_m2'],polygon_boundary_distance_m=edge,geometric_classification='NEAR_GROUP_BOUNDARY_EDGE_OR_VERTEX' if edge<=1e-6 else 'PROJECTED_GROUP_INTERIOR',boundary_descriptor_threshold_m=1e-6))
        result['groups'][level]=entries
    result['scope']='Input may be a penetrated native contact position, not a closest witness pair. Filled-triangle localization and group-boundary descriptors are geometric only.'
    return result

def main():
    started=time.perf_counter();manifest=read(HERE/'input_freeze_v1.json')
    for p in manifest['input_pins'].values():checked(p)
    assert manifest['source']==pin(__file__) and manifest['levels']==LEVELS
    probe=read(P/'geometry_query_v2/saved_candidate_queries_v1.json')
    checked(probe['geometry_input_arrays']);checked(probe['compiled_scene']);checked(probe['source']);checked(probe['query_source'])
    gpdata=npz(checked(probe['geometry_input_arrays']));old={i:npz(checked(probe['saved_candidates'][f'seed{i}'])) for i in [1,2]}
    spec=importlib.util.spec_from_file_location('frozen_convex_triangle_helper',checked(probe['query_source']))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    current=npz(P/'native_equilibrium_v6/completed_v1/seed0_rank0_state.npz');verify=read(P/'native_equilibrium_v6/completed_v1/seed0_rank0_verification.json')
    raw=verify['native']['raw'];assert np.asarray(raw['qpos27']).tobytes()==current['qpos27'].tobytes()
    currentR=current['geom_xmat'].reshape(-1,3,3);currentp=current['geom_xpos']
    native=np.asarray(raw['contact_rows24']);assert native.shape==(3,24)
    assert {int(x) for x in native[:,1]}=={15,28} and np.all(native[:,2]==71)
    metadata=read(P/'retarget_geometry_v2/report.json')['geometry_metadata']
    phoneR=np.asarray(raw['phone_body_R_world']);phonep=np.asarray(raw['phone_root_world_m'])
    prep=read(P/'retarget_geometry_v2/fixed_anchors_and_targets.json');human=read(P/'retarget_geometry_v1/frame42_targets.json')
    arrays={};catalog={};computed={};recovery=[]
    for gid in GIDS:
        local=[]
        for seed in [1,2]:
            key=f'seed{seed}_phone_world_vertices_m' if gid==71 else f'seed{seed}_geom{gid}_world_mesh_vertices_m'
            R=old[seed]['geom_xmat'].reshape(-1,3,3)[gid];pos=old[seed]['geom_xpos'][gid]
            recovered=np.linalg.solve(R,(gpdata[key]-pos).T).T
            f32=recovered.astype(np.float32);local.append(f32)
            assert np.max(np.abs(f32.astype(float)@R.T+pos-gpdata[key]))<1e-12
        assert local[0].shape==local[1].shape and local[0].tobytes()==local[1].tobytes(),'Two saved poses must recover identical compiled FP32 vertices'
        vertices=local[0].astype(float);h=ConvexHull(vertices);assert h.volume>0
        assert len(vertices)==metadata[str(gid)]['source_compiled_vertices'] and len(h.vertices)==metadata[str(gid)]['exact_hull_vertices']
        levels={};maps={}
        for level,threshold in LEVELS.items():levels[level],maps[level]=groups(vertices,h,threshold)
        catalog[str(gid)]=dict(name=metadata[str(gid)]['name'],body=metadata[str(gid)]['body'],coordinate_system='Compiled mesh geom-local (MuJoCo mesh preprocessing frame), meters; not raw STL/body/world coordinates',compiled_FP32_vertex_count=len(vertices),convex_extreme_vertex_count=len(h.vertices),hull_triangle_count=len(h.simplices),hull_volume_m3=float(h.volume),local_AABB_min_m=vertices.min(0).tolist(),local_AABB_max_m=vertices.max(0).tolist(),levels=levels,functional_pad_semantics='UNKNOWN',convex_filled_holes_or_mount_features_possible=True)
        arrays[f'geom{gid}_vertices_geom_local_FP32']=local[0];arrays[f'geom{gid}_hull_triangles']=h.simplices.astype(np.int64);arrays[f'geom{gid}_hull_equations_F64']=h.equations
        for level in LEVELS:arrays[f'geom{gid}_{level}_triangle_group_ids']=maps[level]
        arrays[f'geom{gid}_current_world_R']=currentR[gid];arrays[f'geom{gid}_current_world_p']=currentp[gid]
        computed[gid]=(vertices,h,levels,maps)
        recovery.append(dict(geom_id=gid,two_saved_pose_compiled_FP32_vertices_bit_equal=True,world_reconstruction_max_abs_m=max(float(np.max(np.abs(local[s-1].astype(float)@old[s]['geom_xmat'].reshape(-1,3,3)[gid].T+old[s]['geom_xpos'][gid]-gpdata[f'seed{s}_phone_world_vertices_m' if gid==71 else f'seed{s}_geom{gid}_world_mesh_vertices_m']))) for s in [1,2])))
    contacts=[]
    for ci,row in enumerate(native):
        p=row[6:9];n=row[9:12];hand=int(row[1]);record=dict(original_contact_index=ci,geom1=hand,geom2=71,dist_m=float(row[5]),native_position_world_m=p.tolist(),native_normal_geom1_to_geom2_world=n.tolist(),decoded_normal_force_N=float(row[18]),native_contact_position_is_not_assumed_surface_witness=True,localization={})
        for gid,sign in [(hand,1),(71,-1)]:
            local=np.linalg.solve(currentR[gid],p-currentp[gid]);nl=np.linalg.solve(currentR[gid],sign*n);nl/=np.linalg.norm(nl)
            record['localization'][str(gid)]=locate(local,nl,*computed[gid],helper._closest_triangle_points_origin)
        pobj=np.linalg.solve(phoneR,p-phonep);record['phone_body_material_coordinate_m']=pobj.tolist()
        record['distances_to_ego_frame42_surface_candidates_m']={a['finger']:float(np.linalg.norm(pobj-np.asarray(a['human_scan_target_object_m']))) for a in prep['preparation']}
        contacts.append(record)
    ego=[]
    for a in prep['preparation']:
        target=np.asarray(a['target_projected_phone_convex_object_m']);pworld=phoneR@target+phonep;local=np.linalg.solve(currentR[71],pworld-currentp[71]);V,H,L,M=computed[71]
        # Localization of the fixed material point; normal follows its nearest hull face only as a geometric descriptor.
        cp=helper._closest_triangle_points_origin(V[H.simplices]-local)+local;nearest=int(np.argmin(np.linalg.norm(cp-local,axis=1)));normal=H.equations[nearest,:3]
        ego.append(dict(finger=a['finger'],frame=42,source_hand='left',robot_hand='right',reflection_applied=False,human_source_unsigned_candidate_not_force_label=True,human_scan_target_object_m=a['human_scan_target_object_m'],projected_hull_material_target_object_m=target.tolist(),localization=locate(local,normal,*computed[71],helper._closest_triangle_points_origin),source_scan_triangle_id=next(x['nearest_scan_triangle_id'] for x in human['mapping'] if x['finger']==a['finger'])))
    # Source contact multiplicity is retained; two nearby index points are not granted independent patch capacity.
    indexcontacts=[c for c in contacts if c['geom1']==28]
    index_separation=float(np.linalg.norm(np.asarray(indexcontacts[1]['native_position_world_m'])-np.asarray(indexcontacts[0]['native_position_world_m'])))
    np.savez_compressed(HERE/'geometry_arrays_v1.npz',**arrays,native_contact_rows24=native)
    save(HERE/'facet_catalog_v1.json',dict(schema='saved_convex_contact_facet_catalog:7',thresholds=LEVELS,grouping_rule='Largest-area triangle first; connected triangles must each satisfy fixed seed-plane normal and all-vertex distance limits; no transitive angular/offset chaining.',geometries=catalog,functional_pad_semantics='UNKNOWN',surface_force_patch_qualified=False))
    save(HERE/'contact_localization_v1.json',dict(schema='saved_native_contact_to_facet_localization:7',native_seed=0,native_rank=0,contacts=contacts,ego_frame42_material_candidates=ego,index_two_native_contact_position_separation_m=index_separation,scope='Saved native contact positions are solver collision records with small penetration. Facet proximity is not a unique closest pair, compliant patch, measured pressure, or a force-capacity claim. Human candidates are unsigned scan proximity, not measured force/contact labels.'))
    # Compact preview in each compiled geom-local coordinate system.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,4));
    for col,gid in enumerate(GIDS):
        ax=fig.add_subplot(1,3,col+1,projection='3d');V,H,L,M=computed[gid]
        ax.add_collection3d(Poly3DCollection(V[H.simplices]*1000,facecolor='#b9c4cf',edgecolor='none',alpha=.22))
        top=sorted(L['planar'],key=lambda g:-g['triangle_area_sum_m2'])[:3]
        for g in top:ax.add_collection3d(Poly3DCollection(V[H.simplices[g['triangle_ids']]]*1000,facecolor='#e3a547',edgecolor='none',alpha=.45))
        for c in contacts:
            if str(gid) in c['localization']:
                q=np.asarray(c['localization'][str(gid)]['input_geom_local_m'])*1000;ax.scatter(*q,color='red',s=22);ax.text(*q,str(c['original_contact_index']),fontsize=8)
        if gid==71:
            for e in ego:
                q=np.asarray(e['localization']['input_geom_local_m'])*1000;ax.scatter(*q,color='blue',s=24);ax.text(*q,e['finger'],fontsize=8)
        lo=V.min(0)*1000;hi=V.max(0)*1000;center=(lo+hi)/2;rad=max(hi-lo)/2
        ax.set(xlim=(center[0]-rad,center[0]+rad),ylim=(center[1]-rad,center[1]+rad),zlim=(center[2]-rad,center[2]+rad),xlabel='local x (mm)',ylabel='local y (mm)',zlabel='local z (mm)',title=LABELS[gid]);ax.set_box_aspect((1,1,1))
    fig.suptitle('Convex CAD facets; red=native contact positions, blue=ego material candidates; function/patch UNKNOWN',fontsize=10)
    fig.tight_layout();fig.savefig(HERE/'facet_localization_preview.png',dpi=180);plt.close(fig)
    summary={}
    for gid,g in catalog.items():
        summary[gid]=dict(name=g['name'],vertices=g['compiled_FP32_vertex_count'],hull_triangles=g['hull_triangle_count'],group_counts={k:len(v) for k,v in g['levels'].items()},largest_planar_groups=[{k:x[k] for k in ['group_id','triangle_area_sum_m2','extent_uv_m','outward_reference_normal_geom_local','area_weighted_centroid_geom_local_m']} for x in sorted(g['levels']['planar'],key=lambda x:-x['triangle_area_sum_m2'])[:8]])
    result=dict(schema='saved_contact_patch_geometry_catalog_report:7',status='COMPLETED_GEOMETRY_CATALOG_NOT_FUNCTIONAL_PATCH_QUALIFICATION',source=pin(__file__),input_freeze=pin(HERE/'input_freeze_v1.json'),geometry_summary=summary,local_vertex_recovery=recovery,contact_count=len(contacts),index_two_native_contact_position_separation_m=index_separation,outputs={f:pin(HERE/f) for f in ['facet_catalog_v1.json','contact_localization_v1.json','geometry_arrays_v1.npz','facet_localization_preview.png']},counts=dict(native_model_load=0,MjData_FK_forward_steps_NN_GPU=0,convex_hull_builds=3,saved_localizations=8),elapsed_s=time.perf_counter()-started,functional_pad_semantics='UNKNOWN',force_patch_qualified=False,scope='Exact saved compiled FP32 vertices recovered through two saved proper rigid poses and checked bit-identical; transforms use raw saved inverse, no projection/recentering. Qhull facets and bounded near-coplanar groups are geometric CAD descriptors, not native contact manifolds, soft pads, scan surfaces, pressure patches, or independent wrench capacity. No fresh model/FK/physics/optimization. No left/right mirroring or object scaling.')
    save(HERE/'report.json',result);print(json.dumps({'report':pin(HERE/'report.json'),'summary':summary,'contact_count':len(contacts)}))

if __name__=='__main__':
    try:main()
    except Exception as e:
        save(HERE/'failure.json',dict(error=repr(e),source=pin(__file__),unknown_not_passed=True,no_automatic_retry=True));raise
