"""Saved/source geometry only: frozen stage1 AABB sampling gates, no physics.

All scanned vertices -> signed AABB surface. All six box faces -> original
transformed scan triangles with deterministic <=2mm rectangular grids.
This is sampled evidence, not a Hausdorff bound or physical surface truth.
"""
from pathlib import Path
import hashlib,json,math,time
import numpy as np
import trimesh

ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent
OUT=ROOT/'aabb_geometry_audit_v1'

def pin(path):
    path=Path(path).resolve(strict=True);return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def require(v,s):
    if not v:raise ValueError(s)

def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def describe(a):
    a=np.asarray(a,np.float64)
    return {'count':int(a.size),'minimum_m':float(a.min()),'maximum_m':float(a.max()),'mean_signed_m':float(a.mean()),'MAE_m':float(np.abs(a).mean()),'RMS_m':float(np.sqrt(np.mean(a*a))),'p95_absolute_m':float(np.percentile(np.abs(a),95,method='linear'))}

def main():
    started=time.perf_counter();OUT.mkdir()
    protocol=PILOT/'protocols/phone_physics_stage1_v1.json';report_path=ROOT/'geometry_report.json'
    p=json.loads(protocol.read_text(encoding='utf-8'));r=json.loads(report_path.read_text(encoding='utf-8'))
    require(p['schema']=='phone_physics_stage1_protocol:1' and r['object_bop_id']==24 and p['object']['dataset_revision']==r['revision'],'Frozen same phone/source protocol')
    limits=p['geometry_error_report']['predeclared_engineering_limits_m']
    require(limits=={'maximum_scanned_vertex_outside_box':.0005,'sampled_box_to_scan_p95':.002,'sampled_box_to_scan_max':.004},'Original fixed thresholds, no post-result changes')
    sources={'protocol':pin(protocol),'asset_geometry_report':pin(report_path),'auditor':pin(__file__),'GLB':pin(r['source_pins']['GLB']['path']),'phone_MJCF':pin(ROOT/'phone_asset.xml'),'baked_visual_OBJ':pin(ROOT/'meshes/phone24_visual_m.obj')}
    require(sources['GLB']==r['source_pins']['GLB'],'Original raw GLB unchanged')
    scene=trimesh.load_scene(sources['GLB']['path'],process=False);vertices=[];faces=[];offset=0
    for node in sorted(scene.graph.nodes_geometry):
        T,name=scene.graph[node];g=scene.geometry[name];v=trimesh.transform_points(np.asarray(g.vertices,np.float64),np.asarray(T,np.float64))
        vertices.append(v);faces.append(np.asarray(g.faces,np.int64)+offset);offset+=len(v)
    mesh=trimesh.Trimesh(vertices=np.concatenate(vertices),faces=np.concatenate(faces),process=False)
    center=np.asarray(r['collision_proxy']['center_model_m'],np.float64);half=np.asarray(r['collision_proxy']['halfsize_m'],np.float64);bounds=np.asarray([center-half,center+half]);size=2*half
    require(np.array_equal(mesh.bounds,np.asarray(r['geometry']['bounds_model_m'])) and np.max(np.abs(mesh.bounds-bounds))<1e-16,'Same baked source and original-coordinate AABB')
    q=np.abs(mesh.vertices-center)-half
    signed=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(q.max(axis=1),0)
    maximum_outside=float(np.maximum(signed,0).max())
    spacing_limit=.002;points=[];face_ids=[];grids=[]
    for axis in range(3):
        other=[j for j in range(3) if j!=axis];counts=[int(math.ceil(size[j]/spacing_limit))+1 for j in other]
        coordinates=[np.linspace(bounds[0,j],bounds[1,j],n) for j,n in zip(other,counts)]
        spacing=[float(size[j]/(n-1)) for j,n in zip(other,counts)]
        require(max(spacing)<=spacing_limit*(1+1e-12),'Every box face spacing <=2mm')
        for side in (0,1):
            a,b=np.meshgrid(*coordinates,indexing='ij');face=np.empty((a.size,3),np.float64);face[:,axis]=bounds[side,axis];face[:,other[0]]=a.ravel();face[:,other[1]]=b.ravel()
            fid=len(grids);points.append(face);face_ids.extend([fid]*len(face));grids.append({'face_id':fid,'fixed_axis':'XYZ'[axis],'side':'minimum' if side==0 else 'maximum','sample_count':len(face),'variable_axes':['XYZ'[j] for j in other],'point_counts':counts,'actual_grid_spacing_m':spacing,'edge_corner_duplicates_retained':True})
    samples=np.concatenate(points);face_ids=np.asarray(face_ids,np.int64);nearest=[];distance=[];triangles=[];chunk=64
    for first in range(0,len(samples),chunk):
        closest,d,tid=trimesh.proximity.closest_point_naive(mesh,samples[first:first+chunk]);nearest.append(closest);distance.append(d);triangles.append(tid)
    nearest=np.concatenate(nearest);distance=np.concatenate(distance);triangles=np.concatenate(triangles)
    require(np.isfinite(distance).all() and (distance>=0).all() and np.isfinite(signed).all() and np.isfinite(nearest).all(),'Finite sampled triangle distance evidence; never replace unknown with zero')
    np.savez_compressed(OUT/'all_geometry_samples.npz',scan_vertices_model_m_F64=np.asarray(mesh.vertices,np.float64),scan_faces_int64=np.asarray(mesh.faces,np.int64),scan_vertex_signed_box_surface_distance_m_F64=signed,
        box_face_points_model_m_F64=samples,box_face_id_int64=face_ids,nearest_scan_triangle_point_model_m_F64=nearest,nearest_scan_triangle_id_int64=triangles,box_face_to_scan_distance_m_F64=distance,
        box_center_model_m_F64=center,box_halfsize_m_F64=half)
    checks={'all_raw_vertices_within_outside_limit':maximum_outside<=limits['maximum_scanned_vertex_outside_box'],'sampled_box_to_scan_p95':float(np.percentile(distance,95,method='linear'))<=limits['sampled_box_to_scan_p95'],'sampled_box_to_scan_max':float(distance.max())<=limits['sampled_box_to_scan_max']}
    for value in sources.values():require(pin(value['path'])==value,'Source files unchanged during sampled audit')
    result={'schema':'hot3d_phone24_AABB_geometry_sampling_audit:1','status':'PASS_SAMPLED_GEOMETRY_ONLY' if all(checks.values()) else 'FAIL_PREDECLARED_AABB_GEOMETRY_LIMITS','source_pins':sources,
        'geometry_qualified':all(checks.values()),'checks':checks,'passed':sum(checks.values()),'total':len(checks),'predeclared_engineering_limits_m':limits,'source_vertices_to_box_signed_surface':describe(signed),'maximum_original_vertex_outside_box_m':maximum_outside,
        'sampled_box_faces_to_original_triangles':describe(distance),'box_face_grids':grids,'total_face_sample_count':len(samples),'unique_coordinate_sample_count':len(np.unique(samples,axis=0)),'all_original_vertices_count':len(mesh.vertices),'scanned_faces_count':len(mesh.faces),
        'method':{'vertex_to_box':'Exact axis-aligned box signed-distance formula; negative inside','box_to_scan':'trimesh.proximity.closest_point_naive over all original baked triangles,64 samples per chunk','percentile':'NumPy linear percentile of retained samples; duplicate edge/corner samples explicit','spacing_max_m':spacing_limit,'no_mesh_repair_or_projection':True,'continuous_Hausdorff_bound':False},
        'mesh_defects':{key:r['geometry'][key] for key in ('is_watertight','is_winding_consistent','is_volume','zero_area_faces','connected_components')},
        'interpretation':'Unchanged nominal box remains a no-hand bench candidate only. Failure blocks contact-geometry qualification even if a physics bench is stable. No threshold retuning or asset modification in this audit; visual scan itself is not physical truth.',
        'next_candidate_if_FAIL':'A single convex hull of all baked phone vertices, visual unchanged; preserve model origin and explicitly compare sampled hull-vs-scan error under a new frozen candidate contract. A phone does not require CoACD before trying this convex candidate. Keep mass/COM/inertia assumptions separate; no change or physics performed here.',
        'actual_MjData_physics_training_GPU':0,'software':{'trimesh':trimesh.__version__,'numpy':np.__version__},'elapsed_seconds':time.perf_counter()-started,'outputs':{'all_geometry_samples.npz':pin(OUT/'all_geometry_samples.npz')}}
    write(OUT/'report.json',result);print(json.dumps({'status':result['status'],'geometry_qualified':result['geometry_qualified'],'checks':checks,'box_distance':result['sampled_box_faces_to_original_triangles'],'report':pin(OUT/'report.json')}));return 0

if __name__=='__main__':raise SystemExit(main())
