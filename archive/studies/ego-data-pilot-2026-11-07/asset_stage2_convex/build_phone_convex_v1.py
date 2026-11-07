"""Single convex phone collision candidate and bounded geometry audit; no physics.

Original visual bytes, model origin, COM, mass and uniform-box inertia remain
unchanged. Qhull Qt has no random joggle/fallback. Sampling is deterministic.
"""
from pathlib import Path
import hashlib,importlib.metadata,json,math,os,time,xml.etree.ElementTree as ET
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent
STAGE1=PILOT/'asset_stage1'
SPACING=.002
SEED=0  # Recorded identifier; deterministic sampling performs no random draws.

def require(value,message):
    if not value:raise ValueError(message)

def pin(path):
    path=Path(path).resolve(strict=True);return {'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def stats(a):
    a=np.asarray(a,np.float64);return {'count':int(a.size),'minimum_m':float(a.min()),'maximum_m':float(a.max()),'mean_signed_m':float(a.mean()),'MAE_m':float(np.abs(a).mean()),'RMS_m':float(np.sqrt(np.mean(a*a))),'p95_absolute_m':float(np.percentile(np.abs(a),95,method='linear'))}

def original_scan(path):
    scene=trimesh.load_scene(path,process=False);vertices=[];faces=[];offset=0;nodes=[]
    for node in sorted(scene.graph.nodes_geometry):
        T,name=scene.graph[node];geometry=scene.geometry[name];baked=trimesh.transform_points(np.asarray(geometry.vertices,np.float64),np.asarray(T,np.float64))
        vertices.append(baked);faces.append(np.asarray(geometry.faces,np.int64)+offset);offset+=len(baked)
        nodes.append({'node':node,'geometry':name,'transform_baked_once':np.asarray(T).tolist()})
    return trimesh.Trimesh(vertices=np.concatenate(vertices),faces=np.concatenate(faces),process=False),nodes

def single_hull(scan):
    # Do not call trimesh's wrapper: its QhullError path silently retries QJ.
    raw=ConvexHull(np.asarray(scan.vertices,np.float64),qhull_options='Qt')
    original_ids=np.sort(raw.vertices);mapping=np.full(len(scan.vertices),-1,np.int64);mapping[original_ids]=np.arange(len(original_ids))
    faces=mapping[raw.simplices].copy();vertices=np.asarray(scan.vertices)[original_ids].copy()
    crosses=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    flips=np.sum(crosses*raw.equations[:,:3],axis=1)<0
    faces[flips]=faces[flips][:,[0,2,1]]
    hull=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    require(hull.is_watertight and hull.is_winding_consistent and hull.is_volume and hull.volume>0,'Finite single watertight outward convex collision surface')
    require(np.isfinite(hull.vertices).all() and np.isfinite(hull.face_normals).all() and bool((hull.area_faces>0).all()),'No invalid hull faces')
    return hull,raw.equations.copy(),original_ids,int(flips.sum())

def cover_triangles(hull):
    centroids=[];face_ids=[];areas=[];edge_lengths=[];counts=[]
    for face_id,triangle in enumerate(hull.triangles):
        pending=[triangle.copy()];count=0
        while pending:
            tri=pending.pop();lengths=np.asarray([np.linalg.norm(tri[1]-tri[0]),np.linalg.norm(tri[2]-tri[1]),np.linalg.norm(tri[0]-tri[2])]);edge=int(lengths.argmax())
            if lengths[edge]<=SPACING:
                centroids.append(tri.mean(0));face_ids.append(face_id);areas.append(float(np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2));edge_lengths.append(float(lengths.max()));count+=1
            else:
                a,b,c=((0,1,2),(1,2,0),(2,0,1))[edge];middle=(tri[a]+tri[b])/2
                pending.append(np.asarray([middle,tri[b],tri[c]]));pending.append(np.asarray([tri[a],middle,tri[c]]))
            require(len(centroids)+len(pending)<200000,'Bounded <=200k sampling cells; no silent coarsening')
        counts.append(count)
    centroids=np.asarray(centroids,np.float64);face_ids=np.asarray(face_ids,np.int64);areas=np.asarray(areas,np.float64);edge_lengths=np.asarray(edge_lengths,np.float64)
    require(edge_lengths.size and edge_lengths.max()<=SPACING and np.isfinite(centroids).all() and (areas>0).all(),'All original hull triangles covered by finite small leaf triangles')
    return centroids,face_ids,areas,edge_lengths,counts

def nearest(mesh,points):
    closest=[];distance=[];triangle=[]
    for first in range(0,len(points),64):
        p,d,t=trimesh.proximity.closest_point_naive(mesh,points[first:first+64]);closest.append(p);distance.append(d);triangle.append(t)
    closest=np.concatenate(closest);distance=np.concatenate(distance);triangle=np.concatenate(triangle)
    require(np.isfinite(closest).all() and np.isfinite(distance).all() and (distance>=0).all(),'Finite complete nearest-triangle evidence')
    return closest,distance,triangle

def overlay(scan,hull,path):
    os.environ['MPLBACKEND']='Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection,LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(12,8));fig.subplots_adjust(left=.07,right=.98,bottom=.09,top=.88,hspace=.36,wspace=.24)
    fig.suptitle('Phone24: unchanged scanned visual + single convex collision candidate\nOriginal model frame and assumed COM retained; finite geometric audit only',fontsize=13)
    center=np.asarray(read(STAGE1/'geometry_report.json')['geometry']['AABB_center_model_m'])
    for index,(i,j) in enumerate(((0,1),(0,2),(2,1)),1):
        ax=fig.add_subplot(2,2,index)
        ax.add_collection(PolyCollection(scan.triangles[:,:,[i,j]],facecolors='.6',edgecolors='none',alpha=.16))
        ax.add_collection(LineCollection(hull.vertices[hull.edges_unique][:,:,[i,j]],colors='#126ab3',linewidths=.5,alpha=.65))
        ax.scatter([0],[0],c='crimson',marker='+',s=50,label='original origin');ax.scatter([center[i]],[center[j]],c='#126ab3',marker='x',s=30,label='unchanged assumed COM')
        ax.autoscale();ax.margins(.08);ax.set_xlabel('XYZ'[i]+' (m)');ax.set_ylabel('XYZ'[j]+' (m)');ax.grid(alpha=.2);ax.set_title('XYZ'[i]+'/'+'XYZ'[j]+' projection — axes independently scaled',fontsize=10)
        if index==1:ax.legend(fontsize=8,loc='upper left')
    ax=fig.add_subplot(2,2,4,projection='3d');ax.add_collection3d(Poly3DCollection(scan.triangles,facecolor='.6',edgecolor='none',alpha=.3));ax.add_collection3d(Poly3DCollection(hull.triangles,facecolor='#247fc0',edgecolor='#126ab3',linewidths=.2,alpha=.10))
    bounds=hull.bounds;size=hull.extents;ax.set_xlim(bounds[:,0]);ax.set_ylim(bounds[:,1]);ax.set_zlim(bounds[:,2]);ax.set_box_aspect(size,zoom=.82);ax.view_init(elev=18,azim=-52);ax.set_axis_off();ax.set_title('3D overlay: blue hull / gray original visual',fontsize=10)
    fig.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.18);plt.close(fig)

def error_plot(signed,distance,centroids,leaf_distance,path):
    os.environ['MPLBACKEND']='Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),layout='constrained');fig.suptitle('Phone24 convex candidate: all raw vertices and full hull-surface sampling\nFinite samples; no physics or true-surface guarantee',fontsize=13)
    axes[0,0].hist(signed*1000,bins=45,color='#347ab7');axes[0,0].axvline(.5,c='crimson',ls='--',label='outside limit0.5mm');axes[0,0].set_xlabel('Raw vertex signed distance to hull (mm)');axes[0,0].set_ylabel('Vertex count');axes[0,0].legend(fontsize=8)
    ordered=np.sort(distance*1000);axes[0,1].plot(ordered,np.arange(1,len(ordered)+1)/len(ordered));axes[0,1].axvline(2,c='orange',ls='--',label='p95 limit2mm');axes[0,1].axvline(4,c='crimson',ls='--',label='max limit4mm');axes[0,1].axhline(.95,c='.5',ls=':');axes[0,1].set_xlabel('Hull sample distance to original triangles (mm)');axes[0,1].set_ylabel('Unweighted empirical CDF');axes[0,1].legend(fontsize=8)
    for ax,(i,j) in zip(axes[1],((0,2),(0,1))):
        colors=ax.scatter(centroids[:,i],centroids[:,j],c=leaf_distance*1000,s=4,cmap='magma',vmin=0,vmax=max(4,float(distance.max()*1000)),rasterized=True)
        ax.set_xlabel('XYZ'[i]+' (m)');ax.set_ylabel('XYZ'[j]+' (m)');ax.set_title('All refined-triangle centroid samples; independent axes',fontsize=10);fig.colorbar(colors,ax=ax,label='distance to scan (mm)')
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.savefig(path,dpi=180,bbox_inches='tight',pad_inches=.16);plt.close(fig)

def main():
    started=time.perf_counter();require(not (ROOT/'report.json').exists(),'Fresh stage2 candidate only; no overwriting completed artifact');(ROOT/'meshes').mkdir(parents=True,exist_ok=True)
    sources={name:pin(path) for name,path in {'builder':Path(__file__),'stage1_geometry':STAGE1/'geometry_report.json','stage1_MJCF':STAGE1/'phone_asset.xml','stage1_visual':STAGE1/'meshes/phone24_visual_m.obj','stage1_sampling_report':STAGE1/'aabb_geometry_audit_v1/report.json','threshold_protocol':PILOT/'protocols/phone_physics_stage1_v1.json','original_GLB':PILOT/'hot3d/object_models/obj_000024.glb'}.items()}
    prior=read(sources['stage1_geometry']['path']);protocol=read(sources['threshold_protocol']['path']);limits=protocol['geometry_error_report']['predeclared_engineering_limits_m']
    require(limits=={'maximum_scanned_vertex_outside_box':.0005,'sampled_box_to_scan_p95':.002,'sampled_box_to_scan_max':.004},'Original numerical geometry gates unchanged')
    require(sources['original_GLB']==prior['source_pins']['GLB'] and prior['revision']=='30fe9674782f32e1e5edba98476b6ff4300132c5','Original same-revision GLB')
    scan,nodes=original_scan(sources['original_GLB']['path']);require(np.array_equal(scan.bounds,np.asarray(prior['geometry']['bounds_model_m'])),'Same original-coordinate meter bake')
    hull_started=time.perf_counter();hull,planes,original_vertex_ids,flipped=single_hull(scan);hull_seconds=time.perf_counter()-hull_started
    visual=ROOT/'meshes/phone24_visual_m.obj';collision=ROOT/'meshes/phone24_collision_convex_m.obj'
    with visual.open('xb') as f:f.write(Path(sources['stage1_visual']['path']).read_bytes())
    with collision.open('x',encoding='utf-8',newline='\n') as f:f.write(trimesh.exchange.obj.export_obj(hull,include_normals=True,include_color=False,include_texture=False,digits=17))
    require(pin(visual)['sha256']==sources['stage1_visual']['sha256'],'Visual OBJ unchanged bytes')
    tree=ET.parse(sources['stage1_MJCF']['path']);root=tree.getroot();root.set('model','hot3d_phone24_convex_candidate_stage2')
    ET.SubElement(root.find('asset'),'mesh',name='phone24_convex_collision',file='phone24_collision_convex_m.obj',scale='1 1 1')
    body=root.find('./worldbody/body');geom=body.find("geom[@name='phone24_collision_aabb']");geom.set('name','phone24_collision_convex');geom.set('type','mesh');geom.set('mesh','phone24_convex_collision');geom.set('pos','0 0 0');del geom.attrib['size']
    ET.indent(root,space='  ')
    with (ROOT/'phone_asset.xml').open('x',encoding='utf-8') as f:f.write('<?xml version="1.0" encoding="utf-8"?>\n'+ET.tostring(root,encoding='unicode')+'\n')
    closest_hull,vertex_distance,vertex_triangle=nearest(hull,np.asarray(scan.vertices));plane_max=(scan.vertices@planes[:,:3].T+planes[:,3]).max(axis=1)
    signed=np.where(plane_max>0,vertex_distance,-vertex_distance)
    centroids,leaf_face,leaf_area,leaf_edge,per_face=cover_triangles(hull);samples=np.concatenate((hull.vertices,centroids));sample_kind=np.r_[np.zeros(len(hull.vertices),np.int8),np.ones(len(centroids),np.int8)]
    sampled_started=time.perf_counter();closest_scan,distance,sample_triangle=nearest(scan,samples);sampling_seconds=time.perf_counter()-sampled_started
    maximum_outside=float(np.maximum(signed,0).max());p95=float(np.percentile(distance,95,method='linear'));maximum=float(distance.max())
    checks={'all_raw_vertices_outside_le_0p5mm':maximum_outside<=.0005,'sampled_hull_to_scan_p95_le_2mm':p95<=.002,'sampled_hull_to_scan_max_le_4mm':maximum<=.004}
    np.savez_compressed(ROOT/'geometry_samples.npz',original_scan_vertices_model_m_F64=scan.vertices,original_scan_faces_int64=scan.faces,hull_vertices_model_m_F64=hull.vertices,hull_faces_int64=hull.faces,hull_vertex_original_scan_indices_int64=original_vertex_ids,hull_halfspace_equations_F64=planes,
        scan_vertex_signed_hull_distance_m_F64=signed,scan_vertex_nearest_hull_point_model_m_F64=closest_hull,scan_vertex_nearest_hull_triangle_int64=vertex_triangle,scan_vertex_max_halfspace_value_F64=plane_max,
        hull_surface_samples_model_m_F64=samples,hull_surface_sample_kind_int8=sample_kind,hull_samples_nearest_scan_point_model_m_F64=closest_scan,hull_samples_nearest_scan_triangle_int64=sample_triangle,hull_samples_to_scan_distance_m_F64=distance,
        refined_leaf_centroids_model_m_F64=centroids,refined_leaf_source_hull_triangle_int64=leaf_face,refined_leaf_area_m2_F64=leaf_area,refined_leaf_max_edge_m_F64=leaf_edge)
    overlay(scan,hull,ROOT/'geometry_overlay.png');error_plot(signed,distance,centroids,distance[len(hull.vertices):],ROOT/'geometry_error.png')
    for value in sources.values():require(pin(value['path'])==value,'Original stage1/source unchanged')
    report={'schema':'hot3d_phone24_single_convex_geometry_candidate:1','status':'PASS_FINITE_SAMPLED_CONVEX_GEOMETRY_NOT_TRUE_SURFACE_PROOF' if all(checks.values()) else 'FAIL_ORIGINAL_GEOMETRY_LIMITS_CONVEX_CANDIDATE','source_pins':sources,'geometry_qualified':all(checks.values()),'qualification_scope':'Only this finite geometric sampling scheme, not continuous Hausdorff distance, true physical surface, dynamics, hand contact or ego retargeting. No physics run.',
        'thresholds_m':{'maximum_scanned_vertex_outside_collision':.0005,'sampled_collision_to_scan_p95':.002,'sampled_collision_to_scan_max':.004},'checks':checks,'passed':sum(checks.values()),'total':len(checks),'GLB_scene_nodes':nodes,
        'hull':{'vertices':len(hull.vertices),'triangles':len(hull.faces),'watertight':bool(hull.is_watertight),'winding_consistent':bool(hull.is_winding_consistent),'positive_volume_m3_geometry_only':float(hull.volume),'bounds_model_m':hull.bounds.tolist(),'original_vertex_indices_retained':True,'face_winding_flips_to_outward':flipped,'origin_recentered':False,'qhull_options':'Qt','random_joggle_or_fallback':False,'construction_seconds':hull_seconds,'collision_proxy_count':1},
        'preserved_stage1':{'visual_byte_identical_sha256':pin(visual)['sha256'],'model_origin':'Original HOT3D model frame unchanged','COM_model_m':prior['unmeasured_engineering_assumptions']['COM_model_m'],'assumptions':prior['unmeasured_engineering_assumptions'],'uniform_box_inertia_not_recomputed_from_hull':True,'inertial_XML_attributes':body.find('inertial').attrib,'visual_geom_XML_attributes':body.find("geom[@name='phone24_visual_only']").attrib},
        'all_original_vertices_to_hull_signed_surface':stats(signed),'maximum_original_vertex_outside_hull_m':maximum_outside,'full_sampled_hull_to_scan':stats(distance),
        'sampling':{'seed':SEED,'random_draws':0,'method':'Deterministic longest-edge bisection of every original hull triangle until all leaf edges<=2mm; sample every leaf centroid plus all original hull vertices. Fixed face/stack order.','maximum_leaf_edge_m':float(leaf_edge.max()),'refined_leaf_count':len(centroids),'original_hull_vertex_samples':len(hull.vertices),'total_samples':len(samples),'leaf_count_per_original_hull_triangle':per_face,'leaf_area_m2_range':[float(leaf_area.min()),float(leaf_area.max())],'max_point_to_leaf_centroid_bound_m':float(2*leaf_edge.max()/3),'percentile':'Unweighted NumPy linear sample percentile; triangle refinement densities are not exactly area uniform.','coverage_limits':'All hull triangles covered by refinement cells, but samples are finite and nonuniform; original scan is nonwatertight with defects. No full continuous Hausdorff or true-contact certificate.','nearest_distance_method':'All original baked scan triangles with closest_point_naive,chunk64','sample_to_scan_distance_seconds':sampling_seconds},
        'original_scan_defects':{key:prior['geometry'][key] for key in ('vertices','faces','is_watertight','is_volume','zero_area_faces','connected_components')},'software':{'trimesh':trimesh.__version__,'numpy':np.__version__,'scipy':importlib.metadata.version('scipy')},
        'MuJoCo_Warp_cost_descriptor':{'collision_convex_count':1,'collision_vertices':len(hull.vertices),'collision_triangles':len(hull.faces),'dense_audit_samples_not_used_in_simulation':True,'backend_compile_or_throughput_tested':False},'actual_MjModel_MjData_steps_training_GPU':0,'elapsed_seconds':time.perf_counter()-started,
        'outputs':{path.relative_to(ROOT).as_posix():pin(path) for path in sorted(ROOT.rglob('*')) if path.is_file()}}
    write(ROOT/'report.json',report);write(ROOT/'output_pins.json',{path.relative_to(ROOT).as_posix():pin(path) for path in sorted(ROOT.rglob('*')) if path.is_file() and path.name!='output_pins.json'})
    print(json.dumps({'status':report['status'],'geometry_qualified':report['geometry_qualified'],'hull_vertices':len(hull.vertices),'hull_triangles':len(hull.faces),'sample_count':len(samples),'p95_m':p95,'max_m':maximum,'report':pin(ROOT/'report.json')}));return 0

if __name__=='__main__':raise SystemExit(main())
