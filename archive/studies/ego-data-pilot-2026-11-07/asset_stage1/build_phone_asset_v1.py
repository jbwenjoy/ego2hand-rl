"""Build first HOT3D phone visual/AABB asset. CPU geometry, optional MjModel only.

All new outputs stay below asset_stage1. No source GLB repair, recentering,
dependency installation, MjData, stepping, hand, model training or GPU work.
"""
from __future__ import annotations
import argparse,hashlib,importlib.metadata,json,math,os,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET

BASE=Path(__file__).resolve().parent
PILOT=BASE.parent
REVISION='30fe9674782f32e1e5edba98476b6ff4300132c5'
SOURCE=PILOT/'hot3d/object_models/obj_000024.glb'
MODELS_INFO=PILOT/'hot3d/object_models/models_info.json'
MANIFEST=PILOT/'selected_models_manifest_v1.json'
SOURCE_SHA='f5548e475296967ad9797bc51fa23738648c2def23082dce6fd300c271439b69'
INFO_SHA='351e1b25a250606fa4b7500f5aeb800b55050bf1c638d05ceb464f8ddfb32e2b'
ASSUMPTIONS={'mass_kg':.2,'COM_model':'AABB center, unmeasured','inertia_model':'uniform solid AABB, unmeasured','sliding_friction':.6,'torsional_friction_m':.005,'rolling_friction_m':.0001,'condim':4,'priority':0,'margin_m':0.,'gap_m':0.,'solref':[.02,1.],'solimp':[.9,.95,.001,.5,2.]}

def require(v,s):
    if not v:raise ValueError(s)

def pin(path):
    p=Path(path).resolve(strict=True);return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def vec(value):return ' '.join(format(float(v),'.17g') for v in value)

def xml_scene(center,half,inertia,*,table=False):
    root=ET.Element('mujoco',model='hot3d_phone24_table_stage1' if table else 'hot3d_phone24_asset_stage1')
    ET.SubElement(root,'compiler',angle='radian',meshdir='meshes',inertiafromgeom='false')
    ET.SubElement(root,'option',timestep='.005',integrator='implicitfast',solver='Newton',iterations='10',ls_iterations='20',cone='pyramidal',jacobian='auto',gravity='0 0 -9.81')
    asset=ET.SubElement(root,'asset');ET.SubElement(asset,'mesh',name='phone24_visual',file='phone24_visual_m.obj',scale='1 1 1')
    world=ET.SubElement(root,'worldbody')
    contact={'friction':vec([ASSUMPTIONS['sliding_friction'],ASSUMPTIONS['torsional_friction_m'],ASSUMPTIONS['rolling_friction_m']]),'condim':'4','priority':'0','margin':'0','gap':'0','solref':vec(ASSUMPTIONS['solref']),'solimp':vec(ASSUMPTIONS['solimp'])}
    if table:
        ET.SubElement(world,'geom',name='stage1_table',type='box',pos='0 0 -0.025',size='.25 .20 .025',rgba='.55 .58 .62 1',contype='1',conaffinity='1',**contact)
        ET.SubElement(world,'light',name='stage1_light',pos='.2 -.3 .5',dir='-.2 .3 -.5')
        ET.SubElement(world,'camera',name='stage1_camera',pos='.3 -.4 .3',xyaxes='0.8 0.6 0 -.36 .48 .8')
        # Rx(+90deg): localY(thickness) maps to worldZ. Original mesh/model
        # origin is retained, only this demo body's world pose changes.
        spawn=[0.,0.,.015-(center[1]-half[1])];quat=[math.sqrt(.5),math.sqrt(.5),0.,0.]
    else:spawn=[0.,0.,0.];quat=[1.,0.,0.,0.]
    body=ET.SubElement(world,'body',name='hot3d_phone24',pos=vec(spawn),quat=vec(quat))
    ET.SubElement(body,'freejoint',name='phone24_freejoint')
    ET.SubElement(body,'inertial',pos=vec(center),mass=str(ASSUMPTIONS['mass_kg']),diaginertia=vec(inertia))
    ET.SubElement(body,'geom',name='phone24_visual_only',type='mesh',mesh='phone24_visual',contype='0',conaffinity='0',density='0',group='2',rgba='.22 .24 .28 1')
    ET.SubElement(body,'geom',name='phone24_collision_aabb',type='box',pos=vec(center),size=vec(half),contype='1',conaffinity='1',density='0',group='3',rgba='.1 .45 .95 .16',**contact)
    ET.indent(root,space='  ')
    return '<?xml version="1.0" encoding="utf-8"?>\n'+ET.tostring(root,encoding='unicode')+'\n',{'position_world_m':spawn,'quaternion_wxyz':quat,'initial_AABB_table_clearance_m':.015 if table else None,'model_origin_recentered':False}

def overlay(path,mesh,bounds,center,np):
    os.environ['MPLBACKEND']='Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from itertools import product
    corners=np.asarray(list(product(*zip(bounds[0],bounds[1]))));edges=[(a,b) for a in range(8) for b in range(a+1,8) if np.count_nonzero(corners[a]!=corners[b])==1]
    fig=plt.figure(figsize=(12,8),layout='constrained');fig.suptitle('HOT3D phone24: baked visual mesh and AABB collision proxy\nOriginal model frame retained; box is an engineering approximation',fontsize=14)
    for index,(i,j) in enumerate(((0,1),(0,2),(2,1)),1):
        ax=fig.add_subplot(2,2,index);triangles=mesh.vertices[mesh.faces][:,:,[i,j]]
        ax.add_collection(PolyCollection(triangles,facecolors='.55',edgecolors='none',alpha=.28))
        lo,hi=bounds[:,[i,j]];closed=np.asarray([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]],[lo[0],lo[1]]])
        ax.plot(closed[:,0],closed[:,1],color='#2870cc',lw=1.5,label='AABB proxy');ax.scatter([0],[0],c='crimson',marker='+',s=65,label='model origin');ax.scatter([center[i]],[center[j]],c='#2870cc',marker='x',s=30,label='assumed COM')
        ax.autoscale();ax.margins(.12);ax.set_aspect('equal');ax.set_xlabel('XYZ'[i]+' (m)');ax.set_ylabel('XYZ'[j]+' (m)');ax.grid(alpha=.2)
        if index==1:ax.legend(fontsize=8,loc='upper right')
    ax=fig.add_subplot(2,2,4,projection='3d');ax.add_collection3d(Poly3DCollection(mesh.vertices[mesh.faces],facecolor='.6',edgecolor='none',alpha=.45))
    for a,b in edges:ax.plot(*corners[[a,b]].T,color='#2870cc',lw=1)
    ax.scatter([0],[0],[0],c='crimson',marker='+',s=40);ax.scatter(*center[:,None],c='#2870cc',marker='x',s=25)
    ax.set_xlim(bounds[:,0]);ax.set_ylim(bounds[:,1]);ax.set_zlim(bounds[:,2]);ax.set_box_aspect(bounds[1]-bounds[0]);ax.view_init(elev=22,azim=-55)
    ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)');ax.set_title('Visual surface + enclosing box')
    fig.savefig(path,dpi=180);plt.close(fig)

def build(out,compile_model=False):
    started=time.perf_counter();out=Path(out).resolve();require(out==BASE or BASE in out.parents,'All writes confined to asset_stage1')
    out.mkdir(parents=True,exist_ok=True)
    require(not any((out/name).exists() for name in ('geometry_report.json','phone_asset.xml','table_scene.xml','geometry_overlay.png')),'Fresh output only; never overwrite completed asset')
    original_sources={name:pin(path) for name,path in {'GLB':SOURCE,'models_info':MODELS_INFO,'download_manifest':MANIFEST,'builder':Path(__file__),'pinned_HOT3D_visualizer':PILOT/'sources/independent_parser_source_v1/hot3d__hot3d__clips__vis_clips.py'}.items()}
    require(original_sources['GLB']['sha256']==SOURCE_SHA and original_sources['models_info']['sha256']==INFO_SHA,'Selected original GLB and metadata hashes')
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'));require(manifest['revision']==REVISION and manifest['selection']['24']=='cellphone','Same frozen HOT3D revision/name')
    for name in ('GLB','models_info'):
        require(any(Path(row['path']).samefile(original_sources[name]['path']) and row['sha256']==original_sources[name]['sha256'] for row in manifest['assets']),'Actual downloaded source manifest binding '+name)
    import numpy as np
    import trimesh
    require(trimesh.__version__=='4.11.2','Reviewed local trimesh version')
    info=json.loads(MODELS_INFO.read_text(encoding='utf-8'))['24'];scene=trimesh.load_scene(SOURCE,process=False)
    nodes=[];vertices=[];faces=[];offset=0
    for name in sorted(scene.graph.nodes_geometry):
        transform,geometry=scene.graph[name];g=scene.geometry[geometry]
        require(isinstance(g,trimesh.Trimesh) and g.faces.ndim==2 and g.faces.shape[1]==3,'Triangle visual source geometry')
        raw=np.asarray(g.vertices,np.float64);T=np.asarray(transform,np.float64);require(T.shape==(4,4) and np.isfinite(T).all() and np.array_equal(T[3],[0,0,0,1]),'Finite affine scene transform')
        baked=trimesh.transform_points(raw,T)
        require(np.isfinite(baked).all() and np.isfinite(raw).all() and len(raw)>0 and len(g.faces)>0 and int(g.faces.min())>=0 and int(g.faces.max())<len(raw),'Finite referenced source mesh')
        vertices.append(baked);faces.append(np.asarray(g.faces,np.int64)+offset)
        nodes.append({'node':name,'geometry':geometry,'scene_base_frame':scene.graph.base_frame,'transform_to_model':T.tolist(),'linear_determinant':float(np.linalg.det(T[:3,:3])),'linear_singular_values':np.linalg.svd(T[:3,:3],compute_uv=False).tolist(),'raw_bounds':np.asarray(g.bounds).tolist(),'baked_bounds_m':np.asarray([baked.min(0),baked.max(0)]).tolist(),'vertices':len(raw),'faces':len(g.faces),'vertex_offset':offset})
        offset+=len(raw)
    mesh=trimesh.Trimesh(vertices=np.concatenate(vertices),faces=np.concatenate(faces),process=False)
    bounds=np.asarray(mesh.bounds,np.float64);size=bounds[1]-bounds[0];center=(bounds[1]+bounds[0])/2;half=size/2
    require((size>0).all(),'Positive AABB extents')
    info_bounds=np.asarray([[info['min_'+axis] for axis in 'xyz'],[info['min_'+axis]+info['size_'+axis] for axis in 'xyz']],np.float64)
    require(np.max(np.abs(bounds-info_bounds))<1e-10,'Baked scene is already meters and agrees with same-revision models_info; do not apply an extra .001')
    meshes=out/'meshes';meshes.mkdir(exist_ok=True);visual=meshes/'phone24_visual_m.obj'
    require(not visual.exists(),'Fresh visual mesh')
    # Original vertex order/connectivity is preserved; normals are derived from
    # the baked triangles. No welding, smoothing repair, centering or rescaling.
    text=trimesh.exchange.obj.export_obj(mesh,include_normals=True,include_color=False,include_texture=False,digits=17)
    with visual.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    vertex_lines=[line.split()[1:4] for line in text.splitlines() if line.startswith('v ')]
    face_lines=[line.split()[1:] for line in text.splitlines() if line.startswith('f ')]
    reload_vertices=np.asarray(vertex_lines,np.float64);reload_faces=np.asarray([[int(v.split('/')[0])-1 for v in row] for row in face_lines],np.int64)
    require(reload_vertices.shape==mesh.vertices.shape and np.array_equal(reload_faces,mesh.faces),'OBJ preserves exact face/vertex order')
    area=np.asarray(mesh.area_faces);normals=np.asarray(mesh.face_normals);normal_norm=np.linalg.norm(normals,axis=1)
    from itertools import product
    corners=np.asarray(list(product(*zip(bounds[0],bounds[1]))))
    _,corner_distance,_=trimesh.proximity.closest_point_naive(mesh,corners)
    vertex_box_surface_distance=np.min(half-np.abs(mesh.vertices-center),axis=1)
    box_volume=float(np.prod(size));mass=ASSUMPTIONS['mass_kg'];inertia=mass/12*np.asarray([size[1]**2+size[2]**2,size[0]**2+size[2]**2,size[0]**2+size[1]**2])
    poses={}
    for filename,table in (('phone_asset.xml',False),('table_scene.xml',True)):
        xml,poses[filename]=xml_scene(center,half,inertia,table=table)
        with (out/filename).open('x',encoding='utf-8') as f:f.write(xml)
    overlay(out/'geometry_overlay.png',mesh,bounds,center,np)
    compiled={}
    if compile_model:
        import mujoco
        for filename in ('phone_asset.xml','table_scene.xml'):
            model=mujoco.MjModel.from_xml_path(str(out/filename));bid=model.body('hot3d_phone24').id;cid=model.geom('phone24_collision_aabb').id;vid=model.geom('phone24_visual_only').id
            require(float(model.body_mass[bid])==mass and np.allclose(model.body_ipos[bid],center,rtol=0,atol=1e-15) and np.allclose(model.body_inertia[bid],inertia,rtol=0,atol=1e-15),'Compiled explicit box mass/COM/principal inertia')
            require(int(model.geom_contype[vid])==int(model.geom_conaffinity[vid])==0 and int(model.geom_type[cid])==int(mujoco.mjtGeom.mjGEOM_BOX),'Visual excluded from all collision; primitive box only')
            compiled[filename]={'nq':int(model.nq),'nv':int(model.nv),'nbody':int(model.nbody),'ngeom':int(model.ngeom),'object_mass_kg':float(model.body_mass[bid]),'body_ipos_m':model.body_ipos[bid].tolist(),'body_inertia_kg_m2':model.body_inertia[bid].tolist(),'body_iquat_wxyz':model.body_iquat[bid].tolist(),'visual_geom_contype':int(model.geom_contype[vid]),'visual_geom_conaffinity':int(model.geom_conaffinity[vid]),'collision_halfsize_m':model.geom_size[cid].tolist(),'MjData_created':False,'physical_steps':0}
    for key,value in original_sources.items():require(pin(value['path'])==value,'Original source or builder changed '+key)
    require(not any(key in sys.modules for key in ('torch','warp','mujoco_warp')),'No NN/Warp/GPU runtime imported')
    report={'schema':'hot3d_phone24_asset_geometry_stage1:1','status':'BUILT_VISUAL_AND_AABB_PROXY_NOT_COLLISION_OR_DYNAMICS_ACCURACY','revision':REVISION,'object_bop_id':24,'name':info['name'],'source_pins':original_sources,
        'software':{'trimesh':trimesh.__version__,'numpy':np.__version__,'mujoco':importlib.metadata.version('mujoco') if compile_model else None},'models_info24':info,'scene_declared_units':scene.units,'scene_nodes':nodes,'bake_policy':'Apply every node-to-model transform once; no extra unit scale, recentering, repair, welding or reorientation. OBJ is visual geometry without original PBR materials/textures.',
        'geometry':{'vertices':len(mesh.vertices),'faces':len(mesh.faces),'bounds_model_m':bounds.tolist(),'size_model_m':size.tolist(),'AABB_center_model_m':center.tolist(),'source_model_origin_model_m':[0.,0.,0.],'metadata_bounds_maxabs_difference_m':float(np.abs(bounds-info_bounds).max()),'export_vertex_maxabs_difference_m':float(np.abs(reload_vertices-mesh.vertices).max()),'export_faces_exact':True,'face_normal_norm_range':normal_norm[[int(np.argmin(normal_norm)),int(np.argmax(normal_norm))]].tolist(),'zero_area_faces':int((area==0).sum()),'faces_area_le_1e_18_m2':int((area<=1e-18).sum()),'nonfinite_normals':int((~np.isfinite(normals)).sum()),'is_watertight':bool(mesh.is_watertight),'is_winding_consistent':bool(mesh.is_winding_consistent),'is_volume':bool(mesh.is_volume),'signed_mesh_volume_m3_descriptor':float(mesh.volume),'signed_volume_not_used_for_inertia':True,'connected_components':len(mesh.split(only_watertight=False))},
        'collision_proxy':{'kind':'AABB primitive box enclosing baked visual vertices','center_model_m':center.tolist(),'halfsize_m':half.tolist(),'volume_m3':box_volume,'max_vertex_outside_box_m':float(np.maximum(np.abs(mesh.vertices-center)-half,0).max()),'source_vertex_to_box_surface_distance_m':{'min':float(vertex_box_surface_distance.min()),'mean':float(vertex_box_surface_distance.mean()),'max':float(vertex_box_surface_distance.max())},'eight_box_corners_to_original_triangle_surface_m':corner_distance.tolist(),'distances_are_not_Hausdorff_or_true_contact_error':True,'precise_collision_qualified':False},
        'unmeasured_engineering_assumptions':{**ASSUMPTIONS,'mass_source':'engineering nominal only, not HOT3D measured','COM_model_m':center.tolist(),'principal_inertia_kg_m2':inertia.tolist(),'inertia_principal_axes':'original model XYZ, uniform AABB only','mass_friction_ranges_owned_by_Root_protocol':True},'world_poses':poses,'CPU_compile':compiled,'scope':{'MjModel_compiles':len(compiled),'MjData_Env_Hand2_physical_steps_training_GPU':0,'old_source_GLBs_and_research_unchanged':True},'elapsed_seconds':time.perf_counter()-started}
    write(out/'geometry_report.json',report)
    write(out/'output_pins.json',{path.relative_to(out).as_posix():pin(path) for path in sorted(out.rglob('*')) if path.is_file() and path.name!='output_pins.json'})
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=BASE);parser.add_argument('--compile',action='store_true');args=parser.parse_args()
    result=build(args.output,args.compile);print(json.dumps({'status':result['status'],'geometry_report':pin(args.output/'geometry_report.json'),'size_m':result['geometry']['size_model_m'],'CPU_compile_count':len(result['CPU_compile'])}));return 0

if __name__=='__main__':raise SystemExit(main())
