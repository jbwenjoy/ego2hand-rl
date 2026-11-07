"""Official pinned UmeTrack CPU FK and descriptive hand-phone geometry.

No MANO/NN weights, optimizer, robot retargeting, MuJoCo, or dynamics.
All 22 pose parameters are passed intact to the unmodified official revision;
its own skinning routine uses the first20 and canonical output uses first20
of21 profile landmarks. This is recorded, never a robot joint mapping.
"""
from pathlib import Path
import csv, dataclasses, hashlib, json, os, sys, tarfile, time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent
CLIP = PILOT / 'hot3d/train_aria/clip-001856.tar'
PHONE = PILOT / 'asset_stage2_convex/completed_v2'
FRAMES = list(range(42, 103))


def pin(p):
    p = Path(p).resolve(strict=True)
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def require(v, message):
    if not v:
        raise ValueError(message)


def write(p, obj):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, allow_nan=False)


def transform(T, points, np):
    return points @ T[:3, :3].T + T[:3, 3]


def closest_all(mesh, points, np, trimesh, chunk=32):
    closest, distance, triangle = [], [], []
    for first in range(0, len(points), chunk):
        c, d, t = trimesh.proximity.closest_point_naive(mesh, points[first:first+chunk])
        closest.append(c); distance.append(d); triangle.append(t)
    return np.concatenate(closest), np.concatenate(distance), np.concatenate(triangle)


def main():
    started = time.perf_counter()
    require(not (ROOT/'report.json').exists(), 'Fresh geometry output required')
    import numpy as np
    import torch
    import trimesh
    from vendor_umetrack_9064a2 import math_utils
    from vendor_umetrack_9064a2.hand_models import umetrack_hand_model as official
    torch.set_num_threads(4)
    require(not torch.cuda.is_initialized(), 'CPU FK only')
    vendor = json.loads((ROOT/'vendor_source_manifest.json').read_text())
    for item in vendor:
        require(pin(item['path'])['sha256'] == item['sha256'], 'Exact unmodified vendor bytes')
    sources = {'builder': pin(__file__), 'clip': pin(CLIP),
               'vendor_manifest': pin(ROOT/'vendor_source_manifest.json'),
               'phone_visual': pin(PHONE/'meshes/phone24_visual_m.obj'),
               'phone_convex': pin(PHONE/'meshes/phone24_collision_convex_m.obj'),
               'phone_report': pin(PHONE/'report.json')}
    raw, angles, wrist, objects, records = [], [], [], [], []
    with tarfile.open(CLIP) as tar:
        load = lambda name: json.load(tar.extractfile(name))
        profile_all = load('__hand_shapes.json__')
        profile = profile_all['umetrack']
        for frame in FRAMES:
            hands = load(f'{frame:06d}.hands.json'); info = load(f'{frame:06d}.info.json')
            obj = load(f'{frame:06d}.objects.json')['24']
            require('left' in hands and len(obj) == 1, 'One observed left hand and phone24 per frame')
            pose = hands['left']['umetrack_pose']; a = pose['joint_angles']
            require(len(a) == 22, 'Keep all22 parameters')
            w, o = pose['T_world_from_wrist'], obj[0]['T_world_from_object']
            matrix = lambda p: math_utils.quat_trans_to_matrix(*p['quaternion_wxyz'], *p['translation_xyz'])
            angles.append(a); wrist.append(matrix(w)); objects.append(matrix(o))
            raw.append({'frame_id': frame, 'left_umetrack_pose': pose,
                        'phone24': {k: obj[0][k] for k in ('object_bop_id','object_uid','T_world_from_object')}, 'info': info})
            records.append({'array_row': len(records), 'frame_id': frame,
                            'ref_timestamp_ns': info['ref_timestamp_ns'],
                            'image_timestamps_ns': info['image_timestamps_ns'],
                            'sequence_id': info['sequence_id'], 'participant_id': info['participant_id']})
    write(ROOT/'raw_profile_and_poses.json', {'umetrack_profile': profile, 'frames': raw})
    write(ROOT/'frame_identities.json', records)
    angles = np.asarray(angles, dtype=np.float32); wrist = np.asarray(wrist); objects = np.asarray(objects)
    require(angles.shape == (61,22) and np.isfinite(angles).all(), 'Finite full22 poses')
    model = official.from_json(profile)
    # Official from_json performs the only mm->m conversion on profile positions.
    batched = official.UmeTrackHandModelData(**{
        field.name: (getattr(model,field.name).unsqueeze(0).expand(61,*getattr(model,field.name).shape).clone()
                     if isinstance(getattr(model,field.name),torch.Tensor) else getattr(model,field.name))
        for field in dataclasses.fields(model)})
    pose = official.UmeTrackHandPose(hand_side=torch.zeros(61,dtype=torch.int64),
            joint_angles=torch.from_numpy(angles), wrist_xform=torch.from_numpy(wrist))
    with torch.inference_mode():
        landmarks, vertices, faces = official.forward_kinematics(pose,batched,requires_mesh=True)
        all_landmarks = official.skin_landmarks(batched,pose.joint_angles,pose.wrist_xform)
        local_pose = official.UmeTrackHandPose(hand_side=pose.hand_side,
                joint_angles=pose.joint_angles,wrist_xform=torch.eye(4).expand(61,4,4).clone())
        local_landmarks, local_vertices, _ = official.forward_kinematics(local_pose,batched,requires_mesh=True)
    lworld = landmarks.numpy(); vworld = vertices.numpy(); raw21world = all_landmarks.numpy(); faces = faces.numpy()
    require(lworld.shape == (61,20,3) and vworld.shape == (61,788,3) and faces.shape == (61,1544,3), 'Official output shapes')
    require(np.isfinite(vworld).all() and np.isfinite(raw21world).all(), 'Finite geometry')
    require(np.array_equal(lworld,raw21world[:,:20]), 'Canonical source mapping unchanged')
    require(all(np.array_equal(faces[0],f) for f in faces), 'Fixed mesh topology')
    # Raw official FP32 matrix -> explicit F64 inverse, no projection or new scale.
    invw, invo = np.linalg.inv(wrist.astype(np.float64)), np.linalg.inv(objects.astype(np.float64))
    vwrist = np.stack([transform(t,v,np) for t,v in zip(invw,vworld)])
    vobject = np.stack([transform(t,v,np) for t,v in zip(invo,vworld)])
    lwrist = np.stack([transform(t,v,np) for t,v in zip(invw,raw21world)])
    lobject = np.stack([transform(t,v,np) for t,v in zip(invo,raw21world)])
    wrist_roundtrip = np.stack([transform(t,v,np) for t,v in zip(wrist.astype(np.float64),vwrist)])-vworld.astype(np.float64)
    object_roundtrip = np.stack([transform(t,v,np) for t,v in zip(objects.astype(np.float64),vobject)])-vworld.astype(np.float64)
    local_match = np.stack([transform(t,v,np) for t,v in zip(wrist.astype(np.float64),local_vertices.numpy())])-vworld.astype(np.float64)
    np.savez_compressed(ROOT/'left_hand_phone_geometry61.npz',frame_ids=np.array(FRAMES),joint_angles22_FP32=angles,
        T_world_from_wrist_FP32=wrist,T_world_from_object_FP32=objects,
        landmarks20_world_FP32=lworld,landmarks21_raw_world_FP32=raw21world,
        landmarks21_wrist_F64=lwrist,landmarks21_object_F64=lobject,
        hand_vertices_world_FP32=vworld,hand_vertices_wrist_F64=vwrist,hand_vertices_object_F64=vobject,
        hand_triangles=faces[0],official_identity_wrist_vertices_FP32=local_vertices.numpy(),
        official_identity_wrist_landmarks20_FP32=local_landmarks.numpy())
    # Descriptive proximity only: unsigned nearest ORIGINAL scan triangle.
    # Scan is not watertight; no inside/contact-force claim follows from this.
    scan = trimesh.load(PHONE/'meshes/phone24_visual_m.obj',process=False,force='mesh')
    hull = trimesh.load(PHONE/'meshes/phone24_collision_convex_m.obj',process=False,force='mesh')
    print(json.dumps({'phase':'FK_SAVED_BEGIN_UNSIGNED_TRIANGLE_DISTANCES','frames':61,'mesh_vertex_queries':61*788,'landmark_queries':61*21}),flush=True)
    c,d,tri = closest_all(scan,vobject.reshape(-1,3),np,trimesh)
    lc,ld,ltri = closest_all(scan,lobject.reshape(-1,3),np,trimesh)
    d=d.reshape(61,788);ld=ld.reshape(61,21)
    normals=np.asarray(hull.face_normals);offsets=np.einsum('ti,ti->t',normals,np.asarray(hull.triangles)[:,0,:])
    halfspace_max=np.stack([(v@normals.T-offsets).max(-1) for v in vobject])
    np.savez_compressed(ROOT/'proximity_candidates61.npz',hand_vertex_unsigned_scan_distance_m=d,
        hand_vertex_closest_scan_point_object_m=c.reshape(61,788,3),hand_vertex_closest_scan_triangle=tri.reshape(61,788),
        landmark21_unsigned_scan_distance_m=ld,landmark21_closest_scan_point_object_m=lc.reshape(61,21,3),
        landmark21_closest_scan_triangle=ltri.reshape(61,21),hand_vertex_convex_max_plane_signed_m=halfspace_max)
    with (ROOT/'per_frame_proximity_descriptors.csv').open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['frame_id','timestamp_ns','hand_vertex_min_unsigned_scan_m','hand_vertex_p05_unsigned_scan_m','hand_vertex_median_unsigned_scan_m','landmark21_min_unsigned_scan_m','hull_inside_vertex_count','sampled_vertices']);writer.writeheader()
        for i,identity in enumerate(records):
            writer.writerow({'frame_id':identity['frame_id'],'timestamp_ns':identity['ref_timestamp_ns'],
                'hand_vertex_min_unsigned_scan_m':float(d[i].min()),'hand_vertex_p05_unsigned_scan_m':float(np.quantile(d[i],.05)),
                'hand_vertex_median_unsigned_scan_m':float(np.median(d[i])),'landmark21_min_unsigned_scan_m':float(ld[i].min()),
                'hull_inside_vertex_count':int((halfspace_max[i]<0).sum()),'sampled_vertices':788})
    (ROOT/'mesh_frames').mkdir()
    for i in (0,30,60):
        trimesh.Trimesh(vertices=vobject[i],faces=faces[0],process=False).export(ROOT/f'mesh_frames/frame_{FRAMES[i]:06d}_left_hand_object_m.obj')
    make_plots(scan,vobject,faces[0],lobject,d,records,np)
    report={'schema':'hot3d1856_left_official_umetrack_geometry:1','status':'COMPLETED_REFERENCE_GEOMETRY_NOT_RETARGETING_OR_CONTACT_FORCE',
        'source_pins':sources,'vendor_revision':'9064a2fe7fda51bb1f1451a0d3d0da7f3ef5e64e','official_sources_modified':False,
        'frames_inclusive':[42,102],'frame_count':61,'human_hand':'left','robot_mapping':'NOT_IMPLEMENTED',
        'pose_parameter_count_preserved':22,'official_skinning_uses_pose_indices':list(range(20)),
        'official_unused_tail_indices':[20,21],'tail_angles_minmax_rad':[angles[:,20:].min(0).tolist(),angles[:,20:].max(0).tolist()],
        'canonical_landmark_count':20,'raw_skinned_profile_landmark_count':21,'mesh_vertices':788,'mesh_triangles':1544,
        'units':{'joint_angles':'rad','raw_profile_positions':'mm','official_profile_conversion':'x0.001 exactly once in from_json','world_translation_and_output_geometry':'m','timestamps':'ns'},
        'transform_convention':'Official quat_trans_to_matrix(wxyz,translation_m)->FP32; left hand has no right-hand reflection. Derived local coordinates use explicit F64 inverse of official FP32 transform, no reorthogonalization.',
        'coordinate_checks':{'wrist_roundtrip_max_abs_m':float(abs(wrist_roundtrip).max()),'object_roundtrip_max_abs_m':float(abs(object_roundtrip).max()),'direct_local_FK_to_world_max_abs_m':float(abs(local_match).max()),'wrist_rotation_det_minmax':[float(np.linalg.det(wrist[:,:3,:3].astype(np.float64)).min()),float(np.linalg.det(wrist[:,:3,:3].astype(np.float64)).max())]},
        'proximity':{'query':'All788 skinned mesh vertices and21 raw landmarks per frame to original phone scan triangles; exact brute-force triangle nearest distance, chunk32','mesh_queries':61*788,'landmark_queries':61*21,'unsigned_distance_only':True,'minimum_mesh_vertex_unsigned_scan_distance_m':float(d.min()),'maximum_over_frames_of_min_distance_m':float(d.min(1).max()),'convex_plane_sign':'negative means inside convex envelope; positive outside max halfspace. Not Euclidean signed distance outside edges; not signed original-scan distance.','not_physical_contact_qualification':True,'no_new_threshold_or_frame_selection':True},
        'counts':{'official_CPU_FK_batch61_calls':2,'raw21_landmark_skin_batch61_calls':1,'NN_weights_models_training_optimizer_MuJoCo_physics_GPU':0},
        'CUDA_initialized':torch.cuda.is_initialized(),'elapsed_seconds':time.perf_counter()-started,
        'limitations':['Fixed official revision internally ignores final2 pose angles; they are preserved, not converted to robot DOFs.','Canonical20/profile21 landmarks are not Hand2 joint targets.','World tracking and skinned hand geometry are reference estimates; skin overlap and small unsigned gaps are geometric candidates, not measured contact or force.','Mesh vertex sampling is not a continuous hand-surface collision certificate.','Original phone visual scan is nonwatertight; unsigned scan distances and convex inside flags are separate descriptors.','Frames42–102 are a preselected known clip window, not independent trials.','No robot reflection, scale fit, joint truncation, grasp or holding test has been performed.']}
    require(not report['CUDA_initialized'],'No CUDA initialization')
    report['outputs']={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='report.json'}
    write(ROOT/'report.json',report)
    print(json.dumps({'status':report['status'],'report':pin(ROOT/'report.json'),'seconds':report['elapsed_seconds']}),flush=True)


def make_plots(scan,vertices,faces,landmarks,distances,records,np):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig=plt.figure(figsize=(15,9),layout='constrained');bottom_axes=[]
    for col,i in enumerate((0,30,60)):
        ax=fig.add_subplot(2,3,col+1,projection='3d')
        for mesh,color,alpha in [(scan,'#777777',.28)]:
            ax.add_collection3d(Poly3DCollection(mesh.triangles*1000,facecolor=color,edgecolor='none',alpha=alpha))
        ax.add_collection3d(Poly3DCollection(vertices[i][faces]*1000,facecolor='#46a4bf',edgecolor='none',alpha=.35))
        ax.scatter(*landmarks[i].T*1000,c='#c73535',s=9)
        cloud=np.concatenate([vertices[i],scan.vertices])*1000;lo=cloud.min(0);hi=cloud.max(0);margin=(hi-lo)*.06
        ax.set(xlim=(lo[0]-margin[0],hi[0]+margin[0]),ylim=(lo[1]-margin[1],hi[1]+margin[1]),zlim=(lo[2]-margin[2],hi[2]+margin[2]))
        ax.set_box_aspect(hi-lo);ax.view_init(elev=22,azim=-64);ax.set_axis_off()
        ax.set_title(f"Frame {records[i]['frame_id']}: left hand in phone frame")
        ax2=fig.add_subplot(2,3,col+4);bottom_axes.append(ax2)
        ax2.triplot(scan.vertices[:,0]*1000,scan.vertices[:,2]*1000,scan.faces,color='#888888',lw=.25,alpha=.35)
        points=ax2.scatter(vertices[i,:,0]*1000,vertices[i,:,2]*1000,c=np.minimum(distances[i]*1000,20),s=5,vmin=0,vmax=20,cmap='viridis')
        ax2.scatter(landmarks[i,:,0]*1000,landmarks[i,:,2]*1000,s=15,c='#c73535',marker='x')
        ax2.set_aspect('equal');ax2.set(xlabel='Phone X (mm)',ylabel='Phone Z (mm)',title='Original scan + hand mesh samples')
    fig.colorbar(points,ax=bottom_axes,label='Unsigned scan distance (mm; colors capped20)',shrink=.7)
    fig.suptitle('HOT3D1856 / official UmeTrack left geometry — no robot mapping or force inference')
    fig.savefig(ROOT/'geometry_reference_overlay.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    frames=[r['frame_id'] for r in records]
    ax.plot(frames,distances.min(1)*1000,label='Min of788 mesh vertices')
    ax.plot(frames,np.quantile(distances,.05,axis=1)*1000,label='5th percentile of788 vertices')
    ax.set(xlabel='Clip frame',ylabel='Unsigned scan distance (mm)',title='Geometric proximity descriptors; no contact-force label')
    ax.grid(alpha=.2);ax.legend();fig.savefig(ROOT/'proximity_descriptors.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    main()
