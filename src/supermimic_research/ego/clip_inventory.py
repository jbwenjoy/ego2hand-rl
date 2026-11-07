"""Generic known-mesh object/wrist screening from saved HOT3D clip JSON.

Reads tar members without extracting files or importing historical scripts.
Visibility, motion and wrist proximity are descriptors, never contact labels.
"""
import json,tarfile
import numpy as np


def pose(value):
    if not isinstance(value,dict):return None
    p=np.asarray(value.get('translation_xyz',[]),dtype=float);q=np.asarray(value.get('quaternion_wxyz',[]),dtype=float)
    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(np.r_[p,q]).all() or abs(np.linalg.norm(q)-1)>1e-6:return None
    # Normalize source annotations only for a rotation descriptor, not simulation.
    w,x,y,z=q/np.linalg.norm(q)
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    return p,R


def read_clip(path,object_ids):
    rows=[]
    with tarfile.open(path,'r') as tar:
        frames=sorted(int(m.name.split('.')[0]) for m in tar.getmembers() if m.name.endswith('.info.json'))
        if frames!=list(range(150)):raise ValueError('Explicit150-frame pilot clip required')
        for frame in frames:
            def read(suffix):return json.load(tar.extractfile(f'{frame:06d}.{suffix}.json'))
            info,objects,hands=read('info'),read('objects'),read('hands');selected={}
            for oid in object_ids:
                for ob in objects.get(str(oid),[]):
                    uid=str(ob['object_uid']);key=f'{oid}:{uid}'
                    if key in selected:raise ValueError('Duplicate object instance in frame')
                    selected[key]=dict(id=oid,uid=uid,name=ob['object_name'],pose=ob.get('T_world_from_object'),rgb_visibility=ob.get('visibilities_predicted',{}).get('214-1'))
            h={side:dict(pose=((hands.get(side) or {}).get('umetrack_pose') or {}).get('T_world_from_wrist'),rgb_visibility=((hands.get(side) or {}).get('visibilities_modeled') or {}).get('214-1')) for side in ('left','right')}
            rows.append(dict(frame=frame,timestamp_ns=info['ref_timestamp_ns'],sequence_id=info['sequence_id'],participant_id=info['participant_id'],objects=selected,hands=h))
    if any(b['timestamp_ns']<=a['timestamp_ns'] for a,b in zip(rows[:-1],rows[1:])):raise ValueError('Strict timestamps required')
    return rows


def describe_window(rows,key,side):
    positions=[];relative=[];rotations=[];co=object_vis=hand_vis=0
    for row in rows:
        ob=row['objects'].get(key);hand=row['hands'][side]
        if ob is None:return None
        O,H=pose(ob['pose']),pose(hand['pose'])
        if O is None or H is None:return None
        op,OR=O;hp,HR=H;positions.append(op);relative.append(HR.T@(op-hp));rotations.append(HR.T@OR)
        ov,hv=ob['rgb_visibility'],hand['rgb_visibility']
        object_vis+=ov is not None;hand_vis+=hv is not None
        co+=ov is not None and hv is not None and ov>0 and hv>0
    p=np.array(positions);r=np.array(relative);R=np.array(rotations)
    angles=np.arccos(np.clip((np.einsum('ij,nij->n',R[0],R)-1)/2,-1,1))*180/np.pi
    world=float(np.linalg.norm(p-p[0],axis=1).max());distance=float(np.median(np.linalg.norm(r,axis=1)))
    # Predetermined intake heuristic, never a grasp/physical qualification gate.
    eligible=bool(len(rows)==61 and co>=31 and world>=.01 and distance<=.3)
    return dict(object_id=ob['id'],object_uid=ob['uid'],object_name=ob['name'],hand=side,start_frame=rows[0]['frame'],end_frame=rows[-1]['frame'],frames=len(rows),
        elapsed_s=(rows[-1]['timestamp_ns']-rows[0]['timestamp_ns'])*1e-9,object_world_max_translation_m=world,
        origin_to_wrist_median_m=distance,relative_position_RMS_m=float(np.sqrt(np.mean(np.sum((r-r.mean(0))**2,axis=1)))),
        relative_rotation_max_deg=float(angles.max()),RGB_co_visible_frames=int(co),object_visibility_known_frames=int(object_vis),hand_visibility_known_frames=int(hand_vis),intake_eligible=eligible)


def inventory(rows):
    keys=sorted({key for row in rows for key in row['objects']});whole=[];windows=[]
    for key in keys:
        for side in ('left','right'):
            descriptor=describe_window(rows,key,side)
            if descriptor:whole.append(descriptor)
            for start in range(len(rows)-60):
                descriptor=describe_window(rows[start:start+61],key,side)
                if descriptor:windows.append(descriptor)
    def rank(row):return (-row['RGB_co_visible_frames'],row['origin_to_wrist_median_m'],row['relative_rotation_max_deg'],row['relative_position_RMS_m'],row['start_frame'])
    candidates=sorted((row for row in windows if row['intake_eligible']),key=rank)
    return dict(tracks=len(keys),whole_clip_descriptors=whole,windows=windows,candidates=candidates,
        best_per_object=[next(row for row in candidates if row['object_id']==oid) for oid in sorted({row['object_id'] for row in candidates})],
        scope='Source annotation poses, modeled-hand/predicted-object RGB visibility and wrist proximity only. No finger FK, image reconstruction, mesh collision, contact truth, robot retargeting, gravity calibration or dynamics qualification.')
