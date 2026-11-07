"""Pinned official right-human UmeTrack FK + unsigned phone scan proximity.

No robot, MuJoCo, native NN weights, optimizer, physics or GPU. One batched
official human FK call over at most61 rows; missing rows remain missing.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import csv
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
P = ROOT.parent
OLD = P/'ego_hand_geometry_v1'
PHONE = P/'asset_stage2_convex/completed_v2'
FRAMES = list(range(42,103))
THRESHOLDS = [.001,.002,.005]
QUANTILES = [0.,.01,.05,.25,.5,.75,.95,1.]
DIGITS = ['thumb','index','middle','ring','pinky']
BONE_NAMES = ['root_frame0','wrist_frame1']+[f'{f}_chain_{c}' for f in DIGITS for c in 'ABC']
COARSE_NAMES = ['root_wrist_influenced']+DIGITS


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def require(v,msg):
    if not v:raise ValueError(msg)


def stats(a,np):
    a=np.asarray(a,np.float64).reshape(-1)
    return {'sampled_vertex_observations':len(a),'minimum_unsigned_m':float(a.min()) if len(a) else None,
            'quantile_unsigned_m':{str(q):float(np.quantile(a,q,method='linear')) if len(a) else None for q in QUANTILES},
            'count_at_or_below':{f'{int(t*1000)}mm':int((a<=t).sum()) for t in THRESHOLDS},
            'original_scan_signed_depth_or_inside':'UNKNOWN_UNSIGNED_ONLY'}


def partition(profile,np):
    w=np.asarray(profile['dense_bone_weights'],np.float64)
    require(w.shape==(788,17) and np.isfinite(w).all() and (w>=0).all(),'Raw17weight ABI')
    require((w.sum(1)>0).all() and np.max(abs(w.sum(1)-1))<=1e-5,'Raw weight sums; no renormalization')
    require(np.asarray(profile['landmark_rest_bone_indices'])[:5,0].tolist()==[4,7,10,13,16],'Actual canonical tip bones')
    ties=w==w.max(1,keepdims=True);bone=np.where(ties.sum(1)==1,w.argmax(1),-1)
    fold=np.array([0,0]+[d+1 for d in range(5) for _ in range(3)]);coarse=np.full(788,-1,np.int64)
    for i in range(788):
        c=np.unique(fold[np.flatnonzero(ties[i])])
        if len(c)==1:coarse[i]=c[0]
    regions=[('ALL',np.ones(788,bool),'all_mesh')]
    regions += [(name,bone==i,'dominant_bone') for i,name in enumerate(BONE_NAMES)]+[('unknown_exact_tie',bone<0,'dominant_bone')]
    regions += [(name,coarse==i,'coarse_folded_dominant_bone') for i,name in enumerate(COARSE_NAMES)]+[('unknown_cross_region_tie',coarse<0,'coarse_folded_dominant_bone')]
    return w,bone,coarse,regions


def closest_all(mesh,points,np,trimesh,counts):
    cs=[];ds=[];ts=[]
    for first in range(0,len(points),32):
        take=points[first:first+32];counts['nearest_scan_triangle_API_calls_attempted']+=1
        c,d,t=trimesh.proximity.closest_point_naive(mesh,take)
        counts['nearest_scan_triangle_API_calls_returned']+=1;counts['point_triangle_nearest_queries']+=len(take)
        cs.append(c);ds.append(d);ts.append(t)
    return np.concatenate(cs),np.concatenate(ds),np.concatenate(ts)


def runs(mask,frames,records):
    out=[];run=[]
    for j,yes in enumerate(mask):
        if yes and (not run or frames[j]==frames[run[-1]]+1):run.append(j)
        else:
            if run:
                out.append(run);run=[]
            if yes:run=[j]
    if run:out.append(run)
    return [{'start_frame':frames[r[0]],'end_frame':frames[r[-1]],'sample_count':len(r),
             'first_timestamp_ns':records[r[0]]['ref_timestamp_ns'],'last_timestamp_ns':records[r[-1]]['ref_timestamp_ns'],
             'observed_timestamp_span_s':(records[r[-1]]['ref_timestamp_ns']-records[r[0]]['ref_timestamp_ns'])*1e-9,
             'between_sample_contact_not_observed':True} for r in out]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
    rules=read(ROOT/'frozen_rules_v1.json')
    require(rules['source']==pin(__file__) and rules['frames']==FRAMES and rules['thresholds_m']==THRESHOLDS,'Locked source/rules')
    for v in rules['inputs'].values():require(pin(v['path'])==v,'Unchanged source/input '+v['path'])
    if not a.execute:
        require(not any(m in sys.modules for m in ('torch','numpy','trimesh','mujoco','warp')),'Metadata only')
        print(json.dumps({'status':'METADATA_ONLY_NO_FK_DISTANCE','rules':pin(ROOT/'frozen_rules_v1.json'),'max_human_rows':61,'max_official_FK_API_calls':1,'max_point_nearest_queries':49288}));return
    require(not (ROOT/'report.json').exists() and not (ROOT/'execution_started.json').exists(),'Fresh once execution')
    counts={'official_human_profile_tensor_construction':0,'official_human_FK_batch_calls_attempted':0,'official_human_FK_batch_calls_returned':0,
            'official_human_FK_rows':0,'nearest_scan_triangle_API_calls_attempted':0,'nearest_scan_triangle_API_calls_returned':0,
            'point_triangle_nearest_queries':0,'robot_FK_or_MjData_models_physics_training_NN_weights_GPU':0}
    write(ROOT/'execution_started.json',{'source':pin(__file__),'rules':pin(ROOT/'frozen_rules_v1.json'),'status':'STARTED_NO_AUTOMATIC_RETRY'})
    started=time.perf_counter()
    try:
        import numpy as np
        import torch
        import trimesh
        torch.set_num_threads(4)
        require(not torch.cuda.is_initialized(),'No CUDA context')
        sys.path.insert(0,str(OLD))
        from vendor_umetrack_9064a2 import math_utils
        from vendor_umetrack_9064a2.hand_models import umetrack_hand_model as official
        vendor=read(OLD/'vendor_source_manifest.json')
        for v in vendor:require(pin(v['path'])['sha256']==v['sha256'],'Pinned official unmodified vendor')
        require(Path(official.__file__).samefile(OLD/'vendor_umetrack_9064a2/hand_models/umetrack_hand_model.py'),'Exact official module namespace')
        records=[];raw=[];angles=[];wrists=[];objects=[];available=[]
        with tarfile.open(rules['inputs']['clip']['path']) as tar:
            def load(name):return json.load(tar.extractfile(name))
            profile=load('__hand_shapes.json__')['umetrack']
            require(profile==read(OLD/'raw_profile_and_poses.json')['umetrack_profile'],'Same shared source profile; no fit')
            for i,frame in enumerate(FRAMES):
                hands=load(f'{frame:06d}.hands.json');obj=load(f'{frame:06d}.objects.json');info=load(f'{frame:06d}.info.json')
                row={'array_row':i,'frame_id':frame,'ref_timestamp_ns':info['ref_timestamp_ns'],'sequence_id':info['sequence_id'],
                     'participant_id':info['participant_id'],'available':False,'missing_reason':None}
                if 'right' not in hands or 'umetrack_pose' not in hands['right']:
                    row['missing_reason']='NO_RIGHT_UMETRACK_POSE';raw.append({'frame_id':frame,'right_umetrack_pose':None,'info':info})
                elif '24' not in obj or len(obj['24'])!=1:
                    row['missing_reason']='PHONE24_POSE_MISSING_OR_AMBIGUOUS';raw.append({'frame_id':frame,'right_umetrack_pose':hands['right']['umetrack_pose'],'info':info})
                else:
                    pose=hands['right']['umetrack_pose'];o=obj['24'][0]
                    require(len(pose['joint_angles'])==22,'Full raw22 passed intact')
                    matrix=lambda p:math_utils.quat_trans_to_matrix(*p['quaternion_wxyz'],*p['translation_xyz'])
                    available.append(i);angles.append(pose['joint_angles']);wrists.append(matrix(pose['T_world_from_wrist']));objects.append(matrix(o['T_world_from_object']))
                    row.update(available=True,right_modeled_visibility=hands['right'].get('visibilities_modeled'),phone_modeled_visibility=o.get('visibilities_modeled'))
                    raw.append({'frame_id':frame,'right_umetrack_pose':pose,'phone24':{k:o[k] for k in ('object_bop_id','object_uid','T_world_from_object')},'info':info})
                records.append(row)
        write(ROOT/'raw_profile_and_right_poses.json',{'umetrack_profile':profile,'frames':raw});write(ROOT/'frame_identities61.json',records)
        weights,bone,coarse,regions=partition(profile,np)
        completed=np.zeros(61,bool);angles61=np.full((61,22),np.nan,np.float32);world61=np.full((61,788,3),np.nan,np.float32)
        object61=np.full((61,788,3),np.nan,np.float64);landmark61=np.full((61,20,3),np.nan,np.float64)
        d61=np.full((61,788),np.nan,np.float64);c61=np.full((61,788,3),np.nan,np.float64);tri61=np.full((61,788),-1,np.int64)
        ld61=np.full((61,20),np.nan,np.float64);lc61=np.full((61,20,3),np.nan,np.float64);lt61=np.full((61,20),-1,np.int64)
        raw_wrist61=np.full((61,4,4),np.nan,np.float32);raw_object61=raw_wrist61.copy()
        faces_saved=np.asarray(profile['mesh_triangles'],np.int64)
        checks={}
        if available:
            n=len(available);ang=np.asarray(angles,np.float32);w=np.asarray(wrists);o=np.asarray(objects)
            require(n<=61 and ang.shape==(n,22) and np.isfinite(ang).all(),'Finite actual right rows')
            model=official.from_json(profile);counts['official_human_profile_tensor_construction']+=1
            batched=official.UmeTrackHandModelData(**{f.name:(getattr(model,f.name).unsqueeze(0).expand(n,*getattr(model,f.name).shape).clone() if isinstance(getattr(model,f.name),torch.Tensor) else getattr(model,f.name)) for f in dataclasses.fields(model)})
            handpose=official.UmeTrackHandPose(hand_side=torch.ones(n,dtype=torch.int64),joint_angles=torch.from_numpy(ang),wrist_xform=torch.from_numpy(w))
            # The original official API alone handles right-handed reflection and face winding.
            counts['official_human_FK_batch_calls_attempted']+=1
            with torch.inference_mode():lm,vs,fs=official.forward_kinematics(handpose,batched,requires_mesh=True)
            counts['official_human_FK_batch_calls_returned']+=1;counts['official_human_FK_rows']+=n
            v=vs.numpy();l=lm.numpy();f=fs.numpy()
            require(v.shape==(n,788,3) and l.shape==(n,20,3) and f.shape==(n,1544,3),'Official output ABI')
            require(np.isfinite(v).all() and np.isfinite(l).all(),'Finite official output')
            require(all(np.array_equal(face,f[0]) for face in f) and np.array_equal(f[0],faces_saved[:,::-1]),'Official right-hand face winding flip')
            inv=np.linalg.inv(o.astype(np.float64));vo=np.stack([v0@T[:3,:3].T+T[:3,3] for v0,T in zip(v,inv)]);lo=np.stack([l0@T[:3,:3].T+T[:3,3] for l0,T in zip(l,inv)])
            back=np.stack([v0@T[:3,:3].T+T[:3,3] for v0,T in zip(vo,o.astype(np.float64))])
            scan=trimesh.load(rules['inputs']['phone_original_scan']['path'],process=False,force='mesh')
            c,d,tr=closest_all(scan,vo.reshape(-1,3),np,trimesh,counts);cl,dl,trl=closest_all(scan,lo.reshape(-1,3),np,trimesh,counts)
            completed[available]=True;angles61[available]=ang;world61[available]=v;object61[available]=vo;landmark61[available]=lo
            d61[available]=d.reshape(n,788);c61[available]=c.reshape(n,788,3);tri61[available]=tr.reshape(n,788)
            ld61[available]=dl.reshape(n,20);lc61[available]=cl.reshape(n,20,3);lt61[available]=trl.reshape(n,20)
            raw_wrist61[available]=w;raw_object61[available]=o;faces_saved=f[0]
            checks={'official_right_hand_side':1,'official_internal_wrist_column0_and_face_winding_flip_source_only':True,'custom_mirroring_applied':False,
                    'all22_input_angles_preserved':True,'official_skin_uses_first20_source_indices':list(range(20)),'unused_tail_indices':[20,21],
                    'raw_wrist_rotation_det_minmax':[float(np.linalg.det(w[:,:3,:3]).min()),float(np.linalg.det(w[:,:3,:3]).max())],
                    'world_object_inverse_roundtrip_max_abs_m':float(abs(back-v).max()),'scan_vertices':len(scan.vertices),'scan_triangles':len(scan.faces),
                    'scan_watertight':bool(scan.is_watertight)}
        np.savez_compressed(ROOT/'right_geometry_and_proximity61.npz',frame_ids=np.asarray(FRAMES,np.int64),completed_mask=completed,raw_joint_angles22_FP32=angles61,
                            T_world_from_right_wrist_FP32=raw_wrist61,T_world_from_phone_FP32=raw_object61,hand_vertices_world_FP32=world61,
                            hand_vertices_phone_object_F64=object61,canonical_landmarks20_phone_object_F64=landmark61,hand_faces=faces_saved,
                            hand_vertex_unsigned_scan_distance_m=d61,hand_vertex_closest_scan_material_point_object_m=c61,hand_vertex_closest_scan_triangle=tri61,
                            canonical_landmark20_unsigned_scan_distance_m=ld61,canonical_landmark20_closest_scan_material_point_object_m=lc61,canonical_landmark20_closest_scan_triangle=lt61,
                            dominant_bone_id788=bone,coarse_region_id788=coarse,raw_dense_skin_weights788x17=weights)
        rows=[];temporal=[]
        for name,mask,scheme in regions:
            for i,frame in enumerate(FRAMES):
                rows.append({'frame_id':frame,'available':bool(completed[i]),'region':name,'scheme':scheme,'static_vertex_count':int(mask.sum()),
                             'statistics':stats(d61[i,mask],np) if completed[i] else None})
            by_threshold={}
            for threshold in THRESHOLDS:
                minima=np.full(61,np.inf,np.float64)
                if mask.any():minima[completed]=d61[completed][:,mask].min(axis=1)
                yes=completed & (minima<=threshold)
                by_threshold[f'{int(threshold*1000)}mm']={'sampled_frames':int(yes.sum()),'observed_runs':runs(yes,FRAMES,records)}
            temporal.append({'region':name,'scheme':scheme,'available_frames':int(completed.sum()),'missing_frames':int((~completed).sum()),
                             'pooled_available_statistics':stats(d61[completed][:,mask],np),'threshold_persistence':by_threshold})
        write(ROOT/'all61_region_statistics.json',rows);write(ROOT/'all61_region_temporal_summary.json',temporal)
        with (ROOT/'all61_region_statistics.csv').open('x',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in row.items()})
        pts=[]
        if completed[0]:
            for vi in np.flatnonzero(d61[0]<=.005):pts.append({'vertex_id':int(vi),'dominant_bone':int(bone[vi]),'coarse_region':COARSE_NAMES[coarse[vi]] if coarse[vi]>=0 else 'unknown_tie',
                'unsigned_scan_distance_m':float(d61[0,vi]),'right_hand_vertex_phone_object_m':object61[0,vi].tolist(),
                'closest_original_scan_material_point_object_m':c61[0,vi].tolist(),'closest_scan_triangle':int(tri61[0,vi]),'raw_skin_weights17':weights[vi].tolist()})
        write(ROOT/'frame42_all_near5mm_material_points.json',pts)
        require(counts['official_human_FK_rows']<=61 and counts['official_human_FK_batch_calls_attempted']<=1 and counts['point_triangle_nearest_queries']<=49288,'Fixed numerical budget')
        require(not torch.cuda.is_initialized() and not any(x in sys.modules for x in ('mujoco','warp')),'No robot/backend/GPU')
        report={'schema':'official_right_human_mesh_phone_contact_assignment_audit:1','status':'COMPLETED_RIGHT_HUMAN_UNSIGNED_PROXIMITY_NOT_CONTACT_OR_FORCE_TRUTH',
                'rules':pin(ROOT/'frozen_rules_v1.json'),'source':pin(__file__),'input_pins':rules['inputs'],'counts':counts,
                'frames':FRAMES,'available_frames':int(completed.sum()),'missing_frame_records':[r for r in records if not r['available']],
                'handedness_and_coordinate_checks':checks,'CUDA_initialized':torch.cuda.is_initialized(),
                'frame42_all_mesh':next(r for r in rows if r['frame_id']==42 and r['region']=='ALL'),
                'frame42_coarse_regions':[r for r in rows if r['frame_id']==42 and r['scheme']=='coarse_folded_dominant_bone'],
                'all61_coarse_temporal_summary':[r for r in temporal if r['scheme']=='coarse_folded_dominant_bone'],
                'all61_ALL_temporal_summary':next(r for r in temporal if r['region']=='ALL'),
                'elapsed_seconds':time.perf_counter()-started,'limitations':rules['limitations'],
                'robot_targets_or_left_reference_modified':False,'unimanual_or_bimanual_force_assignment_qualified':False,
                'outputs':{p.name:pin(p) for p in ROOT.iterdir() if p.is_file() and p.name not in ('report.json','description.md')}}
        write(ROOT/'report.json',report)
        desc=['# 右手接触分配审查（官方人体FK）','',f"帧42–102保留全部61行，实际可用{int(completed.sum())}、missing{int((~completed).sum())}。官方人体FK一次batch调用、{counts['official_human_FK_rows']}行；原scan最近triangle点查询{counts['point_triangle_nearest_queries']}。0机器人FK/MjData/NN权重/学习/积分/GPU。",'',
              '右手hand_side=1；原始proper wrist pose按官方API输入，官方代码内部处理第一列反射及face winding翻转。没有自行镜像、copy左手pose或22→20截断；22角完整传入，官方skinning内部仍只用前20。','',
              '|frame42粗区域|顶点数|min mm|≤1/2/5mm顶点数|','|---|---:|---:|---:|']
        for r in report['frame42_coarse_regions']:
            st=r['statistics'];mn='MISSING' if st is None or st['minimum_unsigned_m'] is None else f"{1000*st['minimum_unsigned_m']:.5f}"
            ct='MISSING' if st is None else ' / '.join(str(st['count_at_or_below'][f'{t}mm']) for t in (1,2,5))
            desc.append(f"|{r['region']}|{r['static_vertex_count']}|{mn}|{ct}|")
        desc += ['', '全61帧逐区域最小值、分位数、1/2/5mm数量与逐段持续性见all61_region_temporal_summary.json；frame42的全部≤5mm右手点/phone材料点见frame42_all_near5mm_material_points.json。','',
                 '这只是可见/重建人体与扫描表面的几何近距；不是传感接触、力或独立支持点证书。原scan不封闭、距离无符号，可能穿入或受tracking/skin/profile/scan误差影响；region仅dominant蒙皮影响。缺帧未插值。连续近距样本只报告时间戳跨度，不能证明采样间实际接触。即使右手也近距，仍不能推定其负担了多少重量；右手较远也不能证明unimanual受力。旧左手参考、机器人目标与当前优化均未改。']
        (ROOT/'description.md').write_text('\n'.join(desc)+'\n',encoding='utf8')
        print(json.dumps({'report':pin(ROOT/'report.json'),'description':pin(ROOT/'description.md'),'counts':counts,
                          'available_frames':report['available_frames'],'frame42_minimum_m':report['frame42_all_mesh']['statistics']['minimum_unsigned_m'] if report['frame42_all_mesh']['statistics'] else None,
                          'ALL_persistence':report['all61_ALL_temporal_summary']['threshold_persistence'],'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False))
    except BaseException as exc:
        write(ROOT/'failure.json',{'status':'FAILED_PRESERVED_NO_AUTOMATIC_RETRY','exception':repr(exc),'counts':counts,'elapsed_seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':main()
