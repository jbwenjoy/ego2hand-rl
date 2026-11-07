"""Observation-only surface proximity hypotheses, not contact/force labels.

Reads only the three requested observation fields. New geometric-candidate q
is used for a separately named FK comparison; no upstream robot-q is read.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,hashlib,itertools,json,sys,time
from pathlib import Path
import numpy as np
import mujoco
from pen_outer_surface import PenExterior,quat_matrix
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/contact_intent_v2'
RETARGET=Path('D:/workspace/supermimic-observation-retarget-study')
sys.path.insert(0,str(RETARGET/'scripts'))
from retarget_palmroot_cpu import RobotKinematics,hand_frames,directions
def pin(p):return {'path':str(p),'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();OUT=args.output
    started=time.perf_counter();OUT.mkdir(parents=True,exist_ok=False)
    obs=Path('D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1/observations.npz');manifestpath=obs.with_name('manifest.json');manifest=json.loads(manifestpath.read_text());candidate=RETARGET/'outputs/native_contact_refined_geometry_v1/geometric_retarget.npz';model=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb');context=Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz')
    if pin(obs)['sha256']!=manifest['observation_sha256'] or manifest['is_real_ego'] is not False:raise ValueError('This prototype only admits the frozencontrolledfixture')
    read_fields=['time_s','hand_landmarks_source_w','object_pose_source_w']
    with np.load(obs,allow_pickle=False) as z:t,human,sourcepose=[np.array(z[k],dtype=np.float64) for k in read_fields]
    candidate_fields=['time_s','joint_pos_rad','source_to_fixed_wrist_R','source_to_fixed_wrist_translation_m','morphology_normalized_target_points_w','mapped_object_pose_w']
    with np.load(candidate,allow_pickle=False) as z:ct,q,R,translation,target,mappedpose=[np.array(z[k],copy=True) for k in candidate_fields]
    if human.shape!=(151,21,3) or sourcepose.shape!=(151,7) or q.shape!=(151,20) or not np.array_equal(t,ct) or not np.isfinite(human).all() or not np.isfinite(sourcepose).all() or not (np.diff(t)>0).all():raise ValueError('Invalid frozen151frame layout')
    forbidden={name:0 for name in ['mj_geomDistance','mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_contactForce','mj_collision']}
    def deny(name):
        def f(*a,**kw):forbidden[name]+=1;raise AssertionError('Forbidden '+name)
        return f
    for name in forbidden:setattr(mujoco,name,deny(name))
    pen=PenExterior(model,context);robot=RobotKinematics(model);objqa=int(robot.model.joint('object/cube_freejoint').qposadr[0]);rigid=np.einsum('tij,tpj->tpi',R,human)+translation[:,None,:]
    reconstructed_R=np.einsum('ij,tkj->tik',robot.palm_frame,hand_frames(human));reconstructed_t=robot.neutral[0]-np.einsum('tij,tj->ti',reconstructed_R,human[:,0]);units=np.einsum('tij,tkj->tki',reconstructed_R,directions(human));newtarget=[];fk=[]
    for frame in range(151):
        temp=robot.neutral.copy()
        for finger in range(5):
            first=1+4*finger
            for bone in range(3):temp[first+bone+1]=temp[first+bone]+robot.lengths[3*finger+bone]*units[frame,3*finger+bone]
        newtarget.append(temp);robot.data.qpos[objqa:objqa+7]=mappedpose[frame];robot.last_q=None;fk.append(robot.forward(q[frame])[0].copy())
    fk=np.asarray(fk);newtarget=np.asarray(newtarget)
    checks={'mapping_R_reconstructed':np.max(np.abs(R-reconstructed_R))<1e-12,'mapping_translation_reconstructed':np.max(np.abs(translation-reconstructed_t))<1e-12,'normalized_target_exact_reconstructed':np.array_equal(newtarget,target),'object_mapping_position':np.max(np.abs(mappedpose[:,:3]-(np.einsum('tij,tj->ti',R,sourcepose[:,:3])+translation)))<1e-12,'object_mapping_rotation':np.max(np.abs(quat_matrix(mappedpose[:,3:])-R@quat_matrix(sourcepose[:,3:])))<1e-12,'R_orthonormal':np.max(np.abs(R.transpose(0,2,1)@R-np.eye(3)))<1e-12}
    variants={'source_original':(human,sourcepose),'source_rigid_mapped':(rigid,mappedpose),'morphology_normalized_target':(target,mappedpose),'robot_candidate_FK':(fk,mappedpose)};data={};npz={'time_s':t,'source_to_fixed_wrist_R':R,'source_to_fixed_wrist_translation_m':translation};frame0={'schema':'frame0_surface_proximity_hypotheses_not_contact_labels:1','confidence':None,'confidence_status':'unknown; no confidences loaded or invented','variants':{}}
    def point_record(name,k,frame=0):
        a=data[name];j=int(a['feature_index'][frame,k]);patch=pen.patches[j];ties=[pen.patches[i]['component']+':'+pen.patches[i]['feature'] for i in np.flatnonzero(a['tie_mask'][frame,k])]
        return {'landmark_index':k,'point_w':a['points_w'][frame,k].tolist(),'point_o':a['points_o'][frame,k].tolist(),'object_surface_closest_w':a['closest_w'][frame,k].tolist(),'object_surface_closest_o':a['closest_o_m'][frame,k].tolist(),'outward_normal_w':a['normal_w'][frame,k].tolist(),'outward_normal_o':a['outward_normal_o'][frame,k].tolist(),'signed_distance_m':float(a['signed_distance_m'][frame,k]),'unsigned_distance_m':float(a['unsigned_distance_m'][frame,k]),'component':patch['component'],'feature':patch['feature'],'feature_index':j,'tie_count':int(a['tie_count'][frame,k]),'tie_features':ties,'axis_radial_normal_ambiguity':bool(a['axis_radial_normal_ambiguity'][frame,k]),'confidence':None,'is_known_contact_label':False}
    for name,(points,pose) in variants.items():
        rotation=quat_matrix(pose[:,3:]);po=np.einsum('tji,tpj->tpi',rotation,points-pose[:,None,:3]);a=pen.query(po);a['points_w']=points;a['points_o']=po;a['closest_w']=np.einsum('tij,tpj->tpi',rotation,a['closest_o_m'])+pose[:,None,:3];a['normal_w']=np.einsum('tij,tpj->tpi',rotation,a['outward_normal_o']);data[name]=a
        for key,value in a.items():npz[name+'__'+key]=value
        fingers=[];nearestall=[];nearestdistal=[]
        for finger in range(5):
            indices=list(range(1+4*finger,5+4*finger));points_rec=[point_record(name,k) for k in indices];nearest=min(points_rec,key=lambda p:p['unsigned_distance_m']);distal=min(points_rec[-2:],key=lambda p:p['unsigned_distance_m']);fingers.append({'finger_chain':finger+1,'anatomical_name_assumed':['thumb','index','middle','ring','little'][finger],'mapping_status':'ordinal correspondence assumption, not anatomical calibration','points':points_rec});nearestall.append({'finger_chain':finger+1,**nearest});nearestdistal.append({'finger_chain':finger+1,**distal})
        opposition=[]
        for a,b in itertools.combinations(nearestdistal,2):
            na,nb=np.array(a['outward_normal_o']),np.array(b['outward_normal_o']);dot=float(np.clip(na@nb,-1,1));maxgap=max(a['unsigned_distance_m'],b['unsigned_distance_m'])
            opposition.append({'finger_chains':[a['finger_chain'],b['finger_chain']],'landmark_indices':[a['landmark_index'],b['landmark_index']],'outward_normal_dot':dot,'outward_normal_angle_deg':float(np.rad2deg(np.arccos(dot))),'maximum_unsigned_surface_distance_m':maxgap,'surface_point_separation_m':float(np.linalg.norm(np.array(a['object_surface_closest_o'])-b['object_surface_closest_o'])),'diagnostic_both_within10mm_and_normals_over120deg':bool(maxgap<=.01 and dot<=-.5),'confidence':None,'not_a_force_closure_or_contact_label':True})
        frame0['variants'][name]={'point_semantics':'source observation landmarks' if name=='source_original' else 'rigid transform of same source landmarks; metric relation should be identical' if name=='source_rigid_mapped' else 'robot macrosegment length/root-normalized target, NOT the same anatomical observations' if name=='morphology_normalized_target' else 'newcandidate robotFK macro-landmarks, NOT source robot labels','fingers':fingers,'nearest_per_finger_all4':sorted(nearestall,key=lambda p:p['unsigned_distance_m']),'nearest_per_finger_distal2':sorted(nearestdistal,key=lambda p:p['unsigned_distance_m']),'opposition_pairs_sorted_by_proximity_then_opposition':sorted(opposition,key=lambda p:(p['maximum_unsigned_surface_distance_m'],p['outward_normal_dot']))}
    source=data['source_original'];rigiddata=data['source_rigid_mapped'];normalized=data['morphology_normalized_target'];checks['rigid_framechange_preserves_objectcentric_points']=np.max(np.abs(source['points_o']-rigiddata['points_o']))<1e-12;checks['rigid_framechange_preserves_surface_distance']=np.max(np.abs(source['signed_distance_m']-rigiddata['signed_distance_m']))<1e-12
    # Numerical seam regressions use the true exterior, never min primitive SDF.
    rad,L=pen.radius,pen.half_length;test=np.array([[0,0,0],[L,0,0],[-L,0,0],[L-.001,0,0],[L+.001,0,0],[0,.02,0],[.135,0,0],[L-.0001,.0085,0]])
    expected=np.array([-rad,-rad,-rad,-rad,-np.sqrt(rad**2+.001**2),.02-rad,.010,.0001]);got=pen.query(test)['signed_distance_m'];checks['true_exterior_seam_and_step_regressions']=np.max(np.abs(got-expected))<1e-8
    checks['no_physics_gapquery_force_APIs']=not any(forbidden.values());checks['no_Torch_Warp_import']='torch' not in sys.modules and 'warp' not in sys.modules
    np.savez_compressed(OUT/'surface_proximity_151.npz',**npz);write(OUT/'frame0_rankings.json',frame0)
    frame_ranks=[]
    for f in range(151):
        row={'frame':f,'time_s':float(t[f]),'variants':{}}
        for name,a in data.items():
            row['variants'][name]=[{'finger_chain':i+1,'nearest_all4_landmark':int(1+4*i+np.argmin(a['unsigned_distance_m'][f,1+4*i:5+4*i])),'minimum_all4_unsigned_m':float(a['unsigned_distance_m'][f,1+4*i:5+4*i].min()),'minimum_distal2_unsigned_m':float(a['unsigned_distance_m'][f,3+4*i:5+4*i].min()),'tip_signed_m':float(a['signed_distance_m'][f,4+4*i])} for i in range(5)]
        frame_ranks.append(row)
    write(OUT/'rankings_151.json',frame_ranks)
    difference=normalized['signed_distance_m']-rigiddata['signed_distance_m'];report={'status':'completed_observation_surface_hypothesis_diagnostic' if all(checks.values()) else 'failed_geometry_contract','checks':checks,'surface_geometry':pen.metadata,'observation':pin(obs),'observation_manifest':pin(manifestpath),'candidate_for_FK_only':pin(candidate),'candidate_read_fields':candidate_fields,'observation_read_fields':read_fields,'known_context_read_fields':['geom_size','geom_pos'],'forbidden_API_counts':forbidden,'geometry_module':pin(Path(__file__).with_name('pen_outer_surface.py')),'code':pin(__file__),'frame0_rankings':pin(OUT/'frame0_rankings.json'),'arrays':pin(OUT/'surface_proximity_151.npz'),'rigid_distance_difference_max_m':float(np.max(np.abs(source['signed_distance_m']-rigiddata['signed_distance_m']))),'normalization_signed_distance_change_m':{'all151_max_abs':float(np.abs(difference).max()),'frame0_max_abs':float(np.abs(difference[0]).max()),'frame0_per_point':difference[0].tolist()},'seam_regressions':{'points_o':test.tolist(),'expected_signed_m':expected.tolist(),'actual_signed_m':got.tolist()},'confidence':None,'contact_labels_inferred':False,'force_estimated':False,'physical_reference_qualified':False,'is_real_ego':False,'scope':'Known-template surface proximity under cachedSI/frame assumptions only. Source original/rigid map preserve observed relations; morphology normalization changes root/length/anatomical correspondences. Joint landmark centers are not skin/contact surfaces. Opposing object surface normals at near landmarks are contact-intent hypotheses, not observed contacts, friction/force closure or physicalgrasp. No optimization or wrist/objectpose change.','wall_seconds':time.perf_counter()-started}
    write(OUT/'report.json',report);print(json.dumps({'status':report['status'],'checks':checks,'frame0_rankings':str(OUT/'frame0_rankings.json'),'frame0_nearest_distal_mm':{name:[(p['finger_chain'],p['landmark_index'],round(p['signed_distance_m']*1000,4),p['component'],p['feature']) for p in value['nearest_per_finger_distal2']] for name,value in frame0['variants'].items()}},indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)));raise SystemExit(not all(checks.values()))
if __name__=='__main__':main()
