"""Independent nominal-option151frame FK/metric/collision review, never physics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import csv,hashlib,json,re,sys,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path('D:/workspace/supermimic-observation-retarget-study');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from contact_penetration_geometry_nominal import ContactGeometry,ALLOWANCE_M,NORMALIZATION_M
from retarget_palmroot_cpu import hand_frames,directions,BONES

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')
def arr(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def main():
    started=time.perf_counter();parent=ROOT/'outputs/contact_refined_nominal_geometry_v1';report= json.loads((parent/'report.json').read_text());recipe=json.loads((parent/'recipe.json').read_text())
    model=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb');context=Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz');observation=Path('D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1/observations.npz');baseline=ROOT/'outputs/cached_landmarks_palmroot_v2/geometric_retarget.npz';candidate=parent/'geometric_retarget.npz'
    fixedfiles=[model,context,observation,baseline,candidate,parent/'report.json',parent/'recipe.json',ROOT/'scripts/contact_penetration_geometry.py',ROOT/'scripts/contact_penetration_geometry_nominal.py',ROOT/'scripts/refine_contact_geometry_nominal_cpu.py',ROOT/'scripts/retarget_palmroot_cpu.py']
    before={str(p):sha(p) for p in fixedfiles};z=arr(candidate);base=arr(baseline)
    with np.load(observation,allow_pickle=False) as o:human=o['hand_landmarks_source_w'].copy();objectsource=o['object_pose_source_w'].copy();t=o['time_s'].copy()
    checks={'frozen_inputs_and_modules_hash_match':sha(model)==report['model_sha256'] and sha(context)==report['context_sha256'] and sha(observation)==report['input_npz_sha256'] and sha(baseline)==report['baseline_npz_sha256'] and sha(candidate)==report['output_npz_sha256'] and sha(fixedfiles[8])==report['geometry_module_sha256'] and sha(fixedfiles[9])==report['script_sha256'] and sha(parent/'recipe.json')==report['recipe_sha256'],
        'strict_failure_source_retained':sha(ROOT/'scripts/contact_penetration_geometry.py')=='98f9f2495fa96e357e6b9292c4a896d471eade2769c76761f945458bb457f717',
        'actual_candidate_finite151x20':z['joint_pos_rad'].shape==(151,20) and np.isfinite(z['joint_pos_rad']).all(),
        'fixed_soft_recipe':recipe['bone_direction_weight']==recipe['penetration_weight']==1 and recipe['temporal_weight']==.03 and recipe['nominal_weight']==.001 and recipe['max_nfev_per_frame']==80 and recipe['penetration_allowance_m']==.0005 and recipe['penetration_normalization_m']==.001 and recipe['contact_attraction'] is False and recipe['object_pose_optimized'] is False,
        'all151_no_frame_selection':report['frames']==len(report['optimizer_per_frame'])==151 and [x['frame'] for x in report['optimizer_per_frame']]==list(range(151)),
        'three_nonconverged_retained':[(x['frame'],x['nfev'],x['status']) for x in report['optimizer_per_frame'] if not x['success']]==[(54,80,0),(55,80,0),(56,80,0)],
        'unknown_contact_and_physicalfalse':report['known_contacts_or_forces_or_phase_inferred'] is False and report['physical_reference_qualified'] is False and report['dynamics_steps']==0 and report['source_robot_labels_read'] is False}
    calls={name:0 for name in ['mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_contactForce']}
    def deny(name):
        def f(*a,**kw):calls[name]+=1;raise AssertionError('Forbidden dynamics/forceAPI '+name)
        return f
    for name in calls:setattr(mujoco,name,deny(name))
    g=ContactGeometry(model,context);robot=g.robot;m,d=g.model,g.data;ctx=arr(context)
    checks['original_MJB_query_options_preserved']=m.opt.ccd_tolerance==1e-6 and m.opt.ccd_iterations==50 and g.query_options['original_ccd_tolerance']==1e-6
    checks['actual_context_geom_size_pos_exact']=np.array_equal(m.geom_size,ctx['geom_size'][0]) and np.array_equal(m.geom_pos,ctx['geom_pos'][0])
    hand=set(map(int,g.pairs[:,0]));obj=set(map(int,g.pairs[:,1]));mount={i for i,n in enumerate(g.names) if 'mount' in n.lower()};terrain={i for i,n in enumerate(g.names) if n=='terrain'};table={i for i,n in enumerate(g.names) if 'table' in n.lower()}
    inventory=[{'id':i,'name':g.names[i],'body':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])),'type':int(m.geom_type[i]),'contype':int(m.geom_contype[i]),'conaffinity':int(m.geom_conaffinity[i]),'group':int(m.geom_group[i])} for i in range(m.ngeom)]
    mountcollision={i for i in mount if g.names[i]};enabled=lambda i:int(m.geom_contype[i])!=0 or int(m.geom_conaffinity[i])!=0
    rotations=np.einsum('ij,tkj->tik',robot.palm_frame,hand_frames(human));translations=robot.neutral[0]-np.einsum('tij,tj->ti',rotations,human[:,0]);units=np.einsum('tij,tkj->tki',rotations,directions(human))
    objectR=rotations@Rotation.from_quat(objectsource[:,[4,5,6,3]]).as_matrix();recordedR=Rotation.from_quat(z['mapped_object_pose_w'][:,[4,5,6,3]]).as_matrix()
    checks['same_clock_object_targets_exact']=np.array_equal(t,z['time_s']) and np.array_equal(z['mapped_object_pose_w'],base['mapped_object_pose_w']) and np.array_equal(z['morphology_normalized_target_points_w'],base['morphology_normalized_target_points_w'])
    checks['mapping_from_observation_rebuilt']=np.max(np.abs(rotations-z['source_to_fixed_wrist_R']))<1e-12 and np.max(np.abs(translations-z['source_to_fixed_wrist_translation_m']))<1e-12 and np.max(np.abs(objectR-recordedR))<1e-12 and np.max(np.abs(z['mapped_object_pose_w'][:,:3]-(np.einsum('tij,tj->ti',rotations,objectsource[:,:3])+translations)))<1e-12
    fk=[];angles=[];targets=[];distances=[];baselines=[];witnesses=[];frame_rows=[];native_rows=[];cost_reconstructed=[];side_rows=[];negative_errors=[];positive_errors=[]
    all_primary_guard_passed=True
    def query_pairs(pairs):
        dist=[];seg=[]
        for a,b in pairs:
            witness=np.zeros(6);value=mujoco.mj_geomDistance(m,d,int(a),int(b),1.,witness);dist.append(value);seg.append(witness)
        return np.array(dist),np.array(seg)
    def category(a,b):
        pair={a,b}
        if (a in hand and b in obj) or (b in hand and a in obj):return 'hand_object'
        if a in hand and b in hand:return 'hand_self'
        if (a in hand and b in mount) or (b in hand and a in mount):return 'hand_mount'
        if (a in hand and b in terrain) or (b in hand and a in terrain):return 'hand_terrain'
        if (a in obj and b in terrain) or (b in obj and a in terrain):return 'object_terrain'
        if (a in obj and b in mount) or (b in obj and a in mount):return 'object_mount'
        return 'other'
    for frame,q in enumerate(z['joint_pos_rad']):
        g.set_object_pose(z['mapped_object_pose_w'][frame]);points,_=robot.forward(q);bone,_=robot.bone_features(q)
        fk.append(points.copy());angles.append(np.rad2deg(np.arccos(np.clip(np.sum(bone*units[frame],axis=-1),-1,1))))
        target=robot.neutral.copy()
        for finger in range(5):
            first=1+4*finger
            for segment in range(3):target[first+segment+1]=target[first+segment]+robot.lengths[3*finger+segment]*units[frame,3*finger+segment]
        targets.append(target)
        ds,segments=query_pairs(g.pairs);distances.append(ds);witnesses.append(segments)
        err=np.abs(np.linalg.norm(segments[:,3:]-segments[:,:3],axis=1)-np.abs(ds));negative_errors.extend(err[ds<0].tolist());positive_errors.extend(err[ds>=0].tolist())
        try:md,jac=g.distance_and_jacobian(q)
        except ValueError:all_primary_guard_passed=False;raise
        assert np.array_equal(ds,md)
        pen=np.maximum(-ds-ALLOWANCE_M,0)/NORMALIZATION_M;previous=base['joint_pos_rad'][0] if frame==0 else z['joint_pos_rad'][frame-1]
        rr=np.r_[(bone-units[frame]).ravel(),pen,.03*(q-previous),.001*(q-robot.nominal)];cost_reconstructed.append(float(.5*rr@rr))
        side={}
        for label,others in [('hand_mount_unfiltered',mountcollision),('hand_terrain_unfiltered',terrain)]:
            pairs=[(h,o) for h in sorted(hand) for o in sorted(others)]
            if not pairs:side[label]={'status':'MISSING_GEOMETRY_UNKNOWN','depth_max_m':None};continue
            values,s=query_pairs(pairs);e=np.abs(np.linalg.norm(s[:,3:]-s[:,:3],axis=1)-np.abs(values));neg=values<0;bad=neg&(e>1e-6)
            side[label]={'status':'NEGATIVE_WITNESS_FAILURE_UNKNOWN' if bad.any() else 'DIRECT_GEOMETRY_QUERY_ONLY','pair_count':len(pairs),'negative_pair_count':int(neg.sum()),'unqualified_negative_witness_count':int(bad.sum()),'signed_min_m':float(values.min()),'observed_negative_depth_max_m':float(np.maximum(-values,0).max()) if not bad.any() else None}
        side_rows.append({'frame':frame,**side})
        expected_q=d.qpos.copy();expected_v=d.qvel.copy();clock=d.time;mujoco.mj_collision(m,d)
        native=[]
        for c in d.contact:
            a,b=map(int,c.geom);native.append({'frame':frame,'category':category(a,b),'geom':[a,b],'names':[g.names[a],g.names[b]],'distance_m':float(c.dist),'position_w':c.pos.copy().tolist(),'normal_geom0_to_geom1':c.frame[:3].copy().tolist()})
        assert np.array_equal(d.qpos,expected_q) and np.array_equal(d.qvel,expected_v) and d.time==clock==0
        native_rows.extend(native)
        row={'frame':frame,'time_s':float(t[frame]),'query_max_pair_depth_m':float(np.maximum(-ds,0).max()),'query_negative_pair_count':int((ds<0).sum()),'query_pairs_deeper_than_allowance':int((ds<-ALLOWANCE_M).sum()),'optimizer_success':report['optimizer_per_frame'][frame]['success']}
        for label in ['hand_object','hand_self','hand_mount','hand_terrain','object_terrain','object_mount']:
            hits=[c for c in native if c['category']==label];row[label+'_native_contact_count']=len(hits)
            row[label+'_native_max_depth_m']=float(max(max(-c['distance_m'],0) for c in hits)) if hits else None
            row[label+'_native_negative_count']=sum(c['distance_m']<0 for c in hits)
        frame_rows.append(row)
        robot.forward(base['joint_pos_rad'][frame]);bd,_=query_pairs(g.pairs);baselines.append(bd)
    fk=np.asarray(fk);angles=np.asarray(angles);targets=np.asarray(targets);ds=np.asarray(distances);initial=np.asarray(baselines);witnesses=np.asarray(witnesses)
    checks['reconstructed_FK_and_angles_match']=np.max(np.abs(fk-z['fk_hand_points_w']))<1e-12 and np.max(np.abs(angles-z['bone_angle_error_deg']))<1e-9
    checks['same_morphology_target_rebuilt_exact']=np.array_equal(targets,z['morphology_normalized_target_points_w'])
    checks['all151_primary_distances_and_baseline_reproduced']=np.array_equal(ds,z['geometry_pair_signed_distance_m']) and np.array_equal(initial,z['baseline_pair_signed_distance_m']) and np.array_equal(g.pairs,z['geometry_pair_indices'])
    checks['all151_final_negative_witness_guard']=all_primary_guard_passed and max(negative_errors)<1e-6
    checks['saved_optimizer_cost_reconstructed']=max(abs(a-b['cost']) for a,b in zip(cost_reconstructed,report['optimizer_per_frame']))<1e-8
    metrics={'bone_angle_mean_deg':float(angles.mean()),'bone_angle_max_deg':float(angles.max()),'normalized_target_tip_error_mean_m':float(np.linalg.norm(fk[:,[4,8,12,16,20]]-targets[:,[4,8,12,16,20]],axis=-1).mean()),'hard_joint_excess_max_rad':float(np.maximum(np.maximum(robot.lower-z['joint_pos_rad'],z['joint_pos_rad']-robot.upper),0).max()),'q_ref_fd_rate_abs_max_rad_s':float(np.abs(np.diff(z['joint_pos_rad'],axis=0)/np.diff(t)[:,None]).max()),'q_change_from_geometric_baseline_abs_max_rad':float(np.abs(z['joint_pos_rad']-base['joint_pos_rad']).max()),'baseline_pair_depth_max_m':float(np.maximum(-initial,0).max()),'refined_pair_depth_max_m':float(np.maximum(-ds,0).max()),'frames_depth_over3mm':int((ds.min(1)<-.003).sum()),'frames_depth_over_point5mm':int((ds.min(1)<-.0005).sum())}
    checks['all_saved_metrics_recomputed']=all(abs(metrics[k]-report[k])<1e-9 for k in metrics)
    # Revisit the exact failedstrict state under unchanged nominal queryoptions.
    bad=arr(ROOT/'outputs/contact_refined_geometry_debug_v1/query_failure_snapshot.npz');g.set_object_pose(bad['object_pose']);g.robot.forward(bad['q']);failure_pair=g.pairs[12];segment=np.zeros(6);failure_distance=mujoco.mj_geomDistance(m,d,int(failure_pair[0]),int(failure_pair[1]),1.,segment)
    oldcert=json.loads((ROOT/'outputs/contact_geometry_independent_review_v1/separation_certificate.json').read_text());gap=oldcert['positive_separation_lower_bound_m']
    checks['strict_counterexample_nominal_positive_gap']=failure_distance>0 and abs(failure_distance-gap)<1e-8 and abs(np.linalg.norm(segment[3:]-segment[:3])-failure_distance)<1e-8
    checks['no_dynamics_or_force_calls']=not any(calls.values());checks['model_context_sources_artifacts_unchanged']=before=={p:sha(p) for p in before}
    np.savez_compressed(OUT/'reconstructed_arrays.npz',fk_hand_points_w=fk,bone_angle_error_deg=angles,morphology_normalized_target_points_w=targets,pair_signed_distance_m=ds,baseline_pair_signed_distance_m=initial,pair_witness_w=witnesses,reconstructed_optimizer_cost=np.array(cost_reconstructed))
    (OUT/'native_contacts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in native_rows),encoding='utf-8');write(OUT/'side_geometry_queries.json',side_rows)
    with (OUT/'frames.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(frame_rows[0]));w.writeheader();w.writerows(frame_rows)
    summary_native={}
    for label in ['hand_object','hand_self','hand_mount','hand_terrain','object_terrain','object_mount']:
        hits=[x for x in native_rows if x['category']==label];depths=[max(-x['distance_m'],0) for x in hits]
        summary_native[label]={'status':'OBSERVED_FILTERED_NATIVE_CONTACTS' if hits else 'NO_FILTERED_NATIVE_CONTACTS_OBSERVED_NOT_GEOMETRY_CLEARANCE','contact_records':len(hits),'frames_with_contacts':len(set(x['frame'] for x in hits)),'negative_contact_records':sum(x['distance_m']<0 for x in hits),'max_recorded_penetration_depth_m':max(depths) if depths else None,'pair_names':sorted(set(tuple(x['names']) for x in hits))}
    environment={'hand_geometry_unfiltered_count':len(hand),'hand_geometry_native_enabled_count':sum(enabled(h) for h in hand),'disabled_hand_names':[g.names[h] for h in sorted(hand) if not enabled(h)],'mount_native_status':'DISABLED_NATIVE_COLLISION_UNKNOWN_CLEARANCE' if mountcollision and not any(enabled(i) for i in mountcollision) else 'CHECK_INVENTORY','mount_ids':sorted(mount),'table_status':'ABSENT_IN_MODEL_UNKNOWN_NOT_ZERO' if not table else 'PRESENT','table_ids':sorted(table),'terrain_ids':sorted(terrain),'model_excluded_body_signatures':m.exclude_signature.tolist(),'explicit_collision_pair_count':int(m.npair),'native_parent_filter_enabled':not bool(m.opt.disableflags&int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))}
    result={'status':'PASSED_INDEPENDENT_GEOMETRY_RECOMPUTATION_NO_PHYSICAL_QUALIFICATION' if all(checks.values()) else 'FAILED_INDEPENDENT_RECOMPUTATION','checks':checks,'recomputed_metrics':metrics,'optimizer_nonconverged_frames':[x for x in report['optimizer_per_frame'] if not x['success']],'native_collision_summary':summary_native,'geometry_presence_and_filters':environment,'native_geometry_inventory':inventory,'negative_witness_max_error_m':max(negative_errors),'nonnegative_witness_errors_over1um_count':sum(x>1e-6 for x in positive_errors),'nonnegative_witness_max_error_m':max(positive_errors),'failedstrict_snapshot_nominal_query':{'distance_m':float(failure_distance),'independent_positive_support_gap_m':gap,'source_certificate':pin(ROOT/'outputs/contact_geometry_independent_review_v1/separation_certificate.json')},'model':pin(model),'context':pin(context),'candidate':pin(candidate),'parent_report':pin(parent/'report.json'),'code':pin(__file__),'reconstructed_arrays':pin(OUT/'reconstructed_arrays.npz'),'native_contacts':pin(OUT/'native_contacts.jsonl'),'source_hashes':before,'query_options':g.query_options,'forbidden_API_call_counts':calls,'dynamics_steps':0,'known_contact_labels_inferred':False,'physical_reference_qualified':False,'scope':'Reconstruction of final151frame geometry and declared metrics. Original optimizer execution path is supported by source/guard/recorded completion, not independently rerun. Soft0.5mm allowance remains exceeded slightly at all151frames, three optimizer frames did not converge; do not call this strict collision-free geometry or known/force-bearing contacts. Native collision uses original model masks/parent/exclusion filters. Absent/disabled geometry and absent native hits do not prove no overlap. Unfilteredsidequeries are explicitly separate. Camera/scale/topology/palm-root mappings and dynamic feasibility remain unqualified.','wall_seconds_including_setup':time.perf_counter()-started}
    write(OUT/'report.json',result);print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'metrics':metrics,'native':summary_native,'presence':environment},indent=2));raise SystemExit(not all(checks.values()))
if __name__=='__main__':main()
