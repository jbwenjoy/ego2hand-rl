"""Independent final151frame native-contact IK artifact review, no optimizer."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
ROOT=Path('D:/workspace/supermimic-observation-retarget-study');OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from retarget_palmroot_cpu import RobotKinematics,hand_frames,directions

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def arr(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')
def depth_summary(ds,mask):
    measured=mask.any(1);perframe=np.array([np.maximum(-d[m],0).max() if m.any() else np.nan for d,m in zip(ds,mask)])
    return {'max_generated_depth_m':float(perframe[measured].max()) if measured.any() else None,'frames_with_generated_contact':int(measured.sum()),'frames_without_generated_contact_unknown_gap':int((~measured).sum()),'frames_generated_depth_over3mm':int((perframe[measured]>.003).sum()),'frames_generated_depth_over_allowance':int((perframe[measured]>.0005).sum())}

def main():
    started=time.perf_counter();parent=ROOT/'outputs/native_contact_refined_geometry_v1';report=read(parent/'report.json');recipe=read(parent/'recipe.json');optimizer=[json.loads(l) for l in (parent/'optimizer.jsonl').read_text().splitlines()]
    candidate=parent/'geometric_retarget.npz';baseline=ROOT/'outputs/contact_refined_nominal_geometry_v1/geometric_retarget.npz';observation=Path('D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1/observations.npz');model=Path('D:/workspace/supermimic-chord-study/outputs/rotating_reference_warp_a0/model.mjb');context=Path('D:/workspace/supermimic-chord-study/outputs/a1l_paired_v2_clip007_2000iter_seed1/compiled/fixed_physics_context.npz');module=ROOT/'scripts/native_filtered_contact_geometry.py';caller=ROOT/'scripts/refine_native_contacts_cpu.py'
    files=[candidate,baseline,observation,model,context,module,caller,parent/'report.json',parent/'recipe.json',parent/'optimizer.jsonl'];before={str(p):sha(p) for p in files}
    z=arr(candidate);base=arr(baseline);ctx=arr(context)
    with np.load(observation,allow_pickle=False) as a:t=np.array(a['time_s'],dtype=float);human=np.array(a['hand_landmarks_source_w'],dtype=float);objectsource=np.array(a['object_pose_source_w'],dtype=float)
    checks={'input_model_context_module_caller_hashes':all(sha(p)==report[k] for p,k in [(candidate,'output_npz_sha256'),(baseline,'baseline_npz_sha256'),(observation,'input_npz_sha256'),(model,'model_sha256'),(context,'context_sha256'),(module,'module_sha256'),(caller,'script_sha256'),(parent/'recipe.json','recipe_sha256')]),'module_c964_retained':sha(module)=='c964d5b300baac57a5353afd3b5aa4cfb381eca1da03ceef712691dd10376c72',
        'all151_finite20q_no_selection':z['joint_pos_rad'].shape==(151,20) and np.isfinite(z['joint_pos_rad']).all() and len(optimizer)==151 and [x['frame'] for x in optimizer]==list(range(151)),
        'all_eight_unconverged_frames_retained':[x['frame'] for x in optimizer if not x['success']]==report['optimizer_unconverged_frames']==[54,55,56,82,83,84,85,86] and all(x['nfev']==80 and x['status']==0 for x in optimizer if not x['success']),
        'fixed_recipe_weights_and_allowance':recipe['bone_direction_weight']==recipe['hand_object_overlap_weight']==recipe['hand_self_overlap_weight']==1 and recipe['temporal_weight']==.03 and recipe['nominal_weight']==.001 and recipe['allowance_m']==.0005 and recipe['normalization_m']==.001 and recipe['max_nfev_per_frame']==80 and recipe['contact_attraction'] is False and recipe['object_pose_fixed'] is True,
        'same_time_target_object_exact':np.array_equal(t,z['time_s']) and np.array_equal(z['mapped_object_pose_w'],base['mapped_object_pose_w']) and np.array_equal(z['morphology_normalized_target_points_w'],base['morphology_normalized_target_points_w'])}
    forbidden={n:0 for n in ['mj_geomDistance','mj_forward','mj_step','mj_step1','mj_step2','mj_inverse','mj_contactForce']}
    def deny(n):
        def f(*a,**kw):forbidden[n]+=1;raise AssertionError('Forbidden '+n)
        return f
    for n in forbidden:setattr(mujoco,n,deny(n))
    robot=RobotKinematics(model);m,d=robot.model,robot.data
    for field in ['geom_size','geom_pos']:assert np.array_equal(ctx[field],np.broadcast_to(ctx[field][:1],ctx[field].shape));getattr(m,field)[:]=ctx[field][0]
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(m.ngeom)];pairs=z['geometry_pair_indices'];kinds=z['geometry_pair_kind'];lookup={tuple(p):i for i,p in enumerate(pairs.tolist())};objqa=int(m.joint('object/cube_freejoint').qposadr[0]);hand=set(pairs[kinds=='hand_self'].ravel());objectids={m.geom(n).id for n in ['object/cube','object/A_bottom','object/A_top']}
    checks['catalog273_original_enabled_filters']=len(pairs)==273 and (kinds=='hand_self').sum()==210 and (kinds=='hand_object').sum()==63 and len(hand)==21 and all(m.geom_contype[h] or m.geom_conaffinity[h] for h in hand) and not any('mount' in names[i] or 'link1_col' in names[i] for i in pairs.ravel())
    checks['original_options_context_exact']=m.opt.ccd_tolerance==1e-6 and m.opt.ccd_iterations==50 and np.array_equal(m.geom_size,ctx['geom_size'][0]) and np.array_equal(m.geom_pos,ctx['geom_pos'][0])
    R=np.einsum('ij,tkj->tik',robot.palm_frame,hand_frames(human));translation=robot.neutral[0]-np.einsum('tij,tj->ti',R,human[:,0]);units=np.einsum('tij,tkj->tki',R,directions(human))
    checks['observations_mapping_rebuilt']=np.max(np.abs(R-z['source_to_fixed_wrist_R']))<1e-12 and np.max(np.abs(translation-z['source_to_fixed_wrist_translation_m']))<1e-12 and np.max(np.abs(z['mapped_object_pose_w'][:,:3]-(np.einsum('tij,tj->ti',R,objectsource[:,:3])+translation)))<1e-12 and np.max(np.abs(R@Rotation.from_quat(objectsource[:,[4,5,6,3]]).as_matrix()-Rotation.from_quat(z['mapped_object_pose_w'][:,[4,5,6,3]]).as_matrix()))<1e-12
    def setq(q,pose):
        d.qpos[objqa:objqa+7]=pose;robot.last_q=None;return robot.forward(q)
    native_rows=[];frame_rows=[];fk=[];angles=[];targets=[];dist=[];mask=[];count=[];ties=[];basedist=[];basemask=[];cost=[];normalerrors=[]
    def collide(frame,phase):
        qbefore=d.qpos.copy();vbefore=d.qvel.copy();clock=d.time;mujoco.mj_collision(m,d);assert np.array_equal(qbefore,d.qpos) and np.array_equal(vbefore,d.qvel) and d.time==clock==0
        buckets=[[] for _ in pairs];selecteddist=np.full(len(pairs),np.nan);generated=np.zeros(len(pairs),dtype=bool);counts=np.zeros(len(pairs),int);tie=np.zeros(len(pairs),int)
        for index,c in enumerate(d.contact):
            a,b=map(int,c.geom);pair=tuple(sorted((a,b)));row=lookup.get(pair);category=str(kinds[row]) if row is not None else 'uncatalogued_native_contact'
            n=c.frame[:3].copy();normalerrors.append(abs(np.linalg.norm(n)-1));entry={'frame':frame,'phase':phase,'native_contact_index':index,'catalogue_pair_index':row,'category':category,'geom':[a,b],'names':[names[a],names[b]],'distance_m':float(c.dist),'position_w':c.pos.copy().tolist(),'normal_geom0_to_geom1':n.tolist()};native_rows.append(entry)
            if row is not None:buckets[row].append((float(c.dist),index))
        for row,hits in enumerate(buckets):
            if not hits:continue
            values=np.array([h[0] for h in hits]);minimum=values.min();selecteddist[row]=minimum;generated[row]=True;counts[row]=len(hits);tie[row]=int((np.abs(values-minimum)<=1e-9).sum())
        return selecteddist,generated,counts,tie
    for frame,q in enumerate(z['joint_pos_rad']):
        points,_=setq(q,z['mapped_object_pose_w'][frame]);feature,_=robot.bone_features(q);fk.append(points.copy());angles.append(np.rad2deg(np.arccos(np.clip(np.sum(feature*units[frame],axis=-1),-1,1))))
        target=robot.neutral.copy()
        for finger in range(5):
            first=1+4*finger
            for bone in range(3):target[first+bone+1]=target[first+bone]+robot.lengths[3*finger+bone]*units[frame,3*finger+bone]
        targets.append(target);dd,gg,cc,tt=collide(frame,'refined');dist.append(dd);mask.append(gg);count.append(cc);ties.append(tt)
        residual=np.zeros(len(pairs));residual[gg]=np.maximum(-dd[gg]-.0005,0)/.001;prev=base['joint_pos_rad'][0] if frame==0 else z['joint_pos_rad'][frame-1];r=np.r_[(feature-units[frame]).ravel(),residual,.03*(q-prev),.001*(q-robot.nominal)];cost.append(float(.5*r@r))
        row={'frame':frame,'time_s':float(t[frame]),'generated_pairs':int(gg.sum()),'unmeasured_gap_pairs':int((~gg).sum()),'tied_generated_pairs':int(((tt>1)&gg).sum()),'optimizer_success':optimizer[frame]['success'],'reconstructed_cost':cost[-1]}
        for kind in ['hand_object','hand_self']:
            km=kinds==kind;seen=gg&km;row[kind+'_generated_pairs']=int(seen.sum());row[kind+'_contact_records']=int(cc[km].sum());row[kind+'_max_generated_depth_m']=float(np.maximum(-dd[seen],0).max()) if seen.any() else None
        frame_rows.append(row);setq(base['joint_pos_rad'][frame],z['mapped_object_pose_w'][frame]);bd,bg,_,_=collide(frame,'object_only_baseline');basedist.append(bd);basemask.append(bg)
    fk,angles,targets,dist,mask,count,ties,basedist,basemask=map(np.asarray,[fk,angles,targets,dist,mask,count,ties,basedist,basemask])
    checks['FK_bone_angles_targets_rebuilt']=np.max(np.abs(fk-z['fk_hand_points_w']))<1e-12 and np.max(np.abs(angles-z['bone_angle_error_deg']))<1e-9 and np.array_equal(targets,z['morphology_normalized_target_points_w'])
    checks['all273_selected_distances_generated_masks_rebuilt']=np.array_equal(dist,z['selected_native_contact_distance_m'],equal_nan=True) and np.array_equal(mask,z['generated_contact_mask']) and np.array_equal(count,z['native_contact_count_by_pair']) and np.array_equal(ties,z['native_contact_tie_count_by_pair'])
    checks['baseline_native_distances_masks_rebuilt']=np.array_equal(basedist,z['baseline_selected_native_contact_distance_m'],equal_nan=True) and np.array_equal(basemask,z['baseline_generated_contact_mask'])
    checks['all_unknown_gaps_preserved_NaN']=np.isnan(dist[~mask]).all() and np.isfinite(dist[mask]).all() and np.isnan(basedist[~basemask]).all()
    checks['all_final_optimizer_costs_reconstructed']=max(abs(a-b['cost']) for a,b in zip(cost,optimizer))<1e-8
    summaries={kind:{'before':depth_summary(basedist[:,kinds==kind],basemask[:,kinds==kind]),'after':depth_summary(dist[:,kinds==kind],mask[:,kinds==kind])} for kind in ['hand_object','hand_self']}
    checks['all_native_contact_summary_fields_reproduced']=summaries==report['native_contact_summaries']
    metrics={'bone_angle_mean_deg':float(angles.mean()),'bone_angle_max_deg':float(angles.max()),'normalized_target_tip_error_mean_m':float(np.linalg.norm(fk[:,[4,8,12,16,20]]-targets[:,[4,8,12,16,20]],axis=-1).mean()),'hard_joint_excess_max_rad':float(np.maximum(np.maximum(robot.lower-z['joint_pos_rad'],z['joint_pos_rad']-robot.upper),0).max()),'q_ref_fd_rate_abs_max_rad_s':float(np.abs(np.diff(z['joint_pos_rad'],axis=0)/np.diff(t)[:,None]).max()),'q_change_from_input_baseline_abs_max_rad':float(np.abs(z['joint_pos_rad']-base['joint_pos_rad']).max())}
    checks['all_scalar_metrics_reproduced']=all(abs(metrics[k]-report[k])<1e-9 for k in metrics)
    checks['eighty_tied_pair_frames_retained']=int(((ties>1)&mask).sum())==report['generated_tied_pair_frames']==80
    checks['native_normals_unit']=max(normalerrors)<1e-6;checks['no_geomDistance_dynamics_force_calls']=not any(forbidden.values());checks['all_parent_sources_and_artifacts_unchanged']=before=={str(p):sha(p) for p in files}
    (OUT/'all151_native_contacts.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in native_rows),encoding='utf-8')
    np.savez_compressed(OUT/'reconstructed_arrays.npz',time_s=t,joint_pos_rad=z['joint_pos_rad'],fk_hand_points_w=fk,bone_angle_error_deg=angles,targets=targets,selected_native_contact_distance_m=dist,generated_contact_mask=mask,native_contact_count_by_pair=count,native_contact_tie_count_by_pair=ties,baseline_dist=basedist,baseline_generated=basemask,reconstructed_cost=np.array(cost))
    with (OUT/'frames.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(frame_rows[0]));w.writeheader();w.writerows(frame_rows)
    # Export geometric frame0 inputs and model/context parameter inventory only.
    # qvel0 here is a geometry-query scratch state, not an observed velocity or
    # an already authorized/executed rest-hold physical experiment.
    setq(z['joint_pos_rad'][0],z['mapped_object_pose_w'][0]);firstqpos=d.qpos.copy();_,firstmask,firstcount,firstties=collide(0,'frame0_inventory_duplicate')
    objectbody=int(m.geom_bodyid[next(iter(objectids))]);objectgeoms=[{'id':int(i),'name':names[i],'type_id':int(m.geom_type[i]),'type_name':str(mujoco.mjtGeom(int(m.geom_type[i]))),'size':m.geom_size[i].tolist(),'local_pos':m.geom_pos[i].tolist(),'local_quat_wxyz':m.geom_quat[i].tolist(),'world_pos':d.geom_xpos[i].tolist(),'world_R':d.geom_xmat[i].reshape(3,3).tolist(),'contype':int(m.geom_contype[i]),'conaffinity':int(m.geom_conaffinity[i])} for i in sorted(objectids)]
    frame0={'status':'GEOMETRIC_FRAME0_PARAMETERS_ONLY_NOT_PHYSICAL_INIT_QUALIFICATION','frame':0,'time_s':float(t[0]),'candidate':pin(candidate),'model':pin(model),'actual_context':pin(context),'joint_names':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(m.dof_jntid[v])) for v in robot.va],'q_rad':z['joint_pos_rad'][0].tolist(),'joint_qposadr':robot.qa.tolist(),'joint_dofadr':robot.va.tolist(),'joint_lower_rad':robot.lower.tolist(),'joint_upper_rad':robot.upper.tolist(),'joint_hard_excess_rad':0.,'object_pose_w_SI_wxyz':z['mapped_object_pose_w'][0].tolist(),'object_freejoint_qposadr':objqa,'full_geometric_qpos':firstqpos.tolist(),'object_geometries':objectgeoms,'object_body_id':objectbody,'object_body_name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,objectbody),
        'actual_context_object_mass_kg':float(ctx['body_mass'][0,objectbody]),'actual_context_object_inertia_kg_m2':ctx['body_inertia'][0,objectbody].tolist(),'nominal_MJB_object_mass_kg':float(m.body_mass[objectbody]),'nominal_MJB_object_inertia_kg_m2':m.body_inertia[objectbody].tolist(),
        'mocap':[{'body_id':i,'body_name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i),'mocap_id':int(m.body_mocapid[i]),'position_w':d.mocap_pos[m.body_mocapid[i]].tolist(),'quaternion_wxyz':d.mocap_quat[m.body_mocapid[i]].tolist()} for i in range(m.nbody) if m.body_mocapid[i]>=0],
        'palm_mount_world_poses':[{'body_id':i,'body_name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i),'position_w':d.xpos[i].tolist(),'quaternion_wxyz':d.xquat[i].tolist()} for i in range(m.nbody) if mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) in ['robot/right_palm_link','robot/right_mount']],
        'actuator_names':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)],'actuator_trnid':m.actuator_trnid.tolist(),'actuator_ctrlrange_rad':m.actuator_ctrlrange.tolist(),'actuator_forcerange':m.actuator_forcerange.tolist(),'actuator_gear':m.actuator_gear.tolist(),'actual_context_actuator_gainprm':ctx['actuator_gainprm'][0].tolist(),'actual_context_actuator_biasprm':ctx['actuator_biasprm'][0].tolist(),
        'original_model_options':{'timestep_s':float(m.opt.timestep),'gravity_m_s2':m.opt.gravity.tolist(),'integrator':int(m.opt.integrator),'solver':int(m.opt.solver),'iterations':int(m.opt.iterations),'tolerance':float(m.opt.tolerance),'ccd_tolerance':float(m.opt.ccd_tolerance),'ccd_iterations':int(m.opt.ccd_iterations)},'frame0_native_geometric_contacts':[c for c in native_rows if c['frame']==0 and c['phase']=='refined'],'frame0_geometry_metrics':frame_rows[0],
        'observed_initial_joint_velocity':None,'observed_object_twist':None,'PD_target':None,'applied_torques_or_forces':None,'qvel_note':'Scratch MjData qvel is zero solely for geometry queries. Any later rest-hold zero velocity/target/gravity/physics initialization must be an explicit prospective synthetic intervention, not reconstructed observation.','future_probe_scope':'Mapped object scale/camera/palm-root eligibility and force-bearing grasp are unqualified. This file proposes no physical controller/criterion and authorizes no physics/GPU run.','physical_reference_qualified':False,'dynamics_steps':0}
    write(OUT/'firstframe_geometry_parameters.json',frame0)
    result={'status':'PASSED_INDEPENDENT_NATIVE_GEOMETRY_ARTIFACT_RECONSTRUCTION' if all(checks.values()) else 'FAILED_INDEPENDENT_NATIVE_GEOMETRY_ARTIFACT_RECONSTRUCTION','checks':checks,'recomputed_metrics':metrics,'native_contact_summaries':summaries,'all151_tied_pair_frames':int(((ties>1)&mask).sum()),'unconverged_optimizer_frames':[x for x in optimizer if not x['success']],'normal_unit_max_error':max(normalerrors),'source_files':[pin(p) for p in files],'code':pin(__file__),'all151_raw_native_contacts':pin(OUT/'all151_native_contacts.jsonl'),'reconstructed_arrays':pin(OUT/'reconstructed_arrays.npz'),'firstframe_geometry_parameters':pin(OUT/'firstframe_geometry_parameters.json'),'forbidden_API_call_counts':forbidden,'dynamics_steps':0,'physical_reference_qualified':False,'scope':'Artifact reconstruction plus all151 native filtered geometric contacts, not optimizer rerun, global Jacobian qualification, strict nonpenetration, known observed contact labels or physical grasp. NaNs/generation masks and all8optimizer failures preserved. Parent/exclusion/disabled masks limit native contact coverage; absent hits do not certify clearance. qFD is geometric reference derivative.','wall_seconds':time.perf_counter()-started}
    write(OUT/'report.json',result);print(json.dumps({'status':result['status'],'checks':len(checks),'failed':[k for k,v in checks.items() if not v],'metrics':metrics,'native':summaries,'firstframe':pin(OUT/'firstframe_geometry_parameters.json')},indent=2));raise SystemExit(not all(checks.values()))
if __name__=='__main__':main()
