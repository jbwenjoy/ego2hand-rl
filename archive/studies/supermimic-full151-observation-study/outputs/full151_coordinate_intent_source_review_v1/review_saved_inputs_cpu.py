"""Read-only source/array comparison. No FK/collision/WM/Env/solver imports."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT=Path(__file__).parent
OBS=Path('D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1')
FULL=Path('D:/workspace/supermimic-observation-retarget-study/outputs/native_contact_refined_geometry_v1/geometric_retarget.npz')
RFROOT=Path('D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_filter_prefix_v1')
EXPORT=Path('D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restored_prefix_control_export_v1')
ROUGH=Path('D:/workspace/supermimic-rough-prefix-learning-study')
INTENT=ROUGH/'outputs/contact_intent_v1'
ANCPATH=Path('D:/workspace/supermimic-observation-contact-intent-study/outputs/frame0_h123_seeded_baseline_v3/frozen_anchor_bindings.json')
PROX=Path('D:/workspace/supermimic-observation-contact-intent-study/outputs/contact_intent_v2/surface_proximity_151.npz')
FIRST=Path('D:/workspace/supermimic-observation-retarget-study/outputs/native_contact_refined_independent_review_v1/firstframe_geometry_parameters.json')
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
def load(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in (z.files if keys is None else keys)}
def rotation(q):
    q=np.asarray(q,np.float64);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(q.shape[:-1]+(3,3))
def compare(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return {'shape_same':a.shape==b.shape,'dtype_same':str(a.dtype)==str(b.dtype),'array_equal':bool(np.array_equal(a,b)),'byte_equal':a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),'max_abs_difference':float(np.abs(a.astype(float)-b.astype(float)).max()) if a.shape==b.shape else None}
def locations(p,patterns):
    lines=Path(p).read_text().splitlines()
    return {'file':pin(p),'lines':{pattern:[i+1 for i,v in enumerate(lines) if pattern in v] for pattern in patterns}}

obs=load(OBS/'observations.npz',['time_s','hand_landmarks_source_w','object_pose_source_w','landmark_confidence_known','object_confidence_known'])
obsmeta=J(OBS/'manifest.json');full=load(FULL);rf=load(RFROOT/'candidate.npz');manifest=J(RFROOT/'manifest.json')
reference=load(EXPORT/'reference_bspline.npz');init=load(EXPORT/'init_requested.npz');controls=load(EXPORT/'planned_control.npz')
bridge=load(ROUGH/'outputs/input_bundle_v1/contact_reference.npz');bridgemeta=J(ROUGH/'outputs/input_bundle_v1/contact_reference.json')
targets=load(INTENT/'contact_targets.npz');binding=J(INTENT/'binding_manifest.json');rules=J(INTENT/'rules.json');spec=J(INTENT/'objective_spec.json')
anchors=J(ANCPATH);first=J(FIRST)
q0src=load(manifest['q0_source']['path']);rawfd=load(manifest['rawFD']['path'])
proximity=load(PROX,['source_rigid_mapped__points_w','source_rigid_mapped__points_o','source_rigid_mapped__closest_w','source_rigid_mapped__closest_o_m','source_rigid_mapped__normal_w','source_rigid_mapped__outward_normal_o'])

comparisons={}
def add(n,a,b):comparisons[n]=compare(a,b)
add('observation_time_vs_full151_time',obs['time_s'],full['time_s'])
add('RF16_time_vs_full151_prefix',rf['time_s'],full['time_s'][:16])
add('RF16_object_pose_vs_full151_prefix',rf['mapped_object_pose_w'],full['mapped_object_pose_w'][:16])
add('RF16_mapping_R_vs_full151_prefix',rf['source_to_fixed_wrist_R'],full['source_to_fixed_wrist_R'][:16])
add('RF16_mapping_t_vs_full151_prefix',rf['source_to_fixed_wrist_translation_m'],full['source_to_fixed_wrist_translation_m'][:16])
add('RF16_normalized_target_vs_full151_prefix',rf['morphology_normalized_target_points_w'],full['morphology_normalized_target_points_w'][:16])
add('RF16_q_vs_full151_prefix_different',rf['q_goal_rad'],full['joint_pos_rad'][:16])
add('RF16_q0_vs_full151_q0_different',rf['q_goal_rad'][0],full['joint_pos_rad'][0])
q0key='joint_pos_rad' if 'joint_pos_rad' in q0src else 'q_rad'
q0=q0src[q0key].reshape(-1,20)[0]
add('RF16_q0_vs_canonical_H123_q0',rf['q_goal_rad'][0],q0)
add('RF16_object0_vs_canonical_H123_object0',rf['mapped_object_pose_w'][0],q0src['mapped_object_pose_w'].reshape(-1,7)[0])
add('RF16_q0_cast_vs_requested_actual_init',rf['q_goal_rad'][0].astype(np.float32),init['q_init_requested_FP32_rad'])
add('RF16_object0_cast_vs_requested_actual_init',rf['mapped_object_pose_w'][0].astype(np.float32),init['object_pose_init_requested_FP32_w'])
add('bridge_q_vs_RF16_q',bridge['joint_pos_rad'],rf['q_goal_rad'])
add('bridge_object_vs_RF16_object',bridge['object_pose_w'],rf['mapped_object_pose_w'])
add('bridge_q_vs_contact_target_input_q',bridge['joint_pos_rad'],targets['input_joint_pos_rad'])
add('bridge_object_vs_contact_target_input_object',bridge['object_pose_w'],targets['input_object_pose_w'])
add('bridge_wrist_vs_contact_target_input_wrist',bridge['wrist_pose_w'],targets['input_wrist_pose_w'])
add('rawFD_full151_object_pose_vs_full151_mapping',rawfd['object_pose_mapped_w'],full['mapped_object_pose_w'])
add('bridge_wrist_constant16',bridge['wrist_pose_w'],np.broadcast_to(bridge['wrist_pose_w'][:1],bridge['wrist_pose_w'].shape))
add('bridge_wrist0_vs_saved_original_default_mocap',bridge['wrist_pose_w'][0],np.r_[first['mocap'][0]['position_w'],first['mocap'][0]['quaternion_wxyz']])

Rmap=full['source_to_fixed_wrist_R'];tmap=full['source_to_fixed_wrist_translation_m'];points=obs['hand_landmarks_source_w'].astype(np.float64)
mapped_points=np.einsum('tij,tkj->tki',Rmap,points)+tmap[:,None,:]
expected_p=np.einsum('tij,tj->ti',Rmap,obs['object_pose_source_w'][:,:3])+tmap
expected_R=Rmap@rotation(obs['object_pose_source_w'][:,3:])
mapping={'object_position_rebuild_maxabs_m':float(np.abs(expected_p-full['mapped_object_pose_w'][:,:3]).max()),
         'object_rotation_rebuild_maxabs_entry':float(np.abs(expected_R-rotation(full['mapped_object_pose_w'][:,3:])).max()),
         'all151_source_palmroot_mapped_position_range_m':np.ptp(mapped_points[:,0],axis=0).tolist(),
         'normalized_target_vs_rigid_source_points_maxabs_m':float(np.abs(full['morphology_normalized_target_points_w']-mapped_points).max()),
         'mapping_R_frame0_to_final_maxabs':float(np.abs(Rmap[-1]-Rmap[0]).max()),
         'mapping_t_frame0_to_final_m':(tmap[-1]-tmap[0]).tolist(),
         'world_frame_semantics':'Per-frame source-hand palm-frame rebase to the fixed robot palm; it is not a single calibrated source-world/camera transform. Object pose undergoes the same per-frame rigid transform; morphology-normalized hand targets additionally change segment lengths.'}

anchor_checks=[]
for a in anchors['observation_hypotheses']:
    l=a['landmark_index'];anchor_checks.append({'finger':a['finger_chain'],'landmark_index':l,
        'saved_source_rigid_point_equal':bool(np.array_equal(a['point_w'],proximity['source_rigid_mapped__points_w'][0,l])),
        'saved_source_rigid_surface_equal':bool(np.array_equal(a['object_surface_closest_w'],proximity['source_rigid_mapped__closest_w'][0,l])),
        'saved_source_rigid_normal_equal':bool(np.array_equal(a['outward_normal_w'],proximity['source_rigid_mapped__normal_w'][0,l])),
        'distance_m':a['signed_distance_m'],'confidence':a['confidence'],'known_contact':a['is_known_contact_label']})

vel={'RF_requested_qd0_nonzero_count':int(np.count_nonzero(init['qd_init_requested_FP32_rad_s'])),
     'RF_requested_qd0_absmax_rad_s':float(np.max(np.abs(init['qd_init_requested_FP32_rad_s']))),
     'RF_requested_object_world_linear_m_s':init['object_bodyorigin_v_requested_FP32_world_m_s'].tolist(),
     'RF_requested_object_body_angular_rad_s':init['object_angular_requested_FP32_body_rad_s'].tolist(),
     'RF_requested_object_world_angular_rad_s':init['object_angular_requested_FP32_world_rad_s'].tolist(),
     'RF_requested_u0_rad':init['u_planned0_FP32_rad'].tolist(),
     'full539d_has_velocity_or_init_or_control_fields':any(k in full for k in ['joint_vel_rad_s','qvel','qd_init','u0','ctrl','q_init_requested_FP32_rad']),
     'firstframe_observed_initial_joint_velocity':first['observed_initial_joint_velocity'],'firstframe_observed_object_twist':first['observed_object_twist'],
     'old_zero_rest_is_separate_intervention':'The 539d NPZ has geometric references only. Scratch query qvel=0 and the later explicitly designed zero-velocity rest probe are not observed initial velocities and cannot supply full-motion reset semantics.'}

producer=ROUGH/'scripts/build_prefix_contact_targets_cpu.py';binder=ROUGH/'src/rough_prefix/contact_binding.py';origproducer=Path('D:/workspace/supermimic-observation-retarget-study/scripts/refine_native_contacts_cpu.py')
source_findings=[locations(producer,['CANDIDATE=','CONTACTS=','ANCHORS=','allcontacts=','known[i]=','source_kind','phase=np.where','human_contact_observation_known']),
                 locations(binder,['KNOWN_SEMANTICS=','source_window_half_open','actual_geometry_host_copy_fields','FAILED_RETAINED']),
                 locations(origproducer,['t, human, object_source','rotations =','translations =','mapped_pose[:, :3]','object_pose_fixed','contact_attraction'])]

intent={'frame0_H123_observation_provenance':anchor_checks,
        'H123_is_observed_contact_label':False,
        'H123_selection_basis':'Frame0 allowlist source-rigid landmark proximity is preserved separately from morphology-normalized targets. Finger/robot-material choice is a declared design hypothesis, selected before its optimizer but in the known exploratory sequence.',
        'actual16_support_target_origin':'Saved native generated contacts of the final R/F retarget q16 plus declared H123 lineage; contact_known is validity of that generated geometric hypothesis, not an independent human-contact observation.',
        'actual16_phase_origin':'Geometric shaft speed of mapped object and generated-contact finger signature, not observed contact-phase annotation.',
        'known_semantics':binding['known_semantics'],'source_kind':binding['source_kind'],
        'all16_declared_known':bool(targets['contact_known'].all()),'human_known_count':int(targets['human_contact_observation_known'].sum()),
        'known_no_contact_count':int(targets['known_no_contact'].sum()),'phase_counts':{str(i):int(np.sum(targets['phase']==i)) for i in range(5)},
        'no_policy_contact_leakage_claim':binding['target_defined_from_policy_contacts'] is False,
        'source_circularity_boundary':'A fixed target derived once from an independently frozen retarget reference is an admissible kinematic teacher; it is not independent evidence that the human contacted the object and cannot validate the retarget geometry by agreement with itself.',
        'existing_factory_not_full151_compatible':'Hardcoded variant, [200,216), frame_count16 and exact input hashes. A new version/binding is required; shape-padding, tiling16 or relabeling the old spec is invalid.'}

recommendations=[
 {'priority':1,'topic':'Object mapping','decision':'Use the frozen all151 mapped_object_pose_w and source_to_fixed_wrist R/t from 539d as one explicit fixed-palm task-redesign mapping. Its prefix equals R/F16 exactly; do not introduce an object shift, original source-world poses, or zero-rest semantics under the same identity.'},
 {'priority':2,'topic':'Initial state versus reference','decision':'Keep q_goal151, actual-q_init, designed-qdot_init, object-pose_init/world-v/body-omega, executed-u0 and C2/EMA memory separate. A new full151 curve must define its own continuous derivative/control boundaries. If canonical H123/RF q0 is retained, use that q0 explicitly; old539d q0 is different. No q0+oldtail splice or implicit inheritance of old rest or prefix qualification.'},
 {'priority':3,'topic':'Temporal contact intent','decision':'Do not repeat the 16 targets or assert H123 material anchors hold for3s. With only observations, all human contact/confidence/release labels remain UNKNOWN. An explicit source-local geometric hypothesis may be transported by each object pose, or per-frame proximity hypotheses may be frozen from existing observation-surface diagnostics; either requires a declared temporal schedule/unknown mask and independent new provenance before optimization.'},
 {'priority':4,'topic':'Frozen teacher alternative','decision':'If using saved539d generated contacts as a151 kinematic teacher, name it derived_robot_kinematic_contact_intent and bind the exactq151/model/context/COM/pair-mask/sourcewindow. It is a different teacher from transporting H123 intent, does not establish human-contact correctness, and may omit a force-bearing thumb contact. Missing generated contacts remain UNKNOWN, never known_no_contact.'},
 {'priority':5,'topic':'Independent intent versus q-derived teacher','decision':'Prefer explicitly separating observation-derived point/normal intent, robot material correspondence, and candidate-generated native contact diagnostics. The first can constrain a new trajectory; the latter should assess that trajectory rather than retroactively define observed contact truth. Keep current exploratory proxy/gradient qualification failures and no physical-reference claim.'},
 {'priority':6,'topic':'Next input gate','decision':'Before a producer/solver, freeze full151 source lineage[200,351), mapping/scale assumptions, actual geometry context, q0 and velocity/control design, confidence/phase semantics, and an explicit contact schedule with UNKNOWN intervals. Validate byte identity on shared prefix fields, but require new151 time/candidate/target hashes. All-unknown support must report A2 degeneration to penalty-only rather than fabricate support labels.'}
]

paths=[OBS/'observations.npz',OBS/'manifest.json',FULL,RFROOT/'candidate.npz',RFROOT/'manifest.json',EXPORT/'reference_bspline.npz',EXPORT/'init_requested.npz',EXPORT/'planned_control.npz',ROUGH/'outputs/input_bundle_v1/contact_reference.npz',ROUGH/'outputs/input_bundle_v1/contact_reference.json',INTENT/'contact_targets.npz',INTENT/'binding_manifest.json',INTENT/'objective_spec.json',INTENT/'rules.json',ANCPATH,PROX,FIRST,producer,binder,origproducer,Path(manifest['q0_source']['path']),Path(manifest['rawFD']['path'])]
result={'schema':'full151_coordinate_intent_source_review:1','status':'READ_ONLY_REVIEW_COMPLETE_NEW_FULL151_INPUT_NOT_YET_QUALIFIED',
        'source_pins':[pin(p) for p in paths],'review_source':pin(__file__),
        'observation_contract':{'source_window_half_open':obsmeta['original_source_frames'],'frames':obsmeta['frames'],'read_observation_fields':list(obs),'wrist_observation_values_read':False,'original_video_parent_id':obsmeta['original_video_parent_id'],'is_real_ego':obsmeta['is_real_ego'],'landmark_confidence_known_count':int(obs['landmark_confidence_known'].sum()),'object_confidence_known_count':int(obs['object_confidence_known'].sum()),'camera_scale_frame_semantics_qualified':False},
        'comparisons':comparisons,'mapping_reconstruction':mapping,'init_and_velocity_contract':vel,'intent_provenance':intent,'source_locations':source_findings,'recommendations':recommendations,
        'new_full151_producer_or_targets_created':False,'queries_FK_collision_Env_WM_physics_optimizer_calls':0,
        'source_q_read_scope':'Only already generated539d151/RF16/H123 candidates for comparison, never upstream source robot labels.',
        'physical_reference_qualified':False,'full_motion_training_or_physics_authorized_by_this_review':False}
(OUT/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
np.savez_compressed(OUT/'comparison_evidence.npz',full_time_s=full['time_s'],full151_q0=full['joint_pos_rad'][0],canonical_H123_q0=q0,RF16_q0=rf['q_goal_rad'][0],RF16_minus_full151_prefix_q=rf['q_goal_rad']-full['joint_pos_rad'][:16],object_mapping_position_residual_m=expected_p-full['mapped_object_pose_w'][:,:3],object_mapping_rotation_residual=expected_R-rotation(full['mapped_object_pose_w'][:,3:]),RF_init_qd0=init['qd_init_requested_FP32_rad_s'],RF_init_object_v_world=init['object_bodyorigin_v_requested_FP32_world_m_s'],RF_init_object_omega_body=init['object_angular_requested_FP32_body_rad_s'])
print(json.dumps({'status':result['status'],'comparisons':comparisons,'mapping_reconstruction':mapping,'init_and_velocity_contract':vel,'anchor_checks':anchor_checks},indent=2))
