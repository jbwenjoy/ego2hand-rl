"""Saved observation diagnostics and metadata only; no geometry evaluation."""
from pathlib import Path
import hashlib,json
import numpy as np
OUT=Path(__file__).parent
J=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(p),'sha256':sha(p)}
base=J(OUT/'report.json');roots={
 'obs':Path('D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1'),
 'rf':Path('D:/workspace/supermimic-observation-restoration-prefix-study/outputs/restoration_filter_prefix_v1'),
 'intent':Path('D:/workspace/supermimic-rough-prefix-learning-study/outputs/contact_intent_v1'),
 'bridge':Path('D:/workspace/supermimic-rough-prefix-learning-study/outputs/input_bundle_v1')}
manifest=J(roots['rf']/'manifest.json');obsmeta=J(roots['obs']/'manifest.json');binding=J(roots['intent']/'binding_manifest.json');targetmeta=J(roots['intent']/'manifest.json');spec=J(roots['intent']/'objective_spec.json')
checks={
 'observation_SHA_matches_manifest':sha(roots['obs']/'observations.npz')==obsmeta['observation_sha256'],
 'observation_151_source_window_200_351':obsmeta['frames']==151 and obsmeta['original_source_frames']==[200,351],
 'observation_no_real_ego_no_robot_controls_contacts':all(obsmeta[k] is False for k in ['is_real_ego','contains_robot_joint_labels','contains_executed_controls','contains_reference_contacts']),
 'original_video_camera_scale_confidence_unknown':all(obsmeta[k] is None for k in ['original_video_parent_id','camera_intrinsics','camera_pose','metric_scale_calibration','confidence_values']),
 'RF_candidate_exact_frozen6364':sha(roots['rf']/'candidate.npz')==manifest['candidate']['sha256']=='6364a6e262f7fbedac049d2d9e789b2c9077cbd6ef7656ef2e9b394161bab81a',
 'full151_mapping_parent_exact539d':sha(manifest['mapped_pose_parent']['path'])==manifest['mapped_pose_parent']['sha256']=='539d6247576b9b2e3fb1b143cf17b6ad507e3baea73c5fafcda4e221d9230c8f',
 'canonical_H123_q0_pin_exact':sha(manifest['q0_source']['path'])==manifest['q0_source']['sha256'],
 'rawFD_pose_parent_pin_exact':sha(manifest['rawFD']['path'])==manifest['rawFD']['sha256'],
 'contact_binding_sourcepins_current':all(sha(p['path'])==p['sha256'] for p in binding['source_pins'].values()),
 'contact_target_NPZ_frozen_manifest_binding':sha(roots['intent']/'contact_targets.npz')==targetmeta['target_npz_sha256']==spec['expected_hashes']['target_npz_sha256'],
 'contact_bridge_reference_hash_binding':sha(roots['bridge']/'contact_reference.npz')==targetmeta['reference_npz_sha256']==binding['reference']['sha256']==spec['expected_hashes']['reference_npz_sha256'],
 'contact_bridge_metadata_hash_binding':sha(roots['bridge']/'contact_reference.json')==targetmeta['reference_json_sha256']==binding['reference_json']['sha256']==spec['expected_hashes']['reference_json_sha256'],
 'contact_human_all_unknown_policycontacts_not_used':binding['human_contact_observation_known'] is False and binding['target_defined_from_policy_contacts'] is False,
 'contact_source_kind_robotkinematic':binding['source_kind']=='derived_robot_kinematic_contact_intent',
 'local_proxy_failure_retained':binding['local_proxy_qualification']=='FAILED_RETAINED_EXPLORATORY_ONLY',
 'sharedprefix_mapping_all_exact':all(base['comparisons'][k]['byte_equal'] for k in ['RF16_time_vs_full151_prefix','RF16_object_pose_vs_full151_prefix','RF16_mapping_R_vs_full151_prefix','RF16_mapping_t_vs_full151_prefix','RF16_normalized_target_vs_full151_prefix']),
 'new_q0_canonical_exact_old_q0_different':base['comparisons']['RF16_q0_vs_canonical_H123_q0']['byte_equal'] and not base['comparisons']['RF16_q0_vs_full151_q0_different']['byte_equal'],
 'new_init_FP32_q_and_pose_match_RF':base['comparisons']['RF16_q0_cast_vs_requested_actual_init']['byte_equal'] and base['comparisons']['RF16_object0_cast_vs_requested_actual_init']['byte_equal'],
 'default_wrist_constant_and_same_old_default':base['comparisons']['bridge_wrist_constant16']['byte_equal'] and base['comparisons']['bridge_wrist0_vs_saved_original_default_mocap']['byte_equal'],
 'H123_point_normal_source_observation_diagnostic_exact':all(a['saved_source_rigid_point_equal'] and a['saved_source_rigid_surface_equal'] and a['saved_source_rigid_normal_equal'] for a in base['intent_provenance']['frame0_H123_observation_provenance']),
}
PROX=Path('D:/workspace/supermimic-observation-contact-intent-study/outputs/contact_intent_v2/surface_proximity_151.npz')
rows=[]
with np.load(PROX,allow_pickle=False) as a:
    for finger,l in [(1,3),(2,7),(3,11)]:
        d=a['source_rigid_mapped__signed_distance_m'][:,l]
        c=a['source_rigid_mapped__closest_o_m'][:,l]
        n=a['source_rigid_mapped__outward_normal_o'][:,l]
        rows.append({'finger':finger,'landmark_index':l,'signed_distance_min_m':float(d.min()),'signed_distance_max_m':float(d.max()),
                     'abs_distance_le3mm_frame_count_descriptive_only':int(np.sum(np.abs(d)<=.003)),
                     'objectlocal_closest_surface_shift_from_frame0_max_m':float(np.linalg.norm(c-c[0],axis=1).max()),
                     'objectlocal_outward_normal_angle_from_frame0_max_deg':float(np.rad2deg(np.arccos(np.clip(n@n[0],-1,1))).max()),
                     'human_contact_known':False,'release_known':False,'confidence':None})
result={'schema':'full151_source_binding_temporal_intent_scope:1','status':'PASS_METADATA_AND_ARRAY_RELATIONS_ONLY' if all(checks.values()) else 'REVIEW_BINDING_MISMATCH',
        'passed':sum(checks.values()),'total':len(checks),'checks':checks,'review_source':pin(__file__),'coordinate_review':pin(OUT/'report.json'),'saved_surface_diagnostics':pin(PROX),
        'frozen_frame0_hypothesis_temporal_descriptive_diagnostics':rows,
        'interpretation':'These saved observation-point proximity changes do not label contact or release: points are skeletal landmarks, source scale/camera are unqualified, and confidence is unknown. They demonstrate that a stationary object-material correspondence across151 frames is an additional design assumption, not an observation fact. No target schedule is chosen by this review.',
        'optimization_or_geometry_calls':0,'physical_reference_qualified':False}
(OUT/'input_binding_and_temporal_scope.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'status':result['status'],'passed':result['passed'],'total':result['total'],'failed':[k for k,v in checks.items() if not v],'temporal_rows':rows},indent=2))
