"""Saved reference extraction only; no robot FK or optimization."""
from pathlib import Path
import json, hashlib
import numpy as np
ROOT=Path(__file__).resolve().parent
PILOT=ROOT.parent
REFERENCE=PILOT/'ego_hand_geometry_v1'
NAMES=('thumb','index','middle','ring','pinky')
THRESHOLD_M=.005  # declared before reading per-finger distances; not a force gate.


def pin(p):
    p=Path(p).resolve(strict=True)
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    if (ROOT/'frame42_targets.json').exists():raise ValueError('Fresh targets only')
    report=json.loads((REFERENCE/'report.json').read_text(encoding='utf8'))
    for name in ('left_hand_phone_geometry61.npz','proximity_candidates61.npz','raw_profile_and_poses.json'):
        assert pin(REFERENCE/name)['sha256']==report['outputs'][name]['sha256']
    with np.load(REFERENCE/'left_hand_phone_geometry61.npz',allow_pickle=False) as a:
        assert a['frame_ids'][0]==42
        tips=a['landmarks20_world_FP32'][0,:5].copy()
        tips_object=a['landmarks21_object_F64'][0,:5].copy()
        verts_object=a['hand_vertices_object_F64'][0].copy()
        angles22=a['joint_angles22_FP32'][0].copy()
    with np.load(REFERENCE/'proximity_candidates61.npz',allow_pickle=False) as a:
        distances=a['hand_vertex_unsigned_scan_distance_m'][0].copy()
        closest=a['hand_vertex_closest_scan_point_object_m'][0].copy()
        triangles=a['hand_vertex_closest_scan_triangle'][0].copy()
        tip_distances=a['landmark21_unsigned_scan_distance_m'][0,:5].copy()
    raw=json.loads((REFERENCE/'raw_profile_and_poses.json').read_text(encoding='utf8'))
    weights=np.asarray(raw['umetrack_profile']['dense_bone_weights'],np.float64)
    mapping=[]
    for f,name in enumerate(NAMES):
        bone=4+3*f
        # Equal positive dominant weights are retained and explicitly counted.
        owned=(weights[:,bone]==weights.max(1))&(weights[:,bone]>0)
        ids=np.flatnonzero(owned)
        assert len(ids)>0
        minimum=distances[ids].min();nearest=ids[distances[ids]==minimum];v=int(nearest.min())
        mapping.append({'finger':name,'canonical_tip_index':f,'robot_finger_index':f+1,
            'robot_tip_site':f'right_finger{f+1}_tip','official_distal_bone_frame':bone,
            'dominant_distal_vertex_ids':ids.tolist(),'dominant_distal_vertex_count':len(ids),
            'dominant_weight_tie_vertex_count':int(((weights[ids]==weights[ids].max(1)[:,None]).sum(1)>1).sum()),
            'reference_tip_object_m':tips_object[f].tolist(),'reference_tip_world_m':tips[f].tolist(),
            'anatomical_tip_unsigned_scan_distance_m':float(tip_distances[f]),
            'nearest_distal_vertex_id':v,'nearest_equal_distance_vertex_ids':nearest.tolist(),
            'nearest_distal_vertex_object_m':verts_object[v].tolist(),
            'nearest_scan_surface_point_object_m':closest[v].tolist(),'nearest_scan_triangle_id':int(triangles[v]),
            'distal_mesh_unsigned_scan_distance_m':float(minimum),
            'uniform_near_contact_candidate':bool(minimum<=THRESHOLD_M)})
    result={'schema':'frame42_object_centric_anatomical_targets:1','status':'REFERENCE_EXTRACTED_NOT_SOLVED',
        'frame':42,'frame_selection':'First frame of preselected42–102 window, independent of fitting outcome',
        'source_pins':{'reference_report':pin(REFERENCE/'report.json'),'geometry':pin(REFERENCE/'left_hand_phone_geometry61.npz'),
                       'proximity':pin(REFERENCE/'proximity_candidates61.npz'),'raw_profile_poses':pin(REFERENCE/'raw_profile_and_poses.json'),
                       'mapping_source':pin(ROOT/'mapping_source.json'),'preparer':pin(__file__)},
        'all_human_joint_angles22_rad':angles22.tolist(),'angles_not_robot_joint_targets':True,
        'human_hand':'left','robot_hand':'right','digit_correspondence':'same anatomical digit, no finger permutation',
        'left_to_right_convention':'No reflection of hand targets or phone. Only proper phone SE3 and robot20 joint variables; chirality/shape mismatch remains in residual.',
        'surface_candidate_threshold_m':THRESHOLD_M,'surface_candidate_definition':'Min unsigned original-scan triangle distance among vertices with positive dominant weight on that finger distal official skinning frame. Uniform5mm descriptor, not contact or force label.',
        'all_five_tip_targets_retained':True,'mapping':mapping,'raw21_extra_landmark_not_used_as_robot_target':20,
        'robot_FK_optimization_physics_calls':0}
    (ROOT/'frame42_targets.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'targets':pin(ROOT/'frame42_targets.json'),'fingers':[{'finger':m['finger'],'candidate':m['uniform_near_contact_candidate'],'distal_distance_mm':m['distal_mesh_unsigned_scan_distance_m']*1000,'tip_distance_mm':m['anatomical_tip_unsigned_scan_distance_m']*1000} for m in mapping]}))


if __name__=='__main__':main()
