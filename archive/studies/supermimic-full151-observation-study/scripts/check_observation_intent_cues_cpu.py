"""Minimal synthetic CPU boundary proof; not a surface-query qualification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import copy,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151.observation_intent_cues import *

def fixture():
    time=np.arange(151)*.02;points=np.zeros((151,21,3));points[:,:,0]=np.arange(21)[None,:]*.001;points[:,:,1]=.02
    pose=np.zeros((151,7));pose[:,3]=1;R=np.broadcast_to(np.eye(3),(151,3,3)).copy();t=np.zeros((151,3))
    obs={'time_s':time,'hand_landmarks_source_w':points,'object_pose_source_w':pose,'landmark_confidence_known':np.zeros((151,21),bool),'object_confidence_known':np.zeros(151,bool)}
    mapping={'time_s':time,'source_to_fixed_wrist_R':R,'source_to_fixed_wrist_translation_m':t,'mapped_object_pose_w':pose}
    normal=np.zeros_like(points);normal[:,:,1]=1;closest=points-normal*.01
    ties=np.zeros((151,21,13),bool);ties[:,:,0]=True
    saved={'time_s':time,'source_to_fixed_wrist_R':R,'source_to_fixed_wrist_translation_m':t}
    for k,v in {'points_w':points,'points_o':points,'closest_o_m':closest,'closest_w':closest,'outward_normal_o':normal,'normal_w':normal,'signed_distance_m':np.full((151,21),.01),'unsigned_distance_m':np.full((151,21),.01),'feature_index':np.zeros((151,21),np.int64),'tie_mask':ties,'tie_count':np.ones((151,21),np.int64),'axis_radial_normal_ambiguity':np.zeros((151,21),bool)}.items():saved[PFX+k]=v
    return obs,mapping,saved

class Guard(dict):
    def __init__(self,d,allowed):super().__init__(d);self.allowed=set(allowed);self.read=[]
    def __getitem__(self,k):
        if k not in self.allowed:raise AssertionError('Forbidden field read '+k)
        self.read.append(k);return super().__getitem__(k)

def run(output):
    if output.exists():raise FileExistsError(output)
    checks={}
    def C(k,v):checks[k]=bool(v)
    def reject(k,fn):
        try:fn()
        except (ValueError,AssertionError):checks[k]=True
        else:checks[k]=False
    obs,mapping,saved=fixture()
    mapping['joint_pos_rad']=object();mapping['generated_contact_mask']=object();saved['robot_candidate_FK__points_w']=object()
    go,gm,gs=Guard(obs,OBS_FIELDS),Guard(mapping,MAP_FIELDS),Guard(saved,SAVED_FIELDS)
    a=extract_cues(go,gm,gs)
    C('151x3_all_selected_points_and_indices',a['source_landmark_source_w'].shape==(151,3,3) and np.array_equal(a['landmark_indices'],[3,7,11]))
    C('guarded_no_joint_generatedcontact_FK_field_access',set(go.read)<=set(OBS_FIELDS) and set(gm.read)<=set(MAP_FIELDS) and set(gs.read)<=set(SAVED_FIELDS))
    C('distance_saved_signed_and_unsigned_not_requeried',np.array_equal(a['signed_distance_m'],saved[PFX+'signed_distance_m'][:,LANDMARKS]))
    C('unknown_human_masks_all_frames',not a['human_contact_known'].any() and not a['human_contact_present'].any() and a['human_contact_unknown_mask'].all())
    C('declared_intent_and_source_confidence_all_unknown',not a['declared_robot_intent_known'].any() and not a['landmark_confidence_known'].any() and not a['object_confidence_known'].any())
    C('unique_finite_normals_geometrically_valid_only',a['surface_normal_valid_unambiguous'].all())
    o,m,s=fixture();s[PFX+'tie_mask'][5,3,1]=True;s[PFX+'tie_count'][5,3]=2;b=extract_cues(o,m,s)
    C('tie_preserved_normal_unambiguous_false',b['surface_tie_count'][5,0]==2 and b['geometric_branch_unknown'][5,0] and not b['surface_normal_valid_unambiguous'][5,0])
    o,m,s=fixture();s[PFX+'axis_radial_normal_ambiguity'][6,7]=True;b=extract_cues(o,m,s)
    C('axis_ambiguity_preserved_not_contact_known',b['geometric_branch_unknown'][6,1] and not b['surface_normal_valid_unambiguous'][6,1] and not b['human_contact_known'].any())
    o,m,s=fixture();s[PFX+'outward_normal_o'][8,11]=0;s[PFX+'normal_w'][8,11]=0;b=extract_cues(o,m,s)
    C('invalid_numeric_normal_is_unknown',not b['surface_normal_numeric_valid'][8,2] and not b['surface_normal_valid_unambiguous'][8,2])
    for field in ['human_contact_known','human_contact_present','declared_robot_intent_known']:
        b=copy.deepcopy(a);b[field][0,0]=True;reject('reject_fabricated_'+field,lambda b=b:validate_cues(b))
    b=copy.deepcopy(a);b['human_contact_unknown_mask'][0,0]=False;reject('reject_false_release_unknown_mask',lambda:validate_cues(b))
    o,m,s=fixture();m['mapped_object_pose_w']=m['mapped_object_pose_w'].copy();m['mapped_object_pose_w'][0,0]=.001;reject('reject_wrong_mapping_pose',lambda:extract_cues(o,m,s))
    o,m,s=fixture();o['object_pose_source_w']=o['object_pose_source_w'].copy();o['object_pose_source_w'][0,3]=0;reject('reject_zero_object_quaternion',lambda:extract_cues(o,m,s))
    o,m,s=fixture();o['time_s']=o['time_s'][::-1];reject('reject_noncanonical_time',lambda:extract_cues(o,m,s))
    o,m,s=fixture();o['landmark_confidence_known'][0,3]=True;reject('reject_confidence_fabrication',lambda:extract_cues(o,m,s))
    o,m,s=fixture();s[PFX+'tie_count'][0,3]=2;reject('reject_inconsistent_tie_metadata',lambda:extract_cues(o,m,s))
    o,m,s=fixture();s[PFX+'signed_distance_m'][0,3]=.1;reject('reject_inconsistent_saved_signed_unsigned',lambda:extract_cues(o,m,s))
    reject('objective_loading_always_refused',lambda:require_objective_loading({'target_ready':True,'objective_loading_permitted':True}))
    C('no_Torch_Warp_MuJoCo_import',all(k not in sys.modules for k in ['torch','warp','mujoco']))
    output.mkdir(parents=True,exist_ok=False)
    result={'schema':'full151_observation_cue_CPU_boundary_proof:1','status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_pins':{'module':pin(ROOT/'src/full151/observation_intent_cues.py'),'checker':pin(__file__)},'scope':'Synthetic data/allowlist/unknown mask/branch semantics only; no actual surface query or contact/physics qualification.','solver_FK_collision_WM_Env_physics_GPU_calls':0}
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=run(a.output);print(json.dumps(r,indent=2));sys.exit(0 if r['status']=='PASS' else 1)
