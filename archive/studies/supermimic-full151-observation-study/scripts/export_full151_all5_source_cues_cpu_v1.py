"""Frozen saved151x5 cue extraction and contracts; no new geometry or target."""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from full151.observation_intent_cues_all5 import *

PROPOSAL=ROOT/'proposals/full151_declared_intent_dynamic_mask_and_all5_v2/next_all5_actual_material_302LP_proposal.json'
OLD=ROOT/'outputs/observation_intent_cues_v1'
EXPECTED={'observations':'3bdca0ee6fd8d7956cac1ee2b13719db5cb302ceb0130972719032a04b3ec35b',
 'mapping':'539d6247576b9b2e3fb1b143cf17b6ad507e3baea73c5fafcda4e221d9230c8f',
 'saved_all21_surface_matrix':'18d54c569bb8db059c77a2ed6aa05e5265ab2b97da59a9564a5ef1ae2bc64e71',
 'surface_metadata':'5fc44a9947a277098f85c644eb44ddc5ce3427899e984f40827fe96dbb46a84d',
 'input_bundle':'59406c6214a085afba7efcc2b33880d5c82b409ee4f6485ec71c073fec19dc76'}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def checked(v):
 p=Path(v['path']).resolve(strict=True)
 if set(v)!={'path','sha256'} or sha(p)!=v['sha256']:raise ValueError('Changed frozen file '+str(p))
 return p


class Guard(dict):
    def __init__(self,data,allowed):super().__init__(data);self.allowed=set(allowed);self.access=[]
    def __getitem__(self,key):
        if key not in self.allowed:raise AssertionError('Forbidden field read '+key)
        self.access.append(key);return super().__getitem__(key)


def execute(output):
    started=time.perf_counter();checks={};errors={}
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':raise ValueError('Explicit CPU mask -1 required')
    if output.exists():raise FileExistsError('Keep all prior attempts')
    output.mkdir(parents=True,exist_ok=False);snapshot=output/'source_before';snapshot.mkdir()
    source={'producer':pin(__file__),'module':pin(ROOT/'src/full151/observation_intent_cues_all5.py'),'parent_proposal':pin(PROPOSAL)}
    for i,v in enumerate(source.values()):(snapshot/f'{i:02d}_{Path(v["path"]).name}').write_bytes(Path(v['path']).read_bytes())
    def check(name,condition):
        if time.perf_counter()-started>30:raise TimeoutError('CPU saved extraction deadline30s; preserve partial')
        try:assert bool(condition() if callable(condition) else condition);checks[name]=True
        except BaseException:checks[name]=False;errors[name]=traceback.format_exc()
    def rejects(name,fn):
        try:fn()
        except (ValueError,AssertionError):checks[name]=True
        else:checks[name]=False;errors[name]='Expected rejection did not occur'
    proposal=read(PROPOSAL)
    if proposal['schema']!='full151_all5_actual_material_cone_CPU_proposal:1':raise ValueError('Wrong fixed source proposal')
    inputs=proposal['source_inputs']
    if set(inputs)!=set(EXPECTED) or any(v['sha256']!=EXPECTED[k] for k,v in inputs.items()):raise ValueError('The fixed151 source inputs changed')
    for value in inputs.values():checked(value)
    oldmeta=read(checked(inputs['surface_metadata']));oldpin=oldmeta['arrays'];checked(oldpin)
    if oldpin['sha256']!='a700c7c0ac6d04694830d7c5508a46a382e1b2923a052025217a06d7956232db':raise ValueError('Wrong old H123 cue provenance')
    smpin=oldmeta['source_pins']['saved_surface_report'];sm=read(checked(smpin))
    ompin=oldmeta['source_pins']['observation_manifest'];om=read(checked(ompin))
    if sm['arrays']['sha256']!=EXPECTED['saved_all21_surface_matrix'] or sm['observation']['sha256']!=EXPECTED['observations'] or om['original_source_frames']!=[200,351] or om['frames']!=151 or om['is_real_ego'] is not False:raise ValueError('Old source/frame lineage changed')
    for role in ('geometry_module','code'):checked(sm[role])
    bundle=read(checked(inputs['input_bundle']))
    if bundle['raw_source_identity']['primary_pose_source_frame_range_half_open']!=[200,351] or bundle['raw_source_identity']['local_reference_range_half_open']!=[0,151] or bundle['physical_reference_qualified'] is not False:raise ValueError('Shared007[0,151) must not masquerade as legacy[200,351)')
    obs=load_selected(checked(inputs['observations']),OBS_FIELDS)
    mapping=load_selected(checked(inputs['mapping']),MAP_FIELDS)
    saved=load_selected(checked(inputs['saved_all21_surface_matrix']),SAVED_FIELDS)
    go,gm,gs=Guard(obs,OBS_FIELDS),Guard(mapping,MAP_FIELDS),Guard(saved,SAVED_FIELDS)
    arrays=extract_all5_cues(go,gm,gs)
    check('actual151x5_same_source_landmarks',arrays['source_landmark_source_w'].shape==(151,5,3) and np.array_equal(arrays['landmark_indices'],[3,7,11,15,19]))
    check('actual_reads_explicit_allowlists',set(go.access)<=set(OBS_FIELDS) and set(gm.access)<=set(MAP_FIELDS) and set(gs.access)<=set(SAVED_FIELDS))
    legacy_keys=tuple(k for k in arrays if k not in ('geometry_valid_mask','declared_potential_mask'))
    old=load_selected(checked(oldpin),legacy_keys);compat={}
    for key,value in old.items():
        new=arrays[key]
        if key in ('finger_ids','landmark_indices'):new=new[:3]
        elif new.shape[:2]==(151,5):new=new[:,:3]
        compat[key]=bool(new.dtype==value.dtype and np.array_equal(new,value,equal_nan=True))
    check('H123_first3_every_legacy_field_dtype_bitexact',all(compat.values()))
    check('geometry_and_declared_potential_masks_separate',np.array_equal(arrays['geometry_valid_mask'],arrays['surface_point_numeric_valid']&arrays['surface_normal_valid_unambiguous']&~arrays['geometric_branch_unknown']) and arrays['declared_potential_mask'].all())
    check('human_contact_confidence_and_robotintent_stay_unknown',not arrays['human_contact_known'].any() and not arrays['human_contact_present'].any() and arrays['human_contact_unknown_mask'].all() and not arrays['landmark_confidence_known'].any() and not arrays['object_confidence_known'].any() and not arrays['declared_robot_intent_known'].any())
    # Negative boundary fixtures rerun only the saved-array algebra. They never
    # invoke a surface oracle, and never overwrite the actual extracted array.
    contaminated=Guard({**mapping,'joint_pos_rad':object(),'generated_contact_mask':object()},MAP_FIELDS)
    extracted=extract_all5_cues(Guard(obs,OBS_FIELDS),contaminated,Guard({**saved,'robot_candidate_FK__points_w':object()},SAVED_FIELDS))
    check('forbidden_q_FK_contact_fields_not_accessed',set(contaminated.access)<=set(MAP_FIELDS) and np.array_equal(extracted['object_surface_closest_body_m'],arrays['object_surface_closest_body_m']))
    wrong=copy.deepcopy(obs);wrong['time_s']=wrong['time_s'][::-1]
    rejects('reject_wrong_time',lambda:extract_all5_cues(wrong,mapping,saved))
    wrong=copy.deepcopy(mapping);wrong['mapped_object_pose_w'][0,0]+=.001
    rejects('reject_wrong_object_transform',lambda:extract_all5_cues(obs,wrong,saved))
    wrong=copy.deepcopy(obs);wrong['landmark_confidence_known'][0,19]=True
    rejects('reject_confidence_fabrication_on_finger5',lambda:extract_all5_cues(wrong,mapping,saved))
    wrong=copy.deepcopy(obs);wrong['object_pose_source_w'][0,3:]=0
    rejects('reject_invalid_source_quaternion',lambda:extract_all5_cues(wrong,mapping,saved))
    wrong=copy.deepcopy(saved);wrong[PFX+'tie_count'][0,19]+=1
    rejects('reject_inconsistent_finger5_tie_metadata',lambda:extract_all5_cues(obs,mapping,wrong))
    wrong=copy.deepcopy(saved);ties=wrong[PFX+'tie_mask'][0,15];ties[:]=False;ties[:2]=True;wrong[PFX+'tie_count'][0,15]=2
    tied=extract_all5_cues(obs,mapping,wrong)
    check('tied_finger4_geometry_unknown_potential_not_deleted',tied['geometric_branch_unknown'][0,3] and not tied['geometry_valid_mask'][0,3] and tied['declared_potential_mask'][0,3] and not tied['human_contact_known'].any())
    wrong=copy.deepcopy(saved);wrong[PFX+'axis_radial_normal_ambiguity'][1,19]=True
    ambiguous=extract_all5_cues(obs,mapping,wrong)
    check('finger5_axis_ambiguity_unknown_no_zero_imputation',not ambiguous['geometry_valid_mask'][1,4] and ambiguous['declared_potential_mask'][1,4] and np.array_equal(ambiguous['object_surface_closest_body_m'][1,4],arrays['object_surface_closest_body_m'][1,4]))
    wrong=copy.deepcopy(arrays);wrong['human_contact_known'][0,4]=True
    rejects('reject_human_truth_injection',lambda:validate_cues(wrong))
    wrong=copy.deepcopy(arrays);wrong['declared_potential_mask'][0,4]=False
    rejects('reject_silent_hypothesis_membership_change',lambda:validate_potential_masks(wrong))
    rejects('fivecue_never_objective_even_with_true_flags',lambda:require_objective_loading({'target_ready':True,'objective_loading_permitted':True}))
    pins={**source,**{'input__'+k:v for k,v in inputs.items()},'old_H123_cues':oldpin,'old_surface_report':smpin,'observation_manifest':ompin,'saved_surface_geometry_source':sm['geometry_module'],'saved_surface_producer':sm['code']}
    check('all_inputs_sources_unchanged',all(sha(v['path'])==v['sha256'] for v in pins.values()))
    check('no_Torch_Warp_MuJoCo_or_runtime_import',not any(k in sys.modules for k in ('torch','warp','mujoco','mujoco_warp')))
    np.savez_compressed(output/'observation_potential_five_cues.npz',**arrays)
    metadata={'schema':SCHEMA,'status':'SOURCE_CUES_ONLY_NOT_CONTACT_TARGETS','source_window_half_open':[200,351],'local_frame_range_half_open':[0,151],'frames':151,'slots':5,'finger_ids':FINGERS.tolist(),'landmark_indices':LANDMARKS.tolist(),
      'contact_point_frame':'object_BODY','point_COM_shift_applied':False,'normal_frame':'object_BODY','normal_convention':'outward unit normal where geometry_valid_mask=True; LP consumer flips sign for inward once','geometry_valid_rule':'finite numeric surface point AND unit normal AND unique unambiguous patch/axis; no proximity/contact threshold',
      'declared_potential_mask_rule':'allTrue persistent potential H12345 experiment membership; not observed contact/force or robot-intent truth','missing_geometry':'retain unknown and raw point/normal/ties; never impute zero or remove potential member. Infeasible partial cone cannot certify the full declared hypothesis.',
      'human_contact_labels_present':False,'human_contact_unknown_everywhere':True,'source_confidence_known':False,'declared_robot_intent_labels_present':False,'human_contact_present_false_semantics':'Placeholder under allTrue unknown mask, not negative/release labels',
      'target_ready':False,'objective_loading_permitted':False,'training_authorized':False,'physical_reference_qualified':False,'is_real_ego':False,'saved_matrix_only':True,'surface_function_reexecuted':False,'new_surface_queries':0,'new_FK_physics_Env_GPU':False,
      'surface_metadata':sm['surface_geometry'],'read_fields':{'observations':list(OBS_FIELDS),'mapping':list(MAP_FIELDS),'saved_surface':list(SAVED_FIELDS),'old_H123_cues':list(legacy_keys),'input_bundle_JSON':'identity/frame/window flags only'},'forbidden_fields_not_read':['robotq','original robotproprioception','generatedrobotcontacts','policycontacts/forces/quality','robotFKpoints','normalizedtargets'],
      'H123_first3_every_field_bitexact':compat,'source_pins':pins,'arrays':pin(output/'observation_potential_five_cues.npz'),'summary':descriptive_summary(arrays)}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    report={'schema':'full151_all5_source_cue_CPU_receipt:1','status':'PASS_SAVED_SOURCE_CUES_ONLY' if all(checks.values()) else 'CPU_CONTRACT_FAILURE_PRESERVED','passed':sum(checks.values()),'total':len(checks),'checks':checks,'errors':errors,'arrays':metadata['arrays'],'metadata':pin(output/'metadata.json'),'source_pins':pins,
      'actual_read_fields':{'observations':sorted(set(go.access)),'mapping':sorted(set(gm.access)),'saved_surface':sorted(set(gs.access))},'geometrically_valid_slots_per_finger':arrays['geometry_valid_mask'].sum(0).tolist(),'human_known_slots':0,'target_ready':False,'objective_loading_permitted':False,'LP_calls':0,'new_geometry_query_FK_Env_WM_physics_GPU_calls':0,'wall_s':time.perf_counter()-started,'pid':os.getpid(),'OS_CreationDate':'NOT_SAMPLED','scope':'One NumPy saved-matrix extraction and explicit boundary fixtures; not302LP or A2/supporttarget/force/contact/physicaltraining qualification.'}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'outputs/full151_all5_source_cues_v1');args=ap.parse_args()
    r=execute(args.output.resolve());print(json.dumps({'report':pin(args.output/'report.json'),'passed':r['passed'],'total':r['total'],'status':r['status']}));raise SystemExit(0 if all(r['checks'].values()) else 2)
