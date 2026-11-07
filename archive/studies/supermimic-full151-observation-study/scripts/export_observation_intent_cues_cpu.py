"""Export frozen saved151x3 observation proximity; never re-evaluate geometry."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from full151.observation_intent_cues import *

OBS=Path('D:/workspace/supermimic-chord-study/outputs/observation_only_fixture_v1/observations.npz')
MAP=Path('D:/workspace/supermimic-observation-retarget-study/outputs/native_contact_refined_geometry_v1/geometric_retarget.npz')
SURF=Path('D:/workspace/supermimic-observation-contact-intent-study/outputs/contact_intent_v2/surface_proximity_151.npz')
SURFREPORT=SURF.parent/'report.json'
EXPECTED={str(OBS):'3bdca0ee6fd8d7956cac1ee2b13719db5cb302ceb0130972719032a04b3ec35b',
          str(MAP):'539d6247576b9b2e3fb1b143cf17b6ad507e3baea73c5fafcda4e221d9230c8f',
          str(SURF):'18d54c569bb8db059c77a2ed6aa05e5265ab2b97da59a9564a5ef1ae2bc64e71',
          str(SURFREPORT):'d207c94a684270ef28dfaef4fa55a44ef3c605d6bd6c45ef960e7f037b07acac'}

def run(output):
    if output.exists():raise FileExistsError('Refuse existing output '+str(output))
    for path,value in EXPECTED.items():
        if sha(path)!=value:raise ValueError('Frozen source changed '+path)
    sm=json.loads(SURFREPORT.read_text());om=json.loads(OBS.with_name('manifest.json').read_text())
    if sm['arrays']['sha256']!=EXPECTED[str(SURF)] or sm['observation']['sha256']!=EXPECTED[str(OBS)]:raise ValueError('Saved surface lineage differs')
    for role in ['geometry_module','code']:
        if sha(sm[role]['path'])!=sm[role]['sha256']:raise ValueError('Saved geometry source changed '+role)
    if om['original_source_frames']!=[200,351] or om['frames']!=151 or om['is_real_ego'] is not False:raise ValueError('Wrong source lineage')
    obs=load_selected(OBS,OBS_FIELDS);mapping=load_selected(MAP,MAP_FIELDS);saved=load_selected(SURF,SAVED_FIELDS)
    arrays=extract_cues(obs,mapping,saved)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'observation_intent_cues.npz',**arrays)
    metadata={'schema':SCHEMA,'source_window_half_open':[200,351],'local_frame_range_half_open':[0,151],'frames':151,'dt_s':.02,
              'finger_ids':FINGERS.tolist(),'landmark_indices':LANDMARKS.tolist(),'target_ready':False,'objective_loading_permitted':False,
              'human_contact_labels_present':False,'human_contact_present_false_semantics':'Storage placeholder under human_contact_unknown_mask=True everywhere; never a negative or release annotation.',
              'declared_robot_intent_labels_present':False,'source_confidence':'All original known masks are false; no numerical confidence values fabricated.',
              'distance_sign_convention':'Signed by closed union membership: negative inside, positive outside; absolute value equals unsigned nearest visible exterior-patch distance. This is not min primitive SDF.',
              'normal_convention':'Outward object BODY/world normal, opposite the force-on-object inward contact-ray convention. Validity/feature/ties are geometric diagnostics, not contact certainty.',
              'normal_validity_rule':'Finite unit normal within1e-6 from pinned original source, valid numeric surface point, exactly one nearest patch and no cylinder-axis radial ambiguity. Ties retain the original first-argmin normal but unambiguous-valid=false.',
              'surface_metadata':sm['surface_geometry'],'frozen_surface_function':sm['geometry_module'],'frozen_surface_producer':sm['code'],
              'saved_matrix_only':True,'surface_function_reexecuted':False,'no_material_anchor_transport_or_repeat16':True,
              'read_fields':{'observations':list(OBS_FIELDS),'mapping_npz':list(MAP_FIELDS),'saved_surface_npz':list(SAVED_FIELDS)},
              'forbidden_inputs_not_read':['joint_pos_rad','robot generated contacts','robot FK points','normalized target points','16-frame target/support/phase','source robot labels','policy contacts'],
              'source_pins':{'observations':pin(OBS),'observation_manifest':pin(OBS.with_name('manifest.json')),'mapping':pin(MAP),'saved_surface':pin(SURF),'saved_surface_report':pin(SURFREPORT),'surface_function':sm['geometry_module'],'surface_producer':sm['code'],'module':pin(ROOT/'src/full151/observation_intent_cues.py'),'entry':pin(__file__)},
              'arrays':pin(output/'observation_intent_cues.npz'),'physical_reference_qualified':False,'is_real_ego':False,'metric_scale_and_camera_qualified':False}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')
    result={'schema':'full151_observation_intent_cues_export_receipt:1','status':'EXPORTED_OBSERVATION_CUES_ONLY_NOT_CONTACT_TARGETS','frames':151,'cues':453,'target_ready':False,'objective_loading_permitted':False,
            'metadata':pin(output/'metadata.json'),'arrays':pin(output/'observation_intent_cues.npz'),'summary':descriptive_summary(arrays),
            'geometric_branch_unknown_count':int(arrays['geometric_branch_unknown'].sum()),'human_contact_known_count':0,'declared_robot_intent_known_count':0,
            'source_confidence_known_count':0,'new_geometry_query_FK_solver_Env_WM_physics_GPU_calls':0,'imports_no_Torch_Warp_MuJoCo':all(name not in sys.modules for name in ['torch','warp','mujoco'])}
    (output/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8');return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.output.resolve()),indent=2))
