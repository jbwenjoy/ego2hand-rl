"""Bind existing qualified CPU arrays for a distinct rough-reference pilot."""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rough_prefix.io import read,write,pin,checked,loadnp
PARENT=Path('D:/workspace/supermimic-observation-restoration-prefix-study')
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();started=time.perf_counter()
    export=PARENT/'outputs/restored_prefix_control_export_v1';native_root=PARENT/'outputs/restored_prefix_native_ABI_v1'
    manifests={k:read(export/(k+'_manifest.json')) for k in ('reference','init','control')};native=read(native_root/'native_catalog_manifest.json');receipt=read(export/'receipt.json')
    if not all(v is True for v in receipt['gates'].values()) or any(m['source_pins']!=receipt['source_pins'] for m in manifests.values()):raise ValueError('CPU numeric/source prerequisites failed')
    pins={k+'_manifest':pin(export/(k+'_manifest.json')) for k in manifests}
    pins.update({k+'_arrays':m['artifact'] for k,m in manifests.items()})
    pins.update(native_manifest=pin(native_root/'native_catalog_manifest.json'),native_adapter_input=pin(native_root/'future_reference_adapter_input.json'),native_data=native['native'],native_recipe=native['recipe'],native_provenance=native['native_provenance'],export_receipt=pin(export/'receipt.json'),native_finalizer=pin(PARENT/'outputs/finalization_review_v1/report.json'),native_independent=pin(PARENT/'outputs/rough_prefix_native_ABI_independent_review_v2/report.json'),builder=pin(__file__))
    pins.update({'export_source__'+k:v for k,v in receipt['source_pins'].items()})
    for v in pins.values():checked(v)
    ref,init,control=(loadnp(pins[k+'_arrays']['path']) for k in ('reference','init','control'));n=loadnp(native['native']['path'])
    if native['source_window_half_open']!=[200,216] or native['frames']!=16 or native['physical_reference_qualified'] is not False:raise ValueError('Wrong enrollment/qualification')
    if not np.array_equal(ref['q_goal_rad'],n['reference_qpos']) or not np.array_equal(ref['mapped_object_pose_w'],n['object_pose_w']) or not np.array_equal(n['wrist_pose_w'],np.broadcast_to(n['wrist_pose_w'][:1],(16,7))):raise ValueError('Native/reference/palm binding changed')
    proto=read(manifests['reference']['protocol']['path']);inputs=proto['inputs']
    import mujoco
    model=mujoco.MjModel.from_binary_path(inputs['model']['path']);body=model.body('object/cube').id
    geoms=np.flatnonzero(model.geom_bodyid==body);radius=max(float(np.linalg.norm(model.geom_pos[g]-model.body_ipos[body])+model.geom_rbound[g]) for g in geoms)
    context=loadnp(inputs['actual_context']['path']);mass=float(context['body_mass'][0,body]);inertia=context['body_inertia'][0,body];iquat=context['body_iquat'][0,body]
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);(out/'builder_source.py').write_bytes(Path(__file__).read_bytes())
    contact=out/'contact_reference.npz';np.savez_compressed(contact,time_s=ref['time_s'],joint_pos_rad=ref['q_goal_rad'],object_pose_w=ref['mapped_object_pose_w'],wrist_pose_w=n['wrist_pose_w'])
    meta={'schema':'rough_prefix_derived_kinematic_reference:1','retarget_variant_id':native['retarget_variant_id'],'raw_source_identity':native['raw_source_identity'],'source_window_half_open':[200,216],'joint_names':[f'right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)],'object_radius_m':radius,'object_com_offset_o_m':model.body_ipos[body],'actual_object_mass_kg':mass,'actual_object_principal_inertia_kg_m2':inertia,'actual_body_iquat_wxyz':iquat,'wrist_frame':'robot/right_palm_link world pose from existing native BODY FK; not mount/mocap','units':'SI','quaternion_order':'wxyz','world_frame':'fixed_hand2_mount_world','source_kind':'observation_derived_robot_kinematic_intent','source_robot_labels_used':False,'human_contact_observation_known':False,'physical_reference_qualified':False,'training_authorized':False,'source_pins':pins}
    write(contact.with_suffix('.json'),meta)
    pins.update(contact_reference=pin(contact),contact_reference_metadata=pin(contact.with_suffix('.json')))
    spec={'schema':'rough_prefix_input_bundle:1','retarget_variant_id':native['retarget_variant_id'],'raw_source_identity':native['raw_source_identity'],'native_reference':{'recipe_path':native['recipe']['path'],'clips_root':native['clips_root'],'duration_s':.3,'provenance_path':native['native_provenance']['path']},'source_pins':pins,'model':inputs['model'],'context':inputs['actual_context'],'native_process_terminal':'ORIGINAL_FINAL_JSON_RECEIPT_FAILED_SAVED_DATA_INDEPENDENTLY_VERIFIED','original_terminal_CUDA_flag':'UNKNOWN','physical_reference_qualified':False,'shared10_enrollment':False,'training_authorized':False,'runtime_implementation_ready':False}
    write(out/'bundle.json',spec)
    write(out/'receipt.json',{'status':'INPUT_DATA_AND_CONTACT_REFERENCE_BRIDGE_BOUND_CPU_ONLY','bundle':pin(out/'bundle.json'),'source_pins_unchanged':all(checked(v) for v in pins.values()),'native_queries':0,'physics_steps':0,'GPU_used':False,'wall_seconds':time.perf_counter()-started})
    print(str(out/'bundle.json'))
if __name__=='__main__':main()
