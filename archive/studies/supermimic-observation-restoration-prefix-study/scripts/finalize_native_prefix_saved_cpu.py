"""Finalize the saved native16 ABI proof after an original NumPy-bool JSON failure.

No reader, FK, native query, optimization, physics, Torch, Warp or child process
is executed here. The original failed source/arrays remain unchanged.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def rotation(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def finalize(source,output):
    start=time.perf_counter();source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'finalizer_source.py').write_bytes(Path(__file__).read_bytes());checks=[]
    def check(name,ok,detail=None):checks.append({'name':name,'passed':bool(ok),**({'detail':detail} if detail is not None else {})})
    manifest=read(source/'native_catalog_manifest.json');frozen=read(source/'frozen_inputs.json');initial_manifest=read(frozen['init_manifest']['path']);reference_manifest=read(frozen['reference_manifest']['path']);native=arrays(manifest['native']['path']);reference=arrays(reference_manifest['artifact']['path']);initial=arrays(initial_manifest['artifact']['path']);loaded=arrays(source/'native_reader_loaded_CPU.npz');vel=arrays(source/'velocity_semantics.npz');provenance=read(manifest['native_provenance']['path']);future=read(source/'future_reference_adapter_input.json');recipe=read(manifest['recipe']['path']);index=read(Path(manifest['native']['path']).parent/'index.json')
    inputs={**frozen,'native_catalog_manifest':pin(source/'native_catalog_manifest.json'),'native':manifest['native'],'native_provenance':manifest['native_provenance'],'recipe':manifest['recipe'],'index':pin(Path(manifest['native']['path']).parent/'index.json'),'saved_actual_native_reader_arrays':pin(source/'native_reader_loaded_CPU.npz'),'saved_velocity_semantics':pin(source/'velocity_semantics.npz'),'preserved_analytic_init':pin(source/'analytic_init_requested_original.npz'),'future_reference_adapter_input':pin(source/'future_reference_adapter_input.json')}
    check('all_original_source_input_pins_current',all(sha(p['path'])==p['sha256'] for p in frozen.values()))
    check('adapter_source_snapshot_exact',sha(source/'artifact_adapter_source.py')==frozen['adapter']['sha256'])
    check('original_terminal_receipt_missing_harness_failure_retained',not (source/'report.json').exists())
    check('native_four_key_ABI',set(native)=={'reference_qpos','wrist_pose_w','object_pose_w','link_pos_w'} and native['reference_qpos'].shape==(16,20) and native['link_pos_w'].shape==(16,21,3))
    check('reference_q16_and_mappedpose_preserved64',np.array_equal(native['reference_qpos'],reference['q_goal_rad']) and np.array_equal(native['object_pose_w'],reference['mapped_object_pose_w']))
    check('explicit_legacy_window_variant_distinct_from_shared',manifest['source_window_half_open']==[200,216] and manifest['native_local_range_half_open']==[0,16] and manifest['retarget_variant_id']=='H123_cubic_minimax_restoration_filtered_prefix16_v1' and manifest['shared_registry_enrollment'] is None and manifest['shared_001_to010_track_exposure']==[])
    check('physical_and_training_authority_false',manifest['physical_reference_qualified'] is False and manifest['training_authorized'] is False and future['physical_reference_qualified'] is False and future['training_authorized'] is False)
    check('native_16_states_15_intervals_point3seconds',loaded['trajectory_lengths'].tolist()==[16] and manifest['frames']==16 and manifest['control_intervals']==15 and manifest['duration_s']==.3 and manifest['frame_dt_s']==.02 and future['reference']['duration_s']==.3)
    sid=manifest['source_id'];check('real_recipe_index_select_only_new_native_file',recipe=={'schema':'wuji_pen_spin_recipe:1','clips':[sid]} and index['frame_dt']==.02 and index['clip_count']==1 and index['group_id']==sid.split('/')[0] and index['clips']==[{'clip_id':sid.split('/')[1],'file':Path(manifest['native']['path']).name,'sha256':manifest['native']['sha256']}])
    q32=native['reference_qpos'].astype(np.float32);fd=np.zeros_like(q32);fd[:-1]=(q32[1:]-q32[:-1])/np.float32(.02);fd[-1]=fd[-2]
    check('saved_real_reader_goal32_exact',np.array_equal(loaded['joint_pos'],q32))
    check('saved_real_reader_original_forwardFD32_exact',np.array_equal(loaded['joint_vel'],fd) and np.array_equal(vel['native_forward_FD32_rad_s'],fd))
    wrist=native['wrist_pose_w'];Rw=rotation(wrist[:,3:]);expectedp=np.einsum('tji,tj->ti',Rw,native['object_pose_w'][:,:3]-wrist[:,:3]);expectedR=np.swapaxes(Rw,-1,-2)@rotation(native['object_pose_w'][:,3:]);expectedlinks=np.einsum('tji,tkj->tki',Rw,native['link_pos_w']-wrist[:,None,:3])
    errors={'q_goal_rad':float(np.max(np.abs(loaded['joint_pos']-q32))),'native_FD_rad_s':float(np.max(np.abs(loaded['joint_vel']-fd))),'object_wristlocal_position_m':float(np.max(np.abs(loaded['object_pose'][:,:3]-expectedp))),'object_wristlocal_rotation':float(np.max(np.abs(rotation(loaded['object_pose'][:,3:])-expectedR))),'BODY21_wristlocal_m':float(np.max(np.abs(loaded['link_pos']-expectedlinks)))}
    check('saved_reader_relative_object_transform',errors['object_wristlocal_position_m']<=1e-6 and errors['object_wristlocal_rotation']<=1e-6)
    check('saved_reader_BODY21_relative_transform',errors['BODY21_wristlocal_m']<=1e-6)
    expected_names=['right_palm_link']+[f'right_finger{f}_link{j}' for f in range(1,6) for j in range(1,5)]
    check('original_native_exporter_BODY21_order_provenance',provenance['native_body_link_names']==expected_names and provenance['exporter_sha256']==frozen['native_exporter']['sha256'] and provenance['link_source'].startswith('Fresh BODY FK') and provenance['kinematics_calls']==16)
    check('fixed_default_wrist_saved_reader',np.max(np.abs(wrist-wrist[:1]))<=1e-12 and np.max(np.abs(loaded['wrist_pose_w']-wrist.astype(np.float32)))<=1e-6)
    check('analytic_initialization_original_bytes_preserved',sha(source/'analytic_init_requested_original.npz')==initial_manifest['artifact']['sha256'])
    check('analytic_initial_qd_and_object_twists_separate_unchanged',np.array_equal(initial['qd_init_requested_FP32_rad_s'],reference['analytic_qd16_rad_s'][0].astype(np.float32)) and np.array_equal(vel['analytic_curve_qd16_rad_s'],reference['analytic_qd16_rad_s']) and np.array_equal(vel['qd_init_requested_FP32_rad_s'],initial['qd_init_requested_FP32_rad_s']) and np.array_equal(vel['object_bodyorigin_v_init_requested_FP32_world_m_s'],initial['object_bodyorigin_v_requested_FP32_world_m_s']) and np.array_equal(vel['object_angular_init_requested_FP32_world_rad_s'],initial['object_angular_requested_FP32_world_rad_s']) and np.array_equal(vel['object_angular_init_requested_FP32_body_rad_s'],initial['object_angular_requested_FP32_body_rad_s']))
    delta=initial['qd_init_requested_FP32_rad_s']-fd[0];check('nativeFD_init_is_demonstrably_not_analyticqd0',np.any(delta!=0))
    check('all_saved_arrays_finite',all(np.isfinite(a).all() for d in (native,loaded,vel) for a in d.values() if a.dtype.kind in 'fiu'))
    check('finalizer_no_Torch_Warp_MuJoCo_import',not any(k in sys.modules for k in ('torch','warp','mujoco_warp','mujoco')))
    report={'schema':'saved_native_prefix16_ABI_finalization_review:1','status':'PASS_SAVED_NATIVE_READER_ABI_ORIGINAL_TERMINAL_RECEIPT_FAILED' if all(c['passed'] for c in checks) else 'FAIL_SAVED_ABI_RETAINED','check_count':len(checks),'passed_checks':sum(c['passed'] for c in checks),'checks':checks,'errors':errors,'input_pins':inputs,'finalizer':pin(__file__),'actual_reader_chain':{'real_native_reader_configured_device':'cpu','reader_source':frozen['native_reader'],'recipe_resolver_source':frozen['native_recipe_source'],'evidence':'Original saved adapter source invokes real resolve_recipe + SingleHandTrackingReader before writing native_reader_loaded_CPU.npz; all persisted reader outputs are independently checked here. Original last report write failed afterward.','native_loaded_joint_pos_shape':list(loaded['joint_pos'].shape),'native_loaded_BODY_links_shape':list(loaded['link_pos'].shape),'original_terminal_CUDA_initialized_flag':'UNKNOWN_NOT_SAVED; no claim from a missing report','original_torch_tensor_devices':'CPU constructor source-bound; original runtime device boolean was not saved'},'velocity_difference':{'initial_nativeFD_minus_analytic_max_abs_rad_s':float(np.max(np.abs(delta))),'analytic_initial_qd32_rad_s':initial['qd_init_requested_FP32_rad_s'].tolist(),'native_forwardFD_initial32_rad_s':fd[0].tolist(),'semantics':'Native forwardFD is a reader goal field; reset still must use explicit analytic initial qd/worldtwist and actual readback, not silently use this FD.'},'original_terminal_harness_failure':{'kind':'TypeError','message':'Object of type bool is not JSON serializable (NumPy bool scalar in checks)','location':'Original adapter final report.json write after all artifacts/reader arrays were saved','report_created':False,'source_and_outputs_preserved':True,'original_tool_wall_seconds':4.8849195,'tool_wall_scope':'Observed exec process wall, not measured GPU/CPU kernel time'},'training_authorized':False,'physical_reference_qualified':False,'shared_registry_enrolled':False,'FK_calls_this_finalizer':0,'reader_calls_this_finalizer':0,'solver_calls_this_finalizer':0,'physics_steps_this_finalizer':0,'GPU_used':False,'wall_seconds':time.perf_counter()-start,'scope':'Saved-only recovery of the data ABI proof, not an original clean process PASS, not runtime reset/controller/training qualification.'}
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');return report


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source',type=Path,default=ROOT/'outputs/restored_prefix_native_ABI_v1');ap.add_argument('--output',type=Path,default=ROOT/'outputs/finalization_review_v1');a=ap.parse_args();r=finalize(a.source,a.output);print(json.dumps({k:r[k] for k in ('status','check_count','passed_checks','errors','velocity_difference')},indent=2));raise SystemExit(r['passed_checks']!=r['check_count'])
if __name__=='__main__':main()
