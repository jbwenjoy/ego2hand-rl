"""Independent saved-only reader-ABI audit: no FK, reader, model or physics import."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
R=Path(__file__).resolve().parents[1]
source=R/'outputs/restored_prefix_native_ABI_v1'
out=R/'outputs/rough_prefix_native_ABI_independent_review_v2';out.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def Rmat(q):
    q=np.array(q,dtype=float,copy=True);q/=np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]]).transpose(2,0,1)
f=read(R/'outputs/finalization_review_v1/report.json');frozen=read(source/'frozen_inputs.json');m=read(source/'native_catalog_manifest.json');n=load(m['native']['path']);loaded=load(source/'native_reader_loaded_CPU.npz');v=load(source/'velocity_semantics.npz')
refm=read(frozen['reference_manifest']['path']);im=read(frozen['init_manifest']['path']);ref=load(refm['artifact']['path']);init=load(im['artifact']['path']);provenance=read(m['native_provenance']['path']);recipe=read(m['recipe']['path']);index=read(Path(m['native']['path']).parent/'index.json')
q32=n['reference_qpos'].astype(np.float32);fd=np.vstack(((q32[1:]-q32[:-1])/np.float32(.02),(q32[-1:]-q32[-2:-1])/np.float32(.02)))
wr=Rmat(n['wrist_pose_w'][:,3:]);expected_p=(np.swapaxes(wr,-1,-2)@(n['object_pose_w'][:,:3]-n['wrist_pose_w'][:,:3])[...,None])[...,0]
expected_r=np.swapaxes(wr,-1,-2)@Rmat(n['object_pose_w'][:,3:]);expected_link=(np.swapaxes(wr,-1,-2)[:,None]@(n['link_pos_w']-n['wrist_pose_w'][:,None,:3])[...,None])[...,0]
errors={'q32':float(np.max(np.abs(loaded['joint_pos']-q32))),'nativeFD32':float(np.max(np.abs(loaded['joint_vel']-fd))),'object_relative_position_m':float(np.max(np.abs(loaded['object_pose'][:,:3]-expected_p))),'object_relative_R':float(np.max(np.abs(Rmat(loaded['object_pose'][:,3:])-expected_r))),'BODY21_relative_position_m':float(np.max(np.abs(loaded['link_pos']-expected_link))),'nativeFD0_minus_analytic_init_max_rad_s':float(np.max(np.abs(fd[0]-init['qd_init_requested_FP32_rad_s'])))}
checks={
 'all_saved_finalization_pins_current':all(sha(p['path'])==p['sha256'] for p in f['input_pins'].values()),
 'frozen_original_adapter_source_snapshot_matches':sha(source/'artifact_adapter_source.py')==frozen['adapter']['sha256']==sha(frozen['adapter']['path']),
 'native_exactly4_keys_and_shapes':set(n)=={'reference_qpos','wrist_pose_w','object_pose_w','link_pos_w'} and n['reference_qpos'].shape==(16,20) and n['link_pos_w'].shape==(16,21,3),
 'derived_q_and_object64_preserved':np.array_equal(n['reference_qpos'],ref['q_goal_rad']) and np.array_equal(n['object_pose_w'],ref['mapped_object_pose_w']),
 'saved_realreader_q32_and_perclipFD32_exact':errors['q32']==errors['nativeFD32']==0 and np.array_equal(v['native_forward_FD32_rad_s'],fd),
 'saved_world_to_wrist_relative_fields_match':errors['object_relative_position_m']<=1e-6 and errors['object_relative_R']<=1e-6 and errors['BODY21_relative_position_m']<=1e-6,
 'nativeBODY21_provenance_distinct_from_macro_points':provenance['native_body_link_names']==['right_palm_link']+[f'right_finger{f}_link{j}' for f in range(1,6) for j in range(1,5)] and provenance['exporter_sha256']==frozen['native_exporter']['sha256'] and provenance['link_source'].startswith('Fresh BODY FK'),
 'new_recipe_catalog_binding':recipe['clips']==[m['source_id']] and index['clip_count']==1 and index['clips'][0]['sha256']==m['native']['sha256']==sha(m['native']['path']),
 'legacy_window_not_shared_and16knots15declaredintervals':m['source_window_half_open']==[200,216] and m['native_local_range_half_open']==[0,16] and m['frames']==16 and loaded['trajectory_lengths'].tolist()==[16] and m['duration_s']==.3 and m['control_intervals']==15 and m['frame_dt_s']==.02 and m['shared_registry_enrollment'] is None,
 'fixed_wrist_saved_values':np.array_equal(n['wrist_pose_w'],np.broadcast_to(n['wrist_pose_w'][:1],n['wrist_pose_w'].shape)),
 'analytic_init_bytes_not_overwritten_by_nativeFD':sha(source/'analytic_init_requested_original.npz')==im['artifact']['sha256'] and np.array_equal(v['qd_init_requested_FP32_rad_s'],init['qd_init_requested_FP32_rad_s']) and np.array_equal(init['qd_init_requested_FP32_rad_s'],ref['analytic_qd16_rad_s'][0].astype(np.float32)) and errors['nativeFD0_minus_analytic_init_max_rad_s']>0,
 'world_and_body_twists_preserved_separately':all(np.array_equal(v[a],init[b]) for a,b in [('object_bodyorigin_v_init_requested_FP32_world_m_s','object_bodyorigin_v_requested_FP32_world_m_s'),('object_angular_init_requested_FP32_world_rad_s','object_angular_requested_FP32_world_rad_s'),('object_angular_init_requested_FP32_body_rad_s','object_angular_requested_FP32_body_rad_s')]),
 'not_training_or_physics_qualification':m['physical_reference_qualified'] is False and m['training_authorized'] is False and f['training_authorized'] is False and f['physical_reference_qualified'] is False,
 'original_process_receipt_failure_and_unknown_cuda_preserved':not (source/'report.json').exists() and f['status']=='PASS_SAVED_NATIVE_READER_ABI_ORIGINAL_TERMINAL_RECEIPT_FAILED' and f['actual_reader_chain']['original_terminal_CUDA_initialized_flag'].startswith('UNKNOWN_NOT_SAVED'),
 'no_new_reader_FK_or_native_libraries':not any(k in sys.modules for k in ('torch','warp','mujoco_warp','mujoco')),
}
checks={k:bool(v) for k,v in checks.items()}
report={'status':'PASS_SAVED_DATA_ABI_ORIGINAL_PROCESS_RECEIPT_FAILED' if all(checks.values()) else 'FAIL_SAVED_DATA_ABI','check_count':len(checks),'passed':sum(checks.values()),'checks':checks,'errors':errors,'inputs':{'reviewer':pin(__file__),'finalizer_receipt':pin(R/'outputs/finalization_review_v1/report.json'),'manifest':pin(source/'native_catalog_manifest.json'),'native':m['native'],'reader_arrays':pin(source/'native_reader_loaded_CPU.npz')},'boundary':'Saved reader arrays and immutable original source establish data ABI evidence only. Original adapter terminal JSON serialization failed after saving the artifacts; this is not a clean-process PASS. Its final CUDA initialization state was not persisted, so UNKNOWN remains. This review neither re-runs nor independently measures FK, Reader, CUDA or physics. New native training still requires analytic-init override, baseline15 control,15interval terminal, versioned checkpoint, targetbundle and actual control/readback qualification. Native length16 alone does not enforce15training intervals.','execution':{'reader_calls':0,'FK_calls':0,'geometry_queries':0,'solver_calls':0,'physics_steps':0,'GPU':0,'child_processes':0,'source_or_data_modified':0}}
(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'status':report['status'],'passed':report['passed'],'checks':len(checks),'report_sha256':sha(out/'report.json'),'errors':errors},indent=2));raise SystemExit(not all(checks.values()))
