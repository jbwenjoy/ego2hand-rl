"""Thin artifact adapter: restored prefix16 -> existing native BODY21 CPU reader.

This is not a reader, reset hook, training entry, optimizer or physical loader.
The analytic initialization remains a separate binding from native forwardFD.
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,importlib.util,json,shutil,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
DEX=Path('D:/workspace/dex-supermimic-research')
SHARED=Path('D:/workspace/supermimic-shared-reference-study')
VARIANT='H123_cubic_minimax_restoration_filtered_prefix16_v1'
GROUP='restored_prefix_rough_geometry'
CLIP='legacy_clip007_f0200_n0016_H123_restored_v1'
PINS={
 'reference_manifest':{'path':str(ROOT/'outputs/restored_prefix_control_export_v1/reference_manifest.json'),'sha256':'8e19a415efc502ca10f1a0fe53a81e1fdef858cc819fc38b69ac89344d004311'},
 'init_manifest':{'path':str(ROOT/'outputs/restored_prefix_control_export_v1/init_manifest.json'),'sha256':'2e7322bac27558f4a2a2161e99bdbfeb861572a865aebadebb3eaadc829f95e4'},
 'native_exporter':{'path':str(SHARED/'src/native_reference_abi.py'),'sha256':'8cefd4f1e4ec6939b8201c26670a691c913ab0b5e9e9f87347904e4889fd6e2d'},
 'native_reader':{'path':str(DEX/'src/dex_supermimic/tasks/pen_spin/mdp/commands.py'),'sha256':'e1adb7c183aa28dde49f6dc6e7795ed450567e6c54347bd7f1e0543922fcbcf0'},
 'native_names':{'path':str(DEX/'src/dex_supermimic/tasks/pen_spin/config/wuji_hand2/hand2_constants.py'),'sha256':'2a1a37a0c97f3bcab631006ac6e0dc47a1a63a92c7717ac4c6c712314ba35986'}}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:{'path':str(Path(p).resolve()),'sha256':sha(p)}
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def loadnp(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def rotation(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def validate():
    for p in PINS.values():
        if sha(p['path'])!=p['sha256']:raise ValueError('Pinned input/source changed')
    reference=read(PINS['reference_manifest']['path']);initial=read(PINS['init_manifest']['path'])
    for kind,m in [('reference',reference),('init',initial)]:
        if m['schema']!='restored_prefix_separate_bundles:1' or m['bundle_kind']!=kind or m['retarget_variant_id']!=VARIANT or m['physical_reference_qualified'] is not False:raise ValueError('New unqualified restored bundle required')
        if m['source_identity']['primary_pose_source_frame_range_half_open']!=[200,216] or m['init_measured'] is not None:raise ValueError('Legacy identity/measurement changed')
        for p in [m['artifact'],*m['source_pins'].values()]:
            if sha(p['path'])!=p['sha256']:raise ValueError('Bundle source/artifact drift')
    if reference['source_pins']!=initial['source_pins'] or reference['source_identity']!=initial['source_identity']:raise ValueError('Reference/init are not from the same new variant')
    z=loadnp(reference['artifact']['path']);i=loadnp(initial['artifact']['path'])
    if z['q_goal_rad'].shape!=(16,20) or z['mapped_object_pose_w'].shape!=(16,7) or not np.array_equal(z['time_s'],np.arange(16)*.02):raise ValueError('Only new16-frame50Hz artifact is accepted')
    return reference,initial,z,i


def export(output):
    start=time.perf_counter();ref,init,z,i=validate();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);(output/'artifact_adapter_source.py').write_bytes(Path(__file__).read_bytes())
    sources={**PINS,'adapter':pin(__file__),**{'bundle_source__'+k:p for k,p in ref['source_pins'].items()},'reference_artifact':ref['artifact'],'init_artifact':init['artifact']}
    sources['native_recipe_source']=pin(DEX/'src/dex_supermimic/tasks/pen_spin/tooling/recipe.py');sources['native_clip_store_source']=pin(DEX/'src/dex_supermimic/tasks/pen_spin/tooling/clip_store.py')
    write(output/'frozen_inputs.json',sources)
    import mujoco
    import scipy.optimize as optimize
    blocked=('mj_step','mj_step1','mj_step2','mj_forward','mj_inverse','mj_geomDistance','mj_contactForce','mj_collision')
    forbidden={k:0 for k in blocked};original={k:getattr(mujoco,k) for k in blocked};optimizers={k:0 for k in ('minimize','least_squares','linprog')};opt_original={k:getattr(optimize,k) for k in optimizers};kin_calls=0;original_kinematics=mujoco.mj_kinematics
    def deny(counter,key):
        def fail(*a,**kw):counter[key]+=1;raise AssertionError('Artifact-only adapter forbids '+key)
        return fail
    def kinematics(*a,**kw):
        nonlocal kin_calls
        kin_calls+=1;return original_kinematics(*a,**kw)
    for k in blocked:setattr(mujoco,k,deny(forbidden,k))
    for k in optimizers:setattr(optimize,k,deny(optimizers,k))
    mujoco.mj_kinematics=kinematics
    try:
        spec=importlib.util.spec_from_file_location('frozen_native_ABI_exporter',PINS['native_exporter']['path']);abi=importlib.util.module_from_spec(spec);spec.loader.exec_module(abi)
        model=ref['source_pins']['model'];group=output/'clips'/GROUP
        provenance={'is_real_ego':False,'source_robot_labels_used':False,'source_kind':'observation-derived restored rough prefix; original video/camera/scale/confidence unknown','observation_source':ref['source_pins']['frozen_input__observation_fixture'],'geometric_derivative':ref['source_pins']['candidate'],'retarget_variant_id':VARIANT,'source_identity':ref['source_identity'],'source_plan':ref['source_pins']['source_plan'],'restoration_protocol':ref['protocol'],'input_parent_protocol':ref['input_parent_protocol'],'physical_reference_qualified':False,'training_authorized':False,'enrolled_in_shared_registry':False,'parent_lineage':'Existing legacy controlled observation fixture, not a new independent demo'}
        native,sidecar,info=abi.export_clip(group,CLIP,z['time_s'],z['q_goal_rad'],z['mapped_object_pose_w'],model_path=model['path'],constants_path=PINS['native_names']['path'],provenance=provenance)
        write(group/'index.json',{'schema':'wuji_pen_spin_clips:1','group_id':GROUP,'group_name':'Newlegacy[200,216) restored16 rough goals; not shared[0,151) or physical references','frame_dt':.02,'clip_count':1,'clips':[{'clip_id':CLIP,'file':native.name,'sha256':sha(native)}]})
        recipe=output/'recipe.json';write(recipe,{'schema':'wuji_pen_spin_recipe:1','clips':[GROUP+'/'+CLIP]})
        import torch
        from dex_supermimic.tasks.pen_spin.tooling.recipe import resolve_recipe
        from dex_supermimic.tasks.pen_spin.mdp.commands import SingleHandTrackingReader
        if sha(Path(sys.modules[SingleHandTrackingReader.__module__].__file__))!=PINS['native_reader']['sha256']:raise ValueError('Actual imported native reader differs')
        selection=resolve_recipe(recipe,clips_root=output/'clips',verify_clip_hashes=True);loaded=SingleHandTrackingReader(selection.clip_sets[0],device='cpu').read();n=loadnp(native)
        q32=n['reference_qpos'].astype(np.float32);fd=np.zeros_like(q32);fd[:-1]=(q32[1:]-q32[:-1])/np.float32(.02);fd[-1]=fd[-2]
        wrist=n['wrist_pose_w'];Rw=rotation(wrist[:,3:]);Ro=rotation(n['object_pose_w'][:,3:]);expected_obj_p=np.einsum('tji,tj->ti',Rw,n['object_pose_w'][:,:3]-wrist[:,:3]);expected_obj_R=np.swapaxes(Rw,-1,-2)@Ro;expected_links=np.einsum('tji,tkj->tki',Rw,n['link_pos_w']-wrist[:,None,:3]);native_obj=loaded['object_pose'].numpy()
        errors={'q_goal_FP32_max_abs_rad':float(np.max(np.abs(loaded['joint_pos'].numpy()-q32))),'native_forward_FD_max_abs_rad_s':float(np.max(np.abs(loaded['joint_vel'].numpy()-fd))),'object_wristlocal_position_max_abs_m':float(np.max(np.abs(native_obj[:,:3]-expected_obj_p))),'object_wristlocal_rotation_max_abs':float(np.max(np.abs(rotation(native_obj[:,3:])-expected_obj_R))),'BODY21_wristlocal_max_abs_m':float(np.max(np.abs(loaded['link_pos'].numpy()-expected_links)))}
        analytic=z['analytic_qd16_rad_s'];delta0=analytic[0].astype(np.float32)-fd[0]
        shutil.copy2(init['artifact']['path'],output/'analytic_init_requested_original.npz')
        np.savez_compressed(output/'native_reader_loaded_CPU.npz',**{key:value.cpu().numpy() for key,value in loaded.items() if torch.is_tensor(value)})
        np.savez_compressed(output/'velocity_semantics.npz',time_s=z['time_s'],native_forward_FD32_rad_s=fd,analytic_curve_qd16_rad_s=analytic,qd_init_requested_FP32_rad_s=i['qd_init_requested_FP32_rad_s'],object_bodyorigin_v_init_requested_FP32_world_m_s=i['object_bodyorigin_v_requested_FP32_world_m_s'],object_angular_init_requested_FP32_world_rad_s=i['object_angular_requested_FP32_world_rad_s'],object_angular_init_requested_FP32_body_rad_s=i['object_angular_requested_FP32_body_rad_s'])
        checks={'one_exact_distinct_native_clip':len(selection.clip_sets)==1 and len(selection.clip_sets[0].clips)==1 and selection.clip_sets[0].clips[0].clip_id==GROUP+'/'+CLIP,'actual_native_reader_CPU_executed':True,'real_reader16states15intervals_point3seconds':loaded['trajectory_lengths'].tolist()==[16] and loaded['frame_dt']==.02 and (len(q32)-1)*.02==.3,'native_four_keys_BODY21_only':set(n)=={'reference_qpos','wrist_pose_w','object_pose_w','link_pos_w'} and n['link_pos_w'].shape==(16,21,3),'source64_goal_and_pose_exact':np.array_equal(n['reference_qpos'],z['q_goal_rad']) and np.array_equal(n['object_pose_w'],z['mapped_object_pose_w']),'fixed_default_wrist_from_native_bodyFK':np.max(np.abs(wrist-wrist[:1]))<=1e-12,'q_goal_FP32_exact':errors['q_goal_FP32_max_abs_rad']==0,'native_FD_FP32_exact':errors['native_forward_FD_max_abs_rad_s']==0,'object_relative_transform_parity':errors['object_wristlocal_position_max_abs_m']<=1e-6 and errors['object_wristlocal_rotation_max_abs']<=1e-6,'BODY21_relative_transform_parity':errors['BODY21_wristlocal_max_abs_m']<=1e-6,'new_analytic_qd_initial_preserved':np.array_equal(i['qd_init_requested_FP32_rad_s'],analytic[0].astype(np.float32)),'analytic_init_remains_separate_from_native_FD':bool(np.any(delta0!=0)),'init_original_bytes_unchanged':sha(output/'analytic_init_requested_original.npz')==init['artifact']['sha256'],'only16_FK_calls_no_dynamics_queries':kin_calls==16 and not any(forbidden.values()) and not any(optimizers.values()),'all_native_tensors_CPU':all(not value.is_cuda for value in loaded.values() if torch.is_tensor(value)),'Torch_CUDA_uninitialized':not torch.cuda.is_initialized(),'source_input_hashes_unchanged':all(sha(p['path'])==p['sha256'] for p in sources.values())}
        catalog={'schema':'legacy_rough_prefix_native_adapter:1','retarget_variant_id':VARIANT,'raw_source_identity':ref['source_identity'],'source_window_half_open':[200,216],'native_local_range_half_open':[0,16],'source_id':GROUP+'/'+CLIP,'frame_dt_s':.02,'frames':16,'control_intervals':15,'duration_s':.3,'recipe':pin(recipe),'clips_root':str((output/'clips').resolve()),'native':pin(native),'native_provenance':pin(sidecar),'source_pins':sources,'native_reference_signature':'PENDING_REAL_NEW_TRAINER_PenMDP_SIGNATURE','analytic_init':init['artifact'],'analytic_init_preserved_snapshot':pin(output/'analytic_init_requested_original.npz'),'native_FD_goal_qd_is_not_analytic_init':True,'physical_reference_qualified':False,'training_authorized':False,'training_mode_proposal':'rough_reference_pilot_only_after_separate_runtime_integration_and_authorization','shared_registry_enrollment':None,'shared_001_to010_track_exposure':[],'teacher_or_force_labels_created':False}
        write(output/'native_catalog_manifest.json',catalog)
        write(output/'future_reference_adapter_input.json',{'schema':'rough_prefix_reference_input_proposal:1','status':'DATA_ABI_CPU_ONLY_NOT_TRAINING_READY','reference':{'recipe_path':str(recipe.resolve()),'clips_root':str((output/'clips').resolve()),'duration_s':.3,'provenance_path':str(sidecar.resolve())},'enrollment':pin(output/'native_catalog_manifest.json'),'physical_reference_qualified':False,'training_authorized':False,'required_new_runtime_interfaces':['analytic-init override beforefirstcollect/allpartialresets','u_baseline15 sharedrealimag controller','15interval horizon without extra16thcontrol','replay episode/source/window provenance','new checkpoint identity; no old103 load migration']})
        report={'status':'PASS_NATIVE_PREFIX16_ABI_CPU_ONLY' if all(checks.values()) else 'FAILED_CPU_ABI_RETAINED','checks':checks,'check_count':len(checks),'passed_checks':sum(checks.values()),'errors':errors,'source_pins':sources,'native_catalog':pin(output/'native_catalog_manifest.json'),'native_export_contract':info,'actual_reader':pin(Path(sys.modules[SingleHandTrackingReader.__module__].__file__)),'analytic_vs_native_FD_initial_max_abs_rad_s':float(np.max(np.abs(delta0))),'analytic_qd0_FP32_rad_s':i['qd_init_requested_FP32_rad_s'].tolist(),'native_forward_FD0_FP32_rad_s':fd[0].tolist(),'native_loaded_joint_pos_shape':list(loaded['joint_pos'].shape),'native_loaded_link_pos_shape':list(loaded['link_pos'].shape),'velocity_kind':'Native joint_vel remains its original per-clip forwardFD; startup must separately use analyticqd/worldtwist; no reset was performed','kinematics_calls':kin_calls,'forbidden_calls':forbidden,'optimizer_calls':optimizers,'GPU_started':False,'Torch_CUDA_initialized':torch.cuda.is_initialized(),'Warp_imported_during_native_library_import':any(k in sys.modules for k in ('warp','mujoco_warp')),'physics_steps':0,'training_executed':False,'physical_reference_qualified':False,'training_authorized':False,'wall_seconds_including_imports_and_FK_reader':time.perf_counter()-start,'scope':'One derived native artifact and real CPU reader ABI check. This does not enroll shared catalogs, qualify reset/controller/physical tracking or authorize rough-reference training.'};write(output/'report.json',report);return report
    finally:
        for k,v in original.items():setattr(mujoco,k,v)
        for k,v in opt_original.items():setattr(optimize,k,v)
        mujoco.mj_kinematics=original_kinematics


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'outputs/restored_prefix_native_ABI_v1');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if not a.run:validate();print(json.dumps({'status':'CPU_INPUT_HASH_SHAPE_CHECK_PASS','export_started':False,'physics_steps':0,'training_authorized':False},indent=2));return
    r=export(a.output);print(json.dumps({k:r[k] for k in ('status','check_count','passed_checks','errors','analytic_vs_native_FD_initial_max_abs_rad_s','kinematics_calls','Torch_CUDA_initialized','Warp_imported_during_native_library_import','wall_seconds_including_imports_and_FK_reader')},indent=2));raise SystemExit(r['status']!='PASS_NATIVE_PREFIX16_ABI_CPU_ONLY')
if __name__=='__main__':main()
