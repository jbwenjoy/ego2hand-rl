"""Load the separate observation-derived goals, requested init and baseline.

CPU-only validation does not install a simulator or confer physical/training
qualification. No original source robot labels are read here.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .io import read,checked,loadnp,pin,canonical
VARIANT='H123_cubic_minimax_restoration_filtered_prefix16_v1'
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
CONTROL_KEY='control__fp32_old_export__tracking_u_exec_15_FP32_rad'
def _array(data,key,shape,dtype=None):
    a=data[key]
    if a.shape!=shape or a.dtype.kind!='f' or not np.isfinite(a).all() or (dtype is not None and a.dtype!=np.dtype(dtype)):raise ValueError('Wrong shape/dtype/nonfinite '+key)
    return a
def rotation(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))
def load_bundle(path,expected_sha256=None):
    path=Path(path).resolve(strict=True);spec=read(path)
    if expected_sha256 is not None and pin(path)['sha256']!=expected_sha256:raise ValueError('Input bundle SHA changed')
    if spec.get('schema')!='rough_prefix_input_bundle:1' or spec.get('retarget_variant_id')!=VARIANT or spec.get('physical_reference_qualified') is not False or spec.get('shared10_enrollment') is not False or spec.get('training_authorized') is not False or spec.get('runtime_implementation_ready') is not False:raise ValueError('Wrong mode/identity/qualification')
    identity=spec['raw_source_identity']
    if identity['primary_pose_source_frame_range_half_open']!=[200,216] or identity['goal_states']!=16:raise ValueError('Wrong source window')
    pins=spec['source_pins']
    for p in pins.values():checked(p)
    for name in ('model','context'):
        if spec[name]!=pins['export_source__'+('model' if name=='model' else 'actual_context')]:raise ValueError('Physical alias mismatch')
    manifests={k:read(checked(pins[k+'_manifest'])) for k in ('reference','init','control')}
    receipt=read(checked(pins['export_receipt']))
    if not all(v is True for v in receipt['gates'].values()):raise ValueError('Failed planned controller qualification')
    for k,m in manifests.items():
        if m['schema']!='restored_prefix_separate_bundles:1' or m['retarget_variant_id']!=VARIANT or m['bundle_kind']!=k or m['source_identity']!=identity or m['source_pins']!=receipt['source_pins'] or m['artifact']!=pins[k+'_arrays']:raise ValueError('Different bundle identities')
        if m['init_measured'] is not None or m['u_readback'] is not None or m['physical_reference_qualified'] is not False:raise ValueError('Planned values cannot be renamed measurements')
    native=read(checked(pins['native_manifest']))
    if native.get('physical_reference_qualified') is not False or native.get('training_authorized') is not False or native.get('shared_registry_enrollment','MISSING') is not None:raise ValueError('Native artifact cannot self-qualify training or shared enrollment')
    if native['source_window_half_open']!=[200,216] or native['native_local_range_half_open']!=[0,16] or native['frames']!=16 or native['control_intervals']!=15 or native['recipe']!=pins['native_recipe'] or native['native']!=pins['native_data']:raise ValueError('Wrong independent native enrollment')
    if spec['native_reference']['recipe_path']!=native['recipe']['path'] or spec['native_reference']['clips_root']!=native['clips_root'] or spec['native_reference']['duration_s']!=.3:raise ValueError('Native recipe descriptor differs')
    final=read(checked(pins['native_finalizer']));independent=read(checked(pins['native_independent']))
    if final['status']!='PASS_SAVED_NATIVE_READER_ABI_ORIGINAL_TERMINAL_RECEIPT_FAILED' or final['passed_checks']!=20 or independent['status']!='PASS_SAVED_DATA_ABI_ORIGINAL_PROCESS_RECEIPT_FAILED' or independent['passed']!=15:raise ValueError('Native saved-data proof missing')
    arrays={k:loadnp(checked(pins[k+'_arrays'])) for k in ('reference','init','control')};ref=arrays['reference'];init=arrays['init'];control=arrays['control'];n=loadnp(checked(pins['native_data']))
    t=_array(ref,'time_s',(16,));times=_array(ref,'planned_time61_s',(61,));q=_array(ref,'q_goal_rad',(16,20));qd=_array(ref,'analytic_qd16_rad_s',(16,20));pose=_array(ref,'mapped_object_pose_w',(16,7));v=_array(ref,'object_v_world_m_s',(61,3));omega=_array(ref,'object_omega_world_rad_s',(61,3))
    if not np.array_equal(t,np.arange(16)*.02) or not np.array_equal(times,np.arange(61)*.005) or not np.array_equal(q,n['reference_qpos']) or not np.array_equal(pose,n['object_pose_w']):raise ValueError('Native/continuous reference mismatch')
    bridge=loadnp(checked(pins['contact_reference']));bridge_meta=read(checked(pins['contact_reference_metadata']))
    if set(bridge)!=set(('time_s','joint_pos_rad','object_pose_w','wrist_pose_w')) or not all(np.array_equal(bridge[k],v) for k,v in [('time_s',t),('joint_pos_rad',q),('object_pose_w',pose),('wrist_pose_w',n['wrist_pose_w'])]):raise ValueError('Contact reference differs from native/continuous arrays')
    if bridge_meta['retarget_variant_id']!=VARIANT or bridge_meta['raw_source_identity']!=identity or bridge_meta['source_window_half_open']!=[200,216] or bridge_meta['physical_reference_qualified'] is not False or bridge_meta['training_authorized'] is not False or bridge_meta['source_robot_labels_used'] is not False or bridge_meta['human_contact_observation_known'] is not False:raise ValueError('Contact reference source/qualification differs')
    baseline=_array(control,CONTROL_KEY,(15,20),'float32')
    shapes={'q_init_requested_FP32_rad':(20,),'qd_init_requested_FP32_rad_s':(20,),'object_pose_init_requested_FP32_w':(7,),'object_bodyorigin_v_requested_FP32_world_m_s':(3,),'object_angular_requested_FP32_body_rad_s':(3,),'object_angular_requested_FP32_world_rad_s':(3,),'u_planned0_FP32_rad':(20,)}
    for k,s in shapes.items():_array(init,k,s,'float32')
    if not np.array_equal(init['q_init_requested_FP32_rad'],q[0].astype(np.float32)) or not np.array_equal(init['qd_init_requested_FP32_rad_s'],qd[0].astype(np.float32)) or not np.array_equal(init['object_pose_init_requested_FP32_w'],pose[0].astype(np.float32)) or not np.array_equal(init['u_planned0_FP32_rad'],baseline[0]):raise ValueError('Requested initial q/qd/pose/u0 differs')
    if not np.array_equal(init['object_bodyorigin_v_requested_FP32_world_m_s'],v[0].astype(np.float32)) or not np.array_equal(init['object_angular_requested_FP32_world_rad_s'],omega[0].astype(np.float32)) or not np.array_equal(init['object_angular_requested_FP32_body_rad_s'],(rotation(init['object_pose_init_requested_FP32_w'][3:]) .T@omega[0]).astype(np.float32)):raise ValueError('Requested world/body velocity differs; body_ipos0 ABI required')
    if np.abs(np.diff(baseline.astype(float),axis=0)).max()/.02>15 or not np.any(baseline[1:]!=baseline[0]):raise ValueError('Wrong baseline rate/intervention')
    context=loadnp(checked(spec['context']))
    for k in FIELDS:
        value=context[k]
        if value.ndim<2 or not np.isfinite(value).all() or not np.array_equal(value,np.broadcast_to(value[:1],value.shape)):raise ValueError('Physical context not fixed across worlds')
    descriptor={'schema':'rough_prefix_controller_input_descriptor:1','mode':'rough_observation_reference_prefix16_v1','input_bundle':pin(path),'source_pins':pins,'retarget_variant_id':VARIANT,'source_identity':identity,'physical_reference_qualified':False,'independent_ego_N':0,'source_lineage_N':1,'reference_states':16,'control_intervals':15,'control_dt_s':.02,'physics_dt_s':.005,'goal_velocity_authority':'frozen_Bspline_analytic_joint_qd16_and_quintic_object_world_v_omega16','native_reader_FD':'unmodified_diagnostic_not_initializer','initialization':'explicit_analytic_nonzero_phase0_only','controller':'baseline_u15_plus_original_EMA_residual_then_hard_ctrl_intersection_then_C2_020_once','EMA_history':'original_residual_only_zero_on_reset','reward_phase':'native_current_preaction_phase_k_common_all_arms','physical_evaluation_phase':'poststate_kplus1_vs_reference_kplus1','natural_end':'after_interval14_terminalobs15_no_u15_done_timeout_not_failure','imagination':'finite_end_15interval_zero_survival_and_masked_deadrow_critic_norm_loss','native_cadence':'Manager_4x5ms_steps_then_forward_same_A0_training_evaluation','directWarp_history':'separate_previous_H0_T0_not_native_A0','checkpoint':'new_format104_fromscratch_only_no_old103_or3_migration','original_native_export_terminal_receipt':'FAILED_saved_data_independently_verified','original_native_terminal_CUDA_flag':'UNKNOWN'}
    descriptor['input_signature_sha256']=hashlib.sha256(canonical(descriptor).encode()).hexdigest()
    for p in pins.values():checked(p)
    return {'arrays':arrays,'baseline_u15':baseline,'native':native,'native_reference':spec['native_reference'],'model':spec['model'],'context':context,'descriptor':descriptor,'spec':spec}
