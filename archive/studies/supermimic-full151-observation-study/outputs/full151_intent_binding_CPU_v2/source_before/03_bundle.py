"""Strict full151 known-rough input binding. No environment or device starts.

Accepting a declared rough reference for engineering is separate from changing
its retained numerical/geometry experiment outcome or qualifying dynamics.
"""
from pathlib import Path
import hashlib,importlib.util,sys
import numpy as np
from .io import canonical,checked,loadnp,pin,read
from .metadata import MODE,VARIANT,DESCRIPTOR_SCHEMA,BUNDLE_SCHEMA

FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
CONTROL_KEY='u_postC2_150_FP32_rad'
INIT_SHAPES={'q_init_requested_FP32_rad':(20,),'qd_init_requested_FP32_rad_s':(20,),'object_pose_init_requested_FP32_w':(7,),
 'object_bodyorigin_v_requested_FP32_world_m_s':(3,),'object_angular_requested_FP32_body_rad_s':(3,),
 'object_angular_requested_FP32_world_rad_s':(3,),'u_planned0_FP32_rad':(20,)}


def _array(data,key,shape,dtype=None):
    a=data[key]
    if a.shape!=shape or a.dtype.kind!='f' or not np.isfinite(a).all() or (dtype is not None and a.dtype!=np.dtype(dtype)):raise ValueError('Wrongshape/dtype/nonfinite '+key)
    return a


def _exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def rotation(q):
    q=np.asarray(q,float)
    if q.shape[-1:]!=(4,) or not np.isfinite(q).all() or np.any(np.abs(np.linalg.norm(q,axis=-1)-1)>1e-6):raise ValueError('Expected near-unit finite wxyz quaternion')
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def load_bundle(path,expected_sha256=None):
    path=Path(path).resolve(strict=True);spec=read(path)
    if expected_sha256 is not None and pin(path)['sha256']!=expected_sha256:raise ValueError('Input bundle bytes changed')
    if spec.get('schema')!=BUNDLE_SCHEMA or spec.get('mode')!=MODE or spec.get('retarget_variant_id')!=VARIANT:raise ValueError('Wrong full151 mode/schema/variant')
    for k in ('physical_reference_qualified','shared10_enrollment','training_authorized','runtime_implementation_ready','geometric_candidate_qualified'):
        if spec.get(k) is not False:raise ValueError('Input data cannot qualify '+k)
    if spec.get('known_rough_reference') is not True or spec.get('contact_intent_bundle','MISSING') is not None or spec.get('human_contact_observation_known') is not False:raise ValueError('Known-rough or UNKNOWN-contact scope changed')
    identity=spec['raw_source_identity']
    if identity['primary_pose_source_frame_range_half_open']!=[200,351] or identity['local_reference_range_half_open']!=[0,151] or identity['goal_states']!=151 or identity['control_intervals']!=150:raise ValueError('Wrong legacy151 source window; shared[0,151) is separate')
    pins=spec['source_pins']
    for value in pins.values():checked(value)
    if spec['model']!=pins['model'] or spec['context']!=pins['actual_context']:raise ValueError('Model/context aliases differ')
    parent=read(checked(pins['parent_scp_receipt']))
    if parent['status']!='CPU_COUPLED_CANDIDATE_FAIL_RETAINED' or parent['last_accepted_major']!=8 or {k for k,v in parent['checks'].items() if not v}!={'last_accepted_eta_final_gate','final_fresh_sampled_geometry'}:raise ValueError('Retained source outcome changed; do not disguise a failed qualification')
    native=read(checked(pins['native_manifest']));proof=read(checked(pins['native_reader_proof']))
    if native['schema']!='full151_knownrough_native_ABI:1' or native['retarget_variant_id']!=VARIANT or native['source_window_half_open']!=[200,351] or native['frames']!=151 or native['control_intervals']!=150:raise ValueError('Different native full151 identity')
    if native['native']!=pins['native_data'] or native['recipe']!=pins['native_recipe'] or native['reader_arrays']!=pins['native_reader_arrays']:raise ValueError('Native aliases changed')
    if native['physical_reference_qualified'] is not False or native['training_authorized'] is not False or native['shared_registry_enrollment'] is not None:raise ValueError('Native ABI cannot self-qualify')
    if proof['status']!='PASS_ACTUAL_NATIVE151_CPU_READER_ONLY' or not all(proof['checks'].values()):raise ValueError('New actual native Reader proof missing')
    nr=spec['native_reference']
    if nr['recipe_path']!=pins['native_recipe']['path'] or nr['clips_root']!=native['clips_root'] or nr['duration_s']!=3.:raise ValueError('Native selection differs')
    arrays={k:loadnp(checked(pins[k+'_arrays'])) for k in ('reference','init','control')}
    ref,initial,control=arrays['reference'],arrays['init'],arrays['control'];n=loadnp(checked(pins['native_data']));reader=loadnp(checked(pins['native_reader_arrays']))
    if set(n)!={'reference_qpos','wrist_pose_w','object_pose_w','link_pos_w'} or n['link_pos_w'].shape!=(151,21,3):raise ValueError('Actual BODY21 native four-field ABI required')
    t=_array(ref,'time_s',(151,));times=_array(ref,'planned_time601_s',(601,));q=_array(ref,'q_goal_rad',(151,20));qd=_array(ref,'analytic_qd151_rad_s',(151,20));pose=_array(ref,'mapped_object_pose_w',(151,7))
    v=_array(ref,'object_v_world_m_s',(601,3));omega=_array(ref,'object_omega_world_rad_s',(601,3))
    if not _exact(t,np.arange(151)*.02) or not _exact(times,np.arange(601)*.005) or not _exact(q,n['reference_qpos']) or not _exact(pose,n['object_pose_w']):raise ValueError('Native/continuous reference differs')
    if not _exact(n['wrist_pose_w'],np.broadcast_to(n['wrist_pose_w'][:1],(151,7))):raise ValueError('Default native wrist must stay fixed')
    if not _exact(reader['joint_pos'],q.astype(np.float32)) or not _exact(reader['joint_vel'],ref['native_forward_FD32_rad_s']) or reader['trajectory_lengths'].tolist()!=[151]:raise ValueError('Actual native reader151/FD binding differs')
    baseline=_array(control,CONTROL_KEY,(150,20),'float32')
    for key,shape in INIT_SHAPES.items():_array(initial,key,shape,'float32')
    if set(initial)!=set(INIT_SHAPES):raise ValueError('Unexpected initialization fields')
    for a,b in [(initial['q_init_requested_FP32_rad'],q[0].astype(np.float32)),(initial['qd_init_requested_FP32_rad_s'],qd[0].astype(np.float32)),(initial['object_pose_init_requested_FP32_w'],pose[0].astype(np.float32)),(initial['u_planned0_FP32_rad'],baseline[0]),(initial['object_bodyorigin_v_requested_FP32_world_m_s'],v[0].astype(np.float32)),(initial['object_angular_requested_FP32_world_rad_s'],omega[0].astype(np.float32)),(initial['object_angular_requested_FP32_body_rad_s'],(rotation(initial['object_pose_init_requested_FP32_w'][3:]).T@omega[0]).astype(np.float32))]:
        if not _exact(a,b):raise ValueError('Analytic requested initial q/qd/pose/u/world-body twist differs')
    if np.abs(np.diff(baseline.astype(float),axis=0)).max()/.02>15.:raise ValueError('Planned149 target updates exceed original15rad/s')
    source=checked(pins['pure_control760']);name='_full151_native_frozen760'
    if name not in sys.modules:
        sp=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(sp);sys.modules[name]=module;sp.loader.exec_module(module)
    old=sys.modules[name]
    if Path(old.__file__).resolve()!=source.resolve():raise ValueError('Wrong loaded PD helper')
    params,metadata=old.load_actual_pd(checked(spec['model']),checked(spec['context']),model_sha256=spec['model']['sha256'],context_sha256=spec['context']['sha256'])
    proposed=old.proposal_numpy(q[:150],qd[:150],control['tau_ff_static0_FP32_Nm'],params)
    if not _exact(proposed,baseline) or not _exact(old.c2_numpy(proposed,params),baseline):raise ValueError('150 original proposal/C2 identity differs; no projection in loader')
    context=loadnp(checked(spec['context']))
    for key in FIELDS:
        value=context[key]
        if value.ndim<2 or not np.isfinite(value).all() or not _exact(value,np.broadcast_to(value[:1],value.shape)):raise ValueError('Actual8 not fixed shared per-world context')
    descriptor={'schema':DESCRIPTOR_SCHEMA,'mode':MODE,'input_bundle':pin(path),'source_pins':pins,'retarget_variant_id':VARIANT,'source_identity':identity,
      'known_rough_reference':True,'geometric_candidate_qualified':False,'parent_candidate_status':parent['status'],'physical_reference_qualified':False,'shared10_enrollment':False,
      'independent_ego_N':0,'source_lineage_N':1,'reference_states':151,'control_intervals':150,'control_dt_s':.02,'physics_dt_s':.005,
      'goal_velocity_authority':'frozen_Bspline_analytic_joint_qd151_and_quintic_object_world_v_omega151','native_reader_FD':'unmodified_diagnostic_not_initializer',
      'initialization':'explicit_analytic_nonzero_phase0_only','controller':'baseline_u150_plus_original_EMA_residual_then_hard_ctrl_intersection_then_C2_020_once',
      'EMA_history':'original_residual_only_zero_on_reset','reward_phase':'native_current_preaction_phase_k_common_all_arms','physical_evaluation_phase':'poststate_kplus1_vs_reference_kplus1',
      'natural_end':'after_interval149_terminalobs150_no_u150_done_timeout_not_failure','imagination':'finite_end_150interval_zero_survival_and_masked_deadrow_critic_norm_loss',
      'native_cadence':'Manager_4x5ms_steps_then_forward_same_A0_training_evaluation','checkpoint':'new_format105_fromscratch_only_no_old104_103_or3_migration',
      'contact_intent_target_ready':False,'human_contact_observation_known':False,'rough_input_status':'NEW_ENGINEERING_INPUT_ONLY_NOT_CPU_GEOMETRY_PASS'}
    descriptor['clock_contract']={'schema':'full151_native_clock:1','dtype':'float32','physics_dt_s':.005,'physics_steps':600,'control_substeps':4,
      'expected_rule':'sequential_float32_add_from_zero','bitexact_expected_time':True,'canonical_time_rule':'integer_physics_tick_times_0p005',
      'nominal_drift_gate_s':5e-5,'legacy_prefix_2us_diagnostic_only':True}
    descriptor['input_signature_sha256']=hashlib.sha256(canonical(descriptor).encode()).hexdigest()
    for value in pins.values():checked(value)
    return {'arrays':arrays,'baseline_u150':baseline,'native':native,'native_reference':nr,'model':spec['model'],'context':context,'descriptor':descriptor,'spec':spec}
