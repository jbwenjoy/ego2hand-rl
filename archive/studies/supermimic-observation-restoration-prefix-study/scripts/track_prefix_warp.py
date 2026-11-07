"""New matched nonzero-init hold/track experiment; CPU preflight by default.

Warp is imported only inside an explicitly authorized --run. This is neither
the old zero-rest plan nor a learned actor/EMA/format103 training contract.
"""
from __future__ import annotations
import argparse,hashlib,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json
STARTED=time.perf_counter()
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
VARIANT='H123_cubic_minimax_restoration_filtered_prefix16_v1'
BUNDLE_SCHEMA='restored_prefix_separate_bundles:1'
CONSTRAINED_SHA='0798ea7aa6da6f56684067992906bbba3136f9e15b413f743674f54159ad2e6d'
PARENT_SHA='94ea7cbe03f3e2aaa7bb4f7c8860bae8e599cd4e8e823cbe47220f73ffe101a3'
CONDITIONS=('H0_MATCHED_DYNAMIC_INIT_HOLD','T0_MATCHED_DYNAMIC_INIT_TRACK')
CONTROL_KEYS={'track':'control__fp32_old_export__tracking_u_exec_15_FP32_rad','hold':'control__fp32_old_export__hold_u_exec_15_FP32_rad','proposed':'control__fp32_old_export__u_pre_16_FP32_rad'}
FIXED_GATES={'object_position_max_m':.02,'directed_shaft_error_max_deg':20.,'actual_joint_hard_excess_max_rad':.001,'penetration_any_category_max_m':.003,'object_nonhand_candidate_contact_count_max':0,'external_force_max_abs':0.,'clock_absolute_tolerance_s':2e-6,'target_rate_max_rad_s':15.,'actuator_load_ratio_max':1.01,'load_charge_max':.8,'charge_decay_time_s':2.,'drop_center_below_default_wrist_m':.15}
INIT_KEYS={'q_init_requested_FP32_rad':(20,),'qd_init_requested_FP32_rad_s':(20,),'object_pose_init_requested_FP32_w':(7,),'object_bodyorigin_v_requested_FP32_world_m_s':(3,),'object_angular_requested_FP32_body_rad_s':(3,),'object_angular_requested_FP32_world_rad_s':(3,),'u_planned0_FP32_rad':(20,)}

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
def write(p,x):write_json(p,x)
def same(a,b):return json.dumps(a,sort_keys=True,separators=(',',':'))==json.dumps(b,sort_keys=True,separators=(',',':'))
def deadline():
    if time.perf_counter()-STARTED>40:raise TimeoutError('40s whole-worker softdeadline; retain partial, no retry')
def convert(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [convert(v) for v in x]
    return x
def checkpin(p):
    if set(p)!={'path','sha256'} or not Path(p['path']).is_absolute() or sha(p['path'])!=p['sha256']:raise ValueError('Missing/changed pinned artifact')
def loadnp(p):
    import numpy as np
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def rotation(q):
    import numpy as np
    q=np.asarray(q,float)
    if q.shape[-1]!=4 or not np.isfinite(q).all() or np.any(np.linalg.norm(q,axis=-1)<1e-9):raise ValueError('Invalid quaternion')
    q=q/np.linalg.norm(q,axis=-1,keepdims=True);w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))


def control_index_for_solver_step(step):
    if type(step) is not int or not 1<=step<=60:raise ValueError('Expected solver step1..60')
    return (step-1)//4
def control_index_for_raw_state(index):
    if type(index) is not int or not 0<=index<=60:raise ValueError('Expected raw state0..60')
    return 0 if index==0 else control_index_for_solver_step(index)
def is_target_update_before(step):return control_index_for_solver_step(step)!=control_index_for_raw_state(step-1)


def validate_arrays(reference,initial,control,condition,lower,upper):
    """Pure NumPy ABI/schedule check; never fabricate a missing future bundle."""
    import numpy as np
    def finite(key,d,shape,dtype=None):
        if key not in d:raise ValueError('Missing bundle field '+key)
        a=np.asarray(d[key])
        if a.shape!=shape or a.dtype.kind!='f' or not np.isfinite(a).all() or (dtype is not None and a.dtype!=dtype):raise ValueError('Wrong shape/type/nonfinite '+key)
        return a
    if condition not in CONDITIONS:raise ValueError('Unknown condition/old zero-rest rejected')
    t=finite('time_s',reference,(16,));times=finite('planned_time61_s',reference,(61,))
    if not np.array_equal(t,np.arange(16)*.02) or not np.array_equal(times,np.arange(61)*.005):raise ValueError('Wrong reference clock/window')
    q=finite('q_goal_rad',reference,(16,20));qd=finite('analytic_qd16_rad_s',reference,(16,20));pose=finite('mapped_object_pose_w',reference,(16,7))
    position=finite('object_COM_world_m',reference,(61,3));R=finite('object_rotation_w',reference,(61,3,3));v=finite('object_v_world_m_s',reference,(61,3));omega=finite('object_omega_world_rad_s',reference,(61,3))
    if np.max(np.abs(np.swapaxes(R,-1,-2)@R-np.eye(3)))>1e-8 or np.max(np.abs(np.linalg.det(R)-1))>1e-8:raise ValueError('Reference not SO3')
    for k,shape in INIT_KEYS.items():finite(k,initial,shape,np.dtype('float32'))
    if not np.array_equal(initial['q_init_requested_FP32_rad'],q[0].astype(np.float32)) or not np.array_equal(initial['qd_init_requested_FP32_rad_s'],qd[0].astype(np.float32)) or not np.array_equal(initial['object_pose_init_requested_FP32_w'],pose[0].astype(np.float32)):raise ValueError('Initialization differs from new analytic reference')
    if not any(np.any(initial[k]) for k in ('qd_init_requested_FP32_rad_s','object_bodyorigin_v_requested_FP32_world_m_s','object_angular_requested_FP32_body_rad_s')):raise ValueError('Old designed zero-rest initialization is not this experiment')
    if np.max(np.abs(R[0]-rotation(pose[0,3:])))>1e-10 or np.max(np.abs(position[0]-pose[0,:3]))>1e-10:raise ValueError('Reference time0 pose/COM mismatch')
    if not np.array_equal(initial['object_bodyorigin_v_requested_FP32_world_m_s'],v[0].astype(np.float32)):raise ValueError('This ABI assumes actual body_ipos0; COM velocity mismatch')
    R32=rotation(initial['object_pose_init_requested_FP32_w'][3:].astype(float))
    if not np.array_equal(initial['object_angular_requested_FP32_body_rad_s'],(R32.T@omega[0]).astype(np.float32)) or not np.array_equal(initial['object_angular_requested_FP32_world_rad_s'],omega[0].astype(np.float32)):raise ValueError('World/native-body angular frame or cast mismatch')
    track=finite(CONTROL_KEYS['track'],control,(15,20),np.dtype('float32'));hold=finite(CONTROL_KEYS['hold'],control,(15,20),np.dtype('float32'))
    proposed=finite(CONTROL_KEYS['proposed'],control,(16,20),np.dtype('float32'))
    if not np.array_equal(track[0].view(np.uint32),hold[0].view(np.uint32)) or not np.array_equal(initial['u_planned0_FP32_rad'].view(np.uint32),track[0].view(np.uint32)) or not np.array_equal(hold,np.broadcast_to(track[0],(15,20))):raise ValueError('Hold/track/u0 do not share one new initializer')
    if not np.any(track[1:]!=track[0]):raise ValueError('Tracking is not an actual FP32 target intervention')
    lo=np.asarray(lower,np.float32)+np.float32(.020);hi=np.asarray(upper,np.float32)-np.float32(.020)
    if np.any(lo>hi):raise ValueError('Empty C2 domain')
    if not np.array_equal(track.view(np.uint32),np.minimum(np.maximum(track,lo),hi).view(np.uint32)):raise ValueError('Planned post-C2 targets would be changed; no runtime repair')
    if not np.array_equal(track.view(np.uint32),np.minimum(np.maximum(proposed[:15],lo),hi).view(np.uint32)):raise ValueError('Exported proposal/post-C2 identity mismatch')
    rates=np.diff(track.astype(float),axis=0)/.02
    if np.abs(rates).max()>15:raise ValueError('Planned target rate exceeds original15rad/s')
    selected=hold.copy() if condition==CONDITIONS[0] else track.copy()
    return selected,{'track_target_rate_max_rad_s':float(np.abs(rates).max()),'track_proposal_clamped_components':int(np.count_nonzero(proposed[:15]!=track)),'initial_u0_clamped_components':int(np.count_nonzero(proposed[0]!=track[0])),'same_new_u0_FP32':True,'q_goal_is_not_actual_state':True,'initial_velocities_are_design_not_video_measurement':True,'native_angular_frame':'body','physics_started':False}


def describe_contract():
    return {'plan_schema':'restoration_dynamic_prefix_plan:1','bundle_schema':BUNDLE_SCHEMA,'retarget_variant_id':VARIANT,'conditions':CONDITIONS,'bundle_aliases':['reference_manifest','init_manifest','control_manifest'],'other_required_pins':['protocol','constrained_protocol','input_parent_protocol','export_receipt','model','context','robot_kinematics'],'reference_required':['time_s','q_goal_rad','analytic_qd16_rad_s','mapped_object_pose_w','planned_time61_s','object_COM_world_m','object_rotation_w','object_v_world_m_s','object_omega_world_rad_s'],'initial_fields':INIT_KEYS,'control_fields':CONTROL_KEYS,'steps':60,'physics_dt_s':.005,'planned_targets':15,'postinit_control_updates_before_steps':list(range(5,58,4)),'grade':FIXED_GATES,'qualification':'CPU/input/static/predicted qualification only; no GPU authorization or runtime readback yet'}


def load_inputs(plan_path):
    """CPU artifact checks and kinematics only; no Warp, Torch, dynamics or output."""
    import numpy as np
    import mujoco
    p=read(plan_path)
    if p.get('schema')!='restoration_dynamic_prefix_plan:1' or p.get('retarget_variant_id')!=VARIANT:raise ValueError('New restoration dynamic plan required')
    aliases=('reference_manifest','init_manifest','control_manifest','protocol','constrained_protocol','input_parent_protocol','export_receipt','model','context','robot_kinematics')
    for key in aliases:
        if not same(p[key],p['source_pins'][key]):raise ValueError('Plan alias differs from frozenpin: '+key)
    for pin in p['source_pins'].values():checkpin(pin)
    if not any(Path(pin['path']).resolve()==Path(__file__).resolve() for pin in p['source_pins'].values()):raise ValueError('Worker must be hash-bound in plan')
    if p['steps']!=60 or p['dt_s']!=.005 or p['duration_s']!=.3 or p['nworld']!=1 or p['nconmax']!=192 or p['njmax']!=4096 or p['hard_process_seconds']!=45 or p['soft_worker_seconds']!=40 or p['automatic_retry'] is not False:raise ValueError('Fixed bounded campaign changed')
    if not same(p['gates'],FIXED_GATES):raise ValueError('Original numeric gates differ')
    if p['constrained_protocol']['sha256']!=CONSTRAINED_SHA or p['input_parent_protocol']['sha256']!=PARENT_SHA:raise ValueError('Wrong protocol ancestry')
    protocol=read(p['protocol']['path']);parent=read(p['input_parent_protocol']['path'])
    if protocol.get('schema')!='restoration_prefix_protocol:1' or protocol.get('retarget_variant_id')!=VARIANT or protocol['parent_protocol']!=p['constrained_protocol'] or protocol['input_parent_protocol']!=p['input_parent_protocol']:raise ValueError('Restoration lineage/variant differs')
    if not same(protocol['source_identity'],parent['source_identity']) or protocol['source_identity']['primary_pose_source_frame_range_half_open']!=[200,216]:raise ValueError('Raw source/window identity mismatch')
    for key,source in [('model','model'),('context','actual_context'),('robot_kinematics','kinematics')]:
        if not same(p[key],protocol['inputs'][source]):raise ValueError('Physical/kinematics source differs from protocol')
    manifests=[read(p[k]['path']) for k in ('reference_manifest','init_manifest','control_manifest')]
    for kind,m in zip(('reference','init','control'),manifests):
        if m.get('schema')!=BUNDLE_SCHEMA or m.get('retarget_variant_id')!=VARIANT or m.get('bundle_kind')!=kind:raise ValueError('Wrong separate bundle schema/variant/kind')
        if not same(m['source_identity'],protocol['source_identity']) or m.get('physical_reference_qualified') is not False or m.get('init_measured','MISSING') is not None or m.get('u_readback','MISSING') is not None:raise ValueError('Bundle measured/physical provenance invalid')
        for key in ('protocol','constrained_protocol','input_parent_protocol'):
            if not same(m[key],p[key]):raise ValueError('Bundle protocol identity mismatch')
        for pin in m['source_pins'].values():checkpin(pin)
        checkpin(m['artifact'])
        if not same(m['source_pins'],manifests[0]['source_pins']):raise ValueError('Three bundles not one exported candidate/sourceplan')
        for key in ('candidate','source_plan'):
            if key not in m['source_pins']:raise ValueError('New candidate/source-plan pin missing')
    receipt=read(p['export_receipt']['path'])
    if not receipt.get('gates') or not all(v is True for v in receipt['gates'].values()) or receipt.get('source_input_hashes_unchanged') is not True:raise ValueError('Numeric reference/control qualification has failed or is pending')
    if not same(receipt['source_pins'],manifests[0]['source_pins']):raise ValueError('Export receipt/source bundle mismatch')
    reference,initial,control=[loadnp(m['artifact']['path']) for m in manifests]
    model=mujoco.MjModel.from_binary_path(p['model']['path']);host=mujoco.MjData(model)
    names=[f'robot/right_finger{finger}_joint{joint}' for finger in range(1,6) for joint in range(1,5)]
    jids=np.array([model.joint(n).id for n in names]);qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids]
    aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids]);obj=model.body('object/cube').id;oj=model.joint('object/cube_freejoint').id;oq=int(model.jnt_qposadr[oj]);ov=int(model.jnt_dofadr[oj])
    if abs(model.opt.timestep-.005)>1e-12 or model.opt.integrator!=mujoco.mjtIntegrator.mjINT_IMPLICITFAST or model.nu!=20 or not np.all(model.jnt_limited[jids]) or not np.all(model.actuator_ctrllimited[aids]):raise ValueError('Original model/mapping/timing contract differs')
    if np.any(model.body_ipos[obj]!=0):raise ValueError('Nonzero object COM offset needs a different verified ABI')
    if not np.array_equal(np.asarray(p['joint_lower_rad']),model.jnt_range[jids,0]) or not np.array_equal(np.asarray(p['joint_upper_rad']),model.jnt_range[jids,1]):raise ValueError('Hard-limit grade differs from physical model')
    if not np.all(model.actuator_trntype[aids]==int(mujoco.mjtTrn.mjTRN_JOINT)) or not np.array_equal(model.actuator_gear[aids],np.broadcast_to([1.,0,0,0,0,0],(20,6))):raise ValueError('Actuator force cannot be called unit-gear joint torque')
    lower=np.maximum(model.jnt_range[jids,0],model.actuator_ctrlrange[aids,0]);upper=np.minimum(model.jnt_range[jids,1],model.actuator_ctrlrange[aids,1])
    selected,array_info=validate_arrays(reference,initial,control,p['control_condition'],lower,upper)
    q64=reference['q_goal_rad'][0];pose64=reference['mapped_object_pose_w'][0];q32=initial['q_init_requested_FP32_rad'];pose32=initial['object_pose_init_requested_FP32_w'];qd32=initial['qd_init_requested_FP32_rad_s']
    if np.any(q64<model.jnt_range[jids,0]) or np.any(q64>model.jnt_range[jids,1]):raise ValueError('Source64 initial q outside original hard range; never project')
    host.qpos[qa]=q32;host.qpos[oq:oq+7]=pose32;host.qvel[:]=0;host.qvel[va]=qd32
    host.qvel[ov:ov+3]=initial['object_bodyorigin_v_requested_FP32_world_m_s'];host.qvel[ov+3:ov+6]=initial['object_angular_requested_FP32_body_rad_s'];host.ctrl[aids]=selected[0]
    host.xfrc_applied[:]=0;host.qfrc_applied[:]=0
    physical={};context_report={}
    for field,value in loadnp(p['context']['path']).items():
        if field not in FIELDS:continue
        nominal=getattr(model,field)
        if value.shape[1:]!=nominal.shape or not len(value) or not np.isfinite(value).all() or not np.array_equal(value,np.broadcast_to(value[:1],value.shape)):raise ValueError('Nonfixed actual physical field '+field)
        physical[field]=value[0].copy();context_report[field]={'shape':list(value[0].shape),'dtype':str(value.dtype),'nominal_max_abs_difference':float(np.max(np.abs(nominal-value[0])))}
    if set(physical)!=set(FIELDS):raise ValueError('Eight actual context fields required')
    force_lower=np.where(model.actuator_forcelimited[aids],model.actuator_forcerange[aids,0],-np.inf);force_upper=np.where(model.actuator_forcelimited[aids],model.actuator_forcerange[aids,1],np.inf)
    if hasattr(model,'jnt_actfrclimited'):
        force_lower=np.maximum(force_lower,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,0],-np.inf));force_upper=np.minimum(force_upper,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,1],np.inf))
    if not np.isfinite(force_lower).all() or not np.isfinite(force_upper).all() or np.any(force_lower>=0) or np.any(force_upper<=0):raise ValueError('Finite bipolar actuator/joint force caps required')
    # Original CPU kinematics uses mj_kinematics/comPos only, not forward/step.
    sys.path.insert(0,str(Path(p['robot_kinematics']['path']).parent))
    from retarget_palmroot_cpu import RobotKinematics
    robot=RobotKinematics(p['model']['path']);robot.data.qpos[oq:oq+7]=pose64;robot.last_q=None;fk64=robot.forward(q64)[0]
    robot.data.qpos[oq:oq+7]=pose32;robot.data.mocap_pos[:]=host.mocap_pos.astype(np.float32);robot.data.mocap_quat[:]=host.mocap_quat.astype(np.float32);robot.last_q=None;fk32=robot.forward(q32)[0]
    idx=dict(qa=qa,va=va,aids=aids,jids=jids,obj=obj,oq=oq,ov=ov,lower=lower,upper=upper,schedule=selected,q32=q32,qd32=qd32,pose32=pose32,expected_full_qvel32=host.qvel.astype(np.float32),expected_full_qpos32=host.qpos.astype(np.float32),expected_fk64=fk64,expected_fk32=fk32,points=robot.points,force_lower=force_lower,force_upper=force_upper,default_wrist_z=float(host.mocap_pos[0,2]))
    info={'status':'passed_CPU_new_dynamic_inputs_FK_FP32_control_contract','physics_started':False,'control_condition':p['control_condition'],'retarget_variant_id':VARIANT,'source_identity':protocol['source_identity'],'q_cast_max_abs_rad':float(np.max(np.abs(q64-q32))),'pose_cast_max_abs':float(np.max(np.abs(pose64-pose32))),'CPU_FK_FP32_vs_source64_max_abs_m':float(np.max(np.abs(fk32-fk64))),'context_fields':context_report,'joint_names':names,'initialization_kind':'new analytic qd and frozen object world motion; no zero-rest substitution','mocap_source':'Original MJB defaults unchanged','grade_primary_indices':list(range(0,61,4)),'timephase':'Afterstep4 ctrl stillu0; beforestep5 eventu1; step5 solverprestate4','Warp_imported':any(n in sys.modules for n in ('warp','mujoco_warp')),'Torch_imported':'torch' in sys.modules,**array_info}
    if info['Warp_imported'] or info['Torch_imported']:raise AssertionError('Default CPU validation must not import Torch/Warp')
    return p,protocol,model,host,physical,idx,reference,info


def classify_capacity(flags,states,contact_capacity,constraint_capacity):
    """End-only sticky read; a missing/unreadable value never becomes zero."""
    import numpy as np
    base={'backend_field':'wp_data.overflow','read_count':1,'read_after_step':60,'resets_after_initialization':0,'scope':'No recorded overflow flags since initialization reset, not convergence/physical accuracy'}
    if flags is None:return {**base,'status':'UNKNOWN','overflow_bitmask':None,'reason':'Sticky overflow unavailable/unreadable'}
    a=np.asarray(flags)
    if a.shape!=(1,) or a.dtype.kind not in 'iu':return {**base,'status':'UNKNOWN','overflow_bitmask':None,'reason':'Expected one integer sticky mask'}
    bitmask=int(a[0]);below=all(s['nacon']<contact_capacity and s['nefc']<constraint_capacity for s in states)
    return {**base,'status':'OK' if bitmask==0 and below else 'OVERFLOW','overflow_bitmask':bitmask,'flags_by_world':a.tolist(),'worlds':1,'counts_strictly_below_capacity':bool(below),'contact_capacity':contact_capacity,'constraint_capacity':constraint_capacity}


def gates(states,solver,capacity,p,reference,complete,force_lower,force_upper,wrist_z):
    """Moving-reference16 primary; all61 trace diagnostic plus raw-state gates."""
    import numpy as np
    if not complete:return {'status':'INCOMPLETE_ENGINEERING_NO_DYNAMIC_CONCLUSION','short_dynamic_pass':None,'physics_inference_qualified':False,'complete':False}
    g=p['gates'];q=np.asarray([s['joint_rad'] for s in states]);pose=np.asarray([s['object_pose_w'] for s in states]);R=rotation(pose[:,3:]);expected_R=reference['object_rotation_w']
    pos=np.linalg.norm(pose[:,:3]-reference['object_COM_world_m'],axis=1);angle=np.rad2deg(np.arccos(np.clip(np.sum(R[:,:,0]*expected_R[:,:,0],axis=1),-1,1)))
    full=np.rad2deg(np.arccos(np.clip((np.trace(np.swapaxes(expected_R,-1,-2)@R,axis1=-2,axis2=-1)-1)/2,-1,1)))
    primary=np.arange(0,61,4);lower=np.asarray(p['joint_lower_rad']);upper=np.asarray(p['joint_upper_rad']);excess=np.maximum(np.maximum(lower-q,q-upper),0)
    forces=np.asarray([s['actuator_force_solver_native'] for s in solver]);ratio=np.where(forces>=0,forces/np.asarray(force_upper),forces/np.asarray(force_lower))
    charge=np.zeros(20);maxcharge=charge.copy()
    for j,dt in enumerate(np.diff([s['raw_time_s'] for s in states])):
        charge=np.clip(charge+dt*(np.minimum(ratio[j],1)-charge/g['charge_decay_time_s']),0,1);maxcharge=np.maximum(maxcharge,charge)
    conditions={'moving_object_position_16':float(pos[primary].max())<=g['object_position_max_m'],'moving_shaft_angle_16':float(angle[primary].max())<=g['directed_shaft_error_max_deg'],'actual_joint_state_all61':float(excess.max())<=g['actual_joint_hard_excess_max_rad'],'penetration_solver60':max(s['penetration_m'] for s in solver)<=g['penetration_any_category_max_m'],'nonhand_object_contacts':sum(s['object_environment_contacts'] for s in solver)==0,'no_exogenous':max(s['external_force_max_abs'] for s in states)==0,'actual_ctrl_schedule_exact':all(s['ctrl_exact'] for s in states),'clock':max(abs(s['raw_time_s']-s['expected_time_s']) for s in states)<=g['clock_absolute_tolerance_s'],'raw_coverage':len(states)==61 and len(solver)==60,'finite_raw':bool(all(np.isfinite(np.asarray([s[k] for s in states])).all() for k in ('full_qpos','full_qvel','ctrl_rad','actuator_force_native'))),'load_ratio':float(ratio.max())<=g['actuator_load_ratio_max'],'load_charge':float(maxcharge.max())<=g['load_charge_max'],'drop':bool(np.min(pose[:,2])>=wrist_z-g['drop_center_below_default_wrist_m']),'capacity':capacity.get('status')=='OK'}
    qualified=capacity.get('status')=='OK'
    return {'status':'COMPLETED_NEW_SHORT_DYNAMIC_DIAGNOSTIC' if qualified else 'COMPLETED_CAPACITY_UNQUALIFIED_DYNAMIC_UNKNOWN','short_dynamic_pass':all(conditions.values()) if qualified else None,'physics_inference_qualified':qualified,'gates':conditions,'primary_reference_samples':16,'diagnostic_reference_samples':61,'max_object_position_error_16_m':float(pos[primary].max()),'max_object_position_error_61_m':float(pos.max()),'max_shaft_angle_16_deg':float(angle[primary].max()),'max_shaft_angle_61_deg':float(angle.max()),'max_SO3_error_16_deg':float(full[primary].max()),'max_SO3_error_61_deg':float(full.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':max(s['penetration_m'] for s in solver),'max_actuator_load_ratio':float(ratio.max()),'max_load_charge':float(maxcharge.max()),'all61_position_diagnostic_pass':bool(pos.max()<=g['object_position_max_m']),'all61_shaft_diagnostic_pass':bool(angle.max()<=g['directed_shaft_error_max_deg']),'full_3s_spin_task':'NOT_ASSESSED','scope':'Known-history sequentialN1 per condition, moving reference16. All61 state/60 preintegration solver evidence retained; no learning/forceprediction/contactlabel/hardware/continuous-safety qualification.'}


def run(plan_path,output):
    """Only explicit authorization imports GPU APIs; one full-horizon attempt."""
    import numpy as np
    import mujoco
    p=read(plan_path)
    if p.get('GPU_run_authorized') is not True:raise ValueError('Root GPU authorization absent; default CPU only')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(plan_path),'source_sha256':sha(__file__),'retarget_variant_id':VARIANT,'control_condition':p['control_condition'],'physics_steps':0,'step_calls':0,'startup_Warp_forward_calls':0,'postinit_state_writes':0,'postinit_ctrl_writes':0,'overflow_reads':0,'resets_after_initialization':0,'actor_EMA_training_used':False}
    write(output/'runtime.json',runtime);states=[];solver=[];rawcontacts=[];events=[];capacity={'status':'UNKNOWN','reason':'No completed60step final sticky read'};error=None
    try:
        p,protocol,model,host,physical,idx,reference,cpu=load_inputs(plan_path);write(output/'CPU_input_contract.json',cpu);deadline()
        import warp as wp
        import mujoco_warp as mjwarp
        from measured_contacts import capture_contacts
        wp.set_device('cuda:0');wp_model=mjwarp.put_model(model);context_readbacks={}
        def verify_context(stage):
            report={}
            for key,value in physical.items():
                got=getattr(wp_model,key).numpy().reshape((-1,)+value.shape)[0];context_readbacks[stage+'__'+key]=got.copy();expected=value.astype(got.dtype)
                exact=np.array_equal(got,expected);report[key]={'exact_expected_FP32':bool(exact),'max_abs_error':float(np.max(np.abs(got-expected)))}
                if not exact:raise ValueError('Actual Warp physical field differs '+key+' at '+stage)
            return report
        # Nominalmodel was never mutated on CPU. Copy all8, retain both real readbacks.
        for key,value in physical.items():
            destination=getattr(wp_model,key);wp.copy(destination,wp.array(value[None],dtype=destination.dtype,device='cuda:0'))
        copies={'after_copy':verify_context('after_copy')};data=mjwarp.put_data(model,host,nworld=1,nconmax=192,njmax=4096);mjwarp.set_const_0(wp_model,data,restore=True);copies['after_set_const_0']=verify_context('after_set_const_0')
        write(output/'actual8field_copy.json',copies);np.savez_compressed(output/'expected_context_fields.npz',**physical);np.savez_compressed(output/'actual_context_applied.npz',**{key:context_readbacks['after_set_const_0__'+key] for key in FIELDS});np.savez_compressed(output/'actual_context_readbacks.npz',**context_readbacks);deadline()
        np.savez_compressed(output/'requested_initialization_and_control.npz',q_init_requested_FP32_rad=idx['q32'],qd_init_requested_FP32_rad_s=idx['qd32'],full_qpos_requested_FP32=idx['expected_full_qpos32'],full_qvel_requested_FP32=idx['expected_full_qvel32'],mocap_pos_requested_FP32=host.mocap_pos.astype(np.float32),mocap_quat_requested_FP32=host.mocap_quat.astype(np.float32),selected_planned15_FP32_rad=idx['schedule'],position_ctrl_ids=idx['aids'],joint_qpos_ids=idx['qa'],joint_dof_ids=idx['va'],force_lower_Nm=idx['force_lower'],force_upper_Nm=idx['force_upper'])
        mjwarp.forward(wp_model,data);runtime['startup_Warp_forward_calls']=1
        readback=mujoco.MjData(model);object_geoms=set(np.flatnonzero(model.geom_bodyid==idx['obj']).tolist());import re
        hand_geoms={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
        def sample(k,previous_time):
            deadline();wp.synchronize_device('cuda:0');mjwarp.get_data_into(readback,model,data,world_id=0);h=readback
            if not all(np.isfinite(a).all() for a in [h.qpos,h.qvel,h.ctrl,h.xfrc_applied,h.qfrc_applied,h.actuator_force]):raise FloatingPointError('Nonfinite Warp raw state/force')
            if np.linalg.norm(h.qpos[idx['oq']+3:idx['oq']+7])<1e-9:raise FloatingPointError('Invalid zero object quaternion')
            ci=control_index_for_raw_state(k)
            state={'index':k,'raw_time_s':float(h.time),'expected_time_s':k*.005,'joint_rad':h.qpos[idx['qa']].copy(),'joint_velocity_rad_s':h.qvel[idx['va']].copy(),'full_qpos':h.qpos.copy(),'full_qvel':h.qvel.copy(),'object_pose_w':h.qpos[idx['oq']:idx['oq']+7].copy(),'object_twist_native':h.qvel[idx['ov']:idx['ov']+6].copy(),'ctrl_rad':h.ctrl[idx['aids']].copy(),'full_ctrl_native':h.ctrl.copy(),'control_interval_used':ci,'ctrl_exact':bool(np.array_equal(h.ctrl[idx['aids']].astype(np.float32).view(np.uint32),idx['schedule'][ci].view(np.uint32))),'actuator_force_native':h.actuator_force[idx['aids']].copy(),'force_phase':'startup_forward_only' if k==0 else 'just_completed_solver_preintegration','force_solver_time_s':0. if k==0 else previous_time,'force_prestate_index':0 if k==0 else k-1,'external_force_max_abs':float(max(np.max(np.abs(h.xfrc_applied)),np.max(np.abs(h.qfrc_applied)))),'mocap_pos_w':h.mocap_pos.copy(),'mocap_quat_wxyz':h.mocap_quat.copy(),'nacon':int(data.nacon.numpy().reshape(-1)[0]),'nefc':int(h.nefc)}
            c=capture_contacts(model,h,idx['obj'],object_geoms,hand_geoms)
            measured={'step':k,'phase':'startup_forward_only' if k==0 else 'just_completed_solver_preintegration','prestate_index':0 if k==0 else k-1,'poststate_index':k,'solver_t_s':0. if k==0 else previous_time,'poststate_t_s':float(h.time),'control_interval_used':ci,'ctrl_solver_native':h.ctrl[idx['aids']].copy(),'actuator_force_solver_native':h.actuator_force[idx['aids']].copy(),'solver_object_body_pose_w':np.r_[h.xpos[idx['obj']],h.xquat[idx['obj']]],'solver_object_COM_w':h.xipos[idx['obj']].copy(),'penet_category_m':c['penetration_by_category_m'],'penetration_m':c['penetration_m'],'object_environment_contacts':int(c['object_environment_contacts']),'measured_hand_object':convert(c)}
            for i,contact in enumerate(h.contact):
                if not np.isfinite(contact.dist) or not np.isfinite(contact.pos).all() or not np.isfinite(contact.frame).all():raise FloatingPointError('Nonfinite solved contact geometry')
                force6=np.zeros(6);mujoco.mj_contactForce(model,h,i,force6)
                if not np.isfinite(force6).all():raise FloatingPointError('Nonfinite solved contact force')
                rawcontacts.append({'step':k,'solver_t_s':measured['solver_t_s'],'poststate_t_s':float(h.time),'phase':measured['phase'],'control_interval_used':ci,'geom':list(map(int,contact.geom)),'pos_w':contact.pos.copy().tolist(),'frame_rows_w':contact.frame.reshape(3,3).copy().tolist(),'distance_m':float(contact.dist),'efc_address':int(contact.efc_address),'force6_contact_frame':force6.tolist()})
            if k:solver.append(measured)
            else:write(output/'startup_contact_measurement.json',convert(measured))
            states.append(state)
            if k==0:
                fk=np.array([h.xpos[i] if kind=='body' else h.site_xpos[i] for kind,i in idx['points']]);err=float(np.max(np.abs(fk-idx['expected_fk32'])))
                init={'startup_FK_vs_requested_FP32_max_abs_m':err,'initial_full_qpos_exact_expected_FP32':bool(np.array_equal(h.qpos.astype(np.float32),idx['expected_full_qpos32'])),'initial_full_qvel_exact_expected_FP32':bool(np.array_equal(h.qvel.astype(np.float32),idx['expected_full_qvel32'])),'initial_ctrl_exact':state['ctrl_exact'],'mocap_default_FP32_exact':bool(np.array_equal(h.mocap_pos.astype(np.float32),host.mocap_pos.astype(np.float32)) and np.array_equal(h.mocap_quat.astype(np.float32),host.mocap_quat.astype(np.float32))),'initial_velocities_are_not_old_zero_rest':bool(np.any(h.qvel)),'readback_status':'MEASURED_ONCE_AFTER_STARTUP_FORWARD'}
                write(output/'startup_initialization_checks.json',init)
                if err>1e-6 or not all(v for key,v in init.items() if key not in ('startup_FK_vs_requested_FP32_max_abs_m','readback_status')):raise ValueError('Startup differs from new FP32 nonzero initialization contract')
            runtime['physics_steps']=k
            if k%10==0:write(output/'runtime.json',runtime)
        sample(0,0.)
        for k in range(1,61):
            previous_time=states[-1]['raw_time_s'];deadline()
            if is_target_update_before(k):
                ci=control_index_for_solver_step(k);full_ctrl=host.ctrl.astype(np.float32).copy();full_ctrl[idx['aids']]=idx['schedule'][ci]
                # Both H0 andT0 execute14 identical-position update operations.
                wp.copy(data.ctrl,wp.array(full_ctrl[None],dtype=data.ctrl.dtype,device='cuda:0'));runtime['postinit_ctrl_writes']+=1
                got=data.ctrl.numpy().reshape(1,-1)[0].copy();exact=bool(np.array_equal(got[idx['aids']].astype(np.float32).view(np.uint32),idx['schedule'][ci].view(np.uint32)))
                events.append({'event':'before_solver_step_control_update','before_step':k,'prestate_index':k-1,'raw_prestate_time_s':previous_time,'previous_raw_state_ctrl_interval':control_index_for_raw_state(k-1),'new_solver_ctrl_interval':ci,'selected_planned_FP32_rad':idx['schedule'][ci].tolist(),'actual_ctrl_readback_native':got.tolist(),'position_ctrl_ids':idx['aids'].tolist(),'FP32_bitexact':exact})
                if not exact:raise ValueError('Actual control write/readback mismatch')
            runtime['step_calls']+=1;mjwarp.step(wp_model,data);sample(k,previous_time)
        # End-only sticky read does not change any preceding physics operation.
        runtime['overflow_reads']=1
        try:flags=data.overflow.numpy().reshape(-1).copy();overflow_error=None
        except Exception as exc:flags=None;overflow_error=repr(exc)
        capacity=classify_capacity(flags,states,int(data.contact.geom.shape[0]),int(data.efc.force.shape[-1]));capacity.update(observed_utc=utc(),read_error=overflow_error);write(output/'capacity_audit.json',capacity)
        runtime['status']='completed_engineering_new_dynamic_diagnostic'
    except BaseException as exc:
        error=exc;runtime['status']='failed_engineering_partial';runtime['error']=repr(exc)
    finally:
        if states:
            np.savez_compressed(output/'raw_states.npz',**{key:np.asarray([s[key] for s in states]) for key in states[0]},selected_planned15_FP32_rad=idx['schedule'],q_init_requested_FP32_rad=idx['q32'],qd_init_requested_FP32_rad_s=idx['qd32'],joint_lower_rad=model.jnt_range[idx['jids'],0],joint_upper_rad=model.jnt_range[idx['jids'],1],position_ctrl_ids=idx['aids'])
        write(output/'solver_samples.json',convert(solver));(output/'raw_contacts.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in rawcontacts),encoding='utf-8');(output/'control_events.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in events),encoding='utf-8')
        complete=len(states)==61 and len(solver)==60 and len(events)==14 and runtime['physics_steps']==60 and runtime['step_calls']==60 and runtime['postinit_ctrl_writes']==14 and error is None
        result=gates(states,solver,capacity,p,reference,complete,idx['force_lower'],idx['force_upper'],idx['default_wrist_z']) if 'idx' in locals() else {'status':'INPUT_ENGINEERING_FAILURE','short_dynamic_pass':None,'physics_inference_qualified':False}
        result.update(complete=complete,raw_state_count=len(states),solver_sample_count=len(solver),control_event_count=len(events),plan_sha256=sha(plan_path),worker_sha256=sha(__file__),physics_steps=runtime['physics_steps'],control_condition=p['control_condition'],retarget_variant_id=VARIANT,physical_reference_qualified=False,known_contact_reference_labels_created=False)
        write(output/'evaluation.json',result);runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(output/'runtime.json',runtime)
    if error is not None:raise error
    print(json.dumps({'engineering':runtime['status'],'short_dynamic_pass':result['short_dynamic_pass'],'physics_steps':60,'capacity':capacity},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path);parser.add_argument('--output',type=Path);modes=parser.add_mutually_exclusive_group();modes.add_argument('--run',action='store_true');modes.add_argument('--validate-only',action='store_true');modes.add_argument('--describe-contract',action='store_true');args=parser.parse_args()
    if args.describe_contract:print(json.dumps(describe_contract(),indent=2));return
    if args.plan is None:parser.error('--plan is required; missing future artifacts are never synthesized')
    if not args.run:
        os.environ['CUDA_VISIBLE_DEVICES']='-1';*_,report=load_inputs(args.plan.resolve(strict=True));print(json.dumps(report,indent=2));return
    if args.output is None:parser.error('Explicit new --output is required for authorized --run')
    run(args.plan.resolve(strict=True),args.output.resolve())
if __name__=='__main__':main()
