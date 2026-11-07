"""Independent read-only audit of terminal QREF or static-preload rest controls.

Control-aware successor of the immutable QREF auditor bf6f074ad2840fd409543f039ac10b5b5d283cfefc5eaafbb2fe486cd737fc74.
Expected proposals come from hash-bound design inputs, never the raw header.

--self-test uses synthetic NumPy arrays only. Normal audit is run only after the
owner reports terminal completion; it never imports the worker, Warp or Torch.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def npz(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def expected_proposal(plan,candidate,lower64,upper64):
    q64=candidate['joint_pos_rad'][0];pose64=candidate['mapped_object_pose_w'][0]
    q32=q64.astype(np.float32);pose32=pose64.astype(np.float32)
    condition=plan['control_condition']
    if condition=='QREF_C2':return q32.copy(),{'known_condition':True}
    if condition!='STATIC_PD_PRELOAD_C2':raise ValueError('Unknown fixed control condition')
    pin=plan['preload']
    if plan['source_pins'].get('preload')!=pin or sha(pin['path'])!=pin['sha256']:
        raise ValueError('Preload source binding/hash differs')
    design=read(pin['path'])
    required=[plan[k] for k in ('candidate','model','context')]+[plan['source_pins']['pure_projection']]
    sources=design.get('source_pins',{})
    checks={'known_condition':True,
      'schema':design.get('schema')=='seeded_static_PD_preload:1',
      'feasible_not_dynamic':design.get('feasible') is True and design.get('no_dynamic_qualification') is True and design.get('physical_reference_qualified') is False,
      'canonical_candidate':design.get('canonical_candidate')==plan['candidate'] and design.get('canonical') is True,
      'required_source_bindings':all(p in sources.values() for p in required),
      'all_design_source_hashes':bool(sources) and all(sha(p['path'])==p['sha256'] for p in sources.values()),
      'C2_margin':design.get('C2_margin_rad')==.020,
      'q64':np.array_equal(q64,np.asarray(design['q0_rad'])),
      'pose64':np.array_equal(pose64,np.asarray(design['object_pose_w'])),
      'q32':np.array_equal(q32,np.asarray(design['q0_FP32_rad'],np.float32)),
      'pose32':np.array_equal(pose32,np.asarray(design['object_pose_w_FP32'],np.float32)),
      'intersection_lower64':np.array_equal(lower64,np.asarray(design['target_lower_rad'])),
      'intersection_upper64':np.array_equal(upper64,np.asarray(design['target_upper_rad'])),
      'inset_lower32':np.array_equal(lower64.astype(np.float32)+np.float32(.020),np.asarray(design['inset_lower_rad'],np.float32)),
      'inset_upper32':np.array_equal(upper64.astype(np.float32)-np.float32(.020),np.asarray(design['inset_upper_rad'],np.float32))}
    proposal=np.asarray(design['u0_rad'],np.float32)
    checks['proposal_finite_shape']=proposal.shape==(20,) and bool(np.isfinite(proposal).all())
    checks['proposal_exact_FP32']=np.array_equal(proposal.astype(np.float64),np.asarray(design['u0_rad'],np.float64))
    if not all(checks.values()):raise ValueError('Preload expected-input contract failed: '+str(checks))
    return proposal,checks
def error(a,b):return float(np.max(np.abs(np.asarray(a,float)-np.asarray(b,float)))) if np.size(a) else 0.
def finite(x):
    if isinstance(x,dict):return all(finite(v) for v in x.values())
    if isinstance(x,(list,tuple)):return all(finite(v) for v in x)
    if isinstance(x,(int,float,np.number)) and not isinstance(x,(bool,np.bool_)):return bool(np.isfinite(x))
    if isinstance(x,np.ndarray) and np.issubdtype(x.dtype,np.number):return bool(np.isfinite(x).all())
    return True
def matrix(q):
    q=np.asarray(q,float);norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if not np.isfinite(q).all() or np.any(norm<1e-9):raise ValueError('Invalid quaternion')
    q=q/norm;w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))
def capacity_class(cap,counts,n):
    if not cap or cap.get('status')=='UNKNOWN':return 'UNKNOWN'
    flags=cap.get('flags_by_world');mask=cap.get('overflow_bitmask')
    if not isinstance(flags,list) or len(flags)!=1 or type(flags[0]) is not int or type(mask) is not int:return 'UNKNOWN'
    if flags[0]<0 or mask<0:return 'UNKNOWN'
    if mask!=flags[0]:return 'UNKNOWN'
    if mask!=0:return 'OVERFLOW'
    if not all(k in cap for k in ('contact_capacity','constraint_capacity')):return 'UNKNOWN'
    if not counts or len(counts)!=n:return 'UNKNOWN'
    if any(a>=cap['contact_capacity'] or b>=cap['constraint_capacity'] for a,b in counts):return 'OVERFLOW'
    if cap.get('read_count')!=1 or cap.get('read_after_step')!=60 or cap.get('worlds')!=1 or cap.get('resets_after_initialization')!=0:return 'UNKNOWN'
    return 'OK'
def quality(raw,solver,cap,gates,complete,all_finite):
    if not complete:return {'rest_hold_pass':None,'physics_inference_qualified':False,'gates':None,'metrics':None}
    q=raw['joint_rad'];pose=raw['object_pose_w'];axis=matrix(pose[:,3:])[:,:,0]
    position=np.linalg.norm(pose[:,:3]-pose[0,:3],axis=-1)
    angle=np.rad2deg(np.arccos(np.clip(axis@axis[0],-1,1)))
    excess=np.maximum(np.maximum(raw['joint_lower_rad']-q,q-raw['joint_upper_rad']),0)
    metrics={'max_object_position_error_m':float(position.max()),'max_shaft_angle_deg':float(angle.max()),
      'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':max(s['penetration_m'] for s in solver)}
    flags={'object_position':metrics['max_object_position_error_m']<=gates['object_position_max_m'],
      'shaft_angle':metrics['max_shaft_angle_deg']<=gates['directed_shaft_error_max_deg'],
      'actual_joint_state':metrics['max_actual_joint_excess_rad']<=gates['actual_joint_hard_excess_max_rad'],
      'penetration':metrics['max_penetration_m']<=gates['penetration_any_category_max_m'],
      'nonhand_object_contacts':all(s['object_environment_contacts']==0 for s in solver),
      'no_exogenous':bool(np.all(raw['external_force_max_abs']==0)),
      'constant_actual_u':bool(np.array_equal(raw['ctrl_rad'].astype(np.float32),np.broadcast_to(raw['fixed_target_FP32'],raw['ctrl_rad'].shape))),
      'clock':bool(np.max(np.abs(raw['raw_time_s']-np.arange(61)*.005))<=gates['clock_absolute_tolerance_s']),
      'raw_coverage':len(q)==61 and len(solver)==60,'finite_raw':all_finite,'capacity':cap=='OK'}
    return {'rest_hold_pass':all(flags.values()) if cap=='OK' else None,'physics_inference_qualified':cap=='OK','gates':flags,'metrics':metrics}
def contact_reconstruction(rows,measurement,object_geoms,hand_geoms):
    depths={'hand_object':0.,'hand_self':0.,'other':0.};environment=0;points=[];normals=[];forces=[];torques=[];pairs=[];active=[]
    pose=np.asarray(measurement['solver_object_body_pose_w']);rotation=matrix(pose[3:]);max_depth=0.
    for c in rows:
        g1,g2=c['geom'];has_obj=g1 in object_geoms or g2 in object_geoms;has_hand=g1 in hand_geoms or g2 in hand_geoms
        category='hand_object' if has_obj and has_hand else 'hand_self' if g1 in hand_geoms and g2 in hand_geoms else 'other'
        depth=max(-c['distance_m'],0.);depths[category]=max(depths[category],depth);max_depth=max(max_depth,depth)
        if has_obj and not has_hand:environment+=1
        if has_obj and has_hand:
            sign=1. if g2 in object_geoms else -1.;frame=np.asarray(c['frame_rows_w']);f=np.asarray(c['force6_contact_frame'])
            points.append(rotation.T@(np.asarray(c['pos_w'])-pose[:3]));normals.append(rotation.T@(sign*frame[0]))
            forces.append(sign*frame.T@f[:3]);torques.append(sign*frame.T@f[3:]);pairs.append([g1,g2]);active.append(c['efc_address']>=0 and f[0]>1e-6)
    calc={'pos_o_m':np.asarray(points).reshape(-1,3),'normal_o':np.asarray(normals).reshape(-1,3),
      'force_w_N':np.asarray(forces).reshape(-1,3),'torque_at_contact_w_Nm':np.asarray(torques).reshape(-1,3),
      'geom_pair':np.asarray(pairs,dtype=int).reshape(-1,2),'active':np.asarray(active,dtype=bool)}
    checks={'depth_any':abs(measurement['penetration_m']-max_depth)<=1e-12,
      'depth_categories':all(abs(measurement['penet_category_m'][k]-v)<=1e-12 for k,v in depths.items()),
      'nonhand_candidates':measurement['object_environment_contacts']==environment}
    reported=measurement['measured_hand_object'];errors={}
    for k,value in calc.items():
        other=np.asarray(reported[k]).reshape(value.shape)
        errors[k]=error(value,other)
        checks['raw_contact_reconstruct_'+k]=bool(np.array_equal(value,other)) if k in ('geom_pair','active') else errors[k]<=1e-10
    return checks,errors,{'penetration_m':max_depth,'penetration_by_category_m':depths,'nonhand_candidates':environment,'hand_object_records':len(points),'active_hand_object_records':sum(active)}

def audit(plan_path,result,output):
    started=time.perf_counter();plan=read(plan_path);protocol=read(plan['protocol']['path'])
    # Never create the formal review directory until terminal evidence exists.
    if not result.is_dir():raise FileNotFoundError('No authoritative terminal output; do not inspect future paths')
    runtime=read(result/'runtime.json')
    if runtime.get('status') in ('initializing','running'):raise ValueError('Owner has not reached a terminal state')
    if output.exists():raise FileExistsError('Independent output must be new')
    raw=npz(result/'raw_states.npz');solver=read(result/'solver_samples.json');evaluation=read(result/'evaluation.json')
    cap=read(result/'capacity_audit.json') if (result/'capacity_audit.json').is_file() else {'status':'UNKNOWN'}
    lines=[json.loads(line) for line in (result/'raw_contacts.jsonl').read_text().splitlines() if line.strip()]
    n=len(raw['index']);complete=n==61 and len(solver)==60 and runtime.get('physics_steps')==runtime.get('step_calls')==60 and runtime['status']=='completed_engineering_new_rest_diagnostic'
    source={k:sha(v['path'])==v['sha256'] for k,v in plan['source_pins'].items()}
    source['runtime_plan']=runtime['plan_sha256']==evaluation['plan_sha256']==sha(plan_path)
    source['runtime_worker']=runtime['source_sha256']==evaluation['worker_sha256']==plan['source_pins']['worker']['sha256']
    checks={'source':source,'schema':{'protocol_gates':plan['gates']==protocol['diagnostic_gates'],
      'steps_dt':plan['steps']==60 and plan['dt_s']==.005,'raw_indices':np.array_equal(raw['index'],np.arange(n)),
      'one_startup_forward':runtime['startup_Warp_forward_calls']==1,'no_postinit_state_ctrl_write':runtime['postinit_state_writes']==runtime['postinit_ctrl_writes']==0,
      'design_not_video':protocol['is_real_ego'] is False and evaluation['physical_reference_qualified'] is False and evaluation['known_contact_reference_labels_created'] is False}}
    candidate=npz(plan['candidate']['path']);q0=candidate['joint_pos_rad'][0].astype(np.float32);pose0=candidate['mapped_object_pose_w'][0].astype(np.float32)
    import mujoco # Read-only compiled topology. No MjData/step/forward/solver.
    model=mujoco.MjModel.from_binary_path(plan['model']['path']);first=read(plan['firstframe']['path']);jids=np.asarray([model.joint(x).id for x in first['joint_names']]);qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids]
    aids=np.asarray([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids]);oj=model.joint('object/cube_freejoint').id;oq=model.jnt_qposadr[oj];ov=model.jnt_dofadr[oj]
    lower64=np.maximum(model.jnt_range[jids,0],model.actuator_ctrlrange[aids,0]);upper64=np.minimum(model.jnt_range[jids,1],model.actuator_ctrlrange[aids,1])
    proposed,design_checks=expected_proposal(plan,candidate,lower64,upper64)
    checks['expected_control_inputs']=design_checks
    lower=lower64.astype(np.float32);upper=upper64.astype(np.float32)
    fixed=np.clip(proposed,lower+np.float32(.020),upper-np.float32(.020));full0=model.qpos0.astype(np.float32);full0[qa]=q0;full0[oq:oq+7]=pose0
    checks['initial_and_target']={'initial_q_not_projected':np.array_equal(raw['joint_rad'][0].astype(np.float32),q0),
      'full_initial_qpos':np.array_equal(raw['full_qpos'][0].astype(np.float32),full0),'initial_pose':np.array_equal(raw['object_pose_w'][0].astype(np.float32),pose0),
      'explicit_zero_rest':bool(not raw['full_qvel'][0].any()),
      'initial_mocap':all(np.array_equal(raw['mocap_pos_w'][0,x['mocap_id']].astype(np.float32),np.asarray(x['position_w'],np.float32)) and np.array_equal(raw['mocap_quat_wxyz'][0,x['mocap_id']].astype(np.float32),np.asarray(x['quaternion_wxyz'],np.float32)) for x in first['mocap']),
      'fixed_mocap_all_states':np.array_equal(raw['mocap_pos_w'],np.broadcast_to(raw['mocap_pos_w'][0],raw['mocap_pos_w'].shape)) and np.array_equal(raw['mocap_quat_wxyz'],np.broadcast_to(raw['mocap_quat_wxyz'][0],raw['mocap_quat_wxyz'].shape)),
      'fixed_C2_once':np.array_equal(raw['fixed_target_FP32'],fixed),'qref_header':np.array_equal(raw['q_reference0_FP32'],q0),
      'proposal_header':np.array_equal(raw['control_proposed_once_FP32'],proposed),
      'preload_projection_identity':np.array_equal(fixed,proposed) if plan['control_condition']=='STATIC_PD_PRELOAD_C2' else True,
      'condition_header':evaluation['control_condition']==plan['control_condition'],
      'actual_u_all_states':np.array_equal(raw['ctrl_rad'].astype(np.float32),np.broadcast_to(fixed,raw['ctrl_rad'].shape)),
      'hardlimits_MJB':np.array_equal(raw['joint_lower_rad'],model.jnt_range[jids,0]) and np.array_equal(raw['joint_upper_rad'],model.jnt_range[jids,1]),
      'full_q_joint_mapping':np.array_equal(raw['full_qpos'][:,qa],raw['joint_rad']),
      'full_q_object_mapping':np.array_equal(raw['full_qpos'][:,oq:oq+7],raw['object_pose_w']),
      'full_qvel_mapping':np.array_equal(raw['full_qvel'][:,va],raw['joint_velocity_rad_s']) and np.array_equal(raw['full_qvel'][:,ov:ov+6],raw['object_twist_native'])}
    expected=npz(result/'expected_context_fields.npz');applied=npz(result/'actual_context_applied.npz');readbacks=npz(result/'actual_context_readbacks.npz');context=npz(plan['context']['path']);ctx={}
    for field in FIELDS:
        value=context[field][0].astype(np.float32)
        ctx[field]={label:bool(np.array_equal(array,value)) for label,array in [('expected',expected[field]),('after_copy',readbacks['after_copy__'+field]),('after_set_const_0',readbacks['after_set_const_0__'+field]),('applied',applied[field])]}
    checks['actual_context']=ctx
    object_body=model.body('object/cube').id;objects=set(np.flatnonzero(model.geom_bodyid==object_body).tolist())
    hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
    contactrows=[];phasechecks=[];reconstructed=[]
    startup=read(result/'startup_contact_measurement.json')
    for k,measurement in [(0,startup),*[(i+1,s) for i,s in enumerate(solver)]]:
        group=[c for c in lines if c['step']==k];cs,errors,summary=contact_reconstruction(group,measurement,objects,hand)
        expected_phase='startup_forward_only' if k==0 else 'just_completed_solver_preintegration';pre=max(k-1,0)
        phasechecks.append({'step':k,'step_index':measurement['step']==k,'phase':measurement['phase']==expected_phase,
          'prepost_index':measurement['prestate_index']==pre and measurement['poststate_index']==k,
          'solver_clock':abs(measurement['solver_t_s']-raw['raw_time_s'][pre])<=1e-12,'post_clock':abs(measurement['poststate_t_s']-raw['raw_time_s'][k])<=1e-12,
          'raw_force_phase':str(raw['force_phase'][k])==expected_phase and int(raw['force_prestate_index'][k])==pre and abs(raw['force_solver_time_s'][k]-raw['raw_time_s'][pre])<=1e-12,
          'actuator_force_pair':np.array_equal(raw['actuator_force_native'][k],measurement['actuator_force_solver_native']),
          'all_raw_contact_phase':all(c['phase']==expected_phase and abs(c['solver_t_s']-measurement['solver_t_s'])<=1e-12 and abs(c['poststate_t_s']-raw['raw_time_s'][k])<=1e-12 for c in group)})
        contactrows.append({'step':k,'checks':cs,'max_abs_errors':errors,'recomputed':summary})
        if k:reconstructed.append({'penetration_m':summary['penetration_m'],'object_environment_contacts':summary['nonhand_candidates']})
    checks['measurement_time']={str(r.pop('step')):r for r in phasechecks}
    checks['contact_reconstruction']={str(r['step']):r['checks'] for r in contactrows}
    counts=list(zip(raw['nacon'],raw['nefc']));classification=capacity_class(cap,counts,n)
    checks['capacity_metadata']={'record_matches_reclassification':classification==cap.get('status'),'one_read':runtime['overflow_reads']==1 if complete else True}
    all_finite=finite(raw) and finite(solver) and finite(lines)
    recomputed=quality(raw,reconstructed,classification,plan['gates'],complete,all_finite)
    compare={k:evaluation.get(k)==recomputed[k] for k in ('rest_hold_pass','physics_inference_qualified')}
    if complete:
        compare['gate_dict']=evaluation['gates']==recomputed['gates'];compare.update({k:abs(evaluation[k]-v)<=1e-10 for k,v in recomputed['metrics'].items()})
    checks['evaluation_recomputed']=compare
    watchdog=result.with_name(result.name+'_watchdog');process=read(watchdog/'state.json');preflight=read(watchdog/'preflight.json')
    command=process['command'];os_identity=process['actual_process_OS_observation']
    supervisor_source=Path(plan['source_pins']['supervisor']['path']).read_text()
    lifecycle_source=Path(plan['source_pins']['reviewed_lifecycle']['path']).read_text()
    checks['supervisor_identity_and_scope']={'terminal_exit0':process['status']=='completed' and process['exit_code']==0 and process['timeout'] is False,
      'actual_PID':process['actual_pid']==runtime['pid']==os_identity['pid'] and os_identity['status']=='exact_OS_identity_observed',
      'parent_PID':runtime['parent_pid']==os_identity['parent_pid']==process['launcher_pid'],
      'command_plan':Path(command[command.index('--plan')+1]).resolve()==plan_path.resolve(),
      'command_output':Path(command[command.index('--output')+1]).resolve()==result.resolve(),
      'command_worker':Path(command[2]).resolve()==Path(plan['source_pins']['worker']['path']).resolve(),
      'preflight_hash':preflight['plan_sha256']==sha(plan_path) and preflight['worker_sha256']==plan['source_pins']['worker']['sha256'],
      'limits':process['hard_seconds']==45 and plan['soft_worker_seconds']==40 and process['automatic_retry'] is False,
      'shared_lock_source_contract':protocol['bounded_execution']['shared_GPU_lock']=='D:/workspace/supermimic-chord-study/.locks/single_gpu_training.lock' and 'with life.gpu_lock():' in supervisor_source and 'single_gpu_training.lock' in lifecycle_source,
      'plan_snapshot_exact':sha(watchdog/'plan_snapshot.json')==sha(plan_path)}
    def all_checks(x):return all(all_checks(v) for v in x.values()) if isinstance(x,dict) else bool(x)
    valid=all_checks(checks)
    report={'status':'INDEPENDENT_RAW_RECONSTRUCTION_PASS' if valid else 'INDEPENDENT_REVIEW_FAILED',
      'complete':complete,'integrity_checks_passed':valid,'checks':checks,'raw_state_count':n,'solver_sample_count':len(solver),
      'raw_contact_count':len(lines),'recomputed':recomputed,'capacity_reclassified':classification,'contact_rows':contactrows,
      'C2_clamped_components':int(np.count_nonzero(proposed!=fixed)),
      'target_components_different_from_initial_q':int(np.count_nonzero(q0!=fixed)),
      'control_condition':plan['control_condition'],'physical_reference_qualified':False,'GPU_used_for_audit':False,
      'cost':{'worker_wall_seconds':runtime['wall_seconds'],'supervisor_child_wall_seconds':process['elapsed_s'],'GPU_kernel_busy_seconds':None,'exact_mutex_held_seconds':None},
      'actual_process_OS_identity':os_identity,
      'scope':'Single designed zero-rest intervention. No3s task, video velocity, grasp/force-prediction, continuous safety or real-ego qualification. Native/generated geometry at solver prestate is not refreshed at final.3s.',
      'source_sha256':sha(__file__),'plan_sha256':sha(plan_path),'result_hashes':{p.name:sha(p) for p in result.iterdir() if p.is_file()},'CPU_audit_wall_seconds':time.perf_counter()-started}
    output.mkdir(parents=True,exist_ok=False);(output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
    print(report['status']);return valid

def self_test(out):
    raw={'joint_rad':np.zeros((61,20)),'joint_lower_rad':np.full(20,-1.),'joint_upper_rad':np.ones(20),
      'object_pose_w':np.tile([0.,0.,1.,1.,0.,0.,0.],(61,1)),'external_force_max_abs':np.zeros(61),
      'ctrl_rad':np.zeros((61,20)),'fixed_target_FP32':np.zeros(20,np.float32),'raw_time_s':np.arange(61)*.005}
    solver=[{'penetration_m':0.,'object_environment_contacts':0} for _ in range(60)]
    g={'object_position_max_m':.02,'directed_shaft_error_max_deg':20.,'actual_joint_hard_excess_max_rad':.001,'penetration_any_category_max_m':.003,'clock_absolute_tolerance_s':2e-6}
    checks={'synthetic_pass':quality(raw,solver,'OK',g,True,True)['rest_hold_pass'] is True,
      'unknown_not_false':quality(raw,solver,'UNKNOWN',g,True,True)['rest_hold_pass'] is None,
      'overflow_not_false':quality(raw,solver,'OVERFLOW',g,True,True)['rest_hold_pass'] is None,
      'partial_not_false':quality(raw,solver,'OK',g,False,True)['rest_hold_pass'] is None}
    import copy
    for name,mutate in [('position',lambda r,s:r['object_pose_w'].__setitem__((60,0),.021)),('joint',lambda r,s:r['joint_rad'].__setitem__((60,0),1.00101)),('ctrl',lambda r,s:r['ctrl_rad'].__setitem__((60,0),.01)),('contact',lambda r,s:s[-1].update(penetration_m=.00301))]:
        r,s=copy.deepcopy(raw),copy.deepcopy(solver);mutate(r,s);checks[name+'_failure_preserved']=quality(r,s,'OK',g,True,True)['rest_hold_pass'] is False
    cap={'status':'OK','flags_by_world':[0],'overflow_bitmask':0,'contact_capacity':192,'constraint_capacity':4096,'read_count':1,'read_after_step':60,'worlds':1,'resets_after_initialization':0}
    checks['count_saturation_overflow']=capacity_class(cap,[(192,1)]*61,61)=='OVERFLOW'
    checks['missing_flags_unknown']=capacity_class({**cap,'flags_by_world':None},[(0,0)]*61,61)=='UNKNOWN'
    checks['no_Warp_Torch']=not any(k in sys.modules for k in ('torch','warp','mujoco_warp'))
    out.mkdir(parents=True,exist_ok=False);(out/'checks.json').write_text(json.dumps({'status':'passed' if all(checks.values()) else 'failed','checks':checks,'source_sha256':sha(__file__),'GPU_used':False},indent=2)+'\n')
    print(json.dumps(checks));return all(checks.values())
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path);p.add_argument('--result',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:ok=self_test(a.output)
    else:
        if not a.plan or not a.result:p.error('Audit requires explicit terminal --plan and --result')
        ok=audit(a.plan,a.result,a.output)
    raise SystemExit(0 if ok else 1)
if __name__=='__main__':main()
