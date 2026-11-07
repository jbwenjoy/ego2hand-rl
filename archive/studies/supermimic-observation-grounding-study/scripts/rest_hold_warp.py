"""One frame0, zero-velocity, constant-C2-target rest diagnostic; default CPU preflight.

Only explicit --run and an authorized hash-bound plan may import Warp/startGPU.
One initial Warpforward, then60step/readback samples, no per-step forward.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from runtime_io import write_json
STARTED=time.perf_counter()
FIELDS=('body_mass','body_inertia','body_iquat','geom_size','geom_pos','geom_friction','actuator_gainprm','actuator_biasprm')
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,d):
    write_json(p,d)
def deadline():
    if time.perf_counter()-STARTED>40:raise TimeoutError('40s whole-worker softdeadline; preserve partial, no retry')
def convert(x):
    if hasattr(x,'tolist'):return x.tolist()
    if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [convert(v) for v in x]
    return x

def load_inputs(plan_path):
    """CPU loads/checks/FK/projection only. No native forward/step or Warp."""
    import numpy as np
    import mujoco
    import torch
    from target_inset import target_inset,validate_inset_domain
    p=read(plan_path);protocol=read(p['protocol']['path'])
    for key in ('model','context','candidate','firstframe','robot_kinematics','protocol'):
        if p[key]!=p['source_pins'][key]:raise ValueError('Input alias is not bound to its frozenpin: '+key)
    for name,pin in p['source_pins'].items():
        if sha(pin['path'])!=pin['sha256']:raise ValueError('Frozen input/source changed: '+name)
    if sha(p['protocol']['path'])!=p['protocol']['sha256']:raise ValueError('Protocol changed')
    if p['schema']!='observation_frame0_rest_hold_plan:1' or p['frame']!=0 or p['steps']!=60 or p['dt_s']!=.005 or p['duration_s']!=.3 or p['nworld']!=1 or p['nconmax']!=192 or p['njmax']!=4096:raise ValueError('Fixed rest recipe changed')
    if protocol['diagnostic_gates']!=p['gates'] or protocol['physics']['context_fields']!=list(FIELDS):raise ValueError('Fixed gates/context fields differ')
    model=mujoco.MjModel.from_binary_path(p['model']['path']);host=mujoco.MjData(model)
    if abs(model.opt.timestep-.005)>1e-12:raise ValueError('MJB timestep differs')
    with np.load(p['candidate']['path'],allow_pickle=False) as z:
        q64=z['joint_pos_rad'][0].copy();pose64=z['mapped_object_pose_w'][0].copy();expected_fk64=z['fk_hand_points_w'][0].copy()
        if z['joint_pos_rad'].shape!=(151,20) or z['mapped_object_pose_w'].shape!=(151,7):raise ValueError('Candidate layout differs')
    first=read(p['firstframe']['path'])
    if not np.array_equal(q64,first['q_rad']) or not np.array_equal(pose64,first['object_pose_w_SI_wxyz']):raise ValueError('Frame0 binding differs')
    if q64.shape!=(20,) or pose64.shape!=(7,) or not np.isfinite(q64).all() or not np.isfinite(pose64).all() or abs(np.linalg.norm(pose64[3:])-1)>1e-6:raise ValueError('Invalid frame0 geometry')
    names=first['joint_names'];jids=np.array([model.joint(n).id for n in names]);qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids];aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids]);obj=model.body('object/cube').id;oj=model.joint('object/cube_freejoint').id;oq=int(model.jnt_qposadr[oj]);ov=int(model.jnt_dofadr[oj])
    if model.nu!=20 or not np.all(model.actuator_ctrllimited[aids]) or not np.all(model.jnt_limited[jids]):raise ValueError('Fixed bounded actuator/joint mapping unavailable')
    if not np.array_equal(np.asarray(p['joint_lower_rad']),model.jnt_range[jids,0]) or not np.array_equal(np.asarray(p['joint_upper_rad']),model.jnt_range[jids,1]):raise ValueError('Actual-state gate hardlimits differ from MJB')
    if model.opt.integrator!=mujoco.mjtIntegrator.mjINT_IMPLICITFAST:raise ValueError('Preintegration force timing contract requires original IMPLICITFAST')
    lower=np.maximum(model.jnt_range[jids,0],model.actuator_ctrlrange[aids,0]);upper=np.minimum(model.jnt_range[jids,1],model.actuator_ctrlrange[aids,1])
    if (q64<model.jnt_range[jids,0]).any() or (q64>model.jnt_range[jids,1]).any():raise ValueError('Initial q cannot be projected silently')
    lo,hi=torch.tensor(lower,dtype=torch.float32),torch.tensor(upper,dtype=torch.float32);validate_inset_domain(lo,hi,.020)
    q32=q64.astype(np.float32);pose32=pose64.astype(np.float32);target=target_inset(torch.from_numpy(q32.copy()),lo,hi,.020).numpy().copy()
    if not np.array_equal(target,np.clip(q32,lower.astype(np.float32)+np.float32(.020),upper.astype(np.float32)-np.float32(.020))):raise ValueError('C2 FP32 projection differs')
    host.qpos[qa]=q32;host.qpos[oq:oq+7]=pose32;host.qvel[:]=0;host.ctrl[aids]=target;host.xfrc_applied[:]=0;host.qfrc_applied[:]=0
    # No reset/keyframe/event/scene compilation. MJBdefault mocap is untouched.
    mocap=first['mocap']
    for entry in mocap:
        i=entry['mocap_id']
        if not np.array_equal(host.mocap_pos[i],entry['position_w']) or not np.array_equal(host.mocap_quat[i],entry['quaternion_wxyz']):raise ValueError('Default MJBmocap differs')
    actual={};context_report={}
    with np.load(p['context']['path'],allow_pickle=False) as z:
        for field in FIELDS:
            value=z[field];nominal=getattr(model,field)
            if value.shape[1:]!=nominal.shape or not len(value) or not np.isfinite(value).all() or not np.array_equal(value,np.broadcast_to(value[:1],value.shape)):raise ValueError('Nonfixed physical field '+field)
            actual[field]=value[0].copy();context_report[field]={'shape':list(value[0].shape),'dtype':str(value.dtype),'nominal_max_abs_difference':float(np.max(np.abs(nominal-value[0])))}
    # Independent CPU FK only; do not use it to refresh any Warp measurement.
    sys.path.insert(0,str(Path(p['robot_kinematics']['path']).parent))
    from retarget_palmroot_cpu import RobotKinematics
    robot=RobotKinematics(p['model']['path']);robot.data.qpos[oq:oq+7]=pose64;robot.last_q=None;fk64=robot.forward(q64)[0]
    if np.max(np.abs(fk64-expected_fk64))>1e-12:raise ValueError('Candidate FK differs from originalMJB')
    object_FK_position_error=float(np.max(np.abs(robot.data.xpos[obj]-pose64[:3])))
    if object_FK_position_error>1e-12:raise ValueError('Object initial FK differs from candidate pose')
    robot.data.qpos[oq:oq+7]=pose32;robot.data.mocap_pos[:]=host.mocap_pos.astype(np.float32);robot.data.mocap_quat[:]=host.mocap_quat.astype(np.float32);robot.last_q=None;fk32=robot.forward(q32)[0]
    info={'status':'passed_CPU_inputs_FK_FP32_C2_contract','physics_started':False,'frame':0,'q_cast_max_abs_rad':float(np.max(np.abs(q64-q32))),'pose_cast_max_abs':float(np.max(np.abs(pose64-pose32))),'CPU_FK_vs_candidate_max_abs_m':float(np.max(np.abs(fk64-expected_fk64))),'CPU_object_FK_position_error_m':object_FK_position_error,'CPU_FK_FP32_input_vs_candidate_max_abs_m':float(np.max(np.abs(fk32-expected_fk64))),'C2_clamped_components':int(np.count_nonzero(target!=q32)),'context_fields':context_report,'joint_names':names,'initial_mocap_source':'Original MJB default exact match to firstframe parameterfile','actual_state_gate_limits_match_MJB':True,'integrator':'IMPLICITFAST','query_numerical_options':{'ccd_tolerance':float(model.opt.ccd_tolerance),'ccd_iterations':int(model.opt.ccd_iterations)},'torch_CUDA_initialized':torch.cuda.is_initialized(),'Warp_imported':'warp' in sys.modules or 'mujoco_warp' in sys.modules}
    if info['q_cast_max_abs_rad']>protocol['initialization_tolerances']['candidate_FP32_cast_q_abs_rad'] or info['pose_cast_max_abs']>protocol['initialization_tolerances']['candidate_FP32_cast_pose_abs']:raise ValueError('Frame0cast exceeds frozen tolerance')
    return p,protocol,model,host,actual,dict(qa=qa,va=va,aids=aids,obj=obj,oq=oq,ov=ov,lower=lower,upper=upper,target=target,q32=q32,pose32=pose32,expected_fk64=expected_fk64,expected_fk32=fk32,points=robot.points),info

def gates(states,solver,capacity,p,complete):
    import numpy as np
    if not complete:return {'status':'INCOMPLETE_ENGINEERING_NO_HOLD_CONCLUSION','rest_hold_pass':None,'physics_inference_qualified':False,'complete':False}
    g=p['gates'];q=np.array([x['joint_rad'] for x in states]);pose=np.array([x['object_pose_w'] for x in states]);pos=np.linalg.norm(pose[:,:3]-pose[0,:3],axis=1)
    from measured_contacts import quat_matrix
    axis=quat_matrix(pose[:,3:])[...,0];cos=np.clip(axis@axis[0],-1,1);angle=np.rad2deg(np.arccos(cos));lower=np.array(p['joint_lower_rad']);upper=np.array(p['joint_upper_rad']);excess=np.maximum(np.maximum(lower-q,q-upper),0)
    conditions={'object_position':float(pos.max())<=g['object_position_max_m'],'shaft_angle':float(angle.max())<=g['directed_shaft_error_max_deg'],'actual_joint_state':float(excess.max())<=g['actual_joint_hard_excess_max_rad'],'penetration':max(s['penetration_m'] for s in solver)<=g['penetration_any_category_max_m'],'nonhand_object_contacts':sum(s['object_environment_contacts'] for s in solver)==0,'no_exogenous':max(s['external_force_max_abs'] for s in states)==0,'constant_actual_u':all(s['ctrl_exact'] for s in states),'clock':max(abs(s['raw_time_s']-s['expected_time_s']) for s in states)<=g['clock_absolute_tolerance_s'],'raw_coverage':len(states)==61 and len(solver)==60,'finite_raw':bool(np.isfinite(q).all() and np.isfinite(pose).all()),'capacity':capacity.get('status')=='OK'}
    qualified=capacity.get('status')=='OK'
    return {'status':'COMPLETED_NEW_REST_DIAGNOSTIC_ONLY' if qualified else 'COMPLETED_CAPACITY_UNQUALIFIED_HOLD_UNKNOWN','rest_hold_pass':all(conditions.values()) if qualified else None,'physics_inference_qualified':qualified,'gates':conditions,'max_object_position_error_m':float(pos.max()),'max_shaft_angle_deg':float(angle.max()),'max_actual_joint_excess_rad':float(excess.max()),'max_penetration_m':max(s['penetration_m'] for s in solver),'scope':'One new0.3srest intervention, not original3stask success or observation/force/reference qualification. Contactforces sample preintegration t0..0.295; final0.3poststate contactgeometry is not refreshed. Capacity failure retains rawdata but makes hold verdict UNKNOWN.'}

def run(plan_path,output):
    import numpy as np
    import mujoco
    p=read(plan_path)
    if p.get('GPU_run_authorized') is not True:raise ValueError('Root GPU authorization is absent; default CPU only')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime={'status':'initializing','pid':os.getpid(),'parent_pid':os.getppid(),'started_utc':utc(),'plan_sha256':sha(plan_path),'source_sha256':sha(__file__),'physics_steps':0,'step_calls':0,'startup_Warp_forward_calls':0,'postinit_state_writes':0,'postinit_ctrl_writes':0,'overflow_reads':0}
    write(output/'runtime.json',runtime);states=[];solver=[];rawcontacts=[];capacity={'status':'UNKNOWN','reason':'No completed60step single sticky read yet'};error=None
    try:
        p,protocol,model,host,physical,idx,cpu=load_inputs(plan_path);write(output/'CPU_input_contract.json',cpu);deadline()
        import warp as wp
        import mujoco_warp as mjwarp
        from measured_contacts import capture_contacts
        wp.set_device('cuda:0');wp_model=mjwarp.put_model(model);context_readbacks={}
        def verify_context(stage):
            report={}
            for key,value in physical.items():
                got=getattr(wp_model,key).numpy().reshape((-1,)+value.shape)[0];context_readbacks[stage+'__'+key]=got.copy();expected=value.astype(got.dtype)
                exact=np.array_equal(got,expected);report[key]={'exact_expected_FP32':bool(exact),'max_abs_error':float(np.max(np.abs(got-expected)))}
                if not exact:raise ValueError('ActualWarp physicalfield differs '+key+' at '+stage)
            return report
        for key,value in physical.items():
            destination=getattr(wp_model,key);wp.copy(destination,wp.array(value[None],dtype=destination.dtype,device='cuda:0'))
        copies={'after_copy':verify_context('after_copy')};data=mjwarp.put_data(model,host,nworld=1,nconmax=192,njmax=4096);mjwarp.set_const_0(wp_model,data,restore=True);copies['after_set_const_0']=verify_context('after_set_const_0');write(output/'actual8field_copy.json',copies);np.savez_compressed(output/'expected_context_fields.npz',**physical);np.savez_compressed(output/'actual_context_applied.npz',**{key:context_readbacks['after_set_const_0__'+key] for key in FIELDS});np.savez_compressed(output/'actual_context_readbacks.npz',**context_readbacks);deadline()
        mjwarp.forward(wp_model,data);runtime['startup_Warp_forward_calls']=1
        readback=mujoco.MjData(model);object_geoms=set(np.flatnonzero(model.geom_bodyid==idx['obj']).tolist());import re
        hand_geoms={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])}
        def sample(k,previous_time):
            deadline();wp.synchronize_device('cuda:0');mjwarp.get_data_into(readback,model,data,world_id=0);h=readback
            if not all(np.isfinite(a).all() for a in [h.qpos,h.qvel,h.ctrl,h.xfrc_applied,h.qfrc_applied,h.actuator_force]):raise FloatingPointError('Nonfinite Warp raw state/force')
            if np.linalg.norm(h.qpos[idx['oq']+3:idx['oq']+7])<1e-9:raise FloatingPointError('Invalid zero object quaternion')
            state={'index':k,'raw_time_s':float(h.time),'expected_time_s':k*.005,'joint_rad':h.qpos[idx['qa']].copy(),'joint_velocity_rad_s':h.qvel[idx['va']].copy(),'full_qpos':h.qpos.copy(),'full_qvel':h.qvel.copy(),'object_pose_w':h.qpos[idx['oq']:idx['oq']+7].copy(),'object_twist_native':h.qvel[idx['ov']:idx['ov']+6].copy(),'ctrl_rad':h.ctrl[idx['aids']].copy(),'ctrl_exact':bool(np.array_equal(h.ctrl[idx['aids']].astype(np.float32),idx['target'])),'actuator_force_native':h.actuator_force[idx['aids']].copy(),'force_phase':'startup_forward_only' if k==0 else 'just_completed_solver_preintegration','force_solver_time_s':0. if k==0 else previous_time,'force_prestate_index':0 if k==0 else k-1,'external_force_max_abs':float(max(np.max(np.abs(h.xfrc_applied)),np.max(np.abs(h.qfrc_applied)))),'mocap_pos_w':h.mocap_pos.copy(),'mocap_quat_wxyz':h.mocap_quat.copy(),'nacon':int(data.nacon.numpy().reshape(-1)[0]),'nefc':int(h.nefc)}
            c=capture_contacts(model,h,idx['obj'],object_geoms,hand_geoms)
            measured={'step':k,'phase':'startup_forward_only' if k==0 else 'just_completed_solver_preintegration','prestate_index':0 if k==0 else k-1,'poststate_index':k,'solver_t_s':0. if k==0 else previous_time,'poststate_t_s':float(h.time),'actuator_force_solver_native':h.actuator_force[idx['aids']].copy(),'solver_object_body_pose_w':np.r_[h.xpos[idx['obj']],h.xquat[idx['obj']]],'solver_object_COM_w':h.xipos[idx['obj']].copy(),'penet_category_m':c['penetration_by_category_m'],'penetration_m':c['penetration_m'],'object_environment_contacts':int(c['object_environment_contacts']),'measured_hand_object':convert(c)}
            for i,contact in enumerate(h.contact):
                if not np.isfinite(contact.dist) or not np.isfinite(contact.pos).all() or not np.isfinite(contact.frame).all():raise FloatingPointError('Nonfinite solvedcontactgeometry')
                force6=np.zeros(6);mujoco.mj_contactForce(model,h,i,force6)
                if not np.isfinite(force6).all():raise FloatingPointError('Nonfinite solvedcontactforce')
                rawcontacts.append({'step':k,'solver_t_s':measured['solver_t_s'],'poststate_t_s':float(h.time),'phase':measured['phase'],'geom':list(map(int,contact.geom)),'pos_w':contact.pos.copy().tolist(),'frame_rows_w':contact.frame.reshape(3,3).copy().tolist(),'distance_m':float(contact.dist),'efc_address':int(contact.efc_address),'force6_contact_frame':force6.tolist()})
            if k:solver.append(measured)
            else:write(output/'startup_contact_measurement.json',convert(measured))
            states.append(state)
            if k==0:
                fk=np.array([h.xpos[i] if kind=='body' else h.site_xpos[i] for kind,i in idx['points']]);err=float(np.max(np.abs(fk-idx['expected_fk64'])))
                init={'startup_FK_vs_candidate_max_abs_m':err,'initial_q_exact_expected_FP32':bool(np.array_equal(h.qpos[idx['qa']].astype(np.float32),idx['q32'])),'initial_object_exact_expected_FP32':bool(np.array_equal(h.qpos[idx['oq']:idx['oq']+7].astype(np.float32),idx['pose32'])),'initial_qvel_all_zero':bool(not h.qvel.any()),'initial_ctrl_exact':state['ctrl_exact'],'mocap_default_FP32_exact':bool(np.array_equal(h.mocap_pos.astype(np.float32),host.mocap_pos.astype(np.float32)) and np.array_equal(h.mocap_quat.astype(np.float32),host.mocap_quat.astype(np.float32)))}
                write(output/'startup_initialization_checks.json',init)
                if err>1e-6 or not all(v for key,v in init.items() if key!='startup_FK_vs_candidate_max_abs_m'):raise ValueError('Startup differs from geometric frame0 / FP32cast contract')
            runtime['physics_steps']=k
            if k%10==0:write(output/'runtime.json',runtime)
        sample(0,0.)
        for k in range(1,61):
            previous_time=states[-1]['raw_time_s'];deadline();runtime['step_calls']+=1;mjwarp.step(wp_model,data);sample(k,previous_time)
        # Exactly one sticky-overflow read, after all60steps, never reset/clear.
        flags=data.overflow.numpy().reshape(-1);runtime['overflow_reads']=1;bitmask=int(np.bitwise_or.reduce(flags));contact_cap=int(data.contact.geom.shape[0]);constraint_cap=int(data.efc.force.shape[-1]);below=all(s['nacon']<contact_cap and s['nefc']<constraint_cap for s in states)
        capacity={'status':'OK' if bitmask==0 and len(flags)==1 and below else 'OVERFLOW','overflow_bitmask':bitmask,'flags_by_world':flags.tolist(),'read_count':1,'read_after_step':60,'worlds':len(flags),'resets_after_initialization':0,'counts_strictly_below_capacity':bool(below),'contact_capacity':contact_cap,'constraint_capacity':constraint_cap,'backend_field':'wp_data.overflow','observed_utc':utc()};write(output/'capacity_audit.json',capacity)
        runtime['status']='completed_engineering_new_rest_diagnostic'
    except BaseException as exc:
        error=exc;runtime['status']='failed_engineering_partial';runtime['error']=repr(exc)
    finally:
        if states:
            np.savez_compressed(output/'raw_states.npz',**{key:np.asarray([s[key] for s in states]) for key in states[0]},fixed_target_FP32=idx['target'],q_reference0_FP32=idx['q32'],joint_lower_rad=model.jnt_range[model.dof_jntid[idx['va']],0],joint_upper_rad=model.jnt_range[model.dof_jntid[idx['va']],1])
        write(output/'solver_samples.json',convert(solver));(output/'raw_contacts.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in rawcontacts),encoding='utf-8')
        complete=len(states)==61 and len(solver)==60 and runtime['physics_steps']==60 and error is None
        result=gates(states,solver,capacity,p,complete);result.update(complete=complete,raw_state_count=len(states),solver_sample_count=len(solver),plan_sha256=sha(plan_path),worker_sha256=sha(__file__),physics_steps=runtime['physics_steps'],C2_clamped_components=cpu['C2_clamped_components'] if 'cpu' in locals() else None,control_comparison_scope='C2 is applied once; this frame0 has0clampedcomponents and does not identify a filter treatment effect.',physical_reference_qualified=False,known_contact_reference_labels_created=False)
        write(output/'evaluation.json',result);runtime.update(finished_utc=utc(),wall_seconds=time.perf_counter()-STARTED);write(output/'runtime.json',runtime)
    if error is not None:raise error
    print(json.dumps({'engineering':runtime['status'],'rest_hold_pass':result['rest_hold_pass'],'physics_steps':60,'capacity':capacity},indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=ROOT/'configs/rest-hold-frame0-v1.json');parser.add_argument('--output',type=Path,default=ROOT/'outputs/rest_hold_frame0_v1');modes=parser.add_mutually_exclusive_group();modes.add_argument('--run',action='store_true');modes.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    if not args.run:
        os.environ['CUDA_VISIBLE_DEVICES']='-1';*_,report=load_inputs(args.plan);print(json.dumps(report,indent=2));return
    run(args.plan.resolve(),args.output.resolve())
if __name__=='__main__':main()
