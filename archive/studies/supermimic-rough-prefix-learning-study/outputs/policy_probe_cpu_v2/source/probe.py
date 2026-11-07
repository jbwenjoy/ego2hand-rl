"""Instrument one new native-cadence zero-residual A0 without optimization."""
from __future__ import annotations
import json,re,time
import numpy as np
from .io import jsonable
from .metadata import write
from .bundle_v2 import rotation
GATES={'object_position_max_m':.02,'directed_shaft_error_max_deg':20.,'actual_joint_hard_excess_max_rad':.001,'penetration_any_category_max_m':.003,'object_nonhand_candidate_contact_count_max':0,'external_force_max_abs':0.,'clock_absolute_tolerance_s':2e-6,'target_rate_max_rad_s':15.,'actuator_load_ratio_max':1.01,'load_charge_max':.8,'charge_decay_time_s':2.,'drop_center_below_default_wrist_m':.15}
def grade(states,solver,capacity,reference,lower,upper,force_lower,force_upper,wrist_z):
    if len(states)!=61 or len(solver)!=60:return {'status':'INCOMPLETE_UNKNOWN','native_A0_pass':None}
    pose=np.asarray([s['object_pose_w'] for s in states]);q=np.asarray([s['joint_rad'] for s in states]);R=rotation(pose[:,3:]);pos=np.linalg.norm(pose[:,:3]-reference['object_COM_world_m'],axis=1)
    angle=np.rad2deg(np.arccos(np.clip(np.sum(R[:,:,0]*reference['object_rotation_w'][:,:,0],axis=1),-1,1)));primary=np.arange(0,61,4)
    excess=np.maximum(np.maximum(lower-q,q-upper),0);forces=np.asarray([s['actuator_force_solver_native'] for s in solver]);ratio=np.where(forces>=0,forces/force_upper,forces/force_lower);charge=np.zeros(20);maximum=charge.copy()
    for dt,row in zip(np.diff([s['raw_time_s'] for s in states]),ratio):charge=np.clip(charge+dt*(np.minimum(row,1)-charge/2.),0,1);maximum=np.maximum(maximum,charge)
    actual15=np.asarray([states[k]['ctrl_rad'] for k in [0,*range(5,58,4)]]);rate=np.abs(np.diff(actual15,axis=0)).max()/.02
    checks={'object_position_16':pos[primary].max()<=.02,'directed_shaft_16':angle[primary].max()<=20,'actual_q_all61':excess.max()<=.001,'penetration_solver60':max(s['penetration_m'] for s in solver)<=.003,'no_nonhand_object_contacts':sum(s['object_environment_contacts'] for s in solver)==0,'no_exogenous':max(s['external_force_max_abs'] for s in states)==0,'all_actual_u_exact':all(s['ctrl_exact'] for s in states),'actual_target_rate':rate<=15,'clock':max(abs(s['raw_time_s']-s['expected_time_s']) for s in states)<=2e-6,'load_ratio':ratio.max()<=1.01,'load_charge':maximum.max()<=.8,'drop':pose[:,2].min()>=wrist_z-.15,'capacity':capacity['status']=='OK'}
    return jsonable({'status':'COMPLETED_NATIVE_A0_QUALITY_RECORDED','native_A0_pass':all(checks.values()) if capacity['status']=='OK' else None,'gates':checks,'criteria':GATES,'max_object_position_error_16_m':pos[primary].max(),'max_object_position_error_61_m':pos.max(),'max_shaft_angle_16_deg':angle[primary].max(),'max_shaft_angle_61_deg':angle.max(),'max_actual_joint_excess_rad':excess.max(),'max_penetration_m':max(s['penetration_m'] for s in solver),'max_actuator_load_ratio':ratio.max(),'max_load_charge':maximum.max(),'max_actual_target_rate_rad_s':rate,'scope':'New Manager native-cadence singleprocess A0. Historical directWarp T0 remains separate. No learnedactor/full3s/CHORD/ego qualification.'})
def run_probe(env,runner,bundle,output,emit,deadline):
    import torch,warp as wp,mujoco,mujoco_warp as mjwarp
    from measured_contacts import capture_contacts
    if env.num_envs!=1:raise ValueError('Native A0 is one world only')
    model=env.sim.mj_model;host=mujoco.MjData(model);ref=bundle['arrays']['reference'];baseline=bundle['baseline_u15'];names=[f'robot/right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]
    jids=np.array([model.joint(n).id for n in names]);qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids];aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids]);obj=model.body('object/cube').id;oj=model.joint('object/cube_freejoint').id;oq=int(model.jnt_qposadr[oj]);ov=int(model.jnt_dofadr[oj])
    if np.any(model.body_ipos[obj]!=0):raise ValueError('Nonzero COM offset not qualified')
    hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])};object_geoms=set(np.flatnonzero(model.geom_bodyid==obj).tolist())
    lo=model.actuator_forcerange[aids,0].copy();hi=model.actuator_forcerange[aids,1].copy()
    if hasattr(model,'jnt_actfrclimited'):
        lo=np.maximum(lo,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,0],-np.inf));hi=np.minimum(hi,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,1],np.inf))
    states=[];solver=[];contacts=[];events=[];forward_calls=0;original_step=env.sim.step;original_forward=env.sim.forward;error=None;capacity={'status':'UNKNOWN'}
    def sample(index):
        if time.perf_counter()>deadline:raise TimeoutError('Native A0 finite deadline')
        wp.synchronize_device(env.device);mjwarp.get_data_into(host,model,env.sim.wp_data,world_id=0)
        if not all(np.isfinite(x).all() for x in (host.qpos,host.qvel,host.ctrl,host.actuator_force)):raise FloatingPointError('Nonfinite native A0 state')
        interval=0 if index==0 else (index-1)//4;expected=baseline[interval];actual=host.ctrl[aids].astype(np.float32)
        state={'index':index,'raw_time_s':float(host.time),'expected_time_s':index*.005,'joint_rad':host.qpos[qa].copy(),'joint_velocity_rad_s':host.qvel[va].copy(),'full_qpos':host.qpos.copy(),'full_qvel':host.qvel.copy(),'object_pose_w':host.qpos[oq:oq+7].copy(),'object_twist_native':host.qvel[ov:ov+6].copy(),'ctrl_rad':host.ctrl[aids].copy(),'ctrl_exact':bool(np.array_equal(actual.view(np.uint32),expected.view(np.uint32))),'actuator_force_native':host.actuator_force[aids].copy(),'external_force_max_abs':float(max(np.abs(host.xfrc_applied).max(),np.abs(host.qfrc_applied).max())),'mocap_pos_w':host.mocap_pos.copy(),'mocap_quat_wxyz':host.mocap_quat.copy(),'nacon':int(env.sim.wp_data.nacon.numpy().reshape(-1)[0]),'nefc':int(host.nefc)}
        if not state['ctrl_exact']:raise ValueError('Zeroresidual native actualu differs from frozen baseline')
        states.append(state)
        if index:
            c=capture_contacts(model,host,obj,object_geoms,hand);solver.append({'step':index,'prestate_index':index-1,'poststate_index':index,'solver_t_s':states[-2]['raw_time_s'],'poststate_t_s':state['raw_time_s'],'control_interval_used':interval,'ctrl_solver_native':host.ctrl[aids].copy(),'actuator_force_solver_native':host.actuator_force[aids].copy(),'solver_object_COM_w':host.xipos[obj].copy(),'solver_object_body_pose_w':np.r_[host.xpos[obj],host.xquat[obj]],'penetration_m':c['penetration_m'],'penet_category_m':c['penetration_by_category_m'],'object_environment_contacts':c['object_environment_contacts'],'measured_hand_object':c,'phase':'just_completed_solver_preintegration_before_Manager_forward'})
        for i,c in enumerate(host.contact):
            force=np.zeros(6);mujoco.mj_contactForce(model,host,i,force)
            if not np.isfinite(force).all() or not np.isfinite(c.dist) or not np.isfinite(c.pos).all() or not np.isfinite(c.frame).all():raise FloatingPointError('Nonfinite native A0 solved contact')
            contacts.append({'step':index,'geom':list(map(int,c.geom)),'distance_m':float(c.dist),'pos_w':c.pos.copy(),'frame_rows_w':c.frame.reshape(3,3).copy(),'force6_contact_frame':force,'phase':'startup_after_native_reset' if index==0 else 'preintegration_solver'})
    def step():
        if len(states)>60:raise ValueError('No61st physical step permitted')
        original_step();sample(len(states))
    def forward():
        nonlocal forward_calls
        forward_calls+=1;return original_forward()
    try:
        sample(0);initial=bundle['arrays']['init']
        default=mujoco.MjData(model);expected_q=default.qpos.astype(np.float32);expected_q[qa]=initial['q_init_requested_FP32_rad'];expected_q[oq:oq+7]=initial['object_pose_init_requested_FP32_w'];expected_v=np.zeros(model.nv,np.float32);expected_v[va]=initial['qd_init_requested_FP32_rad_s'];expected_v[ov:ov+3]=initial['object_bodyorigin_v_requested_FP32_world_m_s'];expected_v[ov+3:ov+6]=initial['object_angular_requested_FP32_body_rad_s']
        init_checks={'full_qpos_FP32_exact':np.array_equal(host.qpos.astype(np.float32),expected_q),'joint_qd_FP32_exact':np.array_equal(host.qvel[va].astype(np.float32),expected_v[va]),'object_world_linear_FP32_exact':np.array_equal(host.qvel[ov:ov+3].astype(np.float32),expected_v[ov:ov+3]),'body_omega_API_arithmetic_atol_1e_minus6':np.allclose(host.qvel[ov+3:ov+6],expected_v[ov+3:ov+6],rtol=0,atol=1e-6),'mocap_default_FP32_exact':np.array_equal(host.mocap_pos.astype(np.float32),default.mocap_pos.astype(np.float32)) and np.array_equal(host.mocap_quat.astype(np.float32),default.mocap_quat.astype(np.float32)),'initial_ctrl_exact':states[0]['ctrl_exact'],'no_external_forces':states[0]['external_force_max_abs']==0}
        write(output/'startup_initialization_checks.json',{'checks':init_checks,'body_omega_max_error_rad_s':np.max(np.abs(host.qvel[ov+3:ov+6]-expected_v[ov+3:ov+6])),'arithmetic_tolerance_not_physical_gate':True})
        np.savez_compressed(output/'requested_initialization_and_control.npz',full_qpos_requested_FP32=expected_q,full_qvel_requested_FP32=expected_v,mocap_pos_requested_FP32=default.mocap_pos.astype(np.float32),mocap_quat_requested_FP32=default.mocap_quat.astype(np.float32),selected_planned15_FP32_rad=baseline,position_ctrl_ids=aids,joint_qpos_ids=qa,joint_dof_ids=va,force_lower_Nm=lo,force_upper_Nm=hi)
        if not all(init_checks.values()):raise ValueError('Native A0 reset not same requested nonzero initialization')
        env.sim.step=step;env.sim.forward=forward
        for k in range(15):
            phase=env.command_manager.get_term('motion').time_steps.detach().cpu().numpy().copy()
            if not np.array_equal(phase,np.array([k])):raise ValueError('Native preaction phase mismatch')
            obs,reward,terminated,truncated,_=env.step(torch.zeros((1,20),device=env.device))
            last=env._rough_controller_pending[-1]
            events.append({'phase':k,'preaction_phase':phase,'postaction_phase':env.command_manager.get_term('motion').time_steps.detach().cpu().numpy().copy(),'terminated':terminated.detach().cpu().numpy(),'truncated':truncated.detach().cpu().numpy(),'reward':reward.detach().cpu().numpy(),'actualu':last['actual'].detach().cpu().numpy(),'residual':last['residual'].detach().cpu().numpy()})
            if bool(terminated.any()):raise FloatingPointError('Native A0 unstable hard termination; retain partial, no reset/retry')
            if k%3==2:emit(status='running',phase='native_A0',control_intervals=k+1,physics_steps=len(solver))
        if len(states)!=61 or len(solver)!=60 or forward_calls!=15:raise ValueError('Native A0 cadence/coverage differs')
        if not bool(events[-1]['truncated'][0]) or any(bool(e['truncated'][0]) for e in events[:-1]):raise ValueError('Native finite end must be timeout only at interval14')
        flags=env.sim.wp_data.overflow.numpy().reshape(-1).copy();cc=int(env.sim.wp_data.contact.geom.shape[0]);jc=int(env.sim.wp_data.efc.force.shape[-1])
        ok=flags.shape==(1,) and flags.dtype.kind in 'iu' and not np.any(flags) and max(s['nacon'] for s in states)<cc and max(s['nefc'] for s in states)<jc
        capacity={'status':'OK' if ok else 'UNKNOWN_OR_OVERFLOW','flags_by_world':flags,'read_count':1,'read_after_step':60,'contact_capacity':cc,'constraint_capacity':jc,'resets_during_probe':0}
    except BaseException as exc:error=exc
    finally:
        env.sim.step=original_step;env.sim.forward=original_forward
        if states:np.savez_compressed(output/'raw_states.npz',**{k:np.asarray([s[k] for s in states]) for k in states[0]},joint_lower_rad=model.jnt_range[jids,0],joint_upper_rad=model.jnt_range[jids,1],force_lower_Nm=lo,force_upper_Nm=hi,selected_planned15_FP32_rad=baseline)
        write(output/'solver_samples.json',solver);write(output/'control_events.json',events);write(output/'capacity_audit.json',capacity)
        (output/'raw_contacts.jsonl').write_text(''.join(json.dumps(jsonable(x),allow_nan=False)+'\n' for x in contacts),encoding='utf-8')
        result=grade(states,solver,capacity,ref,model.jnt_range[jids,0],model.jnt_range[jids,1],lo,hi,float(host.mocap_pos[0,2]))
        if error is not None:result.update(status='INCOMPLETE_OR_INVALID_ENGINEERING_UNKNOWN',native_A0_pass=None,error=repr(error))
        result.update(complete=error is None and len(states)==61 and len(solver)==60,physical_steps=len(solver),native_postinterval_forward_calls=forward_calls,control_intervals=len(events),optimizer_updates=0,resets_after_start=0,physical_reference_qualified=False,model_data_readback='All61aftersteps before15nativepostintervalforwards')
        write(output/'evaluation.json',result)
    if error is not None:raise error
    return {'status':'COMPLETED_NATIVE_A0_ENGINEERING_QUALITY_RECORDED','evaluation':result,'optimizer_updates':0}
