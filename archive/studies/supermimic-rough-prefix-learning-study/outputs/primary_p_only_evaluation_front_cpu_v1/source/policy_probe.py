"""Instrument a new native-cadence deterministic policy evaluation, without learning."""
from __future__ import annotations
import json,re,time
import numpy as np
import hashlib,importlib.util
from .io import jsonable
from .metadata import write
from .bundle_v2 import rotation
from .probe import grade as frozen_A0_grade
from .controller import PURE_PATH,PURE_SHA256
GATES={'object_position_max_m':.02,'directed_shaft_error_max_deg':20.,'actual_joint_hard_excess_max_rad':.001,'penetration_any_category_max_m':.003,'object_nonhand_candidate_contact_count_max':0,'external_force_max_abs':0.,'clock_absolute_tolerance_s':2e-6,'target_rate_max_rad_s':15.,'actuator_load_ratio_max':1.01,'load_charge_max':.8,'charge_decay_time_s':2.,'drop_center_below_default_wrist_m':.15}

def expected_control(action,previous_residual,baseline,lower,upper,bias,pure):
    """Independent CPU FP32 prediction, using the actual frozen C2 function."""
    import torch
    a=np.asarray(action,np.float32);previous=np.asarray(previous_residual,np.float32)
    lo=np.asarray(lower,np.float32);hi=np.asarray(upper,np.float32);b=np.asarray(bias,np.float32)
    if any(v.shape!=(1,20) for v in (a,previous,b)) or any(v.shape!=(20,) for v in (np.asarray(baseline),lo,hi)) or not all(np.isfinite(v).all() for v in (a,previous,baseline,lo,hi,b)) or np.any(b):raise ValueError('Expected finite one-world20 control with zero bias')
    residual=np.float32(.5)*np.clip(np.float32(.5)*np.clip(a,-1.,1.),np.float32(-.6),np.float32(.6))+np.float32(.5)*previous
    proposed=np.asarray(baseline,np.float32)[None]+residual-b
    bounded=np.minimum(np.maximum(proposed,lo),hi)
    actual=pure(torch.from_numpy(bounded.copy()),torch.from_numpy(lo.copy()),torch.from_numpy(hi.copy()),.02).detach().cpu().numpy().copy()
    return {'residual':residual,'proposed':proposed,'hard_bounded':bounded,'actual':actual}

def policy_rate(initial_ctrl,events):
    """15 planned20ms differences: startup->learned0, then learned0..14."""
    initial=np.asarray(initial_ctrl,float).reshape(20)
    if len(events)!=15:raise ValueError('Rate qualification requires all15 learned applied targets')
    actual=np.asarray([e['actualu'] for e in events],float).reshape(15,20)
    if not np.isfinite(actual).all() or not np.isfinite(initial).all():raise ValueError('Nonfinite applied control')
    delta=np.diff(np.vstack((initial,actual)),axis=0)
    return {'max_rate_rad_s':float(np.abs(delta).max()/.02),'startup_jump_max_rad':float(np.abs(delta[0]).max()),'startup_rate_rad_s':float(np.abs(delta[0]).max()/.02),'subsequent_updates_max_rate_rad_s':float(np.abs(delta[1:]).max()/.02),'rate_difference_count':15,'startup_rate_convention':'initial ctrl0 to first learned interval0 target divided by declared20ms; not an instantaneous physical derivative'}

def grade_policy(states,solver,capacity,reference,lower,upper,force_lower,force_upper,wrist_z,events):
    result=frozen_A0_grade(states,solver,capacity,reference,lower,upper,force_lower,force_upper,wrist_z)
    result.pop('native_A0_pass',None)
    if len(states)!=61 or len(solver)!=60 or len(events)!=15:
        result.update(status='INCOMPLETE_UNKNOWN',native_policy_pass=None);return result
    rate=policy_rate(states[0]['ctrl_rad'],events)
    result['gates']['actual_target_rate']=rate['max_rate_rad_s']<=GATES['target_rate_max_rad_s']
    result.update(status='COMPLETED_NATIVE_POLICY_QUALITY_RECORDED',native_policy_pass=all(result['gates'].values()) if capacity['status']=='OK' else None,max_actual_target_rate_rad_s=rate['max_rate_rad_s'],startup_and_update_rate=rate,scope='Deterministic learned Actor evaluation under native Manager cadence; full horizon despite quality failure. No training/normalizer update or method-benefit claim.')
    return result

def classify_capacity(flags,contact_count,constraint_count,contact_capacity,constraint_capacity):
    f=np.asarray(flags)
    if f.shape!=(1,) or f.dtype.kind not in 'iu':return 'UNKNOWN'
    if np.any(f):return 'OVERFLOW'
    return 'OK' if contact_count<contact_capacity and constraint_count<constraint_capacity else 'UNKNOWN'

def _policy_obs(obs):
    import torch
    if not isinstance(obs,dict) or not isinstance(obs.get('policy'),torch.Tensor) or tuple(obs['policy'].shape)!=(1,155) or not bool(torch.isfinite(obs['policy']).all()):raise ValueError('Native Actor requires one finite155 policy observation')
    return obs['policy']
def run_policy_probe(env,reader,actor,bundle,output,emit,deadline,*,initial_obs):
    import torch,warp as wp,mujoco,mujoco_warp as mjwarp
    from measured_contacts import capture_contacts
    if env.num_envs!=1 or env.cfg.auto_reset:raise ValueError('Native policy evaluation is one world with auto_resetFalse')
    if reader.env is not env or actor.training:raise ValueError('Use the attached native reader and evaluation-mode Actor')
    obs=initial_obs;_policy_obs(obs)
    actor_before={k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}
    if hashlib.sha256(PURE_PATH.read_bytes()).hexdigest()!=PURE_SHA256:raise ValueError('Frozen C2 source changed')
    spec=importlib.util.spec_from_file_location('_policy_probe_frozen_C2',PURE_PATH);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
    term=env.action_manager.get_term('joint_pos')
    if term.cfg.alpha!=.5 or term.cfg.scale!=.5 or term.cfg.residual_clip!=(-.6,.6) or term._mask_enabled:raise ValueError('Native EMA/action control contract differs')
    lower,upper=term.target_position_limits;lower=lower[0].detach().cpu().numpy().copy();upper=upper[0].detach().cpu().numpy().copy()
    bias=term._entity.data.encoder_bias[:,term._target_ids].detach().cpu().numpy().copy()
    model=env.sim.mj_model;host=mujoco.MjData(model);ref=bundle['arrays']['reference'];baseline=bundle['baseline_u15'];names=[f'robot/right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5)]
    jids=np.array([model.joint(n).id for n in names]);qa=model.jnt_qposadr[jids];va=model.jnt_dofadr[jids];aids=np.array([np.flatnonzero(model.actuator_trnid[:,0]==j).item() for j in jids]);obj=model.body('object/cube').id;oj=model.joint('object/cube_freejoint').id;oq=int(model.jnt_qposadr[oj]);ov=int(model.jnt_dofadr[oj])
    if np.any(model.body_ipos[obj]!=0):raise ValueError('Nonzero COM offset not qualified')
    hand={i for i in range(model.ngeom) if re.fullmatch(r'robot/(right_palm_collision|right_finger[1-5]_(link[1-4]|tip_sensor)_col)',model.geom(i).name) and (model.geom_contype[i] or model.geom_conaffinity[i])};object_geoms=set(np.flatnonzero(model.geom_bodyid==obj).tolist())
    lo=model.actuator_forcerange[aids,0].copy();hi=model.actuator_forcerange[aids,1].copy()
    if hasattr(model,'jnt_actfrclimited'):
        lo=np.maximum(lo,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,0],-np.inf));hi=np.minimum(hi,np.where(model.jnt_actfrclimited[jids],model.jnt_actfrcrange[jids,1],np.inf))
    expected_current=bundle['arrays']['init']['u_planned0_FP32_rad'].copy()
    states=[];solver=[];contacts=[];events=[];forward_calls=0;original_step=env.sim.step;original_forward=env.sim.forward;error=None;capacity={'status':'UNKNOWN','read_count':0};pending_start=len(env._rough_controller_pending)
    def sample(index):
        if time.perf_counter()>deadline:raise TimeoutError('Native policy evaluation finite deadline')
        wp.synchronize_device(env.device);mjwarp.get_data_into(host,model,env.sim.wp_data,world_id=0)
        if not all(np.isfinite(x).all() for x in (host.qpos,host.qvel,host.ctrl,host.actuator_force)):raise FloatingPointError('Nonfinite native policy evaluation state')
        interval=0 if index==0 else (index-1)//4;expected=expected_current;actual=host.ctrl[aids].astype(np.float32)
        state={'index':index,'raw_time_s':float(host.time),'expected_time_s':index*.005,'joint_rad':host.qpos[qa].copy(),'joint_velocity_rad_s':host.qvel[va].copy(),'full_qpos':host.qpos.copy(),'full_qvel':host.qvel.copy(),'object_pose_w':host.qpos[oq:oq+7].copy(),'object_twist_native':host.qvel[ov:ov+6].copy(),'ctrl_rad':host.ctrl[aids].copy(),'ctrl_exact':bool(np.array_equal(actual.view(np.uint32),expected.view(np.uint32))),'actuator_force_native':host.actuator_force[aids].copy(),'external_force_max_abs':float(max(np.abs(host.xfrc_applied).max(),np.abs(host.qfrc_applied).max())),'mocap_pos_w':host.mocap_pos.copy(),'mocap_quat_wxyz':host.mocap_quat.copy(),'nacon':int(env.sim.wp_data.nacon.numpy().reshape(-1)[0]),'nefc':int(host.nefc)}
        if not state['ctrl_exact']:raise ValueError('Actual native ctrl differs from independent baseline+EMA/C2 prediction')
        states.append(state)
        if index:
            c=capture_contacts(model,host,obj,object_geoms,hand);solver.append({'step':index,'prestate_index':index-1,'poststate_index':index,'solver_t_s':states[-2]['raw_time_s'],'poststate_t_s':state['raw_time_s'],'control_interval_used':interval,'ctrl_solver_native':host.ctrl[aids].copy(),'actuator_force_solver_native':host.actuator_force[aids].copy(),'solver_object_COM_w':host.xipos[obj].copy(),'solver_object_body_pose_w':np.r_[host.xpos[obj],host.xquat[obj]],'penetration_m':c['penetration_m'],'penet_category_m':c['penetration_by_category_m'],'object_environment_contacts':c['object_environment_contacts'],'measured_hand_object':c,'phase':'just_completed_solver_preintegration_before_Manager_forward'})
        for i,c in enumerate(host.contact):
            force=np.zeros(6);mujoco.mj_contactForce(model,host,i,force)
            if not np.isfinite(force).all() or not np.isfinite(c.dist) or not np.isfinite(c.pos).all() or not np.isfinite(c.frame).all():raise FloatingPointError('Nonfinite native policy evaluation solved contact')
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
        if not all(init_checks.values()):raise ValueError('Native policy evaluation reset not same requested nonzero initialization')
        env.sim.step=step;env.sim.forward=forward
        for k in range(15):
            phase=env.command_manager.get_term('motion').time_steps.detach().cpu().numpy().copy()
            if not np.array_equal(phase,np.array([k])):raise ValueError('Native preaction phase mismatch')
            prior_residual=term.processed_action.detach().cpu().numpy().copy()
            with torch.no_grad():
                raw_action=actor(_policy_obs(obs))
                if tuple(raw_action.shape)!=(1,20) or not bool(torch.isfinite(raw_action).all()):raise FloatingPointError('Nonfinite or malformed learned action')
                action=raw_action.clamp(-1,1)
            action_cpu=action.detach().cpu().numpy().copy()
            prediction=expected_control(action_cpu,prior_residual,baseline[k],lower,upper,bias,pure.target_inset)
            expected_current=prediction['actual'][0].copy()
            obs,reward,terminated,truncated,_=env.step(action)
            _policy_obs(obs)
            if len(env._rough_controller_pending)!=pending_start+k+1:raise ValueError('Expected exactly one cached-target process per interval')
            last=env._rough_controller_pending[-1]
            event={'phase':k,'preaction_phase':phase,'postaction_phase':env.command_manager.get_term('motion').time_steps.detach().cpu().numpy().copy(),'terminated':terminated.detach().cpu().numpy(),'truncated':truncated.detach().cpu().numpy(),'reward':reward.detach().cpu().numpy(),'actualu':last['actual'].detach().cpu().numpy(),'cached':last['cached'].detach().cpu().numpy(),'residual':last['residual'].detach().cpu().numpy(),'prior_residual':prior_residual,'baseline':baseline[k].copy(),'action':action_cpu,'actor_raw_action':raw_action.detach().cpu().numpy(),'independent_expected':prediction}
            events.append(event)
            same=lambda a,b:np.array_equal(np.asarray(a,np.float32).view(np.uint32),np.asarray(b,np.float32).view(np.uint32))
            if not same(event['residual'],prediction['residual']) or not same(event['actualu'],prediction['actual']) or not same(event['cached'],prediction['actual']):raise ValueError('Independent native EMA/cache/actual target mismatch')
            if not np.array_equal(event['postaction_phase'],np.array([k+1])):raise ValueError('Native postaction observation phase mismatch')
            if bool(terminated.any()):raise FloatingPointError('Native policy evaluation unstable hard termination; retain partial, no reset/retry')
            if k%3==2:emit(status='running',phase='native_policy',control_intervals=k+1,physics_steps=len(solver))
        if len(states)!=61 or len(solver)!=60 or forward_calls!=15:raise ValueError('Native policy evaluation cadence/coverage differs')
        if not bool(events[-1]['truncated'][0]) or any(bool(e['truncated'][0]) for e in events[:-1]):raise ValueError('Native finite end must be timeout only at interval14')
        capacity={'status':'UNKNOWN','read_count':1,'read_after_step':60,'resets_during_probe':0}
        try:
            flags=env.sim.wp_data.overflow.numpy().copy();cc=int(env.sim.wp_data.contact.geom.shape[0]);jc=int(env.sim.wp_data.efc.force.shape[-1])
            capacity.update(status=classify_capacity(flags,max(s['nacon'] for s in states),max(s['nefc'] for s in states),cc,jc),flags_by_world=flags,raw_flags_dtype=str(flags.dtype),raw_flags_shape=list(flags.shape),contact_capacity=cc,constraint_capacity=jc)
        except BaseException as exc:capacity['read_error']=repr(exc)
        if capacity['status']!='OK':raise ValueError('Policy probe capacity is unqualified; retain complete or partial data without retry')
    except BaseException as exc:error=exc
    finally:
        env.sim.step=original_step;env.sim.forward=original_forward
        if states:np.savez_compressed(output/'raw_states.npz',**{k:np.asarray([s[k] for s in states]) for k in states[0]},joint_lower_rad=model.jnt_range[jids,0],joint_upper_rad=model.jnt_range[jids,1],force_lower_Nm=lo,force_upper_Nm=hi,selected_planned15_FP32_rad=baseline,learned_applied15_FP32_rad=np.asarray([e['actualu'][0] for e in events],np.float32))
        write(output/'solver_samples.json',solver);write(output/'control_events.json',events);write(output/'capacity_audit.json',capacity)
        (output/'raw_contacts.jsonl').write_text(''.join(json.dumps(jsonable(x),allow_nan=False)+'\n' for x in contacts),encoding='utf-8')
        norm_unchanged=all(torch.equal(value,actor.state_dict()[key].detach().cpu()) for key,value in actor_before.items())
        write(output/'actor_state_evidence.json',{'all_weights_and_normalizer_unchanged':norm_unchanged,'norm_update_calls':0,'optimizer_objects_created':0,'action_noise_draws':0,'Actor_mode':'eval','policy_observations':'initial_obs supplied from one orchestration reset, then env.step returned policy; no extra observation compute','frozen_C2_source_sha256':PURE_SHA256})
        if not norm_unchanged and error is None:error=ValueError('Actor weights or normalization mutated in evaluation')
        result=grade_policy(states,solver,capacity,ref,model.jnt_range[jids,0],model.jnt_range[jids,1],lo,hi,float(host.mocap_pos[0,2]),events)
        if error is not None:result.update(status='INCOMPLETE_OR_INVALID_ENGINEERING_UNKNOWN',native_policy_pass=None,error=repr(error))
        result.update(complete=error is None and len(states)==61 and len(solver)==60,physical_steps=len(solver),native_postinterval_forward_calls=forward_calls,control_intervals=len(events),optimizer_updates=0,resets_after_start=0,physical_reference_qualified=False,model_data_readback='All61aftersteps before15nativepostintervalforwards')
        write(output/'evaluation.json',result)
    if error is not None:raise error
    return {'status':'COMPLETED_NATIVE_POLICY_ENGINEERING_QUALITY_RECORDED','evaluation':result,'optimizer_updates':0}
