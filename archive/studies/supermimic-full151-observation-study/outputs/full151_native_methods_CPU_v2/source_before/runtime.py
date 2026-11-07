"""Scoped full151 native runtime: analytic velocities, selected reset and end.

No environment, CUDA tensor, simulator, optimizer or training is constructed on
import. This new mode keeps native reader FD diagnostics and source files intact.
Use outside full151_controller so its inner control/save scopes compose normally.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy,hashlib,inspect,json,textwrap
from pathlib import Path
from .metadata import MODE, VARIANT, DESCRIPTOR_SCHEMA
from .controller import NFRAME, NINTERVAL, LAST_INTERVAL

NATIVE_RUNNER_SHA='b19d9ac80e23e4f55b2eb64e458c05516fc5dd4ffb13e1a3050ce9fea2669d46'
JOINT_NAMES=tuple(f'right_finger{f}_joint{j}' for f in range(1,6) for j in range(1,5))


def _torch():
    import torch
    return torch


def _tensor(value,*,device,dtype=None):
    torch=_torch();return torch.as_tensor(value,device=device,dtype=dtype or torch.float32).clone()


def _finite(value,name):
    torch=_torch()
    if not bool(torch.isfinite(value).all()):raise FloatingPointError('Nonfinite full151 '+name)
    return value


def phase_active(memory):
    return memory['time'].squeeze(-1)<NINTERVAL


def reference_timeout(env,command_name='motion'):
    """Before command advancement: interval149 ends at observation150."""
    motion=env.command_manager.get_term(command_name)
    return motion.time_steps>=LAST_INTERVAL


def masked_critic_mse(prediction,target,active):
    """Inactive finite placeholders have zero gradient; preserve minibatch steps."""
    torch=_torch();_finite(prediction,'criticprediction');_finite(target,'critictarget')
    delta=torch.where(active.bool(),prediction-target,torch.zeros_like(prediction))
    return delta.square().sum()/active.sum().clamp_min(1)


def _validated_bundle(bundle,descriptor):
    import numpy as np
    if set(bundle)!={'reference','init','control'}:raise ValueError('Exact separate reference/init/control arrays required')
    if (descriptor.get('schema')!=DESCRIPTOR_SCHEMA or descriptor.get('mode')!=MODE
            or descriptor.get('retarget_variant_id')!=VARIANT
            or descriptor.get('physical_reference_qualified') is not False):raise ValueError('Exact new known-rough full151 descriptor required; prefix16/104 does not qualify')
    r,i,c=bundle['reference'],bundle['init'],bundle['control']
    required={'time_s':(NFRAME,),'planned_time601_s':(601,),
              'q_goal_rad':(NFRAME,20),'analytic_qd151_rad_s':(NFRAME,20),
              'mapped_object_pose_w':(NFRAME,7),'object_v_world_m_s':(601,3),
              'object_omega_world_rad_s':(601,3)}
    for key,shape in required.items():
        if np.asarray(r[key]).shape!=shape or not np.isfinite(r[key]).all():raise ValueError('Reference shape/nonfinite: '+key)
    if not np.array_equal(r['time_s'],np.arange(NFRAME)*.02) or not np.array_equal(r['planned_time601_s'],np.arange(601)*.005):raise ValueError('Exact151/601 full3second clocks required, no padded or repeated prefix')
    for key,shape in {'q_init_requested_FP32_rad':(20,),'qd_init_requested_FP32_rad_s':(20,),'object_pose_init_requested_FP32_w':(7,),'object_bodyorigin_v_requested_FP32_world_m_s':(3,),'object_angular_requested_FP32_world_rad_s':(3,),'object_angular_requested_FP32_body_rad_s':(3,),'u_planned0_FP32_rad':(20,)}.items():
        if np.asarray(i[key]).shape!=shape or np.asarray(i[key]).dtype!=np.float32 or not np.isfinite(i[key]).all():raise ValueError('Exact finite FP32 init field required: '+key)
    baseline=c['u_postC2_150_FP32_rad']
    if np.asarray(baseline).shape!=(NINTERVAL,20) or np.asarray(baseline).dtype!=np.float32 or not np.isfinite(baseline).all():raise ValueError('Exactly150 finite FP32 baseline commands required')
    if not np.array_equal(i['q_init_requested_FP32_rad'],np.asarray(r['q_goal_rad'][0],np.float32)) or not np.array_equal(i['qd_init_requested_FP32_rad_s'],np.asarray(r['analytic_qd151_rad_s'][0],np.float32)) or not np.array_equal(i['u_planned0_FP32_rad'],baseline[0]):raise ValueError('Initialization/reference/baseline mismatch')
    if not np.array_equal(i['object_pose_init_requested_FP32_w'],np.asarray(r['mapped_object_pose_w'][0],np.float32)) or not np.array_equal(i['object_bodyorigin_v_requested_FP32_world_m_s'],np.asarray(r['object_v_world_m_s'][0],np.float32)) or not np.array_equal(i['object_angular_requested_FP32_world_rad_s'],np.asarray(r['object_omega_world_rad_s'][0],np.float32)):raise ValueError('Object initialization must preserve the independently declared pose and analytic world velocities')
    return {key:{k:np.array(v,copy=True) for k,v in value.items()} for key,value in bundle.items()}


def install_velocity_authority(motion,bundle):
    """Called after original fixed-wrist rebase and before the first reset."""
    torch=_torch();r=bundle['reference'];device=motion.device
    if motion.group.frame_dt!=.02 or motion.group.trajectory_end.tolist()!=[NFRAME] or tuple(motion.cfg.joint_names)!=JOINT_NAMES:raise ValueError('Only single native151/50Hz/joint-order command is supported')
    expected=_tensor(r['q_goal_rad'],device=device)
    if not torch.equal(motion._frames['joint_pos'],expected):raise ValueError('Native command q151 differs from frozen full151 variant')
    if motion._frames['link_pos'].shape!=(NFRAME,21,3):raise ValueError('Native151 BODY21 required')
    motion._full151_runtime_native_velocity_diagnostics={name:motion._frames[name].clone() for name in ('joint_vel','object_linvel_w','object_angvel_w')}
    tables={'joint_vel':r['analytic_qd151_rad_s'],'object_linvel_w':r['object_v_world_m_s'][::4],'object_angvel_w':r['object_omega_world_rad_s'][::4]}
    for name,value in tables.items():motion._frames[name]=_finite(_tensor(value,device=device),'velocitytable '+name)
    motion._full151_runtime_velocity_authority={'kind':'analytic_Bspline_joint_and_frozen_quintic_object_world','native_reader_FD_preserved_in_group':True,'frames':NFRAME,'install_stage':'after_original_fixedwrist_before_firstreset'}


def _ids(env,env_ids):
    torch=_torch()
    result=torch.arange(env.num_envs,device=env.device) if env_ids is None else torch.as_tensor(env_ids,device=env.device,dtype=torch.long)
    if result.ndim!=1 or len(result.unique())!=len(result) or bool(((result<0)|(result>=env.num_envs)).any()):raise ValueError('Unique selected reset environment IDs required')
    return result


def reset_readback(env,env_ids=None):
    """Read raw actual state/controller tensors without an extra forward/sense."""
    ids=_ids(env,env_ids);robot=env.scene['robot'];obj=env.scene['object'];term=env.action_manager.get_term('joint_pos');data=env.sim.data
    qa=robot.indexing.joint_q_adr.long();va=robot.indexing.joint_v_adr.long();oq=obj.indexing.free_joint_q_adr.long();ov=obj.indexing.free_joint_v_adr.long();aids=term._position_ctrl_ids.long()
    return {'env_ids':ids.clone(),'generation':env._full151_runtime_episode_generation[ids].clone(),'full_qpos':data.qpos[ids].clone(),'full_qvel':data.qvel[ids].clone(),'joint_pos_rad':data.qpos[ids][:,qa].clone(),'joint_vel_rad_s':data.qvel[ids][:,va].clone(),'object_pose_w':data.qpos[ids][:,oq].clone(),'object_twist_native':data.qvel[ids][:,ov].clone(),'actual_ctrl':data.ctrl[ids][:,aids].clone(),'cached_target':term._targets[ids].clone(),'EMA_residual':term._processed_actions[ids].clone(),'action':env.action_manager.action[ids].clone(),'previous_action':env.action_manager.prev_action[ids].clone(),'mocap_pos':data.mocap_pos[ids].clone(),'mocap_quat':data.mocap_quat[ids].clone(),'time':data.time[ids].clone() if data.time.ndim else data.time.clone(),'phase':env.command_manager.get_term('motion').time_steps[ids].clone()}


def _restore_initial_state(env,ids,bundle):
    """Selected worlds only; world-angular Entity API, no q projection/PD step."""
    torch=_torch();i=bundle['init'];motion=env.command_manager.get_term('motion');robot=env.scene['robot'];obj=env.scene['object'];term=env.action_manager.get_term('joint_pos');n=len(ids)
    if not n:return
    motion.trajectory_ids[ids]=0;motion.time_steps[ids]=0;motion._episode_start_steps[ids]=0
    joint_ids,found=robot.find_joints(JOINT_NAMES,preserve_order=True)
    if tuple(found)!=JOINT_NAMES:raise ValueError('Robot reset joint mapping differs')
    q=_tensor(i['q_init_requested_FP32_rad'],device=env.device).expand(n,-1);qd=_tensor(i['qd_init_requested_FP32_rad_s'],device=env.device).expand(n,-1)
    # Native fixed-wrist command adds env_origins once; do not substitute the
    # source demonstrator wrist or rotate frozen world velocities again.
    wrist=motion.wrist_pose_w[ids].clone();robot.write_mocap_pose_to_sim(wrist,env_ids=ids)
    robot.write_joint_state_to_sim(q,qd,joint_ids=joint_ids,env_ids=ids)
    pose=_tensor(i['object_pose_init_requested_FP32_w'],device=env.device).expand(n,-1).clone();pose[:,:3]+=env.scene.env_origins[ids]
    v=_tensor(i['object_bodyorigin_v_requested_FP32_world_m_s'],device=env.device).expand(n,-1);omega=_tensor(i['object_angular_requested_FP32_world_rad_s'],device=env.device).expand(n,-1)
    obj.write_root_state_to_sim(torch.cat((pose,v,omega),-1),env_ids=ids)
    env.action_manager.reset(env_ids=ids)
    u0=_tensor(i['u_planned0_FP32_rad'],device=env.device).expand(n,-1)
    proposed_all=term._targets.clone();proposed_all[ids]=u0
    if not torch.equal(term.limit_target(proposed_all)[ids],u0):raise ValueError('Requestedu0 outside native hard/ctrl bounds; no repair')
    term._targets[ids]=u0;term._targets_valid[ids]=True
    # Preserve nonselected targets/history. This flush writes the term's entire
    # cached target tensor, whose untouched rows retain their original values.
    term.apply_actions();env.scene.write_data_to_sim();env.sim.forward();env.sim.sense()
    env._full151_runtime_episode_generation[ids]+=1
    env._full151_runtime_last_reset=reset_readback(env,ids)
    actual=env._full151_runtime_last_reset
    if not torch.equal(actual['joint_pos_rad'],q) or not torch.equal(actual['joint_vel_rad_s'],qd) or not torch.equal(actual['object_pose_w'],pose) or not torch.equal(actual['actual_ctrl'],u0) or not torch.equal(actual['cached_target'],u0) or bool(actual['EMA_residual'].any()) or bool(actual['action'].any()) or bool(actual['previous_action'].any()):raise ValueError('Actual reset readback differs from selected new initialization')
    # The Entity world-angular API converts to the freejoint's body convention.
    expected_body=_tensor(i['object_angular_requested_FP32_body_rad_s'],device=env.device).expand(n,-1)
    if not torch.allclose(actual['object_twist_native'][:,3:],expected_body,rtol=0,atol=1e-6) or not torch.equal(actual['object_twist_native'][:,:3],v):raise ValueError('Actual native object twist differs from declared world/body contract')
    return actual


def make_policy_update(original):
    """Pin and minimally adapt the real native method, keeping its draw/update order.

    The generated method is inspectable through __full151_source__ and an exact
    replacement list. No file or original function global is modified. This is
    a new finite-reference objective shared by every newmode arm.
    """
    source_path=Path(inspect.getsourcefile(original)).resolve()
    if hashlib.sha256(source_path.read_bytes()).hexdigest()!=NATIVE_RUNNER_SHA:raise ValueError('Native runner source differs; review finite-end adapter explicitly')
    before=textwrap.dedent(inspect.getsource(original));source=before
    replacements=[
        ('    xs = []\n','    full151_active_rows = []\n    full151_critic_denominators = []\n    full151_start_phases = m["time"].detach().clone()\n    xs = []\n'),
        ('        for t in range(horizon):\n','        for t in range(horizon):\n            full151_active = _full151_phase_active(m)\n            full151_active_rows.append(full151_active)\n            _full151_finite(x, "imagined_input")\n            _full151_finite(history, "actor_history")\n'),
        ('            torque_kwargs = {"limits": m["limits"]} if self.is_pen else {}\n','            _full151_finite(x1, "WM_output_including_inactive")\n            _full151_finite(contact, "contact_output_including_inactive")\n            x1 = torch.where(full151_active[:, None], x1, x.detach())\n            contact = torch.where(full151_active[:, None], contact, torch.zeros_like(contact))\n            torque_kwargs = {"limits": m["limits"]} if self.is_pen else {}\n'),
        ('reference_end = (m["time"] + 1 >= m["end"] - m["start"]).squeeze(-1)','reference_end = (m["time"] + 2 >= m["end"] - m["start"]).squeeze(-1)'),
        ('            objective = objective + gamma_powers[t] * survival * r\n','            _full151_finite(r, "reward_including_inactive")\n            r = torch.where(full151_active, r, torch.zeros_like(r))\n            d = torch.where(full151_active, d, torch.ones_like(d))\n            objective = objective + gamma_powers[t] * survival * r\n'),
        ('        objective = objective + (c.gamma**horizon) * survival * self.critic(final)\n','        full151_final_active = _full151_phase_active(m)\n        full151_final_value = _full151_finite(self.critic(final), "bootstrap_including_inactive")\n        full151_final_value = torch.where(full151_final_active, full151_final_value, torch.zeros_like(full151_final_value))\n        objective = objective + (c.gamma**horizon) * survival * full151_final_value\n'),
        ('    with torch.no_grad():\n        self.critic.norm.update(inputs)\n','    full151_valid = torch.stack(full151_active_rows)\n    full151_norm_valid = torch.cat((full151_valid, full151_final_active[None]), 0)\n    with torch.no_grad():\n        self.critic.norm.update(inputs[full151_norm_valid])\n'),
        ('        values = self.critic(inputs)\n','        values = _full151_finite(self.critic(inputs), "criticvalues_including_inactive")\n'),
        ('    count = features.shape[0]\n','    full151_flat_valid = full151_valid.flatten()\n    count = features.shape[0]\n'),
        ('            critic_loss = (self.critic(features[ids]) - targets[ids]).square().mean()\n','            full151_critic_denominators.append(full151_flat_valid[ids].sum().detach())\n            critic_loss = _full151_masked_critic_mse(self.critic(features[ids]), targets[ids], full151_flat_valid[ids])\n'),
        ('    return dict(\n','    self._full151_runtime_policy_evidence = {"start_phases": full151_start_phases, "active_mask_H_B": full151_valid.detach(), "critic_norm_eligible": full151_norm_valid.detach().sum(), "critic_loss_minibatch_active_counts": full151_critic_denominators, "critic_total_candidate_rows": count, "optimizer_actor_calls": 1, "optimizer_critic_calls": len(full151_critic_denominators), "hard_mask_rule": "phase<150 only; physical soft termination retains native survival semantics"}\n    return dict(\n')]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Native policy adapter anchor changed: '+old.strip())
        source=source.replace(old,new,1)
    namespace=dict(original.__globals__);namespace.update(_full151_phase_active=phase_active,_full151_finite=_finite,_full151_masked_critic_mse=masked_critic_mse)
    exec(compile(source,str(Path(__file__).resolve())+'::native_policy_adapter','exec'),namespace)
    result=namespace[original.__name__];result.__full151_source__=source;result.__full151_native_source__=before;result.__full151_replacements__=tuple((old,new) for old,new in replacements)
    return result


def policy_evidence(runner):
    """Materialize the small latest mask/count receipt explicitly, outside kernels."""
    torch=_torch()
    def convert(value):
        if torch.is_tensor(value):return value.detach().cpu().tolist()
        if isinstance(value,dict):return {k:convert(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [convert(v) for v in value]
        return value
    return convert(getattr(runner,'_full151_runtime_policy_evidence',{'status':'NOT_RUN'}))


def validate_replay_window(window):
    """Validate existing native mask; final transition stays, reset suffix is zero."""
    torch=_torch();done=window['done'];valid=window['valid'];phase=window['full151_phase'];generation=window['full151_generation'];start=window['full151_start'];end=window['full151_end']
    if done.ndim!=2 or done.dtype!=torch.bool or valid.dtype!=torch.bool or any(v.shape!=done.shape for v in (valid,phase,generation,start,end)):raise ValueError('Replay phase/generation/mask shape differs')
    if bool(((phase<0)|(phase>=NINTERVAL)|(start!=0)|(end!=NFRAME)).any()):raise ValueError('Replay contains an unexecuted phase/reference')
    if done.shape[1]>1:
        change=generation[:,1:]!=generation[:,:-1]
        if bool((change&~done[:,:-1]).any()):raise ValueError('Reset generation changed without preceding done')
        continuing=(~done[:,:-1])&valid[:,:-1]
        if bool((continuing&((phase[:,1:]!=phase[:,:-1]+1)|change)).any()):raise ValueError('Unmarked replay phase/episode discontinuity')
    expected=torch.cat((torch.ones_like(done[:,:1]),~done[:,:-1]),1).float().cumprod(1)*valid.float().cumprod(1)
    if not torch.equal(window['mask'],expected):raise ValueError('Native replay mask differs from terminal-retained reset-suffix contract')
    return {'valid_window_entries':int(expected.sum()),'candidate_window_entries':int(expected.numel()),'terminal_valid_entries':int((done&valid&expected.bool()).sum()),'reset_suffix_entries_masked':int((~expected.bool()).sum())}


def install_replay_metadata(runner):
    """Instance-only adapters around the real native buffer, no collector rewrite."""
    torch=_torch();buffer=runner.buffer;old_add=buffer.add;old_sample=buffer.sample
    def add(transition):
        memory=runner.unflatten_memory(transition['m']);raw_phase=memory['time'].squeeze(-1);phase=raw_phase.long();start=memory['start'].squeeze(-1).long();end=memory['end'].squeeze(-1).long()
        if not torch.equal(raw_phase,phase.to(raw_phase.dtype)) or bool(((phase<0)|(phase>=NINTERVAL)|(start!=0)|(end!=NFRAME)).any()):raise ValueError('Collected transition is not one executed interval0..149 of the full151 reference')
        natural=phase==LAST_INTERVAL
        if bool((natural&~transition['done']).any()) or bool((natural&~runner.env.reset_time_outs).any()):raise ValueError('Natural interval149 end is missing its timeout/done label')
        payload={**transition,'full151_phase':phase.clone(),'full151_generation':runner.env._full151_runtime_episode_generation.clone(),'full151_start':start.clone(),'full151_end':end.clone(),'full151_source_frame':phase+200,'full151_reference_id':torch.zeros_like(phase),'full151_natural_end':natural.clone()}
        old_add(payload)
        counts=getattr(runner,'_full151_runtime_replay_counts',{'transitions':0,'natural_end_rows':0,'physical_terminated_rows':0})
        counts.update(transitions=counts['transitions']+len(phase),natural_end_rows=counts['natural_end_rows']+int(natural.sum()),physical_terminated_rows=counts['physical_terminated_rows']+int(transition['terminated'].sum()))
        runner._full151_runtime_replay_counts=counts
    def sample(*a,**kw):
        out=old_sample(*a,**kw);runner._full151_runtime_latest_window=validate_replay_window(out);return out
    buffer.add=add;buffer.sample=sample
    return buffer,old_add,old_sample


@contextmanager
def full151_runtime(env_class,command_class,mdp_class,runner_class,bundle_arrays,descriptor):
    """Outer scope; compose full151_controller inside before constructing env/runner."""
    bundle=_validated_bundle(bundle_arrays,descriptor);torch=_torch();patches=[];cfg_edits=[];replay_edits=[]
    def patch(cls,name,fn):
        own=name in cls.__dict__;old=getattr(cls,name);patches.append((cls,name,own,old));setattr(cls,name,fn);return old
    original_configure=getattr(command_class,'_configure_fixed_wrist_reference');original_prepare=getattr(command_class,'prepare_reset');original_env_init=getattr(env_class,'__init__');original_reset=getattr(env_class,'reset');original_termination=getattr(mdp_class,'termination')
    def configure(self,*a,**kw):
        result=original_configure(self,*a,**kw);install_velocity_authority(self,bundle);return result
    def prepare(self,env_ids):
        if self.cfg.trajectory_sampling_mode!='uniform' or self.cfg.first_frame_hold_prob!=0:raise ValueError('Full151 only permits phase0 single-reference resets, noRSI/hold')
        result=original_prepare(self,env_ids)
        if bool((self.time_steps[env_ids]!=0).any()) or bool((self.trajectory_ids[env_ids]!=0).any()):raise ValueError('Native sampler did not produce phase0')
        return result
    def env_init(self,*a,**kw):
        cfg=kw.get('cfg',a[0] if a else None)
        if cfg is None or cfg.auto_reset is not False:raise ValueError('Manual post-terminal reset is required')
        for name,event in cfg.events.items():
            if getattr(event,'mode',None)=='reset' and name!='reset_to_reference_state':raise ValueError('Unreviewed reset event may mutate fixed physical context: '+name)
        term=cfg.terminations['reference_end'];cfg_edits.append((term,term.func,term.time_out));term.func=reference_timeout;term.time_out=True
        original_env_init(self,*a,**kw);self._full151_runtime_episode_generation=torch.zeros(self.num_envs,device=self.device,dtype=torch.long);self._full151_runtime_descriptor=copy.deepcopy(descriptor);self._full151_runtime_last_reset=None
    def reset(self,*,seed=None,env_ids=None,options=None):
        ids=_ids(self,env_ids);_,info=original_reset(self,seed=seed,env_ids=ids,options=options);_restore_initial_state(self,ids,bundle)
        # Native reset already initialized/reset the observation timelines once.
        # True+ids backfills only selected rows and replaces the stale cache;
        # DelayBuffer.peek does not advance/resample lags. Do not reset again.
        self.obs_buf=self.observation_manager.compute(update_history=True,env_ids=ids)
        return self.obs_buf,info
    def termination(self,x,memory):
        native=original_termination(self,x,memory)
        return torch.where(memory['time'].squeeze(-1)>=LAST_INTERVAL,torch.ones_like(native),native)
    original_runner_init=getattr(runner_class,'__init__')
    def runner_init(self,*a,**kw):
        original_runner_init(self,*a,**kw)
        if not self.is_pen:raise ValueError('Rough runtime only supports the native pen path')
        replay_edits.append(install_replay_metadata(self))
    try:
        patch(command_class,'_configure_fixed_wrist_reference',configure);patch(command_class,'prepare_reset',prepare);patch(env_class,'__init__',env_init);patch(env_class,'reset',reset);patch(mdp_class,'termination',termination)
        # The policy adapter is attached below after native source pin checks;
        # no original source file or function globals are changed.
        if hasattr(runner_class,'policy_update'):patch(runner_class,'policy_update',make_policy_update(getattr(runner_class,'policy_update')))
        patch(runner_class,'__init__',runner_init)
        yield {'descriptor':copy.deepcopy(descriptor),'bundle_arrays':bundle,'reset_readback':reset_readback,'phase_active':phase_active,'policy_evidence':policy_evidence,'validate_replay_window':validate_replay_window,'mode':MODE,'physical_reference_qualified':False}
    finally:
        for buffer,add,sample in reversed(replay_edits):buffer.add=add;buffer.sample=sample
        for term,func,time_out in reversed(cfg_edits):term.func=func;term.time_out=time_out
        for cls,name,own,old in reversed(patches):
            if own:setattr(cls,name,old)
            else:delattr(cls,name)
