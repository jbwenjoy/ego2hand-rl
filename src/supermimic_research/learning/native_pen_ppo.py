"""Actual native pen PPO collection; shared update, no learned dynamics gradient."""
import copy
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from dex_supermimic.rl.dex_supermimic.networks import GaussianPPOActor,Critic
from dex_supermimic.rl.dex_supermimic.ppo import ppo_update
from supermimic_research.runtime.pen_pose_observation import require_full_pose_contract
from supermimic_research.runtime import pen_portable_state as native
from supermimic_research.artifacts import file_pin

CHECKPOINT_SCHEMA='pen_native_full_pose_PPO_training:1'
NORMALIZATION_SCHEMA='BLOCK_FROZEN_PREVIOUS_ROLLOUT:1'


def timeout_reward(reward,terminal_next_value,terminated,truncated,gamma):
    """Bootstrap only real timeouts, using next terminal observation before reset."""
    return reward+gamma*terminal_next_value*(truncated.bool()&~terminated.bool()).to(reward.dtype)


def model_pair(device,cfg):
    actor=GaussianPPOActor(161,20,init_std=cfg.ppo_init_std,std_range=(cfg.ppo_std_min,cfg.ppo_std_max)).to(device)
    critic=Critic(188,amp=False,empirical_norm=True).to(device)
    optimizer=torch.optim.AdamW(tuple(actor.parameters())+tuple(critic.parameters()),lr=cfg.ppo_lr,weight_decay=cfg.weight_decay)
    return actor,critic,optimizer


def cpu_tree(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {key:cpu_tree(v) for key,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def tensor_tree_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and native.tensor_bytes(a)==native.tensor_bytes(b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(tensor_tree_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(tensor_tree_equal(x,y) for x,y in zip(a,b))
    return native.scalar_equal(a,b)


def validate_modules(actor,critic,optimizer,saved):
    for name,module in [('actor',actor),('critic',critic)]:
        current=module.state_dict();state=saved['modules'][name]
        if current.keys()!=state.keys() or any(current[k].shape!=v.shape or current[k].dtype!=v.dtype for k,v in state.items()):
            raise ValueError('PPO module layout mismatch '+name)
    names=[prefix+'.'+name for prefix,module in [('actor',actor),('critic',critic)] for name,_ in module.named_parameters()]
    if names!=saved['parameter_order']:raise ValueError('PPO optimizer parameter order mismatch')
    groups=saved['optimizer']['param_groups'];current=optimizer.state_dict()['param_groups']
    if len(groups)!=len(current) or any(len(a['params'])!=len(b['params']) for a,b in zip(groups,current)):
        raise ValueError('PPO optimizer group layout mismatch')
    for a,b in zip(groups,current):
        if {k:v for k,v in a.items() if k!='params'}!={k:v for k,v in b.items() if k!='params'}:
            raise ValueError('PPO optimizer hyperparameter mismatch')
    params=tuple(actor.parameters())+tuple(critic.parameters())
    ids=[i for group in groups for i in group['params']]
    if not set(saved['optimizer']['state'])<=set(ids):raise ValueError('Unknown optimizer state parameter')
    for index,param in zip(ids,params):
        for key,value in saved['optimizer']['state'].get(index,{}).items():
            if key in ['exp_avg','exp_avg_sq'] and (value.shape!=param.shape or value.dtype!=param.dtype):raise ValueError('Optimizer moment layout mismatch')


class NativePenPPO:
    def __init__(self,adapter,config):
        self.adapter=adapter;self.cfg=SimpleNamespace(**config);self.device=adapter.device
        c=self.cfg
        if c.ppo_rollout_steps*adapter.num_envs % c.ppo_minibatches:raise ValueError('Native PPO minibatches must divide rollout')
        random.seed(c.seed);np.random.seed(c.seed);torch.manual_seed(c.seed)
        self.actor,self.critic,self.optimizer=model_pair(self.device,c)
        self.observations=adapter.observation();self.memory=None;self.normalization_batch=None
        self.normalizers_ready=False;self.iteration=0;self.optimizer_steps=0
        self.training_transitions=0;self.calibration_transitions=0
        import warp as wp
        self.native_views={key:wp.to_torch(getattr(adapter.env.sim.wp_data,key)) for key in
            ['time','ctrl','overflow','qfrc_applied','xfrc_applied']}

    @torch.no_grad()
    def calibrate(self,steps):
        """Fresh TRAIN normalization data, zero raw controls, then pre-training reset."""
        if self.iteration or self.training_transitions or self.normalizers_ready:raise ValueError('Calibrate only before training')
        p=[self.observations['policy']];c=[self.observations['critic']]
        zero=torch.zeros(self.adapter.num_envs,20,device=self.device)
        for _ in range(steps):
            obs,_,term,trunc,_=self.adapter.step(zero)
            ids=(term|trunc).nonzero().flatten()
            if len(ids):obs=self.adapter.reset(ids)
            p.append(obs['policy']);c.append(obs['critic']);self.calibration_transitions+=self.adapter.num_envs
        self.actor.norm.update(torch.cat(p));self.critic.norm.update(torch.cat(c))
        self.observations=self.adapter.reset(torch.arange(self.adapter.num_envs,device=self.device))
        self.normalizers_ready=True

    def _memory(self,steps):
        N=self.adapter.num_envs;device=self.device
        widths=dict(obs=161,critic_obs=188,actions=20,applied_actions=20,terminal_next_policy=161,
            terminal_next_critic=188,EMA_before=20,EMA_after=20,PD_target=20,reference_q=20,transmitted_ctrl=20)
        result={key:torch.empty(steps,N,width,device=device) for key,width in widths.items()}
        for key in ['log_prob','value','raw_reward','reward','terminal_next_value','next_time']:
            result[key]=torch.empty(steps,N,device=device)
        for key in ['terminated','truncated','engine_overflow','no_applied_force']:
            result[key]=torch.empty(steps,N,device=device,dtype=torch.bool)
        for key in ['phase_source','phase_used','phase_next','phase_after_reset']:
            result[key]=torch.empty(steps,N,device=device,dtype=torch.long)
        return result

    @torch.no_grad()
    def collect(self,steps,*,forced_actions=None):
        if self.normalization_batch is not None:
            self.actor.norm.update(self.normalization_batch['policy']);self.critic.norm.update(self.normalization_batch['critic'])
        elif not self.normalizers_ready:
            self.actor.norm.update(self.observations['policy']);self.critic.norm.update(self.observations['critic'])
        self.normalizers_ready=True
        self.memory=self._memory(steps);m=self.memory;env=self.adapter.env
        for t in range(steps):
            obs=self.observations
            # Normal's debugging argument validation synchronizes a CUDA scalar;
            # block-end finite checks cover this frozen-normalizer collection.
            dist=self.actor.distribution(obs['policy'],validate_args=False)
            sampled=dist.sample() if forced_actions is None else forced_actions[t]
            m['obs'][t].copy_(obs['policy']);m['critic_obs'][t].copy_(obs['critic'])
            m['actions'][t].copy_(sampled);m['log_prob'][t].copy_(dist.log_prob(sampled).sum(-1))
            m['value'][t].copy_(self.critic(obs['critic']))
            m['phase_source'][t].copy_(self.adapter.motion.time_steps)
            m['EMA_before'][t].copy_(self.adapter.action._processed_actions)
            nextobs,reward,term,trunc,info=self.adapter.step(sampled)
            nextvalue=self.critic(nextobs['critic'])
            m['raw_reward'][t].copy_(reward);m['reward'][t].copy_(timeout_reward(reward,nextvalue,term,trunc,self.cfg.ppo_gamma))
            m['terminal_next_value'][t].copy_(nextvalue);m['terminated'][t].copy_(term);m['truncated'][t].copy_(trunc)
            m['applied_actions'][t].copy_(info['applied_raw']);m['terminal_next_policy'][t].copy_(nextobs['policy'])
            m['terminal_next_critic'][t].copy_(nextobs['critic']);m['phase_used'][t].copy_(info['reference_phase'])
            m['phase_next'][t].copy_(self.adapter.motion.time_steps);m['next_time'][t].copy_(self.native_views['time'])
            m['EMA_after'][t].copy_(self.adapter.action._processed_actions);m['PD_target'][t].copy_(self.adapter.action._targets)
            m['reference_q'][t].copy_(info['reference_joints'])
            m['transmitted_ctrl'][t].copy_(self.native_views['ctrl'][:,self.adapter.action._position_ctrl_ids])
            m['engine_overflow'][t].copy_(self.native_views['overflow']!=0)
            m['no_applied_force'][t].copy_((self.native_views['qfrc_applied']==0).all(-1)&(self.native_views['xfrc_applied']==0).all((-1,-2)))
            ids=(term|trunc).nonzero().flatten()
            self.observations=self.adapter.reset(ids) if len(ids) else nextobs
            m['phase_after_reset'][t].copy_(self.adapter.motion.time_steps)
            self.training_transitions+=self.adapter.num_envs
        self.normalization_batch=dict(policy=m['obs'].reshape(-1,161),critic=m['critic_obs'].reshape(-1,188))
        return m

    @torch.no_grad()
    def consistency(self):
        m=self.memory
        # Same batch shape as collection; do not confuse GEMM rounding changes
        # with a changing normalizer. No value is read by the CPU inside this loop.
        logp=torch.stack([self.actor.distribution(o,validate_args=False).log_prob(a).sum(-1) for o,a in zip(m['obs'],m['actions'])])
        value=torch.stack([self.critic(o) for o in m['critic_obs']])
        nextvalue=torch.stack([self.critic(o) for o in m['terminal_next_critic']])
        done=m['terminated']|m['truncated'];expected_phase=torch.where(done,torch.ones_like(m['phase_next']),m['phase_next'])
        checks=dict(finite=all(bool(torch.isfinite(v).all()) for v in m.values() if v.is_floating_point()),
            old_logprob_frozen=bool((logp-m['log_prob']).abs().max()<=1e-6),
            old_value_frozen=bool((value-m['value']).abs().max()<=1e-6),
            terminal_next_value=bool((nextvalue-m['terminal_next_value']).abs().max()<=1e-6),
            timeout_correction_once=torch.equal(m['reward'],timeout_reward(m['raw_reward'],nextvalue,m['terminated'],m['truncated'],self.cfg.ppo_gamma)),
            applied_is_clipped_sample=torch.equal(m['applied_actions'],m['actions'].clamp(-1,1)),
            terminal_previous_action=torch.equal(m['terminal_next_policy'][:,:,141:161],m['applied_actions']),
            source_used_phase=torch.equal(m['phase_source'],m['phase_used']),
            phase_advanced_once=torch.equal(m['phase_next'],m['phase_source']+1),reset_phase=torch.equal(m['phase_after_reset'],expected_phase),
            EMA=bool((m['EMA_after']-(.5*m['EMA_before']+.25*m['applied_actions'])).abs().max()<=1e-7),
            transmitted_PD=bool((m['PD_target']-m['transmitted_ctrl']).abs().max()<=2e-7),
            no_force=bool(m['no_applied_force'].all()),no_overflow=not bool(m['engine_overflow'].any()))
        return dict(checks=checks,passed=sum(checks.values()),total=len(checks),
            terminated=int(m['terminated'].sum()),truncated=int(m['truncated'].sum()),
            sample_clipped_components=int((m['actions'].abs()>1).sum()),
            mean_raw_reward=float(m['raw_reward'].mean()),max_logprob_error=float((logp-m['log_prob']).abs().max()))

    def update(self):
        result=ppo_update(self.actor,self.critic,self.optimizer,self.memory,self.observations['critic'],self.cfg)
        self.optimizer_steps+=result['optimizer_steps'];self.iteration+=1
        return result

    def metadata(self):
        return dict(**self.adapter.metadata(),checkpoint_schema=CHECKPOINT_SCHEMA,normalization_schema=NORMALIZATION_SCHEMA,
            joint_order=list(self.adapter.motion.joint_names),actor_hidden=[512,256,128],critic_hidden=[512,512,256,128],
            timeout='TERMINAL_NEXT_CRITIC_BEFORE_RESET_ONCE')

    def checkpoint(self,source_receipt):
        payload=native.capture(self.adapter,source_receipt)
        return dict(schema=CHECKPOINT_SCHEMA,metadata=self.metadata(),config=vars(self.cfg),
            modules=dict(actor=cpu_tree(self.actor.state_dict()),critic=cpu_tree(self.critic.state_dict())),
            parameter_order=[prefix+'.'+name for prefix,module in [('actor',self.actor),('critic',self.critic)] for name,_ in module.named_parameters()],
            optimizer=cpu_tree(self.optimizer.state_dict()),native=payload,observations=cpu_tree(self.observations),
            memory=cpu_tree(self.memory),normalization_batch=cpu_tree(self.normalization_batch),normalizers_ready=self.normalizers_ready,
            iteration=self.iteration,optimizer_steps=self.optimizer_steps,training_transitions=self.training_transitions,
            calibration_transitions=self.calibration_transitions,
            scope='Combined native+network training state. W32 numerical future replay remains unqualified; no CHORD/ego/task-success claim.')

    def load(self,state):
        if state['schema']!=CHECKPOINT_SCHEMA or state['metadata']!=self.metadata() or state['config']!=vars(self.cfg):
            raise ValueError('New native161/188 PPO checkpoint/config required')
        require_full_pose_contract(state['metadata']);validate_modules(self.actor,self.critic,self.optimizer,state)
        audit=native.restore(self.adapter,state['native'])
        if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal']:raise ValueError('Native state restore differs')
        self.actor.load_state_dict(state['modules']['actor']);self.critic.load_state_dict(state['modules']['critic']);self.optimizer.load_state_dict(state['optimizer'])
        def device_tree(value):
            if isinstance(value,torch.Tensor):return value.to(self.device)
            if isinstance(value,dict):return {key:device_tree(v) for key,v in value.items()}
            return value
        self.observations=device_tree(state['observations']);self.memory=device_tree(state['memory']);self.normalization_batch=device_tree(state['normalization_batch'])
        for key in ['normalizers_ready','iteration','optimizer_steps','training_transitions','calibration_transitions']:setattr(self,key,state[key])
        native.restore_rng(state['native']['rng'])
        actual=self.adapter.observation()
        return dict(native=audit,modules_equal=tensor_tree_equal(cpu_tree(self.actor.state_dict()),state['modules']['actor'])
            and tensor_tree_equal(cpu_tree(self.critic.state_dict()),state['modules']['critic']),
            optimizer_equal=tensor_tree_equal(cpu_tree(self.optimizer.state_dict()),state['optimizer']),
            observation_equal=tensor_tree_equal(cpu_tree(actual),state['observations']),
            normalization_equal=tensor_tree_equal(cpu_tree(self.normalization_batch),state['normalization_batch']))
