"""Training2: shared native PPO with owned contact rows and native4 state.

Current reward remains tracking-only for collector/state DIAGNOSTICS. This does
not implement or authorize a geometry/support training claim under that label.
"""
import copy
from contextlib import contextmanager
import torch
from .native_pen_ppo import NativePenPPO,model_pair,cpu_tree,tensor_tree_equal,validate_modules,timeout_reward
from .owned_pen_features import feature_sources,saved_observation
from supermimic_research.runtime.owned_pen_runtime import OwnedPenRuntime
from supermimic_research.runtime import owned_pen_state as native
from supermimic_research.runtime import pen_portable_state as old

SCHEMA='pen_owned_contact_PPO_training:2'
ROLLOUT='native_eight_samples_copied_before_terminal_reset:1'


def validate_recorded_rollout(m,gamma):
    done=m['terminated']|m['truncated']
    checks=dict(eight_samples=bool((m['contact_counts'][...,0]==8).all()),nonnegative_counts=bool((m['contact_counts']>=0).all()),
        valid_contacts=not bool(m['contact_errors'].any() or m['contact_global_error'].any()),
        nonnegative_geometry=bool((m['contact_geometry_max']>=0).all()),nonnegative_normals=bool((m['contact_Fn6_sum']>=0).all() and (m['contact_nonhand_max_sum']>=0).all()),
        contact_phase=torch.equal(m['contact_used_phase'],m['phase_used']),used_source=torch.equal(m['phase_used'],m['phase_source']),
        phase_advance=torch.equal(m['phase_next'],m['phase_source']+1),phase_reset=torch.equal(m['phase_after_reset'],torch.where(done,torch.ones_like(m['phase_next']),m['phase_next'])),
        applied_sample=torch.equal(m['applied_actions'],m['actions'].clamp(-1,1)),terminal_action=torch.equal(m['terminal_next_policy'][...,141:161],m['applied_actions']),
        bootstrap_once=torch.equal(m['reward'],timeout_reward(m['raw_reward'],m['terminal_next_value'],m['terminated'],m['truncated'],gamma)),
        clock=bool(((m['contact_clock'][...,1]-m['contact_clock'][...,0]-.02).abs()<1e-6).all()))
    if not all(checks.values()):raise ValueError('Recorded rollout contract mismatch: '+str([k for k,v in checks.items() if not v]))
    return checks


class OwnedPenPPO(NativePenPPO):
    def __init__(self,adapter,config):
        if not isinstance(adapter,OwnedPenRuntime):raise ValueError('Owned native4 runtime required')
        super().__init__(adapter,config)
        self._collecting=False;self._rollout_complete=False;self._captured_rows=0;self.memory_iteration=-1

    @contextmanager
    def _block(self):
        self.adapter.begin_collection()
        try:yield
        finally:self.adapter.end_collection()

    @torch.no_grad()
    def calibrate(self,steps):
        with self._block():return super().calibrate(steps)

    def _memory(self,steps):
        memory=super()._memory(steps)
        for key,view in self.adapter.contact_views.items():memory['contact_'+key]=torch.empty((steps,)+tuple(view.shape),device=self.device,dtype=view.dtype)
        memory['contact_used_phase']=torch.empty(steps,self.adapter.num_envs,device=self.device,dtype=torch.long)
        return memory

    @torch.no_grad()
    def collect(self,steps,*,forced_actions=None,control_driver='UPSTREAM'):
        if control_driver not in ['UPSTREAM','EMPTY_SIGNALS']:raise ValueError('Explicit supported control driver')
        if self._collecting or 'step' in self.adapter.__dict__ or steps<1:raise ValueError('Exclusive complete rollout required')
        self._collecting=True;self._rollout_complete=False;self._captured_rows=0
        original=self.adapter.step
        def capture_step(sample,**kwargs):
            result=original(sample,**kwargs);info=result[4];row=self._captured_rows
            if row>=steps:raise ValueError('Unexpected extra control transition')
            for key,view in info['contact_window'].items():self.memory['contact_'+key][row].copy_(view)
            self.memory['contact_used_phase'][row].copy_(info['reference_phase'])
            self._captured_rows+=1
            return result
        try:
            with self._block():
                self.adapter.step=capture_step
                try:result=super().collect(steps,forced_actions=forced_actions) if control_driver=='UPSTREAM' else super().collect(steps,forced_actions=forced_actions,control_driver=control_driver)
                finally:del self.adapter.step
            if self._captured_rows!=steps:raise ValueError('Incomplete contact rollout')
            self._rollout_complete=True;self.memory_iteration=self.iteration
            return result
        finally:self._collecting=False

    @torch.no_grad()
    def consistency(self):
        if not self._rollout_complete or self.memory_iteration!=self.iteration:raise ValueError('Frozen-policy consistency only before this rollout is optimized')
        result=super().consistency();m=self.memory
        checks=dict(result['checks'],contact_eight_samples=bool((m['contact_counts'][...,0]==8).all()),
            contact_counts_nonnegative=bool((m['contact_counts']>=0).all()),contact_errors_zero=not bool(m['contact_errors'].any() or m['contact_global_error'].any()),
            contact_geometry_nonnegative=bool((m['contact_geometry_max']>=0).all()),contact_normals_nonnegative=bool((m['contact_Fn6_sum']>=0).all() and (m['contact_nonhand_max_sum']>=0).all()),
            contact_used_phase=torch.equal(m['contact_used_phase'],m['phase_used']),
            contact_clock_20ms=bool(((m['contact_clock'][...,1]-m['contact_clock'][...,0]-.02).abs()<1e-6).all()),
            contact_storage_independent=all(m['contact_'+key].data_ptr()!=view.data_ptr() for key,view in self.adapter.contact_views.items()))
        return dict(result,checks=checks,passed=sum(checks.values()),total=len(checks),contact_rows=self._captured_rows,
            scope='Pre-update collection consistency. Original geometry gates are recorded, not enforced by the unchanged tracking reward.')

    def update(self):
        if not self._rollout_complete or self.memory_iteration!=self.iteration:raise ValueError('Exactly one update per complete new rollout')
        validate_recorded_rollout(self.memory,self.cfg.ppo_gamma)
        return super().update()

    def metadata(self):
        result=super().metadata();result.update(checkpoint_schema=SCHEMA,contact_rollout_schema=ROLLOUT,
            contact_fields={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in self.adapter.contact_views.items()},
            task_reward_contract='native_tracking_only_recorded_contacts:1',policy_update_contract='one_update_per_complete_new_rollout',
            role='COLLECTOR_STATE_DIAGNOSTIC_NO_POLICY_PROMOTION')
        return result

    def checkpoint(self,sources):
        if self._collecting or self.adapter._collection_open or 'step' in self.adapter.__dict__:raise ValueError('Save only outside an active collector callback')
        if self.memory is not None and not self._rollout_complete:raise ValueError('Cannot save partially initialized rollout')
        if not tensor_tree_equal(cpu_tree(self.observations),cpu_tree(self.adapter.observation())):raise ValueError('Stored observation does not match current native source')
        state=native.capture(self.adapter,sources);inv=native.inventory(self.adapter)
        return dict(schema=SCHEMA,metadata=self.metadata(),config=copy.deepcopy(vars(self.cfg)),native=state,feature_sources=feature_sources(self.adapter,inv),
            modules=dict(actor=cpu_tree(self.actor.state_dict()),critic=cpu_tree(self.critic.state_dict())),optimizer=cpu_tree(self.optimizer.state_dict()),
            parameter_order=[prefix+'.'+name for prefix,module in [('actor',self.actor),('critic',self.critic)] for name,_ in module.named_parameters()],
            observations=cpu_tree(self.observations),memory=cpu_tree(self.memory),normalization_batch=cpu_tree(self.normalization_batch),
            normalizers_ready=self.normalizers_ready,iteration=self.iteration,optimizer_steps=self.optimizer_steps,training_transitions=self.training_transitions,
            calibration_transitions=self.calibration_transitions,rollout_complete=self._rollout_complete,captured_rows=self._captured_rows,memory_iteration=self.memory_iteration,
            scope='Combined native4/cache/contact/network/optimizer state for integration diagnostics; original strict task and W32 future physics remain unqualified.')

    @torch.no_grad()
    def preflight(self,state):
        if state.get('schema')!=SCHEMA or state['metadata']!=self.metadata() or state['config']!=vars(self.cfg):raise ValueError('Explicit owned training2 checkpoint required')
        if self._collecting or self.adapter._collection_open or 'step' in self.adapter.__dict__:raise ValueError('Restore outside collection only')
        validate_modules(self.actor,self.critic,self.optimizer,state)
        for key in ['iteration','optimizer_steps','training_transitions','calibration_transitions','captured_rows']:
            if type(state[key]) is not int or state[key]<0:raise ValueError('Training counter type/range '+key)
        if type(state['normalizers_ready']) is not bool or type(state['rollout_complete']) is not bool or type(state['memory_iteration']) is not int:raise ValueError('Training phase flags')
        if state['optimizer_steps']!=state['iteration']*self.cfg.ppo_epochs*self.cfg.ppo_minibatches:raise ValueError('Optimizer update counter mismatch')
        N=self.adapter.num_envs;n=state['captured_rows'];memory=state['memory']
        if state['training_transitions']%N or state['calibration_transitions']%N or state['training_transitions']<n*N:raise ValueError('Transition counter mismatch')
        if memory is None:
            if n or state['rollout_complete'] or state['memory_iteration']!=-1 or state['normalization_batch'] is not None:raise ValueError('Fresh training state layout')
            staged_memory=None;staged_batch=None
        else:
            if not state['rollout_complete'] or n<1 or not state['normalizers_ready'] or state['memory_iteration'] not in [state['iteration'],state['iteration']-1]:raise ValueError('Completed rollout phase mismatch')
            template=self._memory(n)
            if template.keys()!=memory.keys():raise ValueError('Contact memory keys mismatch')
            for key,value in memory.items():
                if not isinstance(value,torch.Tensor) or value.shape!=template[key].shape or value.dtype!=template[key].dtype or (value.is_floating_point() and not bool(torch.isfinite(value).all())):raise ValueError('Contact memory layout/finite mismatch '+key)
            validate_recorded_rollout(memory,self.cfg.ppo_gamma)
            staged_memory={k:v.to(self.device).clone() for k,v in memory.items()}
            expected=dict(policy=memory['obs'].reshape(-1,161),critic=memory['critic_obs'].reshape(-1,188))
            if not tensor_tree_equal(expected,state['normalization_batch']):raise ValueError('Normalization batch must be previous raw rollout')
            staged_batch=dict(policy=staged_memory['obs'].reshape(-1,161),critic=staged_memory['critic_obs'].reshape(-1,188))
        for key,width in [('policy',161),('critic',188)]:
            v=state['observations'][key]
            if not isinstance(v,torch.Tensor) or v.shape!=(N,width) or v.dtype!=torch.float32 or not bool(torch.isfinite(v).all()):raise ValueError('Training observation layout '+key)
        if set(state['observations'])!= {'policy','critic'}:raise ValueError('Training observation keys')
        inv,_=native.preflight(self.adapter,state['native'])
        sources=feature_sources(self.adapter,inv)
        if sources!=state['feature_sources']:raise ValueError('Observation source path contract mismatch')
        reconstructed=saved_observation(self.adapter,state['native'],sources)
        if any(not torch.allclose(reconstructed[k],state['observations'][k],atol=2e-5,rtol=1e-6) for k in reconstructed):raise ValueError('Saved observations disagree with native/Python source')
        with torch.random.fork_rng(devices=[torch.device(self.device).index]):
            actor,critic,opt=model_pair(self.device,self.cfg)
            actor.load_state_dict(state['modules']['actor']);critic.load_state_dict(state['modules']['critic']);opt.load_state_dict(copy.deepcopy(state['optimizer']))
        for module in [actor,critic]:
            if any(not bool(torch.isfinite(v).all()) for v in module.state_dict().values() if v.is_floating_point()):raise ValueError('Nonfinite saved module')
        for values in opt.state_dict()['state'].values():
            if set(values)!= {'step','exp_avg','exp_avg_sq'}:raise ValueError('Declared AdamW state keys')
            for value in values.values():
                if not isinstance(value,torch.Tensor) or not bool(torch.isfinite(value).all()):raise ValueError('Invalid AdamW state tensor')
            step=values['step']
            if step.ndim or step.dtype!=torch.float32 or float(step)<0 or float(step)!=int(step):raise ValueError('Invalid AdamW step counter')
        if not tensor_tree_equal(cpu_tree(opt.state_dict()),state['optimizer']):raise ValueError('Staged optimizer readback mismatch')
        expected_norm=0 if not state['normalizers_ready'] else N+state['calibration_transitions']+state['training_transitions']-n*N
        if int(actor.norm.count)!=expected_norm or int(critic.norm.count)!=expected_norm:raise ValueError('Frozen normalizer data count mismatch')
        return dict(actor=actor,critic=critic,optimizer=opt,memory=staged_memory,normalization_batch=staged_batch,
            observations={k:v.to(self.device).clone() for k,v in state['observations'].items()},
            CPU_observation_max_error={k:float((reconstructed[k]-state['observations'][k]).abs().max()) for k in reconstructed})

    @torch.no_grad()
    def load(self,state):
        staged=self.preflight(state);audit=native.restore(self.adapter,state['native'])
        if not audit['all_array_bytes_equal'] or not audit['all_scalar_values_equal']:raise ValueError('Owned native restoration differs')
        self.actor=staged['actor'];self.critic=staged['critic'];self.optimizer=staged['optimizer']
        self.memory=staged['memory'];self.normalization_batch=staged['normalization_batch'];self.observations=staged['observations']
        for key in ['normalizers_ready','iteration','optimizer_steps','training_transitions','calibration_transitions','memory_iteration']:setattr(self,key,state[key])
        self._rollout_complete=state['rollout_complete'];self._captured_rows=state['captured_rows']
        return dict(native=audit,modules_equal=tensor_tree_equal(cpu_tree(self.actor.state_dict()),state['modules']['actor']) and tensor_tree_equal(cpu_tree(self.critic.state_dict()),state['modules']['critic']),
            optimizer_equal=tensor_tree_equal(cpu_tree(self.optimizer.state_dict()),state['optimizer']),memory_equal=tensor_tree_equal(cpu_tree(self.memory),state['memory']),
            observation_equal=tensor_tree_equal(cpu_tree(self.adapter.observation()),state['observations']),stored_observation_equal=tensor_tree_equal(cpu_tree(self.observations),state['observations']),
            normalization_equal=tensor_tree_equal(cpu_tree(self.normalization_batch),state['normalization_batch']),CPU_observation_max_error=staged['CPU_observation_max_error'])
