"""Explicit cached-reference/contact-window native runtime; no reward changes."""
import types
import torch
import warp as wp
from .pen_native_policy import NativePenPolicyAdapter,NativePenFeatures
from .pen_reference_cache import FrozenPenReferenceCache
from .resident_contact_metrics import ResidentContactMetrics
from .gpu_model_snapshot import model_leaves

CONTRACT='native_pen_owned_cache_and_contacts161:1'
TRANSITION='native_pen_used_reference_and_eight_contact_samples:2'


class OwnedPenFeatures(NativePenFeatures):
    def policy(self):return self.reference_cache.policy(self)


class OwnedPenRuntime(NativePenPolicyAdapter):
    def __init__(self,config_path,num_envs):
        self._owned_ready=False
        super().__init__(config_path,num_envs)
        self.reference_cache=FrozenPenReferenceCache(self.features)
        self.features.__class__=OwnedPenFeatures
        self.features.reference_cache=self.reference_cache
        s=self.env.sim;m=s.mj_model;jids=[m.joint('robot/'+n).id for n in self.motion.joint_names]
        self.contact_metrics=ResidentContactMetrics(m,s.wp_model,s.wp_data,self.features.body,jids)
        self.contact_views={k:wp.to_torch(v) for k,v in self.contact_metrics.window.items()}
        self._native_step_resource=s.step;s.step=self._native_step
        self._window_pending=False;self.owned_native_integrations=0;self._collection_open=False
        e=self.env
        if not hasattr(e,'reset_buf'):e.reset_buf=torch.zeros_like(e._manual_reset_pending)
        if not hasattr(e,'reward_buf'):e.reward_buf=e.reward_manager._reward_buf
        if not hasattr(e,'reset_terminated'):e.reset_terminated=e.termination_manager._terminated_buf
        if not hasattr(e,'reset_time_outs'):e.reset_time_outs=e.termination_manager._truncated_buf
        self._owned_ready=True;self._binding_resources=self._pointers()
        self.reference_cache.validate(self.features)

    def _native_step(self):
        return self._on_owned_stream(self._native_step_impl)

    def _on_owned_stream(self,fn):
        m=self.contact_metrics;caller=torch.cuda.current_stream(m.device.ordinal);target=m.torch_stream
        same=caller.cuda_stream==target.cuda_stream
        if not same:target.wait_stream(caller)
        try:
            with torch.cuda.stream(target),wp.ScopedStream(m.stream):return fn()
        finally:
            if not same:caller.wait_stream(target)

    def _native_step_impl(self):
        if self._window_pending:
            self.contact_metrics.reset_window();self._window_pending=False
        self._native_step_resource();self.owned_native_integrations+=self.num_envs
        self.contact_metrics.sample()

    def _pointers(self):
        result={}
        for group,root in [('model',self.env.sim.wp_model),('data',self.env.sim.wp_data)]:
            arrays,_=model_leaves(root)
            result.update({group+'.'+k:(v.ptr,v.shape,v.strides,str(v.dtype),str(v.device)) for k,v in arrays.items()})
        m=self.contact_metrics
        for group,arrays in [('instant',m.instant),('window',m.window),('static',{k:getattr(m,k) for k in ['object_tag','digit','joint_ids','qids','mask','reset_error']})]:
            result.update({group+'.'+k:(v.ptr,v.shape,v.strides,str(v.dtype),str(v.device)) for k,v in arrays.items()})
        for name in ['prefix','world_pose','mount','origins','world_ids']:
            v=getattr(self.reference_cache,name);result['cache.'+name]=(v.data_ptr(),tuple(v.shape),v.stride(),str(v.dtype),str(v.device))
        for name,v in self.contact_views.items():result['views.'+name]=(v.data_ptr(),tuple(v.shape),v.stride(),str(v.dtype),str(v.device))
        result['resources']=(m.stream.cuda_stream,m.torch_stream.cuda_stream,id(m.graph),str(m.device))
        return result

    def validate_resources(self):
        s=self.env.sim
        if not isinstance(s.step,types.MethodType) or s.step.__self__ is not self or s.step.__func__ is not OwnedPenRuntime._native_step:raise ValueError('Owned native hook replaced')
        if self._native_step_resource.__self__ is not s:raise ValueError('Native step owner changed')
        if self._pointers()!=self._binding_resources:raise ValueError('Captured native/metric/cache allocations replaced')
        return True

    def begin_collection(self):
        if self._collection_open:raise ValueError('Collection already open')
        self.validate_resources();self.reference_cache.validate(self.features);self._collection_open=True

    def end_collection(self):
        if not self._collection_open:raise ValueError('No collection open')
        self.validate_resources();self.reference_cache.validate(self.features);self._collection_open=False

    def reset(self,ids):
        if not self._owned_ready:return super().reset(ids)
        return self._on_owned_stream(lambda:self._reset_impl(ids))

    def _reset_impl(self,ids):
        observation=super().reset(ids)
        self.contact_metrics.reset_window(ids)
        return observation

    def step(self,sampled_raw):
        return self._on_owned_stream(lambda:self._step_impl(sampled_raw))

    def _step_impl(self,sampled_raw):
        # Reset at the first REAL native step, after the original pending/shape
        # guards. A rejected action must not erase an existing metric window.
        self._window_pending=True
        try:ob,reward,term,trunc,info=super().step(sampled_raw)
        finally:self._window_pending=False
        info['contact_window']=self.contact_views
        info['contact_schema']=self.contact_metrics.metadata()['schema']
        return ob,reward,term,trunc,info

    def metadata(self):
        return dict(super().metadata(),owned_runtime_contract=CONTRACT,owned_transition_schema=TRANSITION,
            native_checkpoint_schema='controlled_pen_owned_state:4',
            contact_output='Borrowed device window views; copy before next sample/reset. No reward/termination or applied-force change.')

    def close(self):
        if getattr(self,'_owned_ready',False):
            self.env.sim.step=self._native_step_resource;self._owned_ready=False
        super().close()
