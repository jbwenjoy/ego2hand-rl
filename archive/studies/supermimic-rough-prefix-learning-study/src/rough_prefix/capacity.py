"""Reset-aware sticky capacity observation, independent of actor/WM/runtime.

Import is standard-library only. Production reads only the fixed backend field
env.sim.wp_data.overflow. No flags are cleared or rewritten; no capacity grows,
state write, forward, step, reset retry or world attribution is introduced.
"""
from __future__ import annotations
from contextlib import contextmanager
import copy
from datetime import datetime,timezone
import json

BACKEND_FIELD='env.sim.wp_data.overflow'


class CapacityObservationError(RuntimeError):
    def __init__(self,message,record):super().__init__(message);self.record=copy.deepcopy(record)


class CapacityOverflowError(CapacityObservationError):pass
class CapacityUnknownError(CapacityObservationError):pass


def _utc():return datetime.now(timezone.utc).isoformat()


def _selection(env_ids):
    """Record the native argument; never infer selected worlds from flag shape."""
    if env_ids is None:return {'kind':'native_all_reset_argument','values':None,'dtype':None,'shape':None}
    if isinstance(env_ids,slice):return {'kind':'native_slice_argument','start':env_ids.start,'stop':env_ids.stop,'step':env_ids.step}
    value=env_ids
    if hasattr(value,'detach'):value=value.detach()
    if hasattr(value,'cpu'):value=value.cpu()
    shape=list(value.shape) if hasattr(value,'shape') else None;dtype=str(value.dtype) if hasattr(value,'dtype') else None
    values=value.tolist() if hasattr(value,'tolist') else list(value)
    # Selection is metadata, never fed back into the native reset call.
    json.dumps(values,allow_nan=False)
    return {'kind':'native_explicit_reset_argument','values':values,'dtype':dtype,'shape':shape}


class CapacityObserver:
    def __init__(self,env,*,stage,fixed_capacities):
        if not isinstance(stage,str) or not stage:raise ValueError('Nonempty stage name required')
        if type(fixed_capacities) is not dict or set(fixed_capacities)!={'nconmax','njmax'} or any(type(v) is not int or v<=0 for v in fixed_capacities.values()):raise ValueError('Explicit fixed positive nconmax/njmax metadata required')
        self.env=env;self.stage=stage;self.fixed_capacities=copy.deepcopy(fixed_capacities);self.records=[];self.cumulative_bitmask=0;self.unknown_reads=0;self.finalized=False;self.final_record=None

    def set_stage(self,stage):
        if self.finalized:raise RuntimeError('Capacity observer already finalized')
        if not isinstance(stage,str) or not stage:raise ValueError('Nonempty stage name required')
        self.stage=stage

    @property
    def status(self):
        return 'OVERFLOW' if self.cumulative_bitmask!=0 else 'UNKNOWN' if self.unknown_reads else 'OK'

    def _raise_if_unqualified(self,record):
        if self.cumulative_bitmask!=0:raise CapacityOverflowError('Observed capacity overflow; no reset/physics retry is authorized',record)
        if self.unknown_reads:raise CapacityUnknownError('Capacity observation UNKNOWN; cannot qualify engineering run',record)

    def _read(self,event,env_ids=None):
        record={'sequence':len(self.records)+1,'event':event,'stage':self.stage,'observed_UTC':_utc(),'backend_field':BACKEND_FIELD,'read_attempts_for_record':1,'raw_flags_dtype':None,'raw_flags_shape':None,'raw_flags_values':None,'raw_flags_repr':None,'raw_bitmask':None,'selected_reset':None,'flag_shape_is_world_count':False,'world_attribution':'UNKNOWN_NOT_INFERRED_FROM_GLOBAL_STICKY_FLAGS','flags_cleared_by_observer':False}
        selection_error=None
        try:record['selected_reset']=_selection(env_ids) if event=='before_reset' else {'kind':'not_a_reset','values':None}
        except Exception as exc:selection_error=exc;record['selected_reset']={'kind':'UNKNOWN_ARGUMENT_METADATA','error':repr(exc)}
        try:
            import numpy as np
            # One actual backend read. Copy before the native reset can clear it.
            flags=np.asarray(self.env.sim.wp_data.overflow.numpy()).copy()
            record['raw_flags_dtype']=str(flags.dtype);record['raw_flags_shape']=list(flags.shape)
            if flags.size==0 or flags.dtype.kind not in 'iu':
                record['raw_flags_repr']=repr(flags)
                raise ValueError('Capacity flags must be a nonempty integer array')
            record['raw_flags_values']=flags.tolist();mask=0
            for value in flags.reshape(-1):mask|=int(value)
            record['raw_bitmask']=mask;self.cumulative_bitmask|=mask;record['read_status']='OK' if mask==0 else 'OVERFLOW'
            if selection_error is not None:raise ValueError('Reset selection metadata could not be recorded') from selection_error
        except Exception as exc:
            self.unknown_reads+=1;record['read_status']='UNKNOWN';record['error']=repr(exc)
        record.update(cumulative_OR_bitmask=self.cumulative_bitmask,cumulative_unknown_reads=self.unknown_reads,overall_status=self.status)
        self.records.append(record)
        return record

    def before_reset(self,env_ids=None):
        if self.finalized:raise RuntimeError('No reset may occur after final capacity observation')
        record=self._read('before_reset',env_ids);self._raise_if_unqualified(record);return copy.deepcopy(record)

    def final_read(self):
        """Exactly one end read, idempotent; no GPU work may follow this call."""
        if not self.finalized:
            self.final_record=self._read('final');self.finalized=True
        self._raise_if_unqualified(self.final_record)
        return copy.deepcopy(self.final_record)

    def snapshot(self):
        return {'schema':'rough_prefix_reset_aware_capacity_observation:1','status':self.status,'engineering_capacity_qualified':self.finalized and self.status=='OK','backend_field':BACKEND_FIELD,'configured_fixed_capacities':copy.deepcopy(self.fixed_capacities),'capacity_metadata_scope':'Caller-supplied fixed configuration; observer never allocates or changes capacity. Allocation identity remains a separate native config/source contract.','cumulative_OR_bitmask':self.cumulative_bitmask,'unknown_read_count':self.unknown_reads,'record_count':len(self.records),'before_reset_read_count':sum(r['event']=='before_reset' for r in self.records),'final_read_count':sum(r['event']=='final' for r in self.records),'finalized':self.finalized,'flag_shape_is_world_count':False,'records':copy.deepcopy(self.records),'scope':'No recorded sticky overflow before each observed reset and at final read. This does not establish solver convergence, physical accuracy or a specific offending world. No physics may follow final_read.'}


@contextmanager
def capacity_observer(env,*,stage,fixed_capacities):
    """Wrap this Env instance only; save snapshot in caller finally before close.

    A nonzero or unreadable pre-reset observation raises before native reset,
    preserving its evidence. Normal/exceptional exits make one final read;
    existing body exceptions are preserved if that final read is also unknown.
    The observer never starts another condition or repeats a reset/attempt.
    """
    if hasattr(env,'_rough_capacity_observer'):raise ValueError('Duplicate capacity observer installation')
    observer=CapacityObserver(env,stage=stage,fixed_capacities=fixed_capacities);own='reset' in vars(env);previous=vars(env).get('reset');native=env.reset
    def reset(*args,**kwargs):
        observer.before_reset(kwargs.get('env_ids'))
        return native(*args,**kwargs)
    env.reset=reset;env._rough_capacity_observer=observer;body_error=None
    try:yield observer
    except BaseException as exc:body_error=exc;raise
    finally:
        if own:env.reset=previous
        else:delattr(env,'reset')
        delattr(env,'_rough_capacity_observer')
        try:observer.final_read()
        except CapacityObservationError:
            if body_error is None:raise
