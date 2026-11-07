"""Isolated105 inventory/branch transactions; no framework import on import.

Native WarpTaskSnapshot is the primary restore implementation. This helper adds
an explicit, fail-closed inventory for its NumPy/closure/custom-object gaps and
keeps the CapacityObserver outside the captured object graph. CPU fixtures do
not qualify native restore or physics. No forward, step, reset or state setter
is called here; the worker separately verifies INTEGRATION, three baselines and
the full151 clock. CPU surrogates do not qualify native restore. Generic
sidecar primitives are copied from the hash-bound104 predecessor, not imported.
"""
from __future__ import annotations

from contextlib import contextmanager
import enum
import functools
import hashlib
import inspect
import math
from pathlib import Path
import random
import struct
import sys
from types import SimpleNamespace
import copy
import numpy as np


class TransactionError(RuntimeError):
    pass


class UncoveredSidecarError(TransactionError):
    pass


def _torch():
    return sys.modules.get("torch")


def _kind(value):
    torch = _torch()
    if torch is not None and isinstance(value, torch.Tensor):
        return "torch"
    if type(value).__module__.startswith("warp.") and hasattr(value, "numpy") and hasattr(value, "ptr"):
        return "warp"
    if isinstance(value, np.ndarray):
        return "numpy"
    return None


def _bytes(value):
    kind = _kind(value)
    if kind == "torch":
        raw = value.detach().contiguous().reshape(-1).view(_torch().uint8).cpu().numpy()
        return raw.tobytes()
    if kind == "warp":
        return np.ascontiguousarray(value.numpy()).tobytes()
    if kind == "numpy":
        if value.dtype.hasobject:
            raise UncoveredSidecarError("Object-dtype NumPy array is not a byte snapshot")
        return np.ascontiguousarray(value).tobytes()
    if isinstance(value, np.generic):
        return value.dtype.str.encode()+value.tobytes()
    if isinstance(value, float):
        return struct.pack("!d", value)
    if isinstance(value, complex):
        return struct.pack("!dd", value.real, value.imag)
    return (type(value).__module__+"."+type(value).__qualname__+":"+repr(value)).encode()


def _meta(value):
    kind = _kind(value)
    if kind == "torch":
        return kind, str(value.dtype), tuple(value.shape), tuple(value.stride()), str(value.device), value.data_ptr()
    if kind == "warp":
        shape=tuple(value.shape)
        if value.ptr is None:
            if math.prod(shape)!=0:
                raise TransactionError("Nonempty Warp array has null pointer")
            pointer=None
        else:
            pointer=int(value.ptr)
        return kind, str(value.dtype), shape, str(value.device), pointer
    if kind == "numpy":
        return kind, value.dtype.str, tuple(value.shape), tuple(value.strides), value.__array_interface__["data"][0], bool(value.flags.writeable)
    return type(value).__module__, type(value).__qualname__


def _same_reference_or_scalar(actual, saved):
    if isinstance(saved, (str, bytes, int, float, complex, bool, type(None), enum.Enum, np.generic)):
        return type(actual) is type(saved) and _bytes(actual) == _bytes(saved)
    return actual is saved


def _clone(value):
    kind = _kind(value)
    if kind == "torch":
        return value.detach().clone()
    if kind == "numpy":
        if value.dtype.hasobject:
            raise UncoveredSidecarError("Object-dtype NumPy array rejected")
        return value.copy()
    if kind == "warp":
        # Normally covered by the native snapshot; this path is explicit only.
        return sys.modules["warp"].clone(value)
    raise TypeError(kind)


def _copy_into(target, saved):
    if _kind(target) == "numpy":
        if target.shape != saved.shape or target.dtype != saved.dtype:
            raise TransactionError("NumPy sidecar allocation/shape/dtype changed")
        if not target.flags.writeable and _bytes(target)==_bytes(saved):return
        np.copyto(target, saved, casting="no")
    elif _kind(target) == "torch":
        torch = _torch()
        with torch.no_grad():
            if not any(n > 1 and stride == 0 for n, stride in zip(target.shape, target.stride())):
                target.copy_(saved)
    elif _kind(target) == "warp":
        sys.modules["warp"].copy(target, saved)


def _rng_capture():
    torch = _torch()
    value = {"python": random.getstate(), "numpy": np.random.get_state()}
    if torch is not None:
        value["torch_cpu"] = torch.get_rng_state().clone()
        if torch.cuda.is_initialized():
            value["torch_cuda"] = [v.clone() for v in torch.cuda.get_rng_state_all()]
    return value


def _rng_restore(value):
    random.setstate(value["python"]); np.random.set_state(value["numpy"])
    torch = _torch()
    if "torch_cpu" in value:
        torch.set_rng_state(value["torch_cpu"])
    if "torch_cuda" in value:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def _rng_equal(a, b):
    if set(a) != set(b) or a["python"] != b["python"]:
        return False
    x,y = a["numpy"],b["numpy"]
    if x[0] != y[0] or _bytes(x[1]) != _bytes(y[1]) or x[2:] != y[2:]:
        return False
    for key in ("torch_cpu", "torch_cuda"):
        if key in a:
            left = a[key] if key == "torch_cuda" else [a[key]]
            right = b[key] if key == "torch_cuda" else [b[key]]
            if len(left) != len(right) or any(_bytes(x) != _bytes(y) for x,y in zip(left,right)):
                return False
    return True


class _Sidecars:
    """Inspect only captured native entries and declared execution closures.

    Unknown mutable foreign classes are rejected rather than silently treated
    as restored. Foreign MuJoCo/CUDA handles are identity-only, inventoried
    opaque objects; they are not claimed to have copyable Python state.
    """
    def __init__(self, native, hooks, extra, external):
        self.native = native; self.entries=[]; self.arrays=[]; self.cells=[]; self.generators=[]
        self.opaque=[]; self.external=[]; self.seen=set()
        self.native_ids={id(v) for v,_,_ in native.entries}
        self.native_ids.update(id(v) for v,_ in native.arrays)
        self.native_ids.update(id(v) for v,_ in native.tensors)
        self.extra_ids={id(v) for v in extra.values()}
        self.external_ids={id(v):name for name,v in external.items()}
        if set(self.external_ids)&self.native_ids:
            raise TransactionError("External ledger/log object is part of native snapshot traversal")
        for i,(obj,contents,kind) in enumerate(native.entries):
            values=contents.values() if kind in ("dict","object") else contents
            for j,v in enumerate(values):
                self.walk(v,f"native_entry[{i}].value[{j}]")
        for name,value in extra.items():
            self.walk(value,"extra:"+name,declared=True)
        for i,fn in enumerate(hooks):
            self.walk(fn,f"execution_hook[{i}]")

    def walk(self, value, path, declared=False):
        ident=id(value)
        if ident in self.external_ids:
            self.external.append({"path":path,"external_name":self.external_ids[ident]});return
        if ident in self.native_ids or ident in self.seen:
            return
        self.seen.add(ident)
        if isinstance(value,(str,bytes,int,float,complex,bool,type(None),enum.Enum,np.generic,type,Path,range,slice)):
            return
        kind=_kind(value)
        if kind:
            self.arrays.append((path,value,_clone(value),_meta(value)));return
        if isinstance(value,dict):
            self.entries.append((path,value,dict(value),"dict"))
            for k,v in value.items():self.walk(v,path+"["+repr(k)+"]")
            return
        if isinstance(value,list):
            self.entries.append((path,value,list(value),"list"))
            for i,v in enumerate(value):self.walk(v,f"{path}[{i}]")
            return
        if isinstance(value,tuple):
            for i,v in enumerate(value):self.walk(v,f"{path}[{i}]")
            return
        if isinstance(value,set):
            if not all(isinstance(x,(str,int,float,bool,type(None),enum.Enum)) for x in value):
                raise UncoveredSidecarError("Mutable/non-scalar set member at "+path)
            self.entries.append((path,value,set(value),"set"));return
        if isinstance(value,np.random.Generator):
            self.generators.append((path,value,copy.deepcopy(value.bit_generator.state),"numpy"));return
        if isinstance(value,random.Random):
            self.generators.append((path,value,value.getstate(),"python"));return
        torch=_torch()
        if torch is not None and isinstance(value,torch.Generator):
            self.generators.append((path,value,value.get_state().clone(),"torch"));return
        if isinstance(value,functools.partial):
            self.walk(value.func,path+".func");self.walk(value.args,path+".args");self.walk(value.keywords,path+".keywords");return
        if inspect.ismethod(value):
            self.walk(value.__self__,path+".__self__");self.walk(value.__func__,path+".__func__");return
        if inspect.isfunction(value):
            for name,cell in zip(value.__code__.co_freevars,value.__closure__ or ()):
                try:saved=cell.cell_contents
                except ValueError as exc:raise UncoveredSidecarError("Empty closure cell at "+path) from exc
                if id(saved) in self.external_ids:
                    self.external.append({"path":path+".closure:"+name,"external_name":self.external_ids[id(saved)]});continue
                self.cells.append((path+".closure:"+name,cell,saved));self.walk(saved,path+".closure:"+name)
            return
        if inspect.isbuiltin(value):
            return
        module=type(value).__module__
        if declared or ident in self.extra_ids:
            if not hasattr(value,"__dict__"):
                raise UncoveredSidecarError("Declared custom sidecar has no attribute dictionary: "+path)
            contents=dict(vars(value));self.entries.append((path,value,contents,"object"))
            for k,v in contents.items():self.walk(v,path+"."+k)
            return
        if module.startswith(("mujoco.","mujoco_warp.","warp.","torch","numpy","dataclasses","typing","weakref","_thread")):
            self.opaque.append({"path":path,"type":module+"."+type(value).__qualname__,"id":ident});return
        raise UncoveredSidecarError("Unregistered foreign mutable object at "+path+": "+module+"."+type(value).__qualname__)

    def restore(self):
        for _,cell,saved in self.cells:cell.cell_contents=saved
        for _,obj,contents,kind in self.entries:
            if kind=="dict":obj.clear();obj.update(contents)
            elif kind=="list":obj[:]=contents
            elif kind=="set":obj.clear();obj.update(contents)
            else:vars(obj).clear();vars(obj).update(contents)
        for _,target,saved,_ in self.arrays:_copy_into(target,saved)
        for _,obj,state,kind in self.generators:
            if kind=="numpy":obj.bit_generator.state=copy.deepcopy(state)
            elif kind=="python":obj.setstate(state)
            else:obj.set_state(state)


def _container_equal(obj, contents, kind):
    actual=vars(obj) if kind=="object" else obj
    if kind in ("object","dict"):
        return list(actual)==list(contents) and all(_same_reference_or_scalar(actual[k],v) for k,v in contents.items())
    if kind=="set":return actual==contents
    return len(actual)==len(contents) and all(_same_reference_or_scalar(x,y) for x,y in zip(actual,contents))


MODE = 'rough_observation_reference_full151_v1'
VARIANT = 'legacy_H123_full151_SCP8_knownrough_v1'
FORMAT_VERSION = 105
PREDECESSOR = Path('D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/probe_transaction_v3.py')
CAPACITY_SOURCE = Path('D:/workspace/supermimic-rough-prefix-learning-study/src/rough_prefix/capacity.py')
CAPACITY_SHA256 = '7380bef76a4c6af5c65204d0bf9f5921d68826072a71a77656f1b4b2fda3844a'
ACTUAL8 = ('body_mass','body_inertia','body_iquat','geom_size','geom_pos',
           'geom_friction','actuator_gainprm','actuator_biasprm')
DATA_FIELDS = ('qpos','qvel','act','qacc_warmstart','ctrl','qfrc_applied',
               'xfrc_applied','mocap_pos','mocap_quat','time','qacc',
               'sensordata','actuator_force','qfrc_bias','qfrc_constraint',
               'overflow','nacon','nefc','contact','efc')
PEN_MEMORY_WIDTHS = dict(c1=20,c2=20,a1=20,a2=20,goal=9,hold=1,
    goal_timer=1,window=1,window_timer=1,reward_window=1,reward_hold=1,
    reward_snapshot=1,within=1,cage=20,age=1,count=1,time=1,start=1,end=1,
    limits=20,camera_raw=6,camera_obs=6,camera_holding=1,camera_ticks=1,
    camera_bias_pos=3,camera_bias_rot=9,encoder_bias=20,frozen_target=20,
    mask=20,mask_remaining=1,mask_active=1)
NATIVE_PENDING = ('fullstate_bitexact','integration_roundtrip',
                  'three_fresh_baselines','full151_long_clock',
                  'native_closure_foreign_inventory_review')


def _array(value):
    """Read an already-loaded array without importing a device framework."""
    if _kind(value)=='torch':return value.detach().cpu().numpy().copy()
    if _kind(value)=='warp':return np.asarray(value.numpy()).copy()
    return np.asarray(value)


def _require_fields(obj, names, root):
    missing=[root+'.'+name for name in names if not hasattr(obj,name)]
    if missing:raise TransactionError('Missing105 profile fields: '+repr(missing))


def _phase(value):
    a=_array(value)
    if a.shape not in ((1,),(1,1)) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
        raise TransactionError('Single-world finite local phase required')
    v=float(a.reshape(-1)[0])
    if not v.is_integer() or not 0<=v<=150:raise TransactionError('Phase must be integer0..150')
    return int(v)


def _memory(reader):
    memory=reader.memory()
    if type(memory) is not dict or set(memory)!=set(PEN_MEMORY_WIDTHS):
        raise TransactionError('Exact native PEN memory key inventory required')
    for name,width in PEN_MEMORY_WIDTHS.items():
        a=_array(memory[name])
        if a.shape!=(1,width) or a.dtype.kind not in 'iufb' or not np.isfinite(a).all():
            raise TransactionError('Nonfinite/wrong PEN memory field: '+name)
    return memory


def profile105(env,reader,*,expected_phase,boundary='pre_action',strict_native=True):
    """Inspect exact105 field/phase ABI, without reset/forward/step/state writes.

    terminal_observation is inspectable but can never be branched. Presence and
    selected value guards are not complete native restore qualification.
    """
    if type(strict_native) is not bool:raise TransactionError('Explicit boolean native/surrogate scope required')
    if type(expected_phase) is not int or not 0<=expected_phase<=150:
        raise TransactionError('Explicit local phase integer0..150 required')
    if boundary not in ('pre_action','terminal_observation'):
        raise TransactionError('Unknown105 capture boundary')
    if (boundary=='pre_action' and expected_phase>=150) or (boundary=='terminal_observation' and expected_phase!=150):
        raise TransactionError('Only preaction0..149 or terminal observation150 exists; no u150')
    if strict_native and (not type(env).__module__.startswith('mjlab.') or not type(reader).__module__.startswith('dex_supermimic.')):
        raise TransactionError('Actual installed Manager/PenSimReader required')
    required={
      'env':(env,('sim','scene','command_manager','action_manager','observation_manager','termination_manager','reward_manager',
        'episode_length_buf','common_step_counter','_sim_step_counter','_manual_reset_pending','_command_dt','obs_buf','extras',
        '_full151_controller_pending','_full151_controller_active','_full151_controller_process_calls',
        '_full151_controller_baseline','_full151_controller_lower','_full151_controller_upper',
        '_full151_runtime_episode_generation','_full151_runtime_descriptor','_full151_runtime_last_reset')),
      'action_manager':(env.action_manager,('_action','_prev_action','_prev_prev_action')),
      'action_term':(env.action_manager.get_term('joint_pos'),('_raw_actions','_processed_actions','_targets','_targets_valid',
        '_mask_active','_mask_remaining','_mask_dofs','process_actions')),
      'motion':(env.command_manager.get_term('motion'),('time_steps','trajectory_ids','_episode_start_steps','_reset_prepared',
        '_frames','_full151_runtime_native_velocity_diagnostics','_full151_runtime_velocity_authority')),
      'observation_manager':(env.observation_manager,('_obs_buffer','_group_obs_term_delay_buffer','_group_obs_term_history_buffer','_group_obs_class_instances')),
      'termination_manager':(env.termination_manager,('_term_dones','_truncated_buf','_terminated_buf')),
      'finger_collision':(env.scene['finger_collision'],('_cached_data','_cache_valid')),
      'reader':(reader,('env','robot','obj','kin','actuator_defaults','middle','distal','tips','palm_pos','palm_rot','mdp','xi','reward_snapshot','memory')),
      'MDP':(reader.mdp,('motion','full151_controller_metadata','_full151_controller_baseline','reference_signature')),
      'sim':(env.sim,('wp_model','wp_data','step','forward')),
    }
    for root,(obj,names) in required.items():_require_fields(obj,names,root)
    _require_fields(env.sim.wp_model,ACTUAL8,'wp_model')
    _require_fields(env.sim.wp_data,DATA_FIELDS,'wp_data')
    term=env.action_manager.get_term('joint_pos');motion=env.command_manager.get_term('motion')
    if reader.env is not env or reader.mdp.motion is not motion or _phase(motion.time_steps)!=expected_phase:
        raise TransactionError('Reader/motion identity or local phase differs')
    if env._full151_controller_active is not False:
        raise TransactionError('Capture must be outside active process_actions/env.step')
    if type(env._full151_controller_process_calls) is not int or env._full151_controller_process_calls!=expected_phase:
        raise TransactionError('Fresh single-episode process-call/phase count differs')
    if type(env._full151_controller_pending) is not list or len(env._full151_controller_pending)!=expected_phase:
        raise TransactionError('Fresh donor must retain one pending control row per prior interval')
    if expected_phase and not hasattr(env,'_full151_controller_last'):
        raise TransactionError('Postinterval donor is missing last processed control evidence')
    if _array(env._full151_controller_baseline).shape!=(150,20) or _array(reader.mdp._full151_controller_baseline).shape!=(150,20):
        raise TransactionError('Exactly150 baseline targets required')
    generation=_array(env._full151_runtime_episode_generation)
    if generation.shape!=(1,) or generation.dtype.kind not in 'iu' or np.any(generation<1):
        raise TransactionError('Single-world runtime generation inventory differs')
    descriptor=env._full151_runtime_descriptor
    if type(descriptor) is not dict or descriptor.get('schema')!='full151_knownrough_controller_input_descriptor:1' or descriptor.get('mode')!=MODE or descriptor.get('retarget_variant_id')!=VARIANT or descriptor.get('physical_reference_qualified') is not False:
        raise TransactionError('Knownrough full151 descriptor required; no upgrade/oldprefix')
    c=reader.mdp.full151_controller_metadata
    if type(c) is not dict or c.get('schema')!='full151_knownrough_baseline_EMA_C2:1' or c.get('format_version')!=105 or c.get('mode')!=MODE or c.get('retarget_variant_id')!=VARIANT or c.get('baseline_shape')!=[150,20] or c.get('active_intervals')!=[0,150] or c.get('terminal_observation')!=150:
        raise TransactionError('Exact105 full151 controller metadata required')
    if c.get('root_descriptor')!=descriptor or c.get('reference_controller_signature')!=reader.mdp.reference_signature:
        raise TransactionError('Runtime/MDP/controller descriptor or signature linkage differs')
    for name in ('_raw_actions','_processed_actions','_targets'):
        a=_array(getattr(term,name))
        if a.shape!=(1,20) or a.dtype!=np.float32 or not np.isfinite(a).all():raise TransactionError('One-world FP32 action/cache ABI: '+name)
    valid=_array(term._targets_valid)
    if valid.shape!=(1,) or valid.dtype!=np.bool_ or not bool(valid.all()):raise TransactionError('Native cached targets must be valid at capture')
    for name,shape in (('qpos',(1,27)),('qvel',(1,26)),('ctrl',(1,20))):
        a=_array(getattr(env.sim.wp_data,name))
        if a.shape!=shape or a.dtype!=np.float32 or not np.isfinite(a).all():raise TransactionError('Canonical one-world full state ABI differs: '+name)
    if _array(motion.trajectory_ids).shape!=(1,) or np.any(_array(motion.trajectory_ids)!=0):raise TransactionError('Single frozen reference id0 required')
    xi=_array(reader.xi)
    if xi.shape!=(1,46) or xi.dtype!=np.float32 or not np.isfinite(xi).all():raise TransactionError('Physical xi46 required, not privileged criticcontext368')
    memory=_memory(reader)
    if _phase(memory['time'])!=expected_phase or not np.all(_array(memory['start'])==0) or not np.all(_array(memory['end'])==151):
        raise TransactionError('PEN memory/reference local time0..150/start0/end151 differs')
    for name in ('c1','c2'):
        if _bytes(memory[name])!=_bytes(term._processed_actions):raise TransactionError('c1/c2 must remain native residual EMA, not absolute targets')
    if _bytes(memory['frozen_target'])!=_bytes(term._targets):raise TransactionError('Frozen actual-u/cache differs')
    return {'schema':'full151105_transaction_inventory_profile:1','format_version':105,'mode':MODE,'retarget_variant_id':VARIANT,
        'boundary':boundary,'phase':expected_phase,'action_allowed':boundary=='pre_action','terminal_policy_bootstrap_allowed':False if expected_phase==150 else None,
        'strict_native_types_checked':strict_native,'CPU_surrogate_is_not_native_qualification':not strict_native,
        'required_fields':{root:list(names) for root,(_,names) in required.items()},'actual8_fields':list(ACTUAL8),'wp_data_required_fields':list(DATA_FIELDS),
        'optional_presence':{name:name in vars(env) for name in ('_full151_controller_last','reset_buf','reset_terminated','reset_time_outs')},
        'PEN_memory_widths':dict(PEN_MEMORY_WIDTHS),'qualification':{k:'PENDING' for k in NATIVE_PENDING}}


def _reject_external_in_native_graph(env,reader,external):
    """Reject external objects before native snapshot can capture their state.

    Native WarpTaskSnapshot does not traverse function closures. Declared
    instrumentation closure references are separately marked external by the
    sidecar walker. Direct reachable attributes/containers may never hold them.
    """
    forbidden={id(v):name for name,v in external.items()};seen=set()
    def walk(value,path,root=False):
        ident=id(value)
        if ident in forbidden:raise TransactionError('External ledger/log reachable by native snapshot at '+path+': '+forbidden[ident])
        if ident in seen:return
        seen.add(ident)
        if _kind(value) or isinstance(value,(str,bytes,int,float,complex,bool,type(None),enum.Enum,np.generic,type,Path,range,slice)) or callable(value):return
        if isinstance(value,dict):
            for k,v in value.items():walk(v,path+'['+repr(k)+']')
        elif isinstance(value,(list,tuple,set)):
            for i,v in enumerate(value):walk(v,f'{path}[{i}]')
        elif hasattr(value,'__dict__') and (root or type(value).__module__.startswith(('mjlab.','dex_supermimic.','mujoco_warp.','warp._src.types'))):
            for k,v in vars(value).items():walk(v,path+'.'+k)
    walk(env,'env',True);walk(reader,'reader',True)


def _capacity_owned_external_roots(observer):
    """Protect ledger aliases too, not merely the observer object's identity."""
    roots={'capacity_observer':observer};seen={id(observer),id(observer.env)}
    def walk(value,path):
        if id(value) in seen:return
        seen.add(id(value))
        if _kind(value) or isinstance(value,(dict,list,set)):
            roots[path]=value
        if isinstance(value,dict):
            for k,v in value.items():walk(v,path+'['+repr(k)+']')
        elif isinstance(value,(list,tuple,set)):
            for i,v in enumerate(value):walk(v,f'{path}[{i}]')
    for name,value in vars(observer).items():
        if name!='env':walk(value,'capacity_observer.'+name)
    return roots


class Transaction105:
    """New105 single-world inventory/restore helper; no physics/forward/reset.

    Strict production construction lazily imports the installed native snapshot
    only after the profile and external-ledger guard. Surrogate mode requires an
    explicit snapshot_factory and can never grant native qualification.
    """
    def __init__(self,env,reader,*,capacity_observer,source_action,label,expected_phase,
                 external_roots=None,extra_sidecars=None,snapshot_factory=None,strict_native=True):
        self._profile=profile105(env,reader,expected_phase=expected_phase,strict_native=strict_native)
        if not isinstance(label,str) or not label:raise TransactionError('Nonempty snapshot label required')
        action=np.asarray(source_action)
        if action.dtype!=np.float32 or action.shape!=(1,20) or not np.isfinite(action).all():raise TransactionError('Immutable source action must be finite FP32[1,20]')
        for name in ('env','set_stage','_read','_raise_if_unqualified','records','cumulative_bitmask','unknown_reads','finalized'):
            if not hasattr(capacity_observer,name):raise TransactionError('Capacity observer protocol missing: '+name)
        if capacity_observer.env is not env or capacity_observer.finalized or capacity_observer.cumulative_bitmask or capacity_observer.unknown_reads:
            raise TransactionError('Capacity observer must be external, unfinalized and qualified for this env')
        if strict_native:
            method=getattr(type(capacity_observer),'_read',None)
            filename=Path(method.__code__.co_filename).resolve() if hasattr(method,'__code__') else None
            if filename!=CAPACITY_SOURCE.resolve() or hashlib.sha256(CAPACITY_SOURCE.read_bytes()).hexdigest()!=CAPACITY_SHA256:
                raise TransactionError('Use the exact frozen7380 CapacityObserver implementation')
            if getattr(capacity_observer,'fixed_capacities',None)!={'nconmax':192,'njmax':4096}:raise TransactionError('Original fixed capacity contract required')
            if snapshot_factory is not None:raise TransactionError('Snapshot injection is CPU-surrogate-only')
        elif snapshot_factory is None:raise TransactionError('CPU surrogate requires an explicit fake snapshot factory')
        self.env=env;self.reader=reader;self.observer=capacity_observer;self.label=label;self.expected_phase=expected_phase;self.strict_native=bool(strict_native)
        self.source_action=action.copy();self.source_action.setflags(write=False);self._source_bytes=action.tobytes()
        self.active=None;self.failed=False;self.seen_branches=set();self.events=[]
        external=dict(external_roots or {});extra=dict(extra_sidecars or {})
        external.update(_capacity_owned_external_roots(capacity_observer))
        if any(id(v)==id(capacity_observer) for v in extra.values()):raise TransactionError('Ledger cannot be a restorable sidecar')
        self._external=external
        _reject_external_in_native_graph(env,reader,external)
        if snapshot_factory is None:
            from dex_supermimic.rl.dex_supermimic.snapshot import WarpTaskSnapshot
            snapshot_factory=WarpTaskSnapshot
        self.native=snapshot_factory(SimpleNamespace(env=env,reader=reader))
        for field in ('entries','arrays','tensors','restore'):
            if not hasattr(self.native,field):raise TransactionError('Snapshot inventory API missing: '+field)
        self._native_meta=[(_meta(target),_bytes(saved)) for target,saved in (*self.native.arrays,*self.native.tensors)]
        hooks=[env.step,env.sim.step,env.sim.forward,env.reward_manager.compute,env.action_manager.get_term('joint_pos').process_actions,reader.memory]
        self.sidecars=_Sidecars(self.native,hooks,extra,external)
        # These roots must be visited, rather than only asserted to exist.
        visited=self.sidecars.native_ids|{id(v) for _,v,_,_ in self.sidecars.entries}
        roots={'env':env,'reader':reader,'wp_model':env.sim.wp_model,'wp_data':env.sim.wp_data,
               'action_term':env.action_manager.get_term('joint_pos'),'motion':env.command_manager.get_term('motion'),
               'observation_manager':env.observation_manager,'finger_collision':env.scene['finger_collision']}
        missing=[name for name,value in roots.items() if id(value) not in visited]
        if missing:raise TransactionError('Required objects not actually covered by inventory: '+repr(missing))
        self._memory=self._read_memory();self._rng=_rng_capture();self._initial_inventory=self.verify_current()

    def _read_memory(self):return {name:(_meta(v)[:3],_bytes(v)) for name,v in _memory(self.reader).items()}

    def verify_current(self):
        failures=[];h=hashlib.sha256()
        for i,(obj,saved,kind) in enumerate(self.native.entries):
            if not _container_equal(obj,saved,kind):failures.append(f'native_entry[{i}]')
        for i,((target,_),(meta,expected)) in enumerate(zip((*self.native.arrays,*self.native.tensors),self._native_meta)):
            raw=_bytes(target);h.update(raw)
            if _meta(target)!=meta or raw!=expected:failures.append(f'native_value[{i}]')
        for path,obj,saved,kind in self.sidecars.entries:
            if not _container_equal(obj,saved,kind):failures.append(path)
        for path,cell,saved in self.sidecars.cells:
            if not _same_reference_or_scalar(cell.cell_contents,saved):failures.append(path)
        for path,target,saved,meta in self.sidecars.arrays:
            raw=_bytes(target);h.update(raw)
            if _meta(target)!=meta or raw!=_bytes(saved):failures.append(path)
        for path,obj,state,kind in self.sidecars.generators:
            equal=repr(obj.bit_generator.state)==repr(state) if kind=='numpy' else obj.getstate()==state if kind=='python' else _bytes(obj.get_state())==_bytes(state)
            if not equal:failures.append(path)
        if hasattr(self,'_memory') and self._read_memory()!=self._memory:failures.append('PEN_memory')
        if hasattr(self,'_rng') and not _rng_equal(self._rng,_rng_capture()):failures.append('RNG')
        if self.source_action.tobytes()!=self._source_bytes:failures.append('source_action')
        if _phase(self.env.command_manager.get_term('motion').time_steps)!=self.expected_phase:failures.append('preaction_phase')
        result={'bitexact':not failures,'mismatches':failures,'value_inventory_sha256':h.hexdigest(),'scope':'Visited inventory bytes/identities only, not a full native-state certificate'}
        if failures:raise TransactionError('105 inventory restore mismatch: '+repr(failures))
        return result

    def _capacity(self,event,branch_id):
        self.observer.set_stage(self.label+'/'+branch_id);record=self.observer._read(event)
        self.observer._raise_if_unqualified(record);return copy.deepcopy(record)

    def begin(self,branch_id):
        if self.failed or self.active is not None or not isinstance(branch_id,str) or not branch_id or branch_id in self.seen_branches:
            raise TransactionError('Failed/active/duplicate/invalid branch cannot begin')
        self.seen_branches.add(branch_id)
        receipt={'branch_id':branch_id,'snapshot_label':self.label,'inventory_restored_bitexact':False,'native_restore_qualification':'PENDING_WORKER_GATE'}
        try:
            receipt['capacity_before_restore']=self._capacity('before_restore',branch_id)
            self.native.restore();self.sidecars.restore();_rng_restore(self._rng)
            receipt['inventory']=self.verify_current();receipt['inventory_restored_bitexact']=True
            receipt['source_action_sha256']=hashlib.sha256(self._source_bytes).hexdigest()
            self.active=branch_id;self.events.append(receipt);return copy.deepcopy(receipt)
        except BaseException as exc:
            self.failed=True;receipt['error']=repr(exc);self.events.append(receipt);raise

    def finish(self,branch_id,*,failure=None):
        if self.active!=branch_id:raise TransactionError('Finish must match the one active branch')
        receipt={'branch_id':branch_id,'snapshot_label':self.label,'capacity_qualified':False,'body_completed':failure is None}
        if failure is not None:self.failed=True;receipt['body_error']=repr(failure)
        try:
            receipt['capacity_after_branch']=self._capacity('after_branch',branch_id);receipt['capacity_qualified']=True;return copy.deepcopy(receipt)
        except BaseException as exc:self.failed=True;receipt['error']=repr(exc);raise
        finally:self.active=None;self.events.append(receipt)

    @contextmanager
    def branch(self,branch_id):
        receipt=self.begin(branch_id);body_error=None
        try:yield receipt
        except BaseException as exc:body_error=exc;self.failed=True;raise
        finally:
            try:self.finish(branch_id,failure=body_error)
            except BaseException as exc:
                if body_error is None:raise
                body_error.add_note('Secondary transaction105 finish error: '+repr(exc))

    def describe(self):
        return {'schema':'full151105_probe_transaction_inventory:1','label':self.label,'profile':copy.deepcopy(self._profile),
            'strict_native_types_checked':self.strict_native,'CPU_surrogate_is_not_native_qualification':not self.strict_native,
            'native_entries':len(self.native.entries),'native_Warp_arrays':len(self.native.arrays),'native_Torch_tensors':len(self.native.tensors),
            'explicit_containers':len(self.sidecars.entries),'explicit_arrays':len(self.sidecars.arrays),'closure_cells':len(self.sidecars.cells),
            'local_generators':len(self.sidecars.generators),'opaque_identity_only':copy.deepcopy(self.sidecars.opaque),
            'external_references':copy.deepcopy(self.sidecars.external),'external_names':list(self._external),
            'capacity_ledger_inside_snapshot':False,'source_action_sha256':hashlib.sha256(self._source_bytes).hexdigest(),
            'active_branch':self.active,'failed':self.failed,'qualification':{k:'PENDING' for k in NATIVE_PENDING},
            'native_physics_qualification':'NOT_GRANTED_BY_HELPER','function_globals_and_opaque_native_state':'Worker/source review required; identity alone does not freeze foreign internals',
            'capacity_substep_final_reads':'Caller must latch each substep/donor terminal/final; helper covers only before_restore and branch finish'}


