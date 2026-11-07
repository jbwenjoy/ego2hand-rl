"""Same-world branch transactions; imports no simulator or tensor framework.

Native WarpTaskSnapshot is the primary restore implementation. This helper adds
an explicit, fail-closed inventory for its NumPy/closure/custom-object gaps and
keeps the CapacityObserver outside the captured object graph. CPU fixtures do
not qualify native restore or physics. No forward, step, reset or state setter
is called here; the worker separately verifies the INTEGRATION roundtrip.
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


class ProbeTransaction:
    """Capture once, then begin/finish unique one-interval branches.

    capacity_observer must be an external CapacityObserver(env,...), not the
    context manager that installs it on env. source_action is immutable FP32
    [1,20]. extra_sidecars registers used foreign mutable classes; external_roots
    declares instrumentation/log holders that intentionally must not roll back.
    """
    def __init__(self, env, reader, *, capacity_observer, source_action, label,
                 external_roots=None, extra_sidecars=None, snapshot_factory=None,
                 strict_native=True):
        from .capacity import CapacityObserver
        if not isinstance(capacity_observer,CapacityObserver) or capacity_observer.env is not env:
            raise TransactionError("Use the actual external CapacityObserver for this env")
        if not isinstance(label,str) or not label:raise TransactionError("Snapshot label required")
        action=np.asarray(source_action)
        if action.dtype!=np.float32 or action.shape!=(1,20) or not np.isfinite(action).all():
            raise TransactionError("source_action must be finite FP32[1,20]")
        self.env=env;self.reader=reader;self.observer=capacity_observer;self.label=label
        self.source_action=action.copy();self._source_bytes=action.tobytes();self.strict_native=bool(strict_native)
        self.active=None;self.failed=False;self.seen_branches=set();self.events=[]
        external=dict(external_roots or {});extra=dict(extra_sidecars or {})
        if any(v is capacity_observer for v in vars(env).values()) or any(v is capacity_observer for v in vars(reader).values()):
            raise TransactionError("Capacity ledger must not be attached to env/reader")
        if any(v is capacity_observer for v in extra.values()):raise TransactionError("Capacity ledger cannot be a restorable sidecar")
        external["capacity_observer"]=capacity_observer
        self._external=external
        if strict_native:
            if snapshot_factory is not None:raise TransactionError("Custom snapshot factory is CPU-fixture-only")
            if not type(env).__module__.startswith("mjlab.") or not type(reader).__module__.startswith("dex_supermimic."):
                raise TransactionError("Actual native Manager/PenSimReader required")
            self._profile=self._native_profile()
        else:self._profile={"scope":"CPU_FAKE_CONTRACT_ONLY_NOT_NATIVE_QUALIFICATION"}
        if snapshot_factory is None:
            from dex_supermimic.rl.dex_supermimic.snapshot import WarpTaskSnapshot
            snapshot_factory=WarpTaskSnapshot
        self.native=snapshot_factory(SimpleNamespace(env=env,reader=reader))
        for field in ("entries","arrays","tensors","restore"):
            if not hasattr(self.native,field):raise TransactionError("Snapshot lacks native inventory: "+field)
        self._native_meta=[(_meta(target),_bytes(saved)) for target,saved in (*self.native.arrays,*self.native.tensors)]
        hooks=[]
        for obj,name in ((env,"step"),(getattr(env,"sim",None),"step"),(getattr(env,"sim",None),"forward"),
                         (getattr(env,"reward_manager",None),"compute")):
            if obj is not None and callable(getattr(obj,name,None)):hooks.append(getattr(obj,name))
        if hasattr(env,"action_manager"):hooks.append(env.action_manager.get_term("joint_pos").process_actions)
        self.sidecars=_Sidecars(self.native,hooks,extra,external)
        self._memory=self._read_memory()
        self._rng=_rng_capture()
        self._initial_inventory=self.verify_current()

    def _native_profile(self):
        env=self.env;reader=self.reader
        required={
            "env":(env,("sim","scene","command_manager","action_manager","observation_manager","termination_manager","reward_manager",
                "episode_length_buf","common_step_counter","_sim_step_counter","_manual_reset_pending","_command_dt","obs_buf","extras",
                "_rough_controller_pending","_rough_controller_active","_rough_controller_process_calls","_rough_runtime_episode_generation")),
            "action_manager":(env.action_manager,("_action","_prev_action","_prev_prev_action")),
            "action_term":(env.action_manager.get_term("joint_pos"),("_raw_actions","_processed_actions","_targets","_targets_valid",
                "_mask_active","_mask_remaining","_mask_dofs","process_actions")),
            "motion":(env.command_manager.get_term("motion"),("time_steps","trajectory_ids","_episode_start_steps","_reset_prepared")),
            "observation_manager":(env.observation_manager,("_obs_buffer","_group_obs_term_delay_buffer","_group_obs_term_history_buffer","_group_obs_class_instances")),
            "termination_manager":(env.termination_manager,("_term_dones","_truncated_buf","_terminated_buf")),
            "finger_collision":(env.scene["finger_collision"],("_cached_data","_cache_valid")),
            "reader":(reader,("palm_pos","palm_rot","mdp","xi","reward_snapshot","memory")),
        }
        missing=[root+"."+name for root,(obj,names) in required.items() for name in names if not hasattr(obj,name)]
        if missing:raise TransactionError("Missing native transaction inventory: "+repr(missing))
        profile={root:list(names) for root,(_,names) in required.items()}
        # Manager defines these on the first step, not on reset. Their absence
        # at k0 is state: native vars restoration must remove later additions.
        profile['optional_manual_reset_flags_presence']={name:name in vars(env) for name in ('reset_buf','reset_terminated','reset_time_outs')}
        return profile

    def _read_memory(self):
        if not callable(getattr(self.reader,"memory",None)):
            if self.strict_native:raise TransactionError("Native memory reader missing")
            return {}
        memory=self.reader.memory()
        if self.strict_native:
            from dex_supermimic.rl.dex_supermimic.pen import PEN_MEMORY_DIMS
            if set(memory)!=set(PEN_MEMORY_DIMS):raise TransactionError("PEN memory key inventory differs")
            for key,width in PEN_MEMORY_DIMS.items():
                if tuple(memory[key].shape)!=(1,width):raise TransactionError("Wrong PEN memory shape: "+key)
        return {key:(_meta(v)[:3],_bytes(v)) for key,v in memory.items()}

    def verify_current(self):
        failures=[];h=hashlib.sha256()
        for i,(obj,saved,kind) in enumerate(self.native.entries):
            if not _container_equal(obj,saved,kind):failures.append(f"native_entry[{i}]")
        pairs=(*self.native.arrays,*self.native.tensors)
        for i,((target,_),(meta,expected)) in enumerate(zip(pairs,self._native_meta)):
            raw=_bytes(target);h.update(raw)
            if _meta(target)!=meta or raw!=expected:failures.append(f"native_value[{i}]")
        for path,obj,saved,kind in self.sidecars.entries:
            if not _container_equal(obj,saved,kind):failures.append(path)
        for path,cell,saved in self.sidecars.cells:
            if not _same_reference_or_scalar(cell.cell_contents,saved):failures.append(path)
        for path,target,saved,meta in self.sidecars.arrays:
            raw=_bytes(target);h.update(raw)
            if _meta(target)!=meta or raw!=_bytes(saved):failures.append(path)
        for path,obj,state,kind in self.sidecars.generators:
            if kind=="numpy":equal=repr(obj.bit_generator.state)==repr(state)
            elif kind=="python":equal=obj.getstate()==state
            else:equal=_bytes(obj.get_state())==_bytes(state)
            if not equal:failures.append(path)
        if hasattr(self,"_memory") and self._read_memory()!=self._memory:failures.append("PEN_memory")
        if hasattr(self,"_rng") and not _rng_equal(self._rng,_rng_capture()):failures.append("RNG")
        if self.source_action.tobytes()!=self._source_bytes:failures.append("source_action")
        result={"bitexact":not failures,"mismatches":failures,"value_inventory_sha256":h.hexdigest()}
        if failures:raise TransactionError("Transaction restore inventory mismatch: "+repr(failures))
        return result

    def _capacity(self,event,branch_id):
        self.observer.set_stage(self.label+"/"+branch_id)
        record=self.observer._read(event)
        self.observer._raise_if_unqualified(record)
        return copy.deepcopy(record)

    def begin(self,branch_id):
        if self.failed or self.active is not None or not isinstance(branch_id,str) or not branch_id or branch_id in self.seen_branches:
            raise TransactionError("Failed/active/duplicate/invalid branch cannot begin")
        self.seen_branches.add(branch_id)
        receipt={"branch_id":branch_id,"snapshot_label":self.label,"restore_qualified":False}
        try:
            receipt["capacity_before_restore"]=self._capacity("before_restore",branch_id)
            self.native.restore();self.sidecars.restore();_rng_restore(self._rng)
            receipt["inventory"]=self.verify_current();receipt["restore_qualified"]=True
            receipt["source_action_sha256"]=hashlib.sha256(self._source_bytes).hexdigest()
            self.active=branch_id;self.events.append(receipt);return copy.deepcopy(receipt)
        except BaseException as exc:
            self.failed=True;receipt["error"]=repr(exc);self.events.append(receipt);raise

    def finish(self,branch_id,*,failure=None):
        if self.active!=branch_id:raise TransactionError("Finish must match the one active branch")
        receipt={"branch_id":branch_id,"snapshot_label":self.label,"capacity_qualified":False,"body_completed":failure is None}
        if failure is not None:
            self.failed=True;receipt['body_error']=repr(failure)
        try:
            receipt["capacity_after_branch"]=self._capacity("after_branch",branch_id)
            receipt["capacity_qualified"]=True;return copy.deepcopy(receipt)
        except BaseException as exc:
            self.failed=True;receipt["error"]=repr(exc);raise
        finally:self.active=None;self.events.append(receipt)

    @contextmanager
    def branch(self,branch_id):
        receipt=self.begin(branch_id)
        body_error=None
        try:yield receipt
        except BaseException as exc:
            body_error=exc;self.failed=True;raise
        finally:
            try:self.finish(branch_id,failure=body_error)
            except BaseException as exc:
                if body_error is None:raise
                # Preserve the original physical/observer exception; the final
                # capacity exception remains in both ledger and event receipt.
                body_error.add_note('Secondary transaction finish failure: '+repr(exc))

    def describe(self):
        return {"schema":"current104_probe_transaction:1","label":self.label,"strict_native_profile":self.strict_native,
            "CPU_fixture_is_not_native_qualification":not self.strict_native,"native_profile":self._profile,
            "native_entries":len(self.native.entries),"native_Warp_arrays":len(self.native.arrays),"native_Torch_tensors":len(self.native.tensors),
            "explicit_containers":len(self.sidecars.entries),"explicit_arrays":len(self.sidecars.arrays),"closure_cells":len(self.sidecars.cells),
            "local_generators":len(self.sidecars.generators),"opaque_identity_only":self.sidecars.opaque,
            "external_references":self.sidecars.external,"external_names":list(self._external),
            "source_action_sha256":hashlib.sha256(self._source_bytes).hexdigest(),"integration_roundtrip":"WORKER_SEPARATE_REQUIRED",
            "capacity_ledger_inside_snapshot":False,"active_branch":self.active,"failed":self.failed,
            "native_physics_qualification":"NOT_GRANTED_BY_HELPER"}
