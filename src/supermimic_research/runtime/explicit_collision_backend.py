"""Process-scoped, attested collision binding established BEFORE graph capture.

This changes no installed file. Old runtimes may not execute while a scope is
active; new runtimes carry a distinct metadata/owned-Python contract. A manifest
is a behavioral identity, not a scientific admission or old-state migration.
"""
import copy
import functools
import hashlib
import importlib.util
import json
import marshal
import sys
import weakref
from pathlib import Path
from supermimic_research.artifacts import file_pin,verify_file_pin

_ACTIVE=None
_REGISTRY={}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_contract(path):
    path=Path(path).resolve();packet=json.loads(path.read_text())
    if packet['schema']!='explicit_collision_backend_manifest:1':raise ValueError('Explicit collision backend manifest required')
    identity=packet['identity']
    if identity['schema']!='pinned_EPA_face_normal_backend:1' or packet['backend_id']!=digest(identity):raise ValueError('Backend identity digest differs')
    if identity['settings']!=dict(horizon=96,face_factor=20,max_axis_retries=3,ccd_iterations=50,ccd_tolerance=1e-6,quiet=False):raise ValueError('Only exact static-qualified behavior permitted')
    for p in identity['sources']+identity['installed_sources']+[identity['adapter_source'],identity['static_ledger'],identity['static_report']]:verify_file_pin(p)
    return dict(backend_id=packet['backend_id'],manifest=file_pin(path),**copy.deepcopy(identity))


def _load_module(name,pin):
    path=verify_file_pin(pin)
    if name in sys.modules:raise ValueError('Backend module namespace already occupied: '+name)
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module


def _code(value):
    if hasattr(value,'func'):value=value.func
    return hashlib.sha256(marshal.dumps(value.__code__)).hexdigest()


class CollisionBackend:
    def __init__(self,contract):
        from mujoco_warp._src import collision_driver,collision_convex
        self.contract=copy.deepcopy(contract);self.driver=collision_driver;self.installed=collision_convex.convex_narrowphase
        if collision_driver.convex_narrowphase is not self.installed:raise ValueError('Installed collision binding already changed')
        self.base_narrowphase=collision_driver._narrowphase;self.base_collision=collision_driver.collision
        self.gjk=_load_module('isolated_pen_face_normal_gjk',contract['sources'][0])
        self.convex=_load_module('isolated_pen_face_normal_convex',contract['sources'][1])
        self.expected_entry=self.convex.convex_narrowphase;self.runtime_set=weakref.WeakSet();self.calls=0;self.builds=[];self._entries=[]
        for name in ['ccd','gjk_phase','epa_phase','multicontact','support']:
            value=getattr(self.gjk,name)
            if getattr(self.convex,name) is not value:raise ValueError('Convex dependency is not the pinned GJK function: '+name)
            self._entries.append((self.gjk,name,value,_code(value)))
        for name in ['ccd_kernel_builder_face_normal_v1','ccd_hfield_kernel_builder_face_normal_v1']:
            original=getattr(self.convex,name)
            def instrument(factory,label):
                @functools.wraps(factory)
                def wrapper(*args):
                    kernel=factory(*args)
                    if kernel.func.__globals__ is not self.convex.__dict__:raise ValueError('Cached kernel belongs to another source module')
                    self.builds.append(dict(factory=label,args=list(args),kernel_id=id(kernel),key=kernel.key,module=kernel.module.name))
                    return kernel
                return wrapper
            setattr(self.convex,name,instrument(original,name))
        self.factory_bindings={name:getattr(self.convex,name) for name in ['ccd_kernel_builder_face_normal_v1','ccd_hfield_kernel_builder_face_normal_v1']}

    def _dispatch(self,*args,**kwargs):
        self.verify();self.calls+=1
        return self.expected_entry(*args,**kwargs)

    def verify(self,*,cold=False):
        if _ACTIVE is not self or self.driver.convex_narrowphase is not self.dispatch:raise ValueError('Explicit backend scope/binding not active')
        if self.driver._narrowphase is not self.base_narrowphase or self.driver.collision is not self.base_collision:raise ValueError('Collision driver dependency changed')
        if self.convex.convex_narrowphase is not self.expected_entry:raise ValueError('Backend entry changed')
        for module,name,value,code in self._entries:
            if getattr(module,name) is not value or getattr(self.convex,name) is not value or _code(value)!=code:raise ValueError('Backend function binding/code changed: '+name)
        if any(getattr(self.convex,n) is not v for n,v in self.factory_bindings.items()):raise ValueError('Backend kernel factory changed')
        if cold:
            observed=load_contract(self.contract['manifest']['path'])
            if observed!=self.contract:raise ValueError('Backend source/manifest changed')
        return True

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None or self.driver.convex_narrowphase is not self.installed:raise ValueError('Exclusive fresh backend scope required')
        self.dispatch=self._dispatch;_ACTIVE=self;self.driver.convex_narrowphase=self.dispatch
        try:self.verify(cold=True)
        except BaseException:
            self.driver.convex_narrowphase=self.installed;_ACTIVE=None;raise
        return self

    def __exit__(self,kind,error,tb):
        global _ACTIVE
        try:
            import warp as wp
            wp.synchronize();self.verify(cold=True)
            if len(self.runtime_set):raise ValueError('Close all backend runtimes before leaving scope')
        finally:
            self.driver.convex_narrowphase=self.installed;_ACTIVE=None
        return False

    def audit(self):
        return dict(backend_id=self.contract['backend_id'],entry_calls=self.calls,kernel_factory_calls=len(self.builds),
            builds=copy.deepcopy(self.builds),live_runtimes=len(self.runtime_set),
            scope='Python entry/factory observations include graph construction. Graph replay does not call Python;native counts are measured separately.')


def backend_scope(path):
    contract=load_contract(path);key=contract['backend_id']
    if key not in _REGISTRY:_REGISTRY[key]=CollisionBackend(contract)
    backend=_REGISTRY[key]
    if backend.contract!=contract:raise ValueError('Same backend ID resolved to different manifest')
    return backend


def active_backend(contract=None):
    if _ACTIVE is None:raise ValueError('Explicit collision backend scope required')
    _ACTIVE.verify()
    if contract is not None and contract!=_ACTIVE.contract:raise ValueError('Runtime backend identity differs')
    return _ACTIVE
