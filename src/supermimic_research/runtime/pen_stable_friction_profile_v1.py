"""W1 prospective cost backend, explicit metadata-only historical initialization."""
from contextlib import ExitStack
import copy
import hashlib
import inspect
import json
from pathlib import Path
import weakref

from ..artifacts import file_pin,verify_file_pin
from .pen_stable_friction_binding_v1 import FactoryBinding,code_digest
from .pen_stable_friction_factory_v1 import load_generated
from .pen_termination_rule import _loaded_code_matches

_ACTIVE=None
ALGORITHMS=['INSTALLED_ORIGINAL_OR','STABLE_SAME_REGIME_FRICTION_OR']


def original_method(function,path):
    module=compile(Path(path).read_text(encoding='utf-8'),str(path),'exec',dont_inherit=True)
    classes=[c for c in module.co_consts if isinstance(c,type(module)) and c.co_name=='BackendSupportPenRuntime']
    methods=[c for c in classes[0].co_consts if isinstance(c,type(module)) and c.co_name==function.__name__] if len(classes)==1 else []
    return len(methods)==1 and function.__code__==methods[0]


def execution_identity(algorithm,collision_backend_id,manifest):
    if algorithm not in ALGORITHMS or len(collision_backend_id)!=64 or any(c not in '0123456789abcdef' for c in collision_backend_id):
        raise ValueError('Declared cost algorithm and exact collision digest required')
    for key in ['installed_solver','generated','generator','cost_helper','installed_cache']: verify_file_pin(manifest[key])
    identity=dict(schema='pen_stable_friction_backend:1',algorithm=algorithm,collision_backend_id=collision_backend_id,
        isolated_factory=manifest,runtime_adapter=file_pin(__file__),binding_adapter=file_pin(Path(__file__).with_name('pen_stable_friction_binding_v1.py')),
        import_adapter=file_pin(Path(__file__).with_name('pen_termination_source_import.py')),stopping='INSTALLED_ORIGINAL_OR',
        domain=dict(worlds=1,nv=26,cone='PYRAMIDAL',solver='NEWTON',ls_iterations=50,solver_iterations=50),
        metadata_field='termination_backend',metadata_field_semantics='Legacy field now carries this explicit cost-backend schema;no stopping-rule change.')
    encoded=json.dumps(identity,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    return dict(backend_id=hashlib.sha256(encoded).hexdigest(),identity=identity,
        physical_qualification='PENDING_NATIVE_COST_COMPARISON',source_transfer='EXPLICIT_METADATA_ONLY_IMPORT_REQUIRED')


class CostProfile:
    def __init__(self,contract,manifest):
        self.contract=copy.deepcopy(contract); self.manifest=copy.deepcopy(manifest)
        self.live=weakref.WeakSet(); self.stack=ExitStack(); self.audit={}

    def __enter__(self):
        global _ACTIVE
        from . import pen_termination_profile,pen_termination_rule,pen_stable_friction_binding_v1
        if _ACTIVE is not None or pen_termination_profile._ACTIVE is not None or pen_termination_rule._ACTIVE is not None or pen_stable_friction_binding_v1._ACTIVE is not None:
            raise ValueError('Exclusive cost runtime profile required')
        import warp as wp
        from mujoco_warp._src import solver
        from .backend_pen_runtime import BackendSupportPenRuntime
        from .explicit_collision_backend import active_backend
        base=active_backend()
        if base.runtime_set: raise ValueError('Close existing collision runtimes before cost profile')
        algorithm=self.contract['identity']['algorithm']
        if execution_identity(algorithm,base.contract['backend_id'],self.manifest)!=self.contract: raise ValueError('Cost identity changed')
        load_generated(self.manifest)
        self.wp,self.solver,self.runtime_class=wp,solver,BackendSupportPenRuntime
        self.original_kernel=solver._solve_done; self.original_factory=solver._linesearch_iterative_kernel
        if not _loaded_code_matches(self.original_kernel.func,self.manifest['installed_solver']['path']): raise ValueError('Original OR kernel required')
        if not _loaded_code_matches(inspect.unwrap(self.original_factory),self.manifest['installed_solver']['path']): raise ValueError('Original factory required')
        self.original_methods={n:getattr(BackendSupportPenRuntime,n) for n in ['metadata','validate_resources','close']}
        self.base_source=file_pin(__import__(BackendSupportPenRuntime.__module__,fromlist=['x']).__file__)
        if any(not original_method(f,self.base_source['path']) for f in self.original_methods.values()):
            raise ValueError('Original runtime class methods required')
        self.method_codes={n:code_digest(f) for n,f in self.original_methods.items()}; self.kernel_code=code_digest(self.original_kernel)
        _ACTIVE=self
        try:
            self.binding=None
            if algorithm=='STABLE_SAME_REGIME_FRICTION_OR':
                self.binding=self.stack.enter_context(FactoryBinding(self.manifest,synchronize=wp.synchronize))
            self.expected_factory=solver._linesearch_iterative_kernel
            profile=self
            def metadata(runtime):
                profile.require(runtime)
                return dict(profile.original_methods['metadata'](runtime),termination_backend=copy.deepcopy(profile.contract))
            def validate_resources(runtime):
                profile.require(runtime)
                return profile.original_methods['validate_resources'](runtime)
            def close(runtime):
                profile.require(runtime)
                try: return profile.original_methods['close'](runtime)
                finally: profile.live.discard(runtime)
            self.wrappers=dict(metadata=metadata,validate_resources=validate_resources,close=close)
            for n,f in self.wrappers.items(): setattr(BackendSupportPenRuntime,n,f)
            self.verify(); self.audit.update(contract=self.contract,entered=True,bindings_restored=False,stopping_unchanged=True)
            return self
        except BaseException:
            for n,f in self.original_methods.items(): setattr(BackendSupportPenRuntime,n,f)
            try: self.stack.close()
            finally: _ACTIVE=None
            raise

    def verify(self):
        if _ACTIVE is not self or self.solver._solve_done is not self.original_kernel or code_digest(self.original_kernel)!=self.kernel_code or self.solver._linesearch_iterative_kernel is not self.expected_factory:
            raise ValueError('Live cost profile factory/stopping differs')
        if self.binding is not None: self.binding.verify()
        if any(getattr(self.runtime_class,n) is not f for n,f in self.wrappers.items()) or any(code_digest(f)!=self.method_codes[n] for n,f in self.original_methods.items()):
            raise ValueError('Runtime identity methods changed')
        return True

    def require(self,runtime):
        self.verify()
        if runtime not in self.live: raise ValueError('Runtime not owned by cost profile')

    def create_runtime(self,objective,worlds):
        self.verify()
        if worlds!=1: raise ValueError('W32 not qualified;only W1 cost comparison declared')
        runtime=self.runtime_class.__new__(self.runtime_class); self.live.add(runtime)
        try: self.runtime_class.__init__(runtime,objective,worlds)
        except BaseException:
            self.live.discard(runtime); raise
        self.require(runtime)
        return runtime

    def __exit__(self,kind,error,tb):
        global _ACTIVE
        try:
            self.wp.synchronize(); self.verify()
            if self.live: raise ValueError('Close every cost runtime before profile exit')
            verify_file_pin(self.base_source); verify_file_pin(self.manifest['installed_solver'])
        finally:
            for n,f in self.original_methods.items(): setattr(self.runtime_class,n,f)
            try: self.stack.close()
            finally:
                _ACTIVE=None
                self.audit['factory']=self.binding.audit if self.binding is not None else dict(entered=False,calls=[])
                self.audit['bindings_restored']=(self.solver._solve_done is self.original_kernel and self.solver._linesearch_iterative_kernel is self.original_factory
                    and all(getattr(self.runtime_class,n) is f for n,f in self.original_methods.items()))
        return False
