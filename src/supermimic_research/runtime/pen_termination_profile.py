"""External runtime identity and exclusive original/candidate solver profiles.

No instance leaf/class/schema is silently replaced: runtime metadata gains one
declared field, while the original Python closure description remains exact.
"""
from contextlib import ExitStack
import copy
import hashlib
import json
import marshal
from pathlib import Path
import weakref

from ..artifacts import file_pin, verify_file_pin
from .pen_termination_rule import implementation_identity, kernel_binding, _loaded_code_matches

_ACTIVE = None
ALGORITHMS = ['INSTALLED_ORIGINAL_OR', 'DROP_IMPROVEMENT_ONLY']


def execution_identity(algorithm, collision_backend_id, installed_pin):
    if algorithm not in ALGORITHMS:
        raise ValueError('Declared original/candidate algorithm required')
    candidate = implementation_identity(collision_backend_id, installed_pin)
    identity = dict(schema='pen_termination_comparator_backend:1', algorithm=algorithm,
                    candidate_implementation=candidate,
                    runtime_adapter=file_pin(__file__),
                    import_adapter=file_pin(Path(__file__).with_name('pen_termination_source_import.py')))
    payload = json.dumps(identity, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(backend_id=hashlib.sha256(payload).hexdigest(), identity=identity,
                physical_qualification='PENDING_CONTROLLED_COMPARATOR',
                source_transfer='EXPLICIT_METADATA_ONLY_IMPORT_REQUIRED')


class TerminationProfile:
    def __init__(self, contract, installed_pin):
        self.contract = copy.deepcopy(contract); self.installed_pin = installed_pin
        self.live = weakref.WeakSet(); self.stack = ExitStack(); self.audit = {}

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None:
            raise ValueError('Exclusive runtime termination profile required')
        import warp as wp
        from mujoco_warp._src import solver
        from .backend_pen_runtime import BackendSupportPenRuntime
        from .explicit_collision_backend import active_backend
        from .pen_termination_rule_kernels import candidate_solve_done
        self.wp, self.solver, self.runtime_class = wp, solver, BackendSupportPenRuntime
        base = active_backend()
        if base.runtime_set:
            raise ValueError('Close existing runtimes before solver profile entry')
        algorithm = self.contract['identity']['algorithm']
        if execution_identity(algorithm, base.contract['backend_id'], self.installed_pin) != self.contract:
            raise ValueError('Explicit execution identity/source pins changed')
        self.original_kernel = solver._solve_done
        if not _loaded_code_matches(self.original_kernel.func, self.installed_pin['path']):
            raise ValueError('Installed original solver kernel required at profile entry')
        self.original_methods = {n: getattr(BackendSupportPenRuntime, n) for n in ['metadata', 'validate_resources', 'close']}
        self.method_codes = {n: hashlib.sha256(marshal.dumps(f.__code__)).hexdigest() for n, f in self.original_methods.items()}
        self.base_source = file_pin(__import__(BackendSupportPenRuntime.__module__, fromlist=['x']).__file__)
        _ACTIVE = self
        try:
            if algorithm == 'DROP_IMPROVEMENT_ONLY':
                self.audit['kernel'] = self.stack.enter_context(kernel_binding(solver, self.installed_pin,
                                                        candidate_solve_done, synchronize=wp.synchronize))
            self.expected_kernel = solver._solve_done
            profile = self
            def metadata(runtime):
                profile.require(runtime)
                return dict(profile.original_methods['metadata'](runtime), termination_backend=copy.deepcopy(profile.contract))
            def validate_resources(runtime):
                profile.require(runtime)
                return profile.original_methods['validate_resources'](runtime)
            def close(runtime):
                profile.require(runtime)
                try:
                    return profile.original_methods['close'](runtime)
                finally:
                    profile.live.discard(runtime)
            self.wrappers = dict(metadata=metadata, validate_resources=validate_resources, close=close)
            for n, f in self.wrappers.items():
                setattr(BackendSupportPenRuntime, n, f)
            self.verify()
            self.audit.update(contract=self.contract, entered=True, bindings_restored=False)
            return self
        except BaseException:
            for n, f in self.original_methods.items():
                setattr(BackendSupportPenRuntime, n, f)
            try:
                self.stack.close()
            finally:
                _ACTIVE = None
            raise

    def verify(self):
        if _ACTIVE is not self or self.solver._solve_done is not self.expected_kernel:
            raise ValueError('Live solver profile/kernel differs')
        if any(getattr(self.runtime_class, n) is not f for n, f in self.wrappers.items()):
            raise ValueError('Runtime identity wrappers changed')
        if any(hashlib.sha256(marshal.dumps(f.__code__)).hexdigest() != self.method_codes[n]
               for n, f in self.original_methods.items()):
            raise ValueError('Original runtime method code changed')
        return True

    def require(self, runtime):
        self.verify()
        if runtime not in self.live:
            raise ValueError('Runtime not owned by the explicit termination profile')

    def create_runtime(self, objective, worlds):
        self.verify()
        runtime = self.runtime_class.__new__(self.runtime_class)
        self.live.add(runtime)
        try:
            self.runtime_class.__init__(runtime, objective, worlds)
        except BaseException:
            self.live.discard(runtime)
            raise
        self.require(runtime)
        return runtime

    def __exit__(self, kind, error, tb):
        global _ACTIVE
        try:
            self.wp.synchronize(); self.verify()
            if self.live:
                raise ValueError('Close every comparator runtime before profile exit')
            verify_file_pin(self.installed_pin); verify_file_pin(self.base_source)
        finally:
            for n, f in self.original_methods.items():
                setattr(self.runtime_class, n, f)
            try:
                self.stack.close()
            finally:
                _ACTIVE = None
                self.audit['bindings_restored'] = (self.solver._solve_done is self.original_kernel and
                    all(getattr(self.runtime_class, n) is f for n, f in self.original_methods.items()))
        return False
