"""Exclusive reversible factory binding; installed OR stopping stays intact."""
import hashlib
import inspect
import marshal

from ..artifacts import verify_file_pin,file_pin
from .pen_termination_rule import _loaded_code_matches
from .pen_stable_friction_factory_v4 import load_generated,verify_kernel_owner,NEW_FACTORY,SINGLE,TRIPLE

_ACTIVE=None


def code_digest(value):
    function=getattr(value,'func',inspect.unwrap(value))
    # v2 omits reference/intern flags that can change after nested Warp compilation.
    return hashlib.sha256(marshal.dumps(function.__code__,2)).hexdigest()


class FactoryBinding:
    def __init__(self,manifest,*,synchronize):
        self.manifest=manifest; self.synchronize=synchronize
        self.audit=dict(entered=False,bindings_restored=False,calls=[])

    def __enter__(self):
        global _ACTIVE
        from mujoco_warp._src import solver
        from . import pen_termination_profile,pen_termination_rule,pen_stable_friction_binding_v1,pen_stable_friction_profile_v1
        if _ACTIVE is not None or pen_stable_friction_binding_v1._ACTIVE is not None or pen_stable_friction_profile_v1._ACTIVE is not None or pen_termination_profile._ACTIVE is not None or pen_termination_rule._ACTIVE is not None:
            raise ValueError('Exclusive original-stopping factory scope required')
        self.solver=solver; pin=self.manifest['installed_solver']
        if file_pin(solver.__file__)!=pin: raise ValueError('Installed solver differs')
        self.original=solver._linesearch_iterative_kernel
        if not _loaded_code_matches(inspect.unwrap(self.original),pin['path']): raise ValueError('Original factory source required')
        self.module=load_generated(self.manifest)
        self.entries={n:getattr(solver,n) for n in self.manifest['original_aliases']+[SINGLE,TRIPLE,'_solve_done']}
        for n in [SINGLE,TRIPLE,'_solve_done']:
            if not _loaded_code_matches(self.entries[n].func,pin['path']): raise ValueError('Original installed helper/stopping required: '+n)
        self.codes={n:code_digest(v) for n,v in self.entries.items() if callable(v) or hasattr(v,'func')}
        helper=self.module._stable_cost
        if not _loaded_code_matches(helper.func,self.manifest['cost_helper']['path']): raise ValueError('Stable helper source differs')
        self.helper=helper; self.helper_code=code_digest(helper)
        from . import pen_guarded_stable_friction_v1
        self.guarded_module=pen_guarded_stable_friction_v1
        self.guarded_entries={n:getattr(self.guarded_module,n) for n in ['same_regime','guarded_single','guarded_triple','original_single','original_triple','with_stable_cost']}
        self.guarded_codes={n:code_digest(v) for n,v in self.guarded_entries.items()}
        self.candidate=getattr(self.module,NEW_FACTORY); self.candidate_code=code_digest(self.candidate)
        def dispatch(ls_iterations,cone_type,fuse_jv,is_sparse,incremental):
            self.verify()
            if (ls_iterations,int(cone_type),fuse_jv,is_sparse,incremental)!=(50,0,True,False,True):
                raise ValueError('Only declared dense incremental pyramidal line search admitted')
            kernel=self.candidate(ls_iterations,cone_type,fuse_jv,is_sparse,incremental)
            verify_kernel_owner(self.module,kernel)
            self.audit['calls'].append(dict(ls_iterations=ls_iterations,cone_type=int(cone_type),fuse_jv=fuse_jv,is_sparse=is_sparse,
                incremental=incremental,module=kernel.module.name,key=kernel.key,isolated_helpers=True))
            return kernel
        self.dispatch=dispatch; _ACTIVE=self; solver._linesearch_iterative_kernel=dispatch
        try:
            self.verify(); self.audit.update(entered=True,source_manifest=self.manifest,stopping='INSTALLED_ORIGINAL_OR')
            return self
        except BaseException:
            solver._linesearch_iterative_kernel=self.original; _ACTIVE=None
            raise

    def verify(self):
        if _ACTIVE is not self or self.solver._linesearch_iterative_kernel is not self.dispatch:
            raise ValueError('Stable cost factory binding changed')
        for n,value in self.entries.items():
            if getattr(self.solver,n) is not value or (n in self.codes and code_digest(value)!=self.codes[n]):
                raise ValueError('Original solver dependency/stopping changed: '+n)
            if n in self.manifest['original_aliases'] and getattr(self.module,n) is not value:
                raise ValueError('Generated helper alias changed: '+n)
        if self.module._stable_cost is not self.helper or code_digest(self.helper)!=self.helper_code:
            raise ValueError('Stable cost helper changed')
        if getattr(self.module,NEW_FACTORY) is not self.candidate or code_digest(self.candidate)!=self.candidate_code:
            raise ValueError('Generated factory changed')
        if any(getattr(self.guarded_module,n) is not v or code_digest(v)!=self.guarded_codes[n] for n,v in self.guarded_entries.items()):
            raise ValueError('Guarded helper binding/code changed')
        if getattr(self.module,SINGLE) is not self.guarded_entries['guarded_single'] or getattr(self.module,TRIPLE) is not self.guarded_entries['guarded_triple']:
            raise ValueError('Guarded factory helper alias changed')
        return True

    def __exit__(self,kind,error,tb):
        global _ACTIVE
        try:
            self.synchronize(); self.verify()
            for key in ['installed_solver','generated','generator','cost_helper','installed_cache','guarded_helper','measured_helper_reference','shared_factory_adapter']: verify_file_pin(self.manifest[key])
        finally:
            self.solver._linesearch_iterative_kernel=self.original; _ACTIVE=None
            self.audit['bindings_restored']=self.solver._linesearch_iterative_kernel is self.original
        return False
