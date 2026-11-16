"""Exclusive reversible hooks around original Python graph-construction calls."""
import inspect
import hashlib
import marshal

from ..artifacts import file_pin,verify_file_pin
from .pen_termination_rule import _loaded_code_matches

_ACTIVE=None


class RayBindings:
    def __init__(self,owner):
        self.owner=owner; self.module=owner.solver
        self.original={n:getattr(self.module,n) for n in ['_solve','_solver_iteration','_linesearch_iterative']}
        self.solve_calls=0; self.iteration_callbacks=0; self.line_callbacks=0
        self.bound={'_solve':self.solve,'_solver_iteration':self.iteration,'_linesearch_iterative':self.line}
        self.audit=dict(bindings_restored=False)

    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None: raise ValueError('Exclusive ray hook scope required')
        path=verify_file_pin(self.owner.installed_solver_pin)
        for name,fn in self.original.items():
            if getattr(self.module,name) is not fn or not _loaded_code_matches(inspect.unwrap(fn),path):
                raise ValueError('Unmodified original graph callback required: '+name)
        self.audit['iteration_decorator_source']=file_pin(self.original['_solver_iteration'].__code__.co_filename)
        self.audit['original_code']={name:hashlib.sha256(marshal.dumps(fn.__code__)).hexdigest() for name,fn in self.original.items()}
        _ACTIVE=self
        try:
            for name,fn in self.bound.items(): setattr(self.module,name,fn)
        except BaseException:
            for name,fn in self.original.items(): setattr(self.module,name,fn)
            _ACTIVE=None
            raise
        return self

    def require(self,m,d,compact=False):
        if m is not self.owner.model or d is not self.owner.data or compact:
            raise ValueError('Only owned W1 noncompact graph may be traced')

    def solve(self,m,d,ctx,compact=False):
        self.require(m,d,compact); self.owner.clear_trace()
        result=self.original['_solve'](m,d,ctx,compact=compact)
        self.solve_calls+=1; self.owner._emit(ctx)
        return result

    def line(self,m,d,ctx,fuse_jv):
        self.require(m,d)
        if not fuse_jv: raise ValueError('Declared dense fused-jv path required')
        self.line_callbacks+=1; self.owner.launch_line(ctx)

    def iteration(self,m,d,ctx,nsolving,compact=False):
        self.require(m,d,compact)
        result=self.original['_solver_iteration'](m,d,ctx,nsolving,compact=compact)
        self.iteration_callbacks+=1; self.owner.emit_post(ctx)
        return result

    def __exit__(self,kind,error,tb):
        global _ACTIVE
        unchanged=_ACTIVE is self and all(getattr(self.module,n) is f for n,f in self.bound.items())
        try:
            if not unchanged: raise ValueError('Ray hook binding changed during graph build')
        finally:
            for name,fn in self.original.items(): setattr(self.module,name,fn)
            _ACTIVE=None
            self.audit.update(exit_bindings_verified=unchanged,bindings_restored=all(getattr(self.module,n) is f for n,f in self.original.items()),
                solve_callbacks=self.solve_calls,iteration_callbacks=self.iteration_callbacks,line_callbacks=self.line_callbacks,
                body_error=None if error is None else dict(type=type(error).__name__,message=str(error)))
