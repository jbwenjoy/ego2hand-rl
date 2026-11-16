"""Two externally owned graphs: endpoint-only versus paired ray/updated states."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import verify_source_snapshot
from .pen_solve_exit_observer_v2 import SolveExitGraphObserverV2
from .pen_ray_trace_factory_v1 import GENERATED,RAY_FIELDS,VECTOR_FIELDS,ROW_FIELDS,BRACKET_FIELDS,INT_FIELDS,validate_generated
from .pen_paired_ray_design_v1 import SCOPES


def launch_layout(pin):
    text=verify_file_pin(pin).read_text(); tree=ast.parse(text)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_linesearch_iterative')
    call=[n for n in ast.walk(fn) if isinstance(n,ast.Call) and ast.unparse(n.func)=='wp.launch_tiled']
    if len(call)!=1: raise ValueError('One exact original tiled line-search launch required')
    kws={k.arg:k.value for k in call[0].keywords}; result={}
    for key in ['inputs','outputs']:
        if not isinstance(kws[key],ast.List): raise ValueError('Explicit original argument list required')
        result[key]=[ast.unparse(n) for n in kws[key].elts]
    result['block_dim']=ast.unparse(kws['block_dim']); result['dim']=ast.unparse(kws['dim'])
    for path in result['inputs']+result['outputs']+[result['block_dim'],result['dim']]:
        parts=path.split('.')
        if parts[0] not in ['m','d','ctx'] or len(parts)<2 or any(not p.isidentifier() for p in parts):
            raise ValueError('Original launch must contain only declared attribute paths')
    return result


def resolve(path,m,d,ctx):
    parts=path.split('.'); value=dict(m=m,d=d,ctx=ctx)[parts[0]]
    for part in parts[1:]: value=getattr(value,part)
    return value


def disjoint(diagnostics,others):
    def span(a): return (int(a.ptr),int(a.ptr)+int(a.capacity))
    used=[a for a in diagnostics if a.size]
    for i,a in enumerate(used):
        a0,a1=span(a)
        for b in used[i+1:]+[b for b in others if b.size]:
            b0,b1=span(b)
            if max(a0,b0)<min(a1,b1): raise ValueError('Diagnostic buffer aliases native/context/another diagnostic buffer')


class PairedRayObserver(SolveExitGraphObserverV2):
    def __init__(self,sim,*,factory_qualification,**kwargs):
        qpath=Path(factory_qualification).resolve(); q=json.loads(qpath.read_text())
        if q['passed']!=q['total'] or q['total']!=8 or not all(q['checks'].values()): raise ValueError('Completed CPU factory qualification required')
        verify_source_snapshot(json.loads(verify_file_pin(q['source_receipt']).read_text()),live=True)
        mf=json.loads(verify_file_pin(q['prototype_identity']['generated_manifest']).read_text())
        path=verify_file_pin(mf['generated']); installed=verify_file_pin(mf['installed_solver'])
        validate_generated(installed.read_text(),path.read_text())
        super().__init__(sim,**kwargs)
        if mf['installed_solver']!=self.installed_solver_pin: raise ValueError('Factory and graph solver differ')
        m,d=self.model,self.data
        if (m.nv_pad,int(m.opt.iterations),int(m.opt.ls_iterations),bool(m.is_sparse),d.njmax)!=(28,50,50,False,4096):
            raise ValueError('Exact padded26/28,outer50/line50,dense,njmax4096 trace envelope required')
        name='pen_paired_ray_generated_'+mf['generated']['sha256']
        if name in sys.modules:
            module=sys.modules[name]
            if file_pin(module.__file__)!=mf['generated']: raise ValueError('Generated module namespace changed')
        else:
            spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module
            try: spec.loader.exec_module(module)
            except BaseException: sys.modules.pop(name,None); raise
        for alias in mf['aliased_original_globals']:
            if getattr(module,alias) is not getattr(self.solver,alias): raise ValueError('Ray helper alias changed: '+alias)
        self.factory_module=module; self.factory_manifest=mf; self.qualification=file_pin(qpath)
        self.kernel=getattr(module,GENERATED)(50,m.opt.cone,True,False,True)
        self.layout=launch_layout(self.installed_solver_pin); self.ray_graph=None; self.ray_audit=None
        from .pen_ray_post_kernels_v1 import POST_VECTOR_FIELDS,POST_FLOAT_FIELDS,POST_INT_FIELDS
        self.post_fields=dict(vector=POST_VECTOR_FIELDS,floats=POST_FLOAT_FIELDS,integers=POST_INT_FIELDS)
        with self.wp.ScopedDevice(sim.wp_device):
            z=self.wp.zeros
            self.trace=dict(ray=z((50,1,len(RAY_FIELDS)),dtype=float),vectors=z((50,1,26,len(VECTOR_FIELDS)),dtype=float),
                rows=z((50,1,d.njmax,len(ROW_FIELDS)),dtype=float),brackets=z((50,1,50,len(BRACKET_FIELDS)),dtype=float),
                integers=z((50,1,len(INT_FIELDS)),dtype=int),errors=z((1,),dtype=int),
                post_vectors=z((50,1,26,len(POST_VECTOR_FIELDS)),dtype=float),post_floats=z((50,1,len(POST_FLOAT_FIELDS)),dtype=float),
                post_ints=z((50,1,len(POST_INT_FIELDS)),dtype=int))
        self.check_aliases()

    def check_aliases(self,ctx=None):
        from .gpu_model_snapshot import model_leaves
        native=[]
        for root in [self.model,self.data]: native += list(model_leaves(root)[0].values())
        contexts=[ctx] if ctx is not None else []
        for context in contexts:
            native += [v for v in vars(context).values() if isinstance(v,self.wp.array)]
        disjoint(list(self.trace.values()),native+list(self.buffers.values()))

    def clear_trace(self):
        for array in self.trace.values(): array.zero_()

    def launch_line(self,ctx):
        self.check_aliases(ctx); m,d=self.model,self.data
        inputs=[resolve(p,m,d,ctx) for p in self.layout['inputs']]
        inputs += [d.solver_niter,ctx.grad,ctx.grad_scale,ctx.grad_dot,ctx.newton_decrement]
        outputs=[resolve(p,m,d,ctx) for p in self.layout['outputs']]+[self.trace[k] for k in ['ray','vectors','rows','brackets','integers','errors']]
        self.wp.launch_tiled(self.kernel,dim=d.nworld,inputs=inputs,outputs=outputs,block_dim=m.block_dim.linesearch_iterative)

    def emit_post(self,ctx):
        from .pen_ray_post_kernels_v1 import copy_post_iteration
        d=self.data
        self.wp.launch(copy_post_iteration,dim=(d.nworld,self.model.nv),
            inputs=[d.solver_niter,ctx.grad,ctx.search,d.qacc,d.efc.Ma,d.qfrc_constraint,ctx.grad_scale,ctx.grad_dot,ctx.newton_decrement,
                    ctx.alpha,ctx.improvement,d.time,ctx.state_changed_count,ctx.quad_changed_count,ctx.done,ctx.ls_exhausted,ctx.search_unchanged],
            outputs=[self.trace[k] for k in ['post_vectors','post_floats','post_ints','errors']])

    def build(self):
        super().build()
        from .pen_ray_bindings_v1 import RayBindings
        from . import pen_ray_post_kernels_v1
        with self.wp.ScopedDevice(self.sim.wp_device):
            self.wp.load_module(module=self.kernel.module,device=self.sim.wp_device,block_dim=self.model.block_dim.linesearch_iterative)
            self.wp.load_module(module=pen_ray_post_kernels_v1,device=self.sim.wp_device)
            with RayBindings(self) as hook:
                with self.wp.ScopedCapture() as capture: self.mw.step(self.model,self.data)
            self.ray_audit=hook.audit
            if hook.solve_calls!=1 or hook.line_callbacks!=1 or hook.iteration_callbacks!=1 or len(self.context_owners)!=2:
                raise ValueError('One conditional iteration callback and solve owner per graph required')
        self._verify_binding(); self.ray_graph=capture.graph

    def select(self,scope):
        if self.graph is None or self.ray_graph is None or scope not in SCOPES: raise ValueError('Both built graphs and declared scope required')
        if not any(self.sim.step_graph is g for g in [self.baseline_graph,self.graph,self.ray_graph]): raise ValueError('Unknown step graph')
        self.sim.step_graph=self.graph if scope=='SOLVE_EXIT_ONLY' else self.ray_graph; self.scope=scope

    def read(self):
        if self.scope not in SCOPES: raise ValueError('Selected endpoint-producing graph required')
        result={k:v.numpy().copy() for k,v in self.buffers.items()}
        if self.scope=='SOLVE_EXIT_PLUS_RAY_TRACE': result.update({'trace.'+k:v.numpy().copy() for k,v in self.trace.items()})
        return result

    def close(self):
        self.wp.synchronize()
        try:
            self._verify_binding()
            if not any(self.sim.step_graph is g for g in [self.baseline_graph,self.graph,self.ray_graph]): raise ValueError('Step graph escaped paired owner')
        finally: self.sim.step_graph=self.baseline_graph

    def manifest(self):
        return dict(schema='paired_ray_graphs:1',qualification=self.qualification,factory_manifest=self.factory_manifest,launch_layout=self.layout,
                    scopes=SCOPES,post_fields=self.post_fields,ray_bindings=self.ray_audit,context_owners=len(self.context_owners),
                    buffers={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in self.trace.items()},
                    scope='Two owned endpoint graphs;extra trace is observational,neutrality measured separately,not assumed.')
