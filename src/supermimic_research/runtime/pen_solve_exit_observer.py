"""Guarded process-local solve-exit copies, with external graph/context owners.

CPU binding/readiness tests do not qualify the CUDA graph implementation.
"""
import hashlib
import json
import marshal
from pathlib import Path
import tomllib

from ..artifacts import file_pin, verify_file_pin


DATA_FIELDS = ('qpos','qvel','time','qacc','qacc_smooth','M','efc.Ma',
               'qfrc_smooth','qfrc_constraint','qfrc_bias','qfrc_passive',
               'qfrc_actuator','qfrc_applied','xfrc_applied','nefc','solver_niter')
CONTEXT_VECTOR_FIELDS = ('grad','Mgrad','search')
CONTEXT_FLOAT_FIELDS = ('grad_scale','grad_dot','newton_decrement','improvement','alpha')
CONTEXT_INT_FIELDS = ('state_changed_count','quad_changed_count')
CONTEXT_BOOL_FIELDS = ('done','ls_exhausted','search_unchanged')
_ACTIVE_HOOK = None


def require_closed_full54(config_path):
    """Read/verify terminal workflow ownership before any CUDA import/allocation."""
    config_path = Path(config_path).resolve()
    cfg = tomllib.loads(config_path.read_text())
    paths = {k:(config_path.parent / cfg[k]).resolve() for k in
             ['required_closed_producer','required_closed_review']}
    if any(not p.is_file() for p in paths.values()):
        raise ValueError('Wait for both complete full54 producer and final V3 review')
    if (paths['required_closed_producer'].parent/'failure.json').exists():
        raise ValueError('Producer failure retained; native observation is not admitted')
    p, r = [json.loads(paths[k].read_text()) for k in
            ['required_closed_producer','required_closed_review']]
    good = lambda d,n: d.get('passed') == d.get('total') == n and len(d.get('checks',{})) == n and all(v is True for v in d['checks'].values())
    if (p.get('schema') != 'pen_full54_native_screened_rollout:1' or not good(p,8)
            or r.get('schema') != 'pen_full54_native_screened_saved_review:3'
            or not good(r,5) or r.get('complete') is not True
            or len(p.get('cases',[])) != 54 or len(r.get('reviewed_cases',[])) != 54):
        raise ValueError('Closed successful full54 workflows required; partial records cannot substitute')
    verify_file_pin(r['run'])
    if r['run'] != file_pin(paths['required_closed_producer']) or r['registration'] != p['registration']:
        raise ValueError('Final review belongs to a different producer/registration')
    registration = json.loads(verify_file_pin(p['registration']).read_text())
    names = [c['name'] for c in p['cases']]
    if (len(set(names)) != 54 or names != [c['name'] for c in registration['cases']]
            or names != [c['name'] for c in r['reviewed_cases']]
            or p['actual_native_integrations'] != p['attempted_native_integrations']
            or p['actual_native_integrations'] != 8*sum(c['native_queries'] for c in p['cases'])
            or p['actual_native_integrations'] > 300288
            or p['optimizer_steps'] != 0 or p['attempted_optimizer_steps'] != 0):
        raise ValueError('Exact closed full54 case order/native/optimizer accounting required')
    for pin in [p['source_receipt'],r['source_receipt']]:
        verify_file_pin(pin)
    return dict(config=file_pin(config_path),producer=file_pin(paths['required_closed_producer']),
                review=file_pin(paths['required_closed_review']),registration=p['registration'])


class SolveExitHook:
    """Exclusive temporary Python binding. Always restore it, including failures."""
    def __init__(self, module, model, data, on_exit):
        self.module,self.model,self.data,self.on_exit = module,model,data,on_exit
        self.original = module._solve
        self.calls = 0
        self.bound = self._solve

    def __enter__(self):
        global _ACTIVE_HOOK
        if _ACTIVE_HOOK is not None or self.module._solve is not self.original:
            raise ValueError('Exclusive unchanged solve binding required')
        _ACTIVE_HOOK = self
        self.module._solve = self.bound
        return self

    def _solve(self, model, data, context, compact=False):
        if model is not self.model or data is not self.data or compact:
            raise ValueError('Only the declared noncompact model/data solve may be observed')
        result = self.original(model,data,context,compact=compact)
        self.calls += 1
        self.on_exit(context)
        return result

    def __exit__(self, kind, error, tb):
        global _ACTIVE_HOOK
        unchanged = self.module._solve is self.bound and _ACTIVE_HOOK is self
        try:
            if not unchanged:
                raise ValueError('Solve binding changed inside exclusive scope')
        finally:
            self.module._solve = self.original
            _ACTIVE_HOOK = None


def _field(root, path):
    for name in path.split('.'):
        root = getattr(root,name)
    return root


class SolveExitGraphObserver:
    """Build an extra step graph; all buffers/context owners remain external.

    Construction is forbidden until full54 and its final review close. This
    class has not yet received live GPU qualification; do not infer observer
    neutrality from source review or fake-array tests.
    """
    def __init__(self, sim, *, config_path, installed_solver_pin):
        self.prerequisites = require_closed_full54(config_path)
        # No GPU-related import occurs before the terminal prerequisite guard.
        import warp as wp
        import mujoco_warp as mw
        from mujoco_warp._src import solver
        self.wp,self.mw,self.solver = wp,mw,solver
        self.sim,self.model,self.data = sim,sim.wp_model,sim.wp_data
        self.installed_solver_pin = installed_solver_pin
        if file_pin(solver.__file__) != installed_solver_pin:
            raise ValueError('Exact pinned installed solver file required')
        verify_file_pin(installed_solver_pin)
        self.original_solve = solver._solve
        self.code_digest = hashlib.sha256(marshal.dumps(solver._solve.__code__)).hexdigest()
        if (solver._solve.__module__ != solver.__name__ or solver._solve.__qualname__ != '_solve'
                or Path(solver._solve.__code__.co_filename).resolve() != Path(installed_solver_pin['path']).resolve()):
            raise ValueError('Original installed solve function binding required')
        m,d = self.model,self.data
        if (d.nworld != 1 or m.nv != 26 or int(m.opt.solver) != 2 or int(m.opt.cone) != 0
                or int(m.opt.integrator) != 3 or int(m.opt.enableflags) != 0
                or not m.opt.graph_conditional or not sim.use_cuda_graph or sim.step_graph is None
                or not wp.get_device(sim.wp_device).is_cuda):
            raise ValueError('Qualified W1 noncompact Newton/pyramidal implicitfast CUDA graph required')
        self.baseline_graph = sim.step_graph
        self.graph = None
        self.context_owners = []
        self.buffers = {}
        with wp.ScopedDevice(sim.wp_device):
            for key in DATA_FIELDS:
                self.buffers['data.'+key] = wp.empty_like(_field(d,key))
            for key in CONTEXT_VECTOR_FIELDS:
                self.buffers['context.'+key] = wp.empty((1,m.nv),dtype=float)
            for keys,dtype in [(CONTEXT_FLOAT_FIELDS,float),(CONTEXT_INT_FIELDS,int),(CONTEXT_BOOL_FIELDS,bool)]:
                for key in keys:
                    self.buffers['context.'+key] = wp.empty((1,),dtype=dtype)
            self.executions = wp.zeros((1,),dtype=int)
        self.scope = 'PACKET_BASELINE'

    def _verify_binding(self):
        verify_file_pin(self.installed_solver_pin)
        if (self.solver._solve is not self.original_solve
                or hashlib.sha256(marshal.dumps(self.original_solve.__code__)).hexdigest() != self.code_digest):
            raise ValueError('Original solver binding/code changed')

    def _emit(self, context):
        from .pen_solve_exit_kernels import count_solve_exit
        self.context_owners.append(context)
        for key,target in self.buffers.items():
            group,path = key.split('.',1)
            source = _field(self.data if group == 'data' else context,path)
            if (source.shape,source.dtype,source.device) != (target.shape,target.dtype,target.device):
                raise ValueError('Solve-exit buffer layout differs: '+key)
            if source.ptr == target.ptr:
                raise ValueError('Observer destination aliases native/context state')
            self.wp.copy(target,source)
        self.wp.launch(count_solve_exit,dim=1,inputs=[self.executions])

    def build(self):
        if self.graph is not None or self.sim.step_graph is not self.baseline_graph:
            raise ValueError('Build the separate observed graph exactly once')
        self._verify_binding()
        from . import pen_solve_exit_kernels
        with self.wp.ScopedDevice(self.sim.wp_device):
            self.wp.load_module(module=pen_solve_exit_kernels,device=self.sim.wp_device)
            # Retain the context allocation owners after the temporary binding
            # is removed. The env-forward graph remains the original graph.
            with SolveExitHook(self.solver,self.model,self.data,self._emit) as hook:
                with self.wp.ScopedCapture() as capture:
                    self.mw.step(self.model,self.data)
            if hook.calls != 1 or len(self.context_owners) != 1:
                raise ValueError('Exactly one noncompact solve in the step graph required')
        self._verify_binding()
        self.graph = capture.graph

    def select(self, scope):
        if self.graph is None or scope not in ['PACKET_BASELINE','PACKET_PLUS_SOLVE_EXIT']:
            raise ValueError('Built graph and declared observation scope required')
        if self.sim.step_graph is not self.baseline_graph and self.sim.step_graph is not self.graph:
            raise ValueError('Step graph changed outside observer scope')
        self.sim.step_graph = self.baseline_graph if scope == 'PACKET_BASELINE' else self.graph
        self.scope = scope

    def count(self):
        return int(self.executions.numpy()[0])

    def read(self):
        if self.scope != 'PACKET_PLUS_SOLVE_EXIT':
            raise ValueError('Baseline graph does not produce solve-exit observations')
        return {key:value.numpy().copy() for key,value in self.buffers.items()}

    def close(self):
        self.wp.synchronize()
        try:
            self._verify_binding()
            if self.graph is not None and self.sim.step_graph is not self.baseline_graph and self.sim.step_graph is not self.graph:
                raise ValueError('Observed runtime step graph changed')
        finally:
            self.sim.step_graph = self.baseline_graph

