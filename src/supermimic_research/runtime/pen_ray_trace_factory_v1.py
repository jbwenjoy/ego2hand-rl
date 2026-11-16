"""Generate a pinned diagnostic factory; original physical AST must survive exactly.

This is only the factory building block, not a live graph observer or worker.
"""
import ast
import builtins
import copy
from pathlib import Path

from ..artifacts import file_pin,verify_file_pin

ORIGINAL='_linesearch_iterative_kernel'
GENERATED='_linesearch_iterative_kernel_ray_trace_v1'
INPUTS={'trace_iteration_in':'wp.array[int]','trace_grad_in':'wp.array2d[float]',
        'trace_grad_scale_in':'wp.array[float]','trace_grad_dot_in':'wp.array[float]',
        'trace_newton_decrement_in':'wp.array[float]'}
OUTPUTS={'trace_ray_out':'wp.array3d[float]','trace_vectors_out':'wp.array4d[float]',
         'trace_rows_out':'wp.array4d[float]','trace_brackets_out':'wp.array4d[float]',
         'trace_int_out':'wp.array3d[int]','trace_error_out':'wp.array[int]'}
RAY_PRE={'rows_cost0':'p0_sum[0][0]','rows_grad0':'p0_sum[0][1]','rows_hessian0':'p0_sum[0][2]',
         'gauss_cost0':'ctx_quad_gauss[0]','gauss_grad0':'ctx_quad_gauss[1]','gauss_half_hessian':'ctx_quad_gauss[2]',
         'p0_cost0':'p0[0]','p0_grad0':'p0[1]','p0_hessian0':'p0[2]','gtol':'gtol','noise_floor':'noise_floor',
         'snorm':'snorm','meaninertia':'meaninertia','tolerance':'tolerance','ls_tolerance':'ls_tolerance',
         'pre_grad_scale':'trace_grad_scale_in[worldid]','pre_grad_dot':'trace_grad_dot_in[worldid]',
         'pre_newton_decrement':'trace_newton_decrement_in[worldid]'}
RAY_INITIAL={'newton_alpha':'lo_alpha_in','newton_delta':'lo_in[0]','newton_gradient':'lo_in[1]','newton_hessian':'lo_in[2]'}
RAY_FINAL={'accepted_alpha':'alpha','improvement':'improvement'}
RAY_FIELDS=list(RAY_PRE)+list(RAY_INITIAL)+list(RAY_FINAL)
VECTOR_FIELDS=['search','mv','Ma','smooth','qacc','raw_grad']
ROW_FIELDS=['Jaref','jv','D','frictionloss']
INT_FIELDS=['valid','ne','nf','nefc','ls_used','initial_converged','final_recorded']
BRACKET_PRE={'entry_lo_alpha':'lo_alpha','entry_hi_alpha':'hi_alpha','trial_lo_alpha':'lo_next_alpha',
             'trial_hi_alpha':'hi_next_alpha','trial_mid_alpha':'mid_alpha'}
for label,var in [('entry_lo','lo'),('entry_hi','hi'),('trial_lo','lo_next'),('trial_hi','hi_next'),('trial_mid','mid')]:
    for i,part in enumerate(['delta','gradient','hessian']): BRACKET_PRE[label+'_'+part]=f'{var}[{i}]'
BRACKET_POST={'exit_lo_alpha':'lo_alpha','exit_hi_alpha':'hi_alpha','swap_lo':'float(swap_lo)',
              'swap_hi':'float(swap_hi)','ls_done':'float(ls_done)','improved':'float(improved)',
              'best_alpha':'best_alpha','best_delta':'best_delta','accepted_alpha':'alpha','improvement':'improvement'}
for label,var in [('exit_lo','lo'),('exit_hi','hi')]:
    for i,part in enumerate(['delta','gradient','hessian']): BRACKET_POST[label+'_'+part]=f'{var}[{i}]'
BRACKET_FIELDS=list(BRACKET_PRE)+list(BRACKET_POST)


def function(text,name):
    nodes=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(nodes)!=1: raise ValueError('One exact factory required')
    return nodes[0]


def statements(text): return ast.parse(text).body


def assigned(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)


def after(body,name,extra):
    indices=[i for i,n in enumerate(body) if assigned(n,name)]
    if len(indices)!=1: raise ValueError('Exact insertion assignment required: '+name)
    i=indices[0]+1; body[i:i]=extra


def writes(array,fields,mapping,index):
    body='\n'.join(f'  {array}[{index}, {fields.index(k)}] = {v}' for k,v in mapping.items())
    condition='trace_active and tid == 0'
    if array=='trace_brackets_out': condition+=' and _ < trace_brackets_out.shape[2]'
    return statements(f'if {condition}:\n{body}')


def instrument(original):
    factory=copy.deepcopy(original); factory.name=GENERATED
    if any(isinstance(n,ast.Name) and n.id.startswith('trace_') for n in ast.walk(factory)):
        raise ValueError('Reserved diagnostic prefix already used')
    kernel=[n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='kernel']
    if len(kernel)!=1: raise ValueError('One original inner kernel required')
    kernel=kernel[0]; args=kernel.args.args; offset=[a.arg for a in args].index('qacc_out')
    args[offset:offset]=[ast.arg(arg=k,annotation=ast.parse(v,mode='eval').body) for k,v in INPUTS.items()]
    args += [ast.arg(arg=k,annotation=ast.parse(v,mode='eval').body) for k,v in OUTPUTS.items()]
    # Every inserted physical read is scalar; writes target only explicit diagnostics.
    early=[i for i,n in enumerate(kernel.body) if isinstance(n,ast.If) and ast.unparse(n.test)=='ctx_done_in[worldid]']
    if len(early)!=1: raise ValueError('Original done guard required')
    prefix=statements('''trace_i = trace_iteration_in[worldid]
trace_active = trace_i >= 0 and trace_i < trace_ray_out.shape[0]
trace_ls_used = int(0)
if trace_active and tid == 0:
  trace_int_out[trace_i, worldid, 0] = 1
if not trace_active and tid == 0:
  wp.atomic_add(trace_error_out, 0, 1)
''')
    kernel.body[early[0]+1:early[0]+1]=prefix
    after(kernel.body,'nefc',writes('trace_int_out',INT_FIELDS,dict(ne='ne',nf='nf',nefc='nefc'),'trace_i, worldid'))
    loops=[n for n in kernel.body if isinstance(n,ast.For)]
    rowloops=[n for n in loops if any(isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=='local_p0' for x in ast.walk(n))]
    if len(rowloops)!=1: raise ValueError('Original row reduction loop required')
    rowloops[0].body[0:0]=statements('''if trace_active:
  trace_rows_out[trace_i, worldid, efcid, 0] = ctx_Jaref_in[worldid, efcid]
  trace_rows_out[trace_i, worldid, efcid, 1] = ctx_jv_in[worldid, efcid]
  trace_rows_out[trace_i, worldid, efcid, 2] = efc_D_in[worldid, efcid]
  trace_rows_out[trace_i, worldid, efcid, 3] = efc_frictionloss_in[worldid, efcid]
''')
    gaussloops=[n for n in loops if any(isinstance(x,ast.AugAssign) and isinstance(x.target,ast.Name) and x.target.id=='local_gauss' for x in ast.walk(n))]
    if len(gaussloops)!=1: raise ValueError('Original Gauss reduction loop required')
    values=['ctx_search_in[worldid, dofid]','ctx_mv_in[worldid, dofid]','efc_Ma_out[worldid, dofid]',
            'qfrc_smooth_in[worldid, dofid]','qacc_out[worldid, dofid]','trace_grad_in[worldid, dofid]']
    gaussloops[0].body += statements('if trace_active:\n'+'\n'.join(f'  trace_vectors_out[trace_i, worldid, dofid, {i}] = {v}' for i,v in enumerate(values)))
    after(kernel.body,'p0_delta',writes('trace_ray_out',RAY_FIELDS,RAY_PRE,'trace_i, worldid'))
    after(kernel.body,'initial_converged',writes('trace_ray_out',RAY_FIELDS,RAY_INITIAL,'trace_i, worldid')+
          writes('trace_int_out',INT_FIELDS,dict(initial_converged='int(initial_converged)'),'trace_i, worldid'))
    branch=[n for n in kernel.body if isinstance(n,ast.If) and ast.unparse(n.test)=='not initial_converged']
    if len(branch)!=1: raise ValueError('Original acceptance branch required')
    branch=branch[0]; inner=[n for n in branch.body if isinstance(n,ast.For) and ast.unparse(n.iter)=='range(LS_ITERATIONS)']
    if len(inner)!=1: raise ValueError('Original bracketing loop required')
    inner=inner[0]; inner.body[0:0]=statements('trace_ls_used += 1')
    target=[i for i,n in enumerate(inner.body) if assigned(n,'swap_lo_lo_next')]
    if len(target)!=1: raise ValueError('Original bracket swap point required')
    inner.body[target[0]:target[0]]=writes('trace_brackets_out',BRACKET_FIELDS,BRACKET_PRE,'trace_i, worldid, _')
    after(inner.body,'improvement',writes('trace_brackets_out',BRACKET_FIELDS,BRACKET_POST,'trace_i, worldid, _'))
    idx=kernel.body.index(branch)+1
    kernel.body[idx:idx]=writes('trace_ray_out',RAY_FIELDS,RAY_FINAL,'trace_i, worldid')+writes('trace_int_out',INT_FIELDS,dict(ls_used='trace_ls_used',final_recorded='1'),'trace_i, worldid')
    return ast.fix_missing_locations(factory)


def safe_diagnostic(node):
    for n in ast.walk(node):
        if isinstance(n,(ast.Assign,ast.AugAssign)):
            targets=n.targets if isinstance(n,ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t,ast.Name) and t.id in ['trace_i','trace_active','trace_ls_used']: continue
                if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id in OUTPUTS: continue
                raise ValueError('Diagnostic writes outside dedicated outputs')
        if isinstance(n,ast.Call):
            name=ast.unparse(n.func)
            if name in ['int','float']: continue
            if name=='wp.atomic_add' and ast.unparse(n.args[0])=='trace_error_out': continue
            raise ValueError('Diagnostic has unexpected callable: '+name)


class StripTrace(ast.NodeTransformer):
    def visit_If(self,node):
        if any(isinstance(n,ast.Name) and n.id.startswith('trace_') for n in ast.walk(node.test)):
            safe_diagnostic(node); return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id.startswith('trace_') for t in node.targets):
            safe_diagnostic(node); return None
        return self.generic_visit(node)
    def visit_AugAssign(self,node):
        if isinstance(node.target,ast.Name) and node.target.id.startswith('trace_'):
            safe_diagnostic(node); return None
        return self.generic_visit(node)
    def visit_arguments(self,node):
        node.args=[a for a in node.args if not a.arg.startswith('trace_')]
        return self.generic_visit(node)


def validate_generated(original_text,generated_text):
    original=function(original_text,ORIGINAL); generated=function(generated_text,GENERATED)
    canonical=instrument(original)
    dump=lambda n:ast.dump(n,include_attributes=False)
    if dump(canonical)!=dump(generated): raise ValueError('Generated factory differs from canonical instrumentation')
    stripped=StripTrace().visit(copy.deepcopy(generated)); stripped.name=ORIGINAL
    if dump(stripped)!=dump(original): raise ValueError('Physical factory AST changed')
    return dict(canonical_instrumentation=True,physical_AST_exact_after_removal=True,diagnostic_writes_only=True)


def generate(installed_pin,path):
    installed=verify_file_pin(installed_pin); text=installed.read_text(); original=function(text,ORIGINAL); candidate=instrument(original)
    assigned_names={n.id for n in ast.walk(original) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    local=assigned_names|{n.arg for n in ast.walk(original) if isinstance(n,ast.arg)}|{n.name for n in ast.walk(original) if isinstance(n,ast.FunctionDef)}
    external=sorted({n.id for n in ast.walk(original) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}-local-set(dir(builtins)))
    source='"""Generated diagnostic-only factory; original physical AST verified exactly."""\nfrom mujoco_warp._src.solver import (\n'+''.join('    '+n+',\n' for n in external)+')\n\n'+ast.unparse(candidate)+'\n'
    proof=validate_generated(text,source); path=Path(path)
    if path.exists(): raise ValueError('Fresh generated source required')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(source,encoding='utf-8',newline='\n'); verify_file_pin(installed_pin)
    return dict(installed_solver=installed_pin,generator=file_pin(__file__),generated=file_pin(path),source_proof=proof,aliased_original_globals=external,
        inputs=list(INPUTS),outputs=list(OUTPUTS),ray_fields=RAY_FIELDS,vector_fields=VECTOR_FIELDS,row_fields=ROW_FIELDS,int_fields=INT_FIELDS,bracket_fields=BRACKET_FIELDS,
        scope='Factory only;not a paired-phase observer,registered native worker,CUDA qualification or physical backend change/admission.')
