"""Isolated cost helpers/factory; no installed helper or old cache is mutated."""
import ast
import builtins
import copy
import importlib.util
import inspect
import hashlib
import json
from pathlib import Path
import sys

from ..artifacts import file_pin,verify_file_pin

SINGLE='_compute_efc_eval_pt_pyramidal'
TRIPLE='_compute_efc_eval_pt_3alphas_pyramidal'
FACTORY='_linesearch_iterative_kernel'
NEW_FACTORY='_linesearch_iterative_kernel_stable_friction_v1'


def node(text,name):
    found=[n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name]
    if len(found)!=1: raise ValueError('One exact original function required: '+name)
    return found[0]


def candidate_helper(original):
    result=copy.deepcopy(original)
    branches=[n for n in result.body if isinstance(n,ast.If) and ast.unparse(n.test)=='efcid >= ne']
    if len(branches)!=1: raise ValueError('Exact friction branch required')
    ret=[n for n in branches[0].body if isinstance(n,ast.Return)]
    if len(ret)!=1: raise ValueError('One friction return required')
    def wrap(value,alpha):
        return ast.Call(func=ast.Name(id='_stable_cost',ctx=ast.Load()),args=[value]+[ast.Name(id=k,ctx=ast.Load()) for k in ['ctx_Jaref','ctx_jv','efc_D','f',alpha]],keywords=[])
    if result.name==SINGLE:
        if not isinstance(ret[0].value,ast.Call) or ast.unparse(ret[0].value.func)!='_shift_cost': raise ValueError('Original single cost shift required')
        ret[0].value=wrap(ret[0].value,'alpha')
    elif result.name==TRIPLE:
        if not isinstance(ret[0].value,ast.Tuple) or len(ret[0].value.elts)!=3: raise ValueError('Original three-point return required')
        ret[0].value.elts=[wrap(v,a) for v,a in zip(ret[0].value.elts,['lo_alpha','hi_alpha','mid_alpha'])]
    else: raise ValueError('Only declared pyramidal helpers may change')
    return ast.fix_missing_locations(result)


class RemoveCostWrapper(ast.NodeTransformer):
    def visit_Call(self,n):
        if isinstance(n.func,ast.Name) and n.func.id=='_stable_cost':
            if len(n.args)!=6 or n.keywords: raise ValueError('Exact scalar cost-only wrapper required')
            return self.visit(n.args[0])
        return self.generic_visit(n)


def validate_generated(original_text,generated_text):
    dump=lambda x:ast.dump(x,include_attributes=False)
    for name in [SINGLE,TRIPLE]:
        old=node(original_text,name); new=node(generated_text,name)
        if dump(new)!=dump(candidate_helper(old)) or dump(RemoveCostWrapper().visit(copy.deepcopy(new)))!=dump(old):
            raise ValueError('Unexpected non-cost helper change: '+name)
    candidates=[n for n in ast.parse(generated_text).body if isinstance(n,ast.FunctionDef) and n.name.startswith(NEW_FACTORY+'_owner_')]
    if len(candidates)!=1: raise ValueError('One isolated owner factory required')
    new=candidates[0]; unique_name=new.name
    aliases=[n for n in ast.parse(generated_text).body if isinstance(n,ast.Assign) and ast.unparse(n)=='%s = %s'%(NEW_FACTORY,unique_name)]
    if len(aliases)!=1: raise ValueError('Exact isolated owner factory export required')
    old=node(original_text,FACTORY); new.name=FACTORY
    if dump(old)!=dump(new): raise ValueError('Original factory physical AST changed')
    return dict(single_cost_wrapper_only=True,triple_cost_wrappers_only=True,factory_AST_unchanged_except_owner_name=True)


WRAPPERS='''
@wp.kernel
def evaluate_cost_helpers(jaref:wp.array[float],jv:wp.array[float],d:wp.array[float],loss:wp.array[float],alpha:wp.array[float],
                         old_single:wp.array2d[float],new_single:wp.array2d[float],old_three:wp.array3d[float],new_three:wp.array3d[float]):
    i=wp.tid(); n=jaref.shape[0]
    old=_original_single(i,alpha[i],0,n,d[i],loss,jaref[i],jv[i])
    new=_compute_efc_eval_pt_pyramidal(i,alpha[i],0,n,d[i],loss,jaref[i],jv[i])
    a,b,c=_original_triple(i,alpha[i],alpha[i],alpha[i],0,n,d[i],loss,jaref[i],jv[i])
    x,y,z=_compute_efc_eval_pt_3alphas_pyramidal(i,alpha[i],alpha[i],alpha[i],0,n,d[i],loss,jaref[i],jv[i])
    for k in range(3):
        old_single[i,k]=old[k]; new_single[i,k]=new[k]
        old_three[i,0,k]=a[k]; old_three[i,1,k]=b[k]; old_three[i,2,k]=c[k]
        new_three[i,0,k]=x[k]; new_three[i,1,k]=y[k]; new_three[i,2,k]=z[k]

@wp.kernel
def evaluate_declared_rows(efcid:wp.array[int],ne:wp.array[int],nf:wp.array[int],jaref:wp.array[float],jv:wp.array[float],d:wp.array[float],loss:wp.array[float],alphas:wp.array2d[float],
                          old_single:wp.array2d[float],new_single:wp.array2d[float],old_three:wp.array3d[float],new_three:wp.array3d[float]):
    i=wp.tid()
    old=_original_single(efcid[i],alphas[i,0],ne[i],nf[i],d[i],loss,jaref[i],jv[i])
    new=_compute_efc_eval_pt_pyramidal(efcid[i],alphas[i,0],ne[i],nf[i],d[i],loss,jaref[i],jv[i])
    a,b,c=_original_triple(efcid[i],alphas[i,0],alphas[i,1],alphas[i,2],ne[i],nf[i],d[i],loss,jaref[i],jv[i])
    x,y,z=_compute_efc_eval_pt_3alphas_pyramidal(efcid[i],alphas[i,0],alphas[i,1],alphas[i,2],ne[i],nf[i],d[i],loss,jaref[i],jv[i])
    for k in range(3):
        old_single[i,k]=old[k]; new_single[i,k]=new[k]
        old_three[i,0,k]=a[k]; old_three[i,1,k]=b[k]; old_three[i,2,k]=c[k]
        new_three[i,0,k]=x[k]; new_three[i,1,k]=y[k]; new_three[i,2,k]=z[k]
'''


def generate(installed_pin,path):
    from mujoco_warp._src import warp_util
    path=Path(path).resolve()
    text=verify_file_pin(installed_pin).read_text(); funcs=[candidate_helper(node(text,n)) for n in [SINGLE,TRIPLE]]
    factory=copy.deepcopy(node(text,FACTORY)); factory.name=NEW_FACTORY+'_owner_'+hashlib.sha256(str(path).encode()).hexdigest()[:24]; funcs.append(factory)
    assigned=set(); loaded=set()
    for fn in funcs:
        assigned|={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        assigned|={n.arg for n in ast.walk(fn) if isinstance(n,ast.arg)}|{n.name for n in ast.walk(fn) if isinstance(n,ast.FunctionDef)}
        loaded|={n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    external=sorted(loaded-assigned-set(dir(builtins))-{SINGLE,TRIPLE,'_stable_cost','cache_kernel'})
    cache_pin=file_pin(warp_util.__file__); cache_node=node(verify_file_pin(cache_pin).read_text(),'cache_kernel')
    header='"""Generated isolated same-regime friction-cost backend;original OR stopping."""\nfrom mujoco_warp._src.solver import (\n'+''.join('    '+n+',\n' for n in external)+')\n'
    header+=f'from mujoco_warp._src.solver import {SINGLE} as _original_single, {TRIPLE} as _original_triple\n'
    header+='from supermimic_research.runtime.pen_stable_friction_cost_v1 import with_stable_cost as _stable_cost\n\n'
    header+='from mujoco_warp._src.warp_util import functools\n_KERNEL_CACHE = {}\n'+ast.unparse(cache_node)+'\n\n'
    source=header+'\n\n'.join(ast.unparse(f) for f in funcs)+'\n'+NEW_FACTORY+' = '+factory.name+'\n'+WRAPPERS
    proof=validate_generated(text,source); path=Path(path)
    if path.exists(): raise ValueError('Fresh isolated factory source required')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(source,encoding='utf-8',newline='\n'); verify_file_pin(installed_pin)
    return dict(schema='isolated_stable_friction_factory:1',installed_solver=installed_pin,generated=file_pin(path),generator=file_pin(__file__),
        cost_helper=file_pin(Path(__file__).with_name('pen_stable_friction_cost_v1.py')),installed_cache=cache_pin,
        cache_scope='MODULE_LOCAL_ORIGINAL_CACHE_DECORATOR_AST',source_proof=proof,original_aliases=external,
        stopping='INSTALLED_ORIGINAL_OR',cost_rule='STABLE_SAME_REGIME_WITH_ORIGINAL_CROSS_REGIME_FALLBACK',
        scope='Two pyramidal cost-return wrappers only;all derivative/Hessian expressions,other branches,limits and factory algorithm preserved.')


def load_generated(manifest):
    from mujoco_warp._src import solver
    path=verify_file_pin(manifest['generated']); verify_file_pin(manifest['generator']); verify_file_pin(manifest['cost_helper'])
    validate_generated(verify_file_pin(manifest['installed_solver']).read_text(),path.read_text())
    if ast.dump(node(verify_file_pin(manifest['installed_cache']).read_text(),'cache_kernel'),include_attributes=False)!=ast.dump(node(path.read_text(),'cache_kernel'),include_attributes=False):
        raise ValueError('Original local cache decorator AST required')
    # Warp retains module/kernel registries after a Python module is unloaded.
    # Include the pinned path as well as bytes so distinct owners cannot collide.
    name='isolated_stable_friction_'+hashlib.sha256(json.dumps(manifest['generated'],sort_keys=True).encode()).hexdigest()
    if name in sys.modules:
        module=sys.modules[name]
        if file_pin(module.__file__)!=manifest['generated']: raise ValueError('Generated namespace collision')
    else:
        spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module
        try: spec.loader.exec_module(module)
        except BaseException: sys.modules.pop(name,None); raise
    for n in manifest['original_aliases']:
        if getattr(module,n) is not getattr(solver,n): raise ValueError('Original factory helper alias changed: '+n)
    if module._original_single is not getattr(solver,SINGLE) or module._original_triple is not getattr(solver,TRIPLE): raise ValueError('Original reference helper differs')
    return module


def verify_kernel_owner(module,kernel):
    from mujoco_warp._src import solver
    if kernel.func.__globals__ is not module.__dict__: raise ValueError('Old cached kernel returned for cost candidate')
    closure=inspect.getclosurevars(kernel.func).nonlocals
    if closure['_compute_efc_eval_pt'] is not getattr(module,SINGLE) or closure['_compute_efc_eval_pt_3alphas'] is not getattr(module,TRIPLE):
        raise ValueError('Cost kernel closure uses wrong helpers')
    if closure['_compute_efc_eval_pt_alpha_zero'] is not solver._compute_efc_eval_pt_alpha_zero_pyramidal:
        raise ValueError('Unchanged alpha-zero helper required')
    return True
