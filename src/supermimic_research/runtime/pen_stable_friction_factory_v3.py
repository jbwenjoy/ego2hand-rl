"""V3 uses isolated FP64 cost intermediates; reuses the frozen AST/owner checks."""
import ast
import builtins
import copy
import hashlib
from pathlib import Path

from ..artifacts import file_pin,verify_file_pin
from .pen_stable_friction_factory_v1 import (
    SINGLE,TRIPLE,FACTORY,NEW_FACTORY,node,candidate_helper,validate_generated,WRAPPERS,load_generated,verify_kernel_owner)


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
    header='"""Generated V3 FP64 cost / original FP32 derivatives;original OR stopping."""\nfrom mujoco_warp._src.solver import (\n'+''.join('    '+n+',\n' for n in external)+')\n'
    header+=f'from mujoco_warp._src.solver import {SINGLE} as _original_single, {TRIPLE} as _original_triple\n'
    header+='from supermimic_research.runtime.pen_stable_friction_cost_v3 import with_stable_cost as _stable_cost\n\n'
    header+='from mujoco_warp._src.warp_util import functools\n_KERNEL_CACHE = {}\n'+ast.unparse(cache_node)+'\n\n'
    source=header+'\n\n'.join(ast.unparse(f) for f in funcs)+'\n'+NEW_FACTORY+' = '+factory.name+'\n'+WRAPPERS
    proof=validate_generated(text,source)
    if path.exists(): raise ValueError('Fresh isolated V3 factory source required')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(source,encoding='utf-8',newline='\n'); verify_file_pin(installed_pin)
    return dict(schema='isolated_stable_friction_factory:3',installed_solver=installed_pin,generated=file_pin(path),generator=file_pin(__file__),
        cost_helper=file_pin(Path(__file__).with_name('pen_stable_friction_cost_v3.py')),installed_cache=cache_pin,
        shared_factory_adapter=file_pin(Path(__file__).with_name('pen_stable_friction_factory_v1.py')),
        precision='FP64_COST_INTERMEDIATES_FP32_RETURN_ORIGINAL_FP32_DERIVATIVES',
        cache_scope='MODULE_LOCAL_ORIGINAL_CACHE_DECORATOR_AST',source_proof=proof,original_aliases=external,
        stopping='INSTALLED_ORIGINAL_OR',cost_rule='STABLE_SAME_REGIME_WITH_ORIGINAL_CROSS_REGIME_FALLBACK',
        scope='Same two cost wrappers;V3 isolates new intermediates in FP64. Derivative/Hessian expressions,other branches,limits and factory algorithm preserved. No native qualification implied.')
