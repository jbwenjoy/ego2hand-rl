"""Isolated unchanged factory calling the measured guarded helper topology."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from ..artifacts import file_pin,verify_file_pin
from .pen_termination_rule import _loaded_code_matches
from .pen_stable_friction_factory_v1 import SINGLE,TRIPLE,FACTORY,NEW_FACTORY,node,WRAPPERS,verify_kernel_owner


def qualify_helpers(installed_pin):
    from mujoco_warp._src import solver
    from . import pen_guarded_stable_friction_v1 as h
    source=file_pin(h.__file__); text=verify_file_pin(source).read_text()
    reference=file_pin(Path(__file__).with_name('pen_friction_rounding_probe_v2.py')); measured=verify_file_pin(reference).read_text()
    for n in ['same_regime','guarded_single','guarded_triple']:
        if ast.dump(node(text,n),include_attributes=False)!=ast.dump(node(measured,n),include_attributes=False) or not _loaded_code_matches(getattr(h,n).func,source['path']):
            raise ValueError('Measured guarded helper AST/code required: '+n)
    for n,s in [('original_single',SINGLE),('original_triple',TRIPLE)]:
        if getattr(h,n) is not getattr(solver,s) or not _loaded_code_matches(getattr(h,n).func,installed_pin['path']):
            raise ValueError('Untouched original fallback helper required')
    stable=file_pin(Path(__file__).with_name('pen_stable_friction_cost_v3.py'))
    if not _loaded_code_matches(h.with_stable_cost.func,stable['path']): raise ValueError('Qualified FP64 cost body required')
    return dict(guarded_helper=source,measured_helper_reference=reference,cost_helper=stable)


def render(original,cache,path):
    factory=copy.deepcopy(node(original,FACTORY)); factory.name=NEW_FACTORY+'_owner_'+hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:24]
    assigned={n.id for n in ast.walk(factory) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
    assigned|={n.arg for n in ast.walk(factory) if isinstance(n,ast.arg)}|{n.name for n in ast.walk(factory) if isinstance(n,ast.FunctionDef)}
    loaded={n.id for n in ast.walk(factory) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
    external=sorted(loaded-assigned-set(dir(builtins))-{SINGLE,TRIPLE,'cache_kernel'})
    header='"""Generated guarded cost factory;original OR stopping and fallbacks."""\nfrom mujoco_warp._src.solver import (\n'+''.join('    '+n+',\n' for n in external)+')\n'
    header+=f'from mujoco_warp._src.solver import {SINGLE} as _original_single, {TRIPLE} as _original_triple\n'
    header+=f'from supermimic_research.runtime.pen_guarded_stable_friction_v1 import guarded_single as {SINGLE}, guarded_triple as {TRIPLE}\n'
    header+='from supermimic_research.runtime.pen_stable_friction_cost_v3 import with_stable_cost as _stable_cost\n'
    header+='from mujoco_warp._src.warp_util import functools\n_KERNEL_CACHE = {}\n'+ast.unparse(node(cache,'cache_kernel'))+'\n\n'
    return header+ast.unparse(factory)+'\n'+NEW_FACTORY+' = '+factory.name+'\n'+WRAPPERS,external


def validate_generated(installed_pin,cache_pin,path):
    expected,_=render(verify_file_pin(installed_pin).read_text(),verify_file_pin(cache_pin).read_text(),path)
    if Path(path).read_text()!=expected: raise ValueError('Exact unchanged-factory/guarded-helper source required')
    return dict(factory_AST_unchanged_except_owner_name=True,original_fallback_helpers_unmodified=True,measured_guarded_helper_AST=True,module_local_original_cache=True)


def generate(installed_pin,path):
    from mujoco_warp._src import warp_util
    path=Path(path).resolve(); cache=file_pin(warp_util.__file__); helpers=qualify_helpers(installed_pin)
    source,external=render(verify_file_pin(installed_pin).read_text(),verify_file_pin(cache).read_text(),path)
    if path.exists(): raise ValueError('Fresh guarded factory source required')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(source,encoding='utf-8',newline='\n')
    proof=validate_generated(installed_pin,cache,path)
    return dict(schema='isolated_stable_friction_factory:4',installed_solver=installed_pin,generated=file_pin(path),generator=file_pin(__file__),installed_cache=cache,
        **helpers,shared_factory_adapter=file_pin(Path(__file__).with_name('pen_stable_friction_factory_v1.py')),source_proof=proof,original_aliases=external,
        stopping='INSTALLED_ORIGINAL_OR',cost_rule='GUARDED_STABLE_SAME_REGIME_WITH_UNWRAPPED_ORIGINAL_FALLBACK',
        precision='FP64_COST_INTERMEDIATES_FP32_RETURN_ORIGINAL_FP32_DERIVATIVES',cache_scope='MODULE_LOCAL_ORIGINAL_CACHE_DECORATOR_AST',
        scope='Original factory AST,friction-only guarded calls. No installed functions or caches changed. Arithmetic evidence does not qualify native dynamics.')


def load_generated(manifest):
    from mujoco_warp._src import solver
    from . import pen_guarded_stable_friction_v1 as helpers
    for key in ['generated','generator','guarded_helper','measured_helper_reference','cost_helper','shared_factory_adapter','installed_solver','installed_cache']: verify_file_pin(manifest[key])
    if qualify_helpers(manifest['installed_solver'])!={k:manifest[k] for k in ['guarded_helper','measured_helper_reference','cost_helper']}: raise ValueError('Guarded helper identity changed')
    path=Path(manifest['generated']['path']); validate_generated(manifest['installed_solver'],manifest['installed_cache'],path)
    name='isolated_guarded_stable_friction_'+hashlib.sha256(json.dumps(manifest['generated'],sort_keys=True).encode()).hexdigest()
    if name in sys.modules:
        module=sys.modules[name]
        if file_pin(module.__file__)!=manifest['generated']: raise ValueError('Guarded module namespace collision')
    else:
        spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec); sys.modules[name]=module
        try: spec.loader.exec_module(module)
        except BaseException: sys.modules.pop(name,None); raise
    for n in manifest['original_aliases']:
        if getattr(module,n) is not getattr(solver,n): raise ValueError('Original factory alias changed: '+n)
    for n,expected in [(SINGLE,helpers.guarded_single),(TRIPLE,helpers.guarded_triple),('_original_single',solver._compute_efc_eval_pt_pyramidal),('_original_triple',solver._compute_efc_eval_pt_3alphas_pyramidal),('_stable_cost',helpers.with_stable_cost)]:
        if getattr(module,n) is not expected: raise ValueError('Guarded/original helper alias changed: '+n)
    return module
