"""Portable controlled-pen state; array-copy audit is separate from replay qualification."""
import copy
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
from .portable_inventory import runtime_inventory,describe_inventory,path_value,assign_path,readonly_scalar_path
from .gpu_model_snapshot import model_leaves
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot

SCHEMA='controlled_pen_native_portable_state:3'
EXCLUDED=((('attr','env'),('attr','extras')),(('attr','env'),('attr','obs_buf')),
    (('attr','env'),('attr','observation_manager'),('attr','_obs_buffer')))
STATIC_OPAQUE={'functools.partial','warp._src.context.Device','warp._src.context.Graph',
    'mujoco._structs.MjModel','mujoco._structs.MjData',
    *{'mujoco._specs.'+n for n in ['MjSpec','MjsBody','MjsJoint','MjsGeom','MjsActuator',
        'MjsMaterial','MjsSite','MjsTexture','MjsLight']}}


def inventory(adapter):
    sim=adapter.env.sim
    group=adapter.env.cfg.observations['policy']
    if group.enable_corruption or group.history_length not in [None,0] or set(group.terms)!= {'full_pose'}:
        raise ValueError('Portable output-cache exclusion requires perfect historyless full-pose group')
    term=group.terms['full_pose']
    if any(getattr(term,key)!=0 for key in ['history_length','delay_min_lag','delay_max_lag','delay_hold_prob','delay_update_period']):
        raise ValueError('No observation history/delay state may be excluded')
    # ManagerBasedRlEnv creates these four boundary fields on its FIRST step.
    # Materialize their declared storage/aliases without executing physics.
    # No other missing runtime field is repaired or silently skipped.
    env=adapter.env
    if not hasattr(env,'reset_buf'):env.reset_buf=torch.zeros_like(env._manual_reset_pending)
    if not hasattr(env,'reward_buf'):env.reward_buf=env.reward_manager._reward_buf
    if not hasattr(env,'reset_terminated'):env.reset_terminated=env.termination_manager._terminated_buf
    if not hasattr(env,'reset_time_outs'):env.reset_time_outs=env.termination_manager._truncated_buf
    inv=runtime_inventory(adapter,native_roots=(sim.wp_model,sim.wp_data,sim._model_bridge,sim._data_bridge,sim._reset_mask),
        excluded_paths=EXCLUDED)
    bad=[r for r in inv['opaque'] if r['kind']!='callable' and r['type'] not in STATIC_OPAQUE]
    if bad:raise ValueError('Unclassified mutable closure leaves: '+repr(bad[:5]))
    return inv


def host_model_digest(model):
    import mujoco
    buf=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8);mujoco.mj_saveModel(model,None,buf)
    return hashlib.sha256(buf.tobytes()).hexdigest()


def tensor_bytes(value):return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def reference_signature(adapter):
    digest=hashlib.sha256()
    for key,value in sorted(adapter.motion._frames.items()):
        digest.update(key.encode());digest.update(str((tuple(value.shape),value.dtype)).encode());digest.update(tensor_bytes(value))
    return digest.hexdigest()


def scalar_equal(actual,expected):
    if type(actual)!=type(expected):return False
    if isinstance(actual,(float,np.floating)) and np.isnan(actual) and np.isnan(expected):
        return np.asarray(actual).tobytes()==np.asarray(expected).tobytes()
    return bool(actual==expected)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def engine_sources():
    return [file_pin(m.__file__) for name,m in sorted(sys.modules.items())
        if name.startswith(('mjlab.','mujoco_warp.','warp.')) and getattr(m,'__file__','').endswith('.py')]


def validate_inventory(current,saved):
    # A deliberate strict contract: extra/missing mutable leaves are never ignored.
    now=describe_inventory(current)
    if now!=saved['description']:
        changes={key:dict(saved=len(saved['description'][key]),current=len(now[key]))
            for key in now if now[key]!=saved['description'][key]}
        raise ValueError('Portable Python closure layout/alias mismatch: '+repr(changes))
    for path,target in current['tensors'].items():
        value=saved['tensors'][path]
        if target.shape!=value.shape or target.dtype!=value.dtype:raise ValueError('Tensor layout mismatch')
    for path,target in current['arrays'].items():
        value=saved['arrays'][path];template=target if isinstance(target,np.ndarray) else target.numpy()
        if template.shape!=value.shape or template.dtype!=value.dtype:raise ValueError('Array layout mismatch')
    for path,value in saved['scalars'].items():
        if path and readonly_scalar_path(current['_root'],path):
            if path_value(current['_root'],path)!=value:raise ValueError('Changed immutable tuple/frozen scalar')


def capture(adapter,source_receipt):
    import warp as wp
    verify_source_snapshot(source_receipt,live=True);torch.cuda.synchronize();wp.synchronize()
    inv=inventory(adapter)
    py=dict(description=describe_inventory(inv),tensors={p:v.detach().cpu().clone() for p,v in inv['tensors'].items()},
        arrays={p:(v.copy() if isinstance(v,np.ndarray) else v.numpy().copy()) for p,v in inv['arrays'].items()},
        scalars=copy.deepcopy(inv['scalars']))
    native={}
    for key,root in [('model',adapter.env.sim.wp_model),('data',adapter.env.sim.wp_data)]:
        arrays,scalars=model_leaves(root)
        native[key]=dict(arrays={k:v.numpy().copy() for k,v in arrays.items()},scalars=scalars)
    return dict(schema=SCHEMA,metadata=adapter.metadata(),config=file_pin(adapter.config_path),worlds=adapter.num_envs,
        host_model_digest=host_model_digest(adapter.env.sim.mj_model),reference_signature=reference_signature(adapter),python=py,native=native,
        rng=rng_state(),sources=source_receipt,engine_sources=engine_sources(),
        exclusions=dict(bridge='WarpBridge caches are native-array views/stream resources; model/Data copied separately.',
            paths=EXCLUDED,reason='Controlled historyless perfect observation is regenerated; extras/obs caches are overwritten, not future-physics state.',
            opaque='Compiled specs/host model/host setup data and CUDA graph/device resources are reconstructed; exact host digest/layout/source pins required.'),
        scope='Restricted native controlled adapter. No trained-module payload or physics replay qualification implied.')


@torch.no_grad()
def restore(adapter,state):
    import warp as wp
    if state['schema']!=SCHEMA or state['metadata']!=adapter.metadata() or state['worlds']!=adapter.num_envs:
        raise ValueError('Portable pen schema/161/188/world contract mismatch')
    if file_pin(adapter.config_path)!=state['config'] or host_model_digest(adapter.env.sim.mj_model)!=state['host_model_digest']:
        raise ValueError('Portable config/host topology mismatch')
    if reference_signature(adapter)!=state['reference_signature']:raise ValueError('Portable reference content mismatch')
    verify_source_snapshot(state['sources'],live=True)
    for pin in state['engine_sources']:verify_file_pin(pin)
    torch.cuda.synchronize();wp.synchronize()
    current=inventory(adapter);current['_root']=adapter;validate_inventory(current,state['python'])
    targets={};pointers={}
    for key,root in [('model',adapter.env.sim.wp_model),('data',adapter.env.sim.wp_data)]:
        arrays,scalars=model_leaves(root);saved=state['native'][key]
        if scalars!=saved['scalars'] or set(arrays)!=set(saved['arrays']):raise ValueError('Native scalar/key mismatch')
        for field,target in arrays.items():
            value=saved['arrays'][field];template=target.numpy()
            if template.shape!=value.shape or template.dtype!=value.dtype:raise ValueError('Native layout mismatch '+key+'.'+field)
            pointers[(key,field)]=target.ptr
        targets[key]=arrays
    # Every destination has been checked before any state mutation.
    for key,arrays in targets.items():
        for field,target in arrays.items():
            if target.size:target.assign(state['native'][key]['arrays'][field])
    for path,target in current['arrays'].items():
        value=state['python']['arrays'][path]
        if isinstance(target,np.ndarray):target[:]=value
        elif target.size:target.assign(value)
    for path,target in current['tensors'].items():
        value=state['python']['tensors'][path].to(target.device)
        for dim in reversed(range(target.ndim)):
            if target.shape[dim]>1 and target.stride(dim)==0:
                # Expanded static views have one storage value per repeated row.
                first=value.select(dim,0).unsqueeze(dim).expand_as(value)
                if tensor_bytes(first)!=tensor_bytes(value):raise ValueError('Incompatible expanded tensor values')
                target=target.select(dim,0);value=value.select(dim,0)
        target.copy_(value)
    for path,value in state['python']['scalars'].items():assign_path(adapter,path,copy.deepcopy(value))
    torch.cuda.synchronize();wp.synchronize();restore_rng(state['rng'])
    differences=[];count=0
    for key,arrays in targets.items():
        for field,target in arrays.items():
            count+=1
            if target.ptr!=pointers[(key,field)] or target.numpy().tobytes()!=state['native'][key]['arrays'][field].tobytes():
                differences.append(key+'.'+field)
    for path,target in current['arrays'].items():
        count+=1;value=target if isinstance(target,np.ndarray) else target.numpy()
        if value.tobytes()!=state['python']['arrays'][path].tobytes():differences.append(repr(path))
    for path,target in current['tensors'].items():
        count+=1
        if tensor_bytes(target)!=tensor_bytes(state['python']['tensors'][path]):differences.append(repr(path))
    scalar_differences=[repr(path) for path,value in state['python']['scalars'].items() if not scalar_equal(path_value(adapter,path),value)]
    return dict(array_fields=count,all_array_bytes_equal=not differences,differing_fields=differences,
        all_scalar_values_equal=not scalar_differences,differing_scalar_fields=scalar_differences,
        scalar_fields=len(current['scalars']),scope='Immediate array/closure restore only, no physical replay qualification.')
