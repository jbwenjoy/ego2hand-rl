"""Owned native4 state, including cached tables and metric window scratch.

All saved view/expanded aliases are checked before any copy. Graphs, bridges,
streams and the synchronized last reset-ID upload are reconstruction resources.
This contract does not turn W32 byte restoration into deterministic physics.
"""
import copy
import hashlib
import numpy as np
import torch
import warp as wp
from . import pen_portable_state as base
from .owned_array_writes import prepare_writes
from .owned_rng import validate_rng
from .owned_pen_runtime import OwnedPenRuntime
from .pen_reference_cache import tensor_digest
from .portable_inventory import runtime_inventory,describe_inventory,path_value,assign_path
from .gpu_model_snapshot import model_leaves
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import verify_source_snapshot

SCHEMA='controlled_pen_owned_state:4'
EXCLUDED=base.EXCLUDED+((('attr','_binding_resources'),),(('attr','contact_metrics'),('attr','_reset_ids_owner')))


def inventory(runtime):
    if not isinstance(runtime,OwnedPenRuntime) or not runtime._owned_ready:raise ValueError('Live owned runtime required')
    runtime.validate_resources();env=runtime.env;s=env.sim;m=runtime.contact_metrics
    group=env.cfg.observations['policy'];term=group.terms['full_pose']
    if group.enable_corruption or group.history_length not in [None,0] or set(group.terms)!= {'full_pose'} or any(getattr(term,k)!=0 for k in ['history_length','delay_min_lag','delay_max_lag','delay_hold_prob','delay_update_period']):raise ValueError('Historyless controlled feature contract required')
    if any(not hasattr(env,k) for k in ['reset_buf','reward_buf','reset_terminated','reset_time_outs']):raise ValueError('Constructor-owned output aliases missing')
    inv=runtime_inventory(runtime,native_roots=(s.wp_model,s.wp_data,s._model_bridge,s._data_bridge,s._reset_mask,m.stream,m.graph,m.torch_stream),excluded_paths=EXCLUDED)
    for record in inv['excluded']:
        if record['path']==EXCLUDED[-1]:record['type']='synchronized_transient_reset_ID_resource'
    bad=[r for r in inv['opaque'] if r['kind']!='callable' and r['type'] not in base.STATIC_OPAQUE]
    if bad:raise ValueError('Unclassified owned runtime fields '+repr(bad[:5]))
    return inv


def immutable_paths(runtime,inv):
    cache=runtime.reference_cache
    wanted=[getattr(cache,k) for k in ['prefix','world_pose','mount','origins','world_ids']]+list(runtime.motion._frames.values())+[runtime.env.scene.env_origins]
    result={}
    for value in wanted:
        paths=[p for p,v in inv['tensors'].items() if v is value]
        if len(paths)!=1:raise ValueError('Exact canonical immutable tensor owner required')
        result[paths[0]]=tensor_digest(value)
    return result


def array_digest(value):
    value=np.asarray(value)
    return hashlib.sha256(str((value.shape,value.dtype)).encode()+value.tobytes()).hexdigest()


def immutable_arrays(runtime,inv):
    result={}
    for key in ['object_tag','digit','joint_ids','qids']:
        value=getattr(runtime.contact_metrics,key);paths=[p for p,v in inv['arrays'].items() if v is value]
        if len(paths)!=1:raise ValueError('Canonical immutable metric array required')
        result[paths[0]]=array_digest(value.numpy())
    return result


def immutable_scalars(runtime,inv):
    roots=[p for p in inv['objects'] if path_value(runtime,p) is runtime.reference_cache or path_value(runtime,p) is runtime.contact_metrics]
    result={p:copy.deepcopy(v) for p,v in inv['scalars'].items() if any(p[:len(root)]==root for root in roots)}
    result[(('attr','_owned_ready'),)]=True;result[(('attr','_window_pending'),)]=False
    return result


def snapshot_python(inv):
    return dict(description=describe_inventory(inv),tensors={p:v.detach().cpu().clone() for p,v in inv['tensors'].items()},
        arrays={p:(v.copy() if isinstance(v,np.ndarray) else v.numpy().copy()) for p,v in inv['arrays'].items()},scalars=copy.deepcopy(inv['scalars']))


def capture(runtime,sources):
    verify_source_snapshot(sources,live=True);torch.cuda.synchronize();wp.synchronize()
    runtime.validate_resources();runtime.reference_cache.validate(runtime.features);inv=inventory(runtime)
    native={}
    for k,root in [('model',runtime.env.sim.wp_model),('data',runtime.env.sim.wp_data)]:
        arrays,scalars=model_leaves(root);native[k]=dict(arrays={p:v.numpy().copy() for p,v in arrays.items()},scalars=scalars)
    return dict(schema=SCHEMA,metadata=runtime.metadata(),config=file_pin(runtime.config_path),worlds=runtime.num_envs,
        host_model_digest=base.host_model_digest(runtime.env.sim.mj_model),reference_signature=base.reference_signature(runtime),
        cache_metadata=runtime.reference_cache.metadata(),metric_metadata=runtime.contact_metrics.metadata(),python=snapshot_python(inv),native=native,
        immutable=immutable_paths(runtime,inv),immutable_arrays=immutable_arrays(runtime,inv),immutable_scalars=immutable_scalars(runtime,inv),
        rng=base.rng_state(),sources=sources,engine_sources=base.engine_sources(),
        exclusions=dict(paths=EXCLUDED,resources='Model/Data/bridges/reset-mask, metric Warp stream/graph/Torch external stream and process-local pointer bindings are reconstructed and verified.',
            reset_ID_owner='After global synchronization the last reset-ID tensor is only upload lifetime scratch. The reset mask, error, all window/source/instantaneous arrays ARE saved.',
            cache='Static tables ARE saved and checked against independently built current immutable signatures.'),
        scope='Owned cache/metric/native state. No trained module payload or future physical-repeat qualification.')


def preflight(runtime,state):
    if state.get('schema')!=SCHEMA or state['metadata']!=runtime.metadata() or state['worlds']!=runtime.num_envs:raise ValueError('Owned native4 runtime/schema contract mismatch')
    if state['config']!=file_pin(runtime.config_path) or state['host_model_digest']!=base.host_model_digest(runtime.env.sim.mj_model):raise ValueError('Owned config/host model mismatch')
    if state['reference_signature']!=base.reference_signature(runtime) or state['cache_metadata']!=runtime.reference_cache.metadata() or state['metric_metadata']!=runtime.contact_metrics.metadata():raise ValueError('Owned immutable reference/metric contract mismatch')
    verify_source_snapshot(state['sources'],live=True)
    for pin in state['engine_sources']:verify_file_pin(pin)
    validate_rng(state['rng'])
    torch.cuda.synchronize();wp.synchronize();inv=inventory(runtime);inv['_root']=runtime
    if any(set(inv[k])!=set(state['python'][k]) for k in ['arrays','tensors','scalars']):raise ValueError('Owned Python payload keys mismatch')
    base.validate_inventory(inv,state['python'])
    for path,value in state['python']['scalars'].items():
        if type(value)!=type(inv['scalars'][path]):raise ValueError('Saved scalar type mismatch '+repr(path))
    static=immutable_scalars(runtime,inv)
    if static!=state['immutable_scalars']:raise ValueError('Owned static scalar manifest mismatch')
    for path,value in static.items():
        if not base.scalar_equal(state['python']['scalars'][path],value):raise ValueError('Saved static scalar changed '+repr(path))
    static_arrays=immutable_arrays(runtime,inv)
    if static_arrays!=state['immutable_arrays']:raise ValueError('Owned static metric manifest mismatch')
    for path,digest in static_arrays.items():
        if array_digest(state['python']['arrays'][path])!=digest:raise ValueError('Saved static metric bytes changed '+repr(path))
    # Shapes alone do not protect immutable cached values or duplicated views.
    expected=immutable_paths(runtime,inv)
    if expected!=state['immutable']:raise ValueError('Immutable source/table fingerprint mismatch')
    for path,digest in expected.items():
        if tensor_digest(state['python']['tensors'][path])!=digest:raise ValueError('Saved immutable bytes changed '+repr(path))
    entries=[]
    for key,root in [('model',runtime.env.sim.wp_model),('data',runtime.env.sim.wp_data)]:
        arrays,scalars=model_leaves(root);saved=state['native'][key]
        if scalars!=saved['scalars'] or set(arrays)!=set(saved['arrays']):raise ValueError('Owned native scalar/array keys mismatch '+key)
        entries.extend((key+'.'+name,target,saved['arrays'][name]) for name,target in arrays.items())
    for key in ['arrays','tensors']:entries.extend(('python.'+repr(path),target,state['python'][key][path]) for path,target in inv[key].items())
    return inv,prepare_writes(entries)


@torch.no_grad()
def restore(runtime,state):
    inv,writes=preflight(runtime,state)
    for write in writes:write.apply()
    for path,value in state['python']['scalars'].items():assign_path(runtime,path,copy.deepcopy(value))
    torch.cuda.synchronize();wp.synchronize();base.restore_rng(state['rng'])
    runtime.validate_resources();runtime.reference_cache.validate(runtime.features)
    differences=[w.name for w in writes if not w.equal()]
    scalar_diff=[repr(p) for p,v in state['python']['scalars'].items() if not base.scalar_equal(path_value(runtime,p),v)]
    return dict(schema=SCHEMA,array_fields=len(writes),scalar_fields=len(inv['scalars']),all_array_bytes_equal=not differences,
        differing_fields=differences,all_scalar_values_equal=not scalar_diff,differing_scalar_fields=scalar_diff,
        pointers_preserved=True,scope='Immediate complete owned/native byte restoration only; physics future repeatability separate.')
