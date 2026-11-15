"""Compatible independent one-world model/Data from explicitly saved public arrays."""
import json
import numpy as np
import mujoco

from ..artifacts import verify_file_pin
from .gpu_model_snapshot import capture_model,restore_model_arrays
from .saved_dynamic_arrays import restore_dynamic_arrays
from .gpu_model_snapshot import model_leaves


def observed_data_builder(observation,source,model_nv):
    expected={'nworld','naconmax','naccdmax','njmax','nvmax','nvmax_pad','njmax_pad','njmax_nnz'}
    scalars=observation.get('scalars',{})
    if (observation.get('schema')!='observed_live_dynamic_Data_layout:1' or set(scalars)!=expected
            or set(observation.get('arrays',{}))!=set(source)
            or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in scalars.values())
            or scalars['nworld']!=1 or scalars['nvmax']>model_nv):
        raise ValueError('Exact recorded one-world Data observations required')
    builder={k:scalars[k] for k in ['nworld','naconmax','naccdmax','njmax','njmax_nnz']}
    # Specifying nvmax enables extra compact workspaces even when equal to nv.
    # Empty observed workspaces require the original noncompact constructor.
    if source['cM'].size:builder['nvmax']=scalars['nvmax']
    elif scalars['nvmax']!=model_nv:raise ValueError('Noncompact Data must use model nv')
    return builder


def saved_native_plant(case,source,*,observed_layout=None):
    import mujoco_warp as mw
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['host_topology'])))
    for key,value in dict(np.load(verify_file_pin(case['GPU_physical_context']))).items():getattr(m,key)[:]=value[0]
    values=dict(np.load(verify_file_pin(case['fixed_model_arrays'])));scalars=json.loads(verify_file_pin(case['fixed_model_scalars']).read_text())
    dm=mw.put_model(m);restore_model_arrays(dm,values,scalars)
    current,current_scalars,_=capture_model(dm)
    if current_scalars!=scalars or any(current[k].tobytes()!=v.tobytes() for k,v in values.items()):raise ValueError('Saved model not exact')
    observation=json.loads(verify_file_pin(observed_layout).read_text()) if observed_layout is not None else None
    builder=(observed_data_builder(observation,source,m.nv) if observation is not None else
        dict(nworld=source['qpos'].shape[0],naconmax=len(source['contact.geom']),njmax=source['efc.force'].shape[1]))
    data=mw.make_data(m,**builder);receipt=restore_dynamic_arrays(data,source)
    if observation is not None:
        arrays,current=model_leaves(data)
        metadata={k:dict(shape=list(v.shape),dtype=str(v.dtype),strides=list(v.strides)) for k,v in arrays.items()}
        if current!=observation['scalars'] or metadata!=observation['arrays']:raise ValueError('Fresh Data does not match recorded scalars/layout')
    return m,dm,data,dict(builder=builder,Data_static_scalars=receipt['fresh_static_scalars'],
        Data_static_role='MATCHED_RECORDED_OBSERVATIONS' if observation is not None else 'RECONSTRUCTED_COMPATIBLE_NOT_ORIGINAL_OBSERVATION',
        Data_observation=observed_layout,model_arrays=len(values),model_scalars=len(scalars),model_exact=True)
