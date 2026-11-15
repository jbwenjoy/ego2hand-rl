"""Compatible independent one-world model/Data from explicitly saved public arrays."""
import json
import numpy as np
import mujoco

from ..artifacts import verify_file_pin
from .gpu_model_snapshot import capture_model,restore_model_arrays
from .saved_dynamic_arrays import restore_dynamic_arrays


def saved_native_plant(case,source):
    import mujoco_warp as mw
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(case['host_topology'])))
    for key,value in dict(np.load(verify_file_pin(case['GPU_physical_context']))).items():getattr(m,key)[:]=value[0]
    values=dict(np.load(verify_file_pin(case['fixed_model_arrays'])));scalars=json.loads(verify_file_pin(case['fixed_model_scalars']).read_text())
    dm=mw.put_model(m);restore_model_arrays(dm,values,scalars)
    current,current_scalars,_=capture_model(dm)
    if current_scalars!=scalars or any(current[k].tobytes()!=v.tobytes() for k,v in values.items()):raise ValueError('Saved model not exact')
    builder=dict(nworld=source['qpos'].shape[0],naconmax=len(source['contact.geom']),njmax=source['efc.force'].shape[1])
    data=mw.make_data(m,**builder);receipt=restore_dynamic_arrays(data,source)
    return m,dm,data,dict(builder=builder,Data_static_scalars=receipt['fresh_static_scalars'],
        Data_static_role='RECONSTRUCTED_COMPATIBLE_NOT_ORIGINAL_OBSERVATION',model_arrays=len(values),model_scalars=len(scalars),model_exact=True)
