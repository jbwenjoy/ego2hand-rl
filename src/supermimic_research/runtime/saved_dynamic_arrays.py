"""Strict saved-array import into an independent compatible dynamic Data object."""
import numpy as np

from .gpu_model_snapshot import model_leaves


def restore_dynamic_arrays(data,values):
    """Validate ALL keys/shapes/dtypes before writing; preserve scratch NaN bytes.

    This restores arrays only. It does not assert that fresh static Data
    scalars were observed in the old packet. The caller must label any
    reconstructed capacities and check active physical fields separately.
    """
    arrays,scalars=model_leaves(data)
    if set(arrays)!=set(values):raise ValueError('Exact saved dynamic array keys required')
    layout={}
    for key,target in arrays.items():
        template=target.numpy();value=np.asarray(values[key])
        if value.shape!=template.shape or value.dtype!=template.dtype:
            raise ValueError(f'Saved dynamic array layout mismatch: {key}')
        layout[key]=(target.ptr,target.shape,target.dtype,target.strides)
    for key,target in arrays.items():
        if target.size:target.assign(values[key])
    after,current=model_leaves(data)
    if current!=scalars:raise ValueError('Array restore changed static scalars')
    for key,target in after.items():
        if (target.ptr,target.shape,target.dtype,target.strides)!=layout[key] or target.numpy().tobytes()!=np.asarray(values[key]).tobytes():
            raise ValueError(f'Saved dynamic bytes/pointer not restored: {key}')
    return dict(arrays=len(arrays),array_bytes_exact=True,pointers_unchanged=True,
                fresh_static_scalars=scalars,scope='Exact arrays, not proof of original unrecorded static scalar values.')
