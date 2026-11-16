"""Keep full-row proxy scores separate from source-base derivative outputs."""
import numpy as np


def add_reward_arrays(output,prefix,row_values,base_arrays):
    expected={'values','RAW','PD_partial','RAW_direct_partial','controller_diagonal','computed_PD'}
    if set(base_arrays)!=expected or row_values.ndim!=2 or row_values.shape[1]!=2:
        raise ValueError('Explicit full-row scores and all source-base derivative arrays required')
    b=len(base_arrays['values'])
    shapes=dict(values=(b,2),RAW=(b,2,20),PD_partial=(b,2,20),RAW_direct_partial=(b,2,20),
                controller_diagonal=(b,20),computed_PD=(b,20))
    if any(v.shape!=shapes[k] or not np.isfinite(v).all() for k,v in base_arrays.items()) or not np.isfinite(row_values).all():
        raise ValueError('Finite shape-aligned original arrays required')
    packed={prefix+'values':row_values,**{prefix+('base_values' if k=='values' else k):v for k,v in base_arrays.items()}}
    if len(packed)!=7 or any(k in output for k in packed):raise ValueError('Refuse score/base namespace overwrite')
    output.update(packed)
