"""Keep full padded copies; use only actual physical DOFs for force arithmetic."""
import numpy as np
from .pen_solve_exit_epoch import review as original_review


def physical_view(observations,nv):
    raw=np.asarray(observations['context.grad'])
    if nv==26:
        mg=np.asarray(observations['context.Mgrad'])
        if (raw.ndim!=3 or raw.shape[1:]!=(1,32) or mg.shape!=(raw.shape[0],1,0)
                or not np.isfinite(raw).all() or not np.all(raw[...,26:]==0)):
            raise ValueError('Full finite32-gradient/zero padding/empty Newton Mgrad required')
    elif raw.shape[-1]!=nv:
        raise ValueError('Unpadded analytic layout required outside the fixed Hand2 case')
    result=dict(observations);result['context.grad']=raw[...,:nv]
    return result


def review(packet,observations,model_arrays,case,decoded,decoder_gravity,iteration_cap):
    nv=packet['post_qvel'].shape[-1]
    if nv!=26:raise ValueError('Fixed Hand2 physical26-DOF epoch required')
    physical=physical_view(observations,nv)
    result,arrays=original_review(packet,physical,model_arrays,case,decoded,decoder_gravity,iteration_cap)
    result['context_layout']=dict(version=2,physical_nv=26,padded_gradient_width=32,Mgrad_shape=list(observations['context.Mgrad'].shape),padding_all_zero=True,raw_copies_preserved=True)
    return result,arrays
