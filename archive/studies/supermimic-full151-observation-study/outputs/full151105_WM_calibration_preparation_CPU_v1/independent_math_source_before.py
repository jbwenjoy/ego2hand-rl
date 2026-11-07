"""Independent NumPy F64 reference for common-pre52 calibration mathematics.

No model, data IO, native imports, autodiff, or backend is used. State rotations
are raw9 palm-frame matrices. Quaternion normalization is part of the actual
native Log definition; no external polar/SVD/orthogonality repair is performed.
This reference is not a bitwise replacement for native FP32 or a gradient proof.
"""
import numpy as np

CHART_MARGIN_RAD=.01
BLOCKS52={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),
          'rotation_log':(43,46,'rad'),'v':(46,49,'m/s'),'omega':(49,52,'rad/s')}


def finite(value,tail,label):
    a=np.asarray(value)
    if a.shape[-len(tail):]!=tuple(tail) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():raise ValueError('Finite '+label+' ending '+str(tail))
    return a.astype(np.float64)


def mm(a,b):
    return (np.asarray(a)[..., :, :,None]*np.asarray(b)[...,None,:,:]).sum(-2)


def native_log_reference(R):
    """Same canonical-quaternion mathematical mapping, independently assembled.

    Native mat_to_quat uses four clamped candidates then normalizes the chosen
    quaternion. Here a symmetric quaternion outer-product matrix supplies the
    selected column. Proper SO3 truth in fixtures is separate axis-angle algebra.
    """
    R=finite(R,(3,3),'raw relative rotation');shape=R.shape[:-2]
    K=np.empty(shape+(4,4),dtype='f8')
    a,b,c=R[...,0,0],R[...,1,1],R[...,2,2]
    diagonal=np.stack((1+a+b+c,1+a-b-c,1-a+b-c,1-a-b+c),-1)
    for i in range(4):K[...,i,i]=diagonal[...,i]
    K[...,0,1]=K[...,1,0]=R[...,2,1]-R[...,1,2]
    K[...,0,2]=K[...,2,0]=R[...,0,2]-R[...,2,0]
    K[...,0,3]=K[...,3,0]=R[...,1,0]-R[...,0,1]
    K[...,1,2]=K[...,2,1]=R[...,0,1]+R[...,1,0]
    K[...,1,3]=K[...,3,1]=R[...,0,2]+R[...,2,0]
    K[...,2,3]=K[...,3,2]=R[...,1,2]+R[...,2,1]
    clamped=diagonal<1e-10;roots=np.sqrt(np.maximum(diagonal,1e-10));idx=roots.argmax(-1)
    chosen=np.take_along_axis(K,idx[...,None,None],axis=-1)[...,0]
    denom=2*np.take_along_axis(roots,idx[...,None],axis=-1)
    rawq=chosen/denom;norm=np.linalg.norm(rawq,axis=-1,keepdims=True)
    q=rawq/np.maximum(norm,1e-12);q=np.where(q[...,:1]<0,-q,q)
    v=q[...,1:];sh=np.linalg.norm(v,axis=-1,keepdims=True);half=np.arctan2(sh,q[...,:1])
    scale=np.where(sh<1e-6,2/np.maximum(q[...,:1],1e-6),2*half/(sh+1e-12))
    rv=v*scale;theta=(2*half)[...,0];trace=R.trace(axis1=-2,axis2=-1);cosine=(trace-1)/2
    return {'rotation_log':rv,'canonical_unit_quaternion':q,'chosen_quaternion_pre_normalization':rawq,
       'chosen_quaternion_norm':norm[...,0],'candidate_diagonal_raw':diagonal,'candidate_sqrt_clamp_mask':clamped,
       'selected_candidate':idx,'native_quaternion_angle_rad':theta,'raw_trace':trace,'raw_trace_cosine':cosine,
       'raw_trace_cosine_outside_unit_interval':(cosine< -1)|(cosine>1),
       'strict_pi_margin_passed':theta<np.pi-CHART_MARGIN_RAD}


def difference52(a,b):
    a,b=finite(a,(58,),'state58'),finite(b,(58,),'target58')
    if a.shape!=b.shape:raise ValueError('Same state shapes required')
    Ra=a[...,43:52].reshape(a.shape[:-1]+(3,3));Rb=b[...,43:52].reshape(b.shape[:-1]+(3,3))
    info=native_log_reference(mm(Ra.swapaxes(-1,-2),Rb))
    delta=np.concatenate((b[...,:43]-a[...,:43],info['rotation_log'],b[...,52:58]-a[...,52:58]),-1)
    return {'delta52':delta,'chart':info}


def fixed_scale52(delta_std,error_floor):
    std,floor=finite(delta_std,(52,),'frozen std52'),finite(error_floor,(52,),'fixed error floor52')
    if std.shape!=(52,) or floor.shape!=(52,) or (std<0).any() or (floor<=0).any():raise ValueError('One frozen nonnegative std and positive floor52')
    return np.maximum(std,floor)


def calibration_losses(point_prediction,point_target,shared_pre,pair_prediction,pair_target,actual_pair_u_FP32,joint_indices,s52):
    """Reference pair order arbitrary but explicit; axis1 is [minus,plus].

    Actual production cardinalities152/64 and common-state/source identity belong
    to the producer's source guards. Here no zero denominator or chart violation
    is repaired/dropped; any such pair blocks the complete result.
    """
    u=np.asarray(actual_pair_u_FP32);js=np.asarray(joint_indices)
    if u.ndim!=3 or u.shape[1:]!=(2,20) or u.dtype!=np.float32 or not np.isfinite(u).all():raise ValueError('Original actual pair controls FP32 N2x20')
    n=len(u)
    if js.shape!=(n,) or js.dtype.kind not in 'iu' or (js<0).any() or (js>=20).any():raise ValueError('Explicit native control coordinate per pair')
    h_all=u[:,1].astype('f8')-u[:,0].astype('f8');h=h_all[np.arange(n),js]
    if not n or not np.isfinite(h).all() or (h==0).any():raise ValueError('ZERO_ACTUAL_DENOMINATOR_BLOCKS_ALL_NO_SUBSET')
    pre=finite(shared_pre,(58,),'common pre');pp=finite(pair_prediction,(2,58),'pair prediction');pt=finite(pair_target,(2,58),'pair target')
    if pre.shape!=(n,58) or pp.shape!=(n,2,58) or pt.shape!=(n,2,58):raise ValueError('All pair states and one shared pre per pair')
    scale=finite(s52,(52,),'fixed s52')
    if scale.shape!=(52,) or (scale<=0).any():raise ValueError('Frozen positive mixed-unit scale52')
    point=difference52(point_prediction,point_target)
    if not point['delta52'].size:raise ValueError('All point rows required, not empty')
    cm=difference52(np.broadcast_to(pre[:,None,:],pp.shape),pp)
    cp=difference52(np.broadcast_to(pre[:,None,:],pt.shape),pt)
    if not(point['chart']['strict_pi_margin_passed'].all() and cm['chart']['strict_pi_margin_passed'].all() and cp['chart']['strict_pi_margin_passed'].all()):raise ValueError('CHART_PI_MARGIN_BLOCKS_ALL_NO_SWITCH_OR_ROW_DROP')
    residual=(cm['delta52'][:,1]-cm['delta52'][:,0])-(cp['delta52'][:,1]-cp['delta52'][:,0])
    normalized_point=point['delta52']/scale
    normalized_pair=residual/scale/h[:,None]  # h/(1 rad) is dimensionless.
    if not(np.isfinite(normalized_point).all() and np.isfinite(normalized_pair).all()):raise ValueError('NONFINITE_COMPLETE_LOSS_BLOCK')
    lp=float(np.mean(normalized_point**2));lr=float(np.mean(normalized_pair**2))
    return {'Lpoint':lp,'Lpair':lr,'control_loss':lp,'method_loss':lp+lr,'actual_delta_u20_rad_F64':h_all,
       'actual_h_over_one_rad':h,'s52_frozen':scale.copy(),'point_delta52':point['delta52'],
       'common_pre_model52':cm['delta52'],'common_pre_physical52':cp['delta52'],'common_pre_pair_residual52':residual,
       'normalized_point_residual52':normalized_point,'normalized_pair_residual52':normalized_pair,
       'point_chart':point['chart'],'model_pair_chart':cm['chart'],'physical_pair_chart':cp['chart'],
       'finite_native_backward_qualified':False}
