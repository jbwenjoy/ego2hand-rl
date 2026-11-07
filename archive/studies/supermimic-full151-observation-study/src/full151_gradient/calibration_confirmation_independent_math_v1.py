"""Saved-only confirmation arithmetic, all152/all64; no model or native Log.

The authoritative native point FP32 scalar can be replayed by frozen component
auditor64c pure-Torch arithmetic. This module independently retains native52
captured-coordinate arrays in F64 and raw58 units separately. No masks/filters.
"""
import numpy as np

BLOCKS58={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),
          'R_raw9':(43,52,'dimensionless'),'v':(52,55,'m/s'),'omega':(55,58,'rad/s')}
BLOCKS52={'q':(0,20,'rad'),'qd':(20,40,'rad/s'),'p':(40,43,'m'),
          'R_commonpre_Log3':(43,46,'rad'),'v':(46,49,'m/s'),'omega':(49,52,'rad/s')}


def exact_finite(a,shape,dtype,label):
    a=np.asarray(a)
    if a.shape!=shape or a.dtype!=dtype or not np.isfinite(a).all():raise ValueError('All finite original '+label+' rows/dtype required')
    return a


def coordinate_statistics(a,blocks):
    a=np.asarray(a)
    if a.dtype!=np.float64 or a.ndim!=2 or a.shape[1]!=max(x[1] for x in blocks.values()) or not len(a) or not np.isfinite(a).all():raise ValueError('Complete F64 coordinate table')
    result={'rows':len(a),'per_feature':{'signed_mean':a.mean(0).tolist(),'RMSE':np.sqrt((a*a).mean(0)).tolist(),
       'MAE':abs(a).mean(0).tolist(),'maximum_absolute':abs(a).max(0).tolist()},'blocks':{}}
    for name,(lo,hi,units) in blocks.items():
        v=a[:,lo:hi];r={'start':lo,'stop_exclusive':hi,'units':units,'component_RMSE':float(np.sqrt((v*v).mean())),
           'MAE':float(abs(v).mean()),'maximum_absolute':float(abs(v).max())}
        if hi-lo==3:r['vector_RMS']=float(np.sqrt((v*v).sum(-1).mean()))
        if name=='R_raw9':r['Frobenius_RMS']=float(np.sqrt((v*v).sum(-1).mean()))
        result['blocks'][name]=r
    return result


def all_confirmation_arithmetic(point_delta52_FP32,pred_common52_FP32,target_common52_FP32,s52_FP32,minus64,plus64,actual_du64_F64,prediction58_FP32,target58_FP32):
    point=exact_finite(point_delta52_FP32,(152,52),np.float32,'point52')
    model=exact_finite(pred_common52_FP32,(152,52),np.float32,'pred common52')
    physical=exact_finite(target_common52_FP32,(152,52),np.float32,'target common52')
    scale=exact_finite(s52_FP32,(52,),np.float32,'scale52')
    den=exact_finite(actual_du64_F64,(64,),np.float64,'actualdu64')
    mi,pl=np.asarray(minus64),np.asarray(plus64)
    if mi.dtype!=np.int64 or pl.dtype!=np.int64 or mi.shape!=pl.shape or mi.shape!=(64,) or (mi<0).any() or (mi>=152).any() or (pl<0).any() or (pl>=152).any():raise ValueError('All64 explicit index pairs')
    if (scale<=0).any() or (den==0).any():raise ValueError('Any zero scale/actualdu blocks all rows, no subset')
    p=exact_finite(prediction58_FP32,(152,58),np.float32,'prediction58');t=exact_finite(target58_FP32,(152,58),np.float32,'target58')
    mr=model[pl].astype('f8')-model[mi].astype('f8');pr=physical[pl].astype('f8')-physical[mi].astype('f8')
    error=mr-pr;secant_error=error/den[:,None];normalized=secant_error/scale.astype('f8')
    pair_MSE=(normalized*normalized).mean(-1)
    raw=p.astype('f8')-t.astype('f8')
    arrays={'model_commonpre_response64x52_F64':mr,'physical_commonpre_response64x52_F64':pr,
       'commonpre_response_error64x52_F64':error,'actualdu_coordinate_secant_error64x52_F64':secant_error,
       'normalized_pair_residual64x52_F64':normalized,'pair_normalized_MSE64_F64':pair_MSE,
       'point_delta152x52_F64':point.astype('f8'),'raw_state_error152x58_F64':raw}
    return {'arrays':arrays,'all64_normalized_pair_MSE_F64':float((normalized*normalized).mean()),
       'per_pair_MSE_mean_F64':float(pair_MSE.mean()),
       'native52_point_coordinates':coordinate_statistics(point.astype('f8'),BLOCKS52),
       'native52_response_coordinates':coordinate_statistics(error,BLOCKS52),
       'native52_secant_error_per_command_rad':coordinate_statistics(secant_error,{k:(a,b,u+'/control_rad') for k,(a,b,u) in BLOCKS52.items()}),
       'raw58_state_error_original_units':coordinate_statistics(raw,BLOCKS58),
       'all152_and64_retained':True,'native_Log_recomputed':False,
       'scope':'F64 postprocessing of captured native52. FP32 point mean is separately replayed by frozen pure-Torch helper, not this F64 table.'}


def confirmation_rule(control_point_FP32,response_point_FP32,control_pair_F64,response_pair_F64):
    values=np.asarray([control_point_FP32,response_point_FP32,control_pair_F64,response_pair_F64],dtype='f8')
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Four complete finite nonnegative losses')
    pair_improved=bool(response_pair_F64<control_pair_F64);point_not_increased=bool(response_point_FP32<=control_point_FP32)
    return {'response_error_strictly_lower':pair_improved,'native_pointwise_not_increased':point_not_increased,
       'local_component_confirmation':pair_improved and point_not_increased,
       'task_policy_or_physical_gradient_qualified':False,'tolerance_or_selection_applied':False}
