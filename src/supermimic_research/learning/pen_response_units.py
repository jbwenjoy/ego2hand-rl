"""TRAIN-only derivative units;shared trainer/model,unchanged forward loss."""
import hashlib
import numpy as np

MODELS=['POINT_DIRECT','POINT_IMPULSE','RAW_DIRECT','RAW_IMPULSE','RMS01_DIRECT','RMS01_IMPULSE','RMS1_DIRECT','RMS1_IMPULSE']


def model_config(spec):
    expected=dict(schema=4,models=MODELS,validation_role='INSPECTED_DEVELOPMENT_NEVER_FIT_NOT_FINAL_TEST',
        width=512,batch_size=256,pair_batch_size=32,learning_rate=.0003,weight_decay=.00001,
        gradient_clip_norm=1.,auxiliary_weight=.1,contrast_weight=.1,response_scale='VALID_TRAIN_PAIR_TARGET_RMS',
        normalized_response_weights=[.1,1.],response_floor_relative=.001,point_anchor_weight=1.,
        init_seed=7411,sampler_seed=7411,pair_sampler_seed=7751,first_checkpoint_update=100,
        planned_updates=2000,continue_after_first_healthy_checkpoint=True,validation_evaluation_update=2000)
    if any(spec.get(k)!=v for k,v in expected.items()) or 'model_comparison' in spec:
        raise ValueError('Explicit registered derivative-unit/weight study4 required')
    return {**spec,'schema':1,'study_schema':4,'dataset_schema':'contact_pen_fullrank_response_data:3'}


def derivative_scale(targets,point_scale,*,role,floor_relative=.001):
    target=np.asarray(targets,dtype=np.float64);point=np.asarray(point_scale,dtype=np.float64)
    if role!='TRAIN_CONTROLLED_PHYSICAL_RESPONSE' or target.ndim!=2 or target.shape[1]!=90 or len(target)<2 or point.shape!=(90,) or not np.isfinite(target).all() or not np.isfinite(point).all() or np.any(point<=0) or floor_relative!=.001:
        raise ValueError('Eligible TRAIN finite contrasts and positive fixed point scales required')
    rms=np.sqrt(np.mean(target**2,axis=0));floor=floor_relative*point
    return dict(raw_RMS=rms,floor=floor,response_scale=np.maximum(rms,floor))


def response_contract(scale,mode,weight,scale_report):
    a=np.asarray(scale,dtype=np.float64)
    if a.shape!=(90,) or not np.isfinite(a).all() or np.any(a<=0) or mode not in ['POINT_Y_SCALE','TRAIN_DERIVATIVE_RMS'] or weight not in [0.,.1,1.]:raise ValueError('Explicit finite response-scale/cache contract required')
    return dict(schema='physical_response_units:1',mode=mode,weight=float(weight),scale_sha256=hashlib.sha256(a.tobytes()).hexdigest(),scale_report=scale_report)
