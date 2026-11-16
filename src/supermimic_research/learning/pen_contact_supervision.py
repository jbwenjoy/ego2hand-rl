"""Study5:explicit Fn6 response addition,with unchanged existing objectives."""
MODELS=['RMS01_DIRECT','RMS01_IMPULSE','FN01_DIRECT','FN01_IMPULSE','RMS1_DIRECT','RMS1_IMPULSE','FN1_DIRECT','FN1_IMPULSE']


def model_config(spec):
    expected=dict(schema=5,models=MODELS,validation_role='INSPECTED_DEVELOPMENT_NEVER_FIT_NOT_FINAL_TEST',
        width=512,batch_size=256,pair_batch_size=32,learning_rate=.0003,weight_decay=.00001,
        gradient_clip_norm=1.,auxiliary_weight=.1,contrast_weight=.1,response_scale='VALID_TRAIN_PAIR_TARGET_RMS',
        normalized_response_weights=[.1,1.],fn_response_coefficients=[0.,1.],point_anchor_weight=1.,
        init_seed=7411,sampler_seed=7411,pair_sampler_seed=7751,first_checkpoint_update=100,
        planned_updates=2000,continue_after_first_healthy_checkpoint=True,validation_evaluation_update=2000)
    if any(spec.get(k)!=v for k,v in expected.items()) or 'model_comparison' in spec:
        raise ValueError('Explicit registered Fn6-addition study5 required')
    return {**spec,'schema':1,'study_schema':5,'dataset_schema':'contact_pen_fullrank_response_data:3'}
