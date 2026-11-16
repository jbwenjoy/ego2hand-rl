"""Study3 adapters for the maintained contrast trainer;no new training loop."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ..artifacts import file_pin,verify_file_pin
from ..frozen_sources import verify_source_snapshot
from .pen_fullrank_dataset import full_rank_response_data,VALIDATION_ROLE

MODELS=['POINT_DIRECT','POINT_IMPULSE','RANK4_DIRECT','RANK4_IMPULSE','RANK20_DIRECT','RANK20_IMPULSE']


def model_config(spec):
    expected=dict(schema=3,models=MODELS,validation_role='DEVELOPMENT_RESERVED_NEVER_FIT_NOT_FINAL_TEST',
        width=512,batch_size=256,pair_batch_size=32,learning_rate=.0003,weight_decay=.00001,
        gradient_clip_norm=1.,auxiliary_weight=.1,contrast_weight=.1,response_scale='POINT_Y_SCALE',
        init_seed=7411,sampler_seed=7411,pair_sampler_seed=7751,first_checkpoint_update=100,
        planned_updates=2000,continue_after_first_healthy_checkpoint=True,validation_evaluation_update=2000)
    if any(spec.get(k)!=v for k,v in expected.items()) or 'model_comparison' in spec:
        raise ValueError('Explicit registered six-arm rank-only study3 required')
    return {**spec,'schema':1,'study_schema':3,'dataset_schema':'contact_pen_fullrank_response_data:3'}


def load_common(cp,prepared):
    pp=Path(prepared).resolve();prep=json.loads(pp.read_text())
    import tomllib
    spec=tomllib.loads(cp.read_text())
    if (pp!=(cp.parent/spec['prepared_data']).resolve() or prep['schema']!='contact_pen_fullrank_response_data:3'
            or prep['config']!=file_pin((cp.parent/spec['common_data_config']).resolve())
            or prep['passed']!=prep['total'] or not all(prep['checks'].values())
            or prep['validation_role']!=VALIDATION_ROLE or prep['validation_rows_used_for_TRAIN']
            or prep['CHECK_response_rows_used'] or prep['scale_input_role']!='VALID_TRAIN_ONLY'
            or (prep['TRAIN_valid'],prep['VALIDATE_valid'],prep['TRAIN_pairs'])!=(3866,1935,949)
            or prep['pair_counts']!={'rank4':189,'rank20':949}):
        raise ValueError('Exact reviewed full20 TRAIN/reserved-validation products required')
    # Historical DATA bytes are verified as frozen data,not rebound native state.
    verify_source_snapshot(json.loads(verify_file_pin(prep['source_receipt']).read_text()))
    tr=json.loads(verify_file_pin(prep['TRAIN_preparation']).read_text());vr=json.loads(verify_file_pin(prep['VALIDATE_preparation']).read_text())
    data={r:dict(np.load(verify_file_pin(p))) for r,p in prep['outputs'].items()}
    directions=np.load(verify_file_pin(prep['directions']))['directions']
    rebuilt=full_rank_response_data(tr,data['TRAIN'],vr,data['VALIDATE'],directions)
    for key in ['scales','head_scales','pairs','pair_subsets']:
        saved=dict(np.load(verify_file_pin(prep[key])))
        if saved.keys()!=rebuilt[key].keys() or any(not np.array_equal(v,saved[k]) for k,v in rebuilt[key].items()):
            raise ValueError('Current TRAIN-only reconstruction differs from frozen products')
    review=json.loads(verify_file_pin(prep['data_review']).read_text())
    if review['passed']!=review['total'] or not all(review['checks'].values()) or review['inputs']!={'TRAIN':prep['TRAIN_preparation'],'VALIDATE':prep['VALIDATE_preparation']}:
        raise ValueError('Exact independent physical data review required')
    return pp,prep,data,rebuilt['pairs'],rebuilt['pair_subsets']


def evaluation_roles(update):
    if update not in [100,500,1000,2000]:raise ValueError('Declared curve checkpoints only')
    return ['TRAIN','VALIDATE'] if update==2000 else ['TRAIN']


def pair_contract(prep,rank,indices):
    indices=np.asarray(indices,dtype=np.int64)
    if rank not in [0,4,20] or (rank==0)!=(len(indices)==0):raise ValueError('Explicit point/rank4/rank20 cache identity')
    return dict(rank=rank,rows=len(indices),index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        point_bank=prep['outputs']['TRAIN'],pair_bank=prep['pairs'],subset_bank=prep['pair_subsets'])
