"""Independent fixed-budget/normalization/baseline reproduction and window audit."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin
from supermimic_research.learning.world_model import raw_tree_equal
from supermimic_research.learning.pen_stage import write


def load(pin):return dict(np.load(verify_file_pin(pin)))


def main(a):
    if a.output.exists():raise ValueError('Fresh independent prior review')
    docs={key:json.loads(getattr(a,key).read_text()) for key in ['health','first100','baseline_health','evaluation','rollout','baseline_evaluation','baseline_rollout']}
    new=torch.load(verify_file_pin(docs['health']['checkpoint']),map_location='cpu',weights_only=False)
    old=torch.load(verify_file_pin(docs['baseline_health']['checkpoint']),map_location='cpu',weights_only=False)
    res=json.loads(a.baseline_sources.read_text())
    for pin in old['sources']+old['parent_sources']:verify_source_pin(pin,res)
    for pin in new['sources']+new['parent_sources']:verify_file_pin(pin)
    prep=json.loads(verify_file_pin(new['prepared']).read_text());data=load(prep['outputs']['TRAIN']);mask=data['valid']
    x=data['x'][mask].astype(np.float32).astype(np.float64);y=data['y'][mask]
    # Independent direct construction in the documented output coordinate chart.
    baseline=np.zeros_like(y);baseline[:,:20]=x[:,20:40]*.02
    baseline[:,40:43]=x[:,49:52]*.02;baseline[:,43:46]=x[:,52:55]*.02
    expected=new['arms']['DIRECT']['head_scales']['head_mean'].copy()
    expected[:46]=np.mean(y[:,:46]-baseline[:,:46],axis=0)
    A,B=new['arms']['DIRECT'],new['arms']['DIRECT_CV'];O=old['arms']['DIRECT']
    cfg=[c['config'] for c in [O,A,B]];drop={'schema','model_comparison'}
    checks=dict(schema_v2=new['schema']=='controlled_pen_response_pair_checkpoint:2' and docs['health']['schema']=='pen_response_continuation_health:2',
        exact_arms=set(new['arms'])=={'DIRECT','DIRECT_CV'},fixed2000=new['update']==old['update']==A['update']==B['update']==2000,
        hyperparameters_fixed=all({k:v for k,v in c.items() if k not in drop}=={k:v for k,v in cfg[0].items() if k not in drop} for c in cfg),
        prepared_unchanged=new['prepared']==old['prepared'],roles_unchanged=new['dataset_pins']==old['dataset_pins']==prep['outputs'],
        source_prior_flags=A['pose_baseline']=='ZERO' and B['pose_baseline']=='CONSTANT_VELOCITY' and not A['coupled'] and not B['coupled'],
        full_head_scale_fixed=raw_tree_equal(A['head_scales']['head_scale'],B['head_scales']['head_scale']),
        residual_mean_TRAIN_only=np.array_equal(B['model']['head_mean'].numpy(),expected.astype(np.float32)),
        outside_prior_heads_unchanged=np.array_equal(A['model']['head_mean'][46:].numpy(),B['model']['head_mean'][46:].numpy()),
        same_parameter_shapes={k:tuple(v.shape) for k,v in A['model'].items() if k.startswith('net.')}=={k:tuple(v.shape) for k,v in B['model'].items() if k.startswith('net.')},
        sampler_stream_exact=raw_tree_equal(A['sampler'],B['sampler']),
        health_first100=all(docs['first100']['checks'].values()),health_final=all(docs['health']['checks'].values()),
        same_physical_scales=all(raw_tree_equal(A['model'][k],B['model'][k]) for k in ['x_mean','x_scale','y_mean','y_scale']),
        same_physical_tables=raw_tree_equal(A['physics'],B['physics'])==True)
    repeat=dict(model=raw_tree_equal(A['model'],O['model']),optimizer=raw_tree_equal(A['optimizer'],O['optimizer']),
        sampler=raw_tree_equal(A['sampler'],O['sampler']),loss_logs=new['logs']['DIRECT']==old['logs']['DIRECT'])
    for key in ['evaluation','rollout']:
        checks[key+'_checkpoint_binding']=docs[key]['checkpoint']==docs['health']['checkpoint']
        checks[key+'_preparation_binding']=docs[key]['prepared']==new['prepared']
    for key in ['baseline_evaluation','baseline_rollout']:
        checks[key+'_checkpoint_binding']=docs[key]['checkpoint']==docs['baseline_health']['checkpoint']
    old_tail,new_tail=[load(docs[k]['records']) for k in ['baseline_rollout','rollout']]
    checks['actual_tail_records_exact']=raw_tree_equal(old_tail,new_tail)
    checks['native_pair_metadata_exact']=docs['evaluation']['pairs']==docs['baseline_evaluation']['pairs']
    old_np,new_np=[load(docs[k]['outputs']['DIRECT']) for k in ['baseline_rollout','rollout']]
    repeat['recursive_all_arrays']=raw_tree_equal(old_np,new_np)
    repeat['CHECK_all_predictions']=np.array_equal(load(docs['evaluation']['outputs']['DIRECT'])['all_predictions'],
        load(docs['baseline_evaluation']['outputs']['DIRECT'])['all_predictions'])
    checks['all_DIRECT_reproduction']=all(repeat.values())
    a.output.mkdir();receipt=freeze_sources([Path(__file__).resolve(),ROOT/'src/supermimic_research/learning/world_model.py'],
        a.output/'source_snapshot',ROOT,'independent-cv-mean-budget-parity-review');write(a.output/'source_receipt.json',receipt)
    result=dict(schema='pen_cv_prior_independent_review:1',inputs={k:file_pin(getattr(a,k)) for k in docs},baseline_sources=file_pin(a.baseline_sources),
        source_receipt=file_pin(a.output/'source_receipt.json'),checks={k:bool(v) for k,v in checks.items()},passed=sum(checks.values()),total=len(checks),
        DIRECT_bitwise_reproduction=repeat,parameter_count=sum(v.numel() for k,v in A['model'].items() if k.startswith('net.')),
        head_mean_TRAIN_rows=int(mask.sum()),baseline_input_arithmetic='F32 inference inputs promoted to F64; fixed control H=.02, not native dt',
        attempted_pairs=len(docs['evaluation']['pairs']),eligible_pairs=sum(p['eligible'] for p in docs['evaluation']['pairs']),
        window_coverage=docs['rollout']['window_coverage'],optimizer_updates=0,native_integrations=0,
        scope='Fixed capacity/data/scales/optimizer comparison and exact repeated DIRECT, exact previous development pairs/tails. Physical/recursive improvement is not inferred from these provenance/implementation checks. No Actor or ego result.')
    write(a.output/'report.json',result);verify_source_snapshot(receipt,live=True)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),checks=[result['passed'],result['total']],DIRECT_repeat=repeat,parameter_count=result['parameter_count'])))
    if not all(checks.values()):raise ValueError('Keep failed independent comparison; no qualification')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['health','first100','baseline-health','evaluation','rollout','baseline-evaluation','baseline-rollout','baseline-sources','output']:
        p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())
