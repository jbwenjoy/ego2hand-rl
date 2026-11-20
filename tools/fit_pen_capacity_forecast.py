"""Fit and evaluate causal 20 ms geometry-capacity heads on a completed native collection."""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.pen_capacity_forecast import (
    CAPACITY_WIDTH, SCHEMA, CapacityHead, capacity_features, capacity_mismatch, load_capacity_head)
from supermimic_research.mechanics.pen_wrench_capacity import balanced_directions
from supermimic_research.strict_json import write_json


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def episode_rows(x, valid, ticks, phase, target, eligible):
    """Index k is the control from reference frame k to k+1. k=0 is phase 1."""
    n = len(x)
    if (not n or n > len(eligible) or x.shape != (n,102) or
            ticks.shape != (n,8,CAPACITY_WIDTH) or valid.shape != (n,) or
            valid.dtype != np.bool_ or not np.array_equal(phase,np.arange(1,n+1)) or
            target.shape != (len(eligible)+1,CAPACITY_WIDTH)):
        raise ValueError('Misaligned source controls, capacity ticks, or reference clock')
    if not np.isfinite(ticks).all() or np.min(ticks)<0 or not np.isfinite(x[valid]).all():
        raise ValueError('Invalid native capacity or valid source')
    mean = ticks.mean(axis=1)
    present = np.r_[False,valid[:-1]]
    previous = np.zeros_like(mean)
    previous[1:] = mean[:-1]
    previous[~present] = 0.
    cast = lambda a: torch.as_tensor(np.ascontiguousarray(a[valid]),dtype=torch.float32)
    state, pd, prev = cast(x[:,:55]),cast(x[:,81:101]),cast(previous)
    observed = torch.from_numpy(present[valid])
    return dict(x=capacity_features(state,pd,use_history=True,previous_capacity=prev,history_present=observed),
                y=cast(mean),previous=prev,present=observed,target=cast(target[:n]),
                eligible=torch.from_numpy(eligible[:n][valid]),phase=torch.from_numpy(phase[valid]))


def load_role(episodes, target, eligible):
    parts=[]
    for ep in episodes:
        with np.load(verify_file_pin(ep['trajectory'])) as f:
            x,valid=f['x'],f['valid']
        with np.load(verify_file_pin(ep['capacity'])) as f:
            ticks=f['capacity']
        audit=read(ep['audit'])
        phases=np.array([row['phase'] for row in audit['rows']],dtype=np.int64)
        if len(x)!=ep['controls'] or valid.sum()!=ep['valid']:
            raise ValueError('Episode accounting failure')
        data=episode_rows(x,valid,ticks,phases,target,eligible)
        data['seed']=torch.full_like(data['phase'],ep['seed'])
        parts.append(data)
    if not parts:
        raise ValueError('Missing policy split')
    return {k:torch.cat([r[k] for r in parts]) for k in parts[0]}


def train_variant(train, val, has_history, out):
    torch.manual_seed(7411)
    net=CapacityHead(has_history)
    x=train['x'] if has_history else train['x'][:,:75]
    vx=val['x'] if has_history else val['x'][:,:75]
    net.set_training_stats(x,train['y'])
    optimizer=torch.optim.Adam(net.parameters(),lr=.001)
    best=float('inf')
    curve=[]
    for epoch in range(1,61):
        for ids in torch.randperm(len(x)).split(128):
            error=(net.forward_features(x[ids])-train['y'][ids])/net.target_scale
            loss=error.square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            score=float((net.forward_features(vx)-val['y']).abs().mean())
        if not np.isfinite(score):
            raise ValueError('Validation score nonfinite')
        curve.append(score)
        if score<best:
            best, chosen, weights=score,epoch,copy.deepcopy(net.state_dict())
    name='history' if has_history else 'state'
    path=out/(name+'.pt')
    torch.save(dict(schema=SCHEMA,use_history=has_history,state_dict=weights),path)
    model=load_capacity_head(torch.load(path,map_location='cpu',weights_only=True))
    return model,dict(model=file_pin(path),selected_epoch=chosen,val_mae=best,val_curve=curve)


def mean_errors(estimate,native,mask):
    if not mask.any():
        return dict(n=0,mae=None,rmse=None)
    err=estimate[mask].double()-native[mask].double()
    return dict(n=int(mask.sum()),mae=float(err.abs().mean()),rmse=float(err.square().mean().sqrt()))


def target_errors(estimate,native,target,mask):
    if not mask.any():
        return dict(n=0,predicted=None,measured=None,mae=None,rmse=None,bias=None)
    a=capacity_mismatch(estimate[mask].double(),target[mask].double())
    b=capacity_mismatch(native[mask].double(),target[mask].double())
    err=a-b
    return dict(n=len(a),predicted=float(a.mean()),measured=float(b.mean()),mae=float(err.abs().mean()),
                rmse=float(err.square().mean().sqrt()),bias=float(err.mean()))


@torch.no_grad()
def evaluate(forecast, data, target_stages):
    if not bool(torch.isfinite(forecast).all()):
        raise ValueError('Nonfinite forecast')
    y=data['y']
    stages=target_stages[data['phase'].numpy()-1]
    eligible=data['eligible']
    return dict(components=mean_errors(forecast,y,torch.ones_like(eligible)),
                known_history=mean_errors(forecast,y,data['present']),
                missing_history=mean_errors(forecast,y,~data['present']),
                mismatch={name:target_errors(forecast,y,data['target'],mask) for name,mask in (
                    ('eligible',eligible),('eligible_slow',eligible&torch.from_numpy(stages=='slow')),
                    ('eligible_turn',eligible&torch.from_numpy(stages=='turn')))})


def main(root,out):
    root=root.resolve()
    src=file_pin(root/'report.json')
    report=read(src)
    intent=read(report['intent'])
    refs=read(intent['reference_capacity'])
    ref_pin=refs['refs']['clip_001']['series']
    with np.load(verify_file_pin(ref_pin)) as f:
        target,eligible,stages=f['capacity'],f['eligible'],f['stage']
        np.testing.assert_array_equal(f['directions'],balanced_directions())
    eps=report['episodes']
    split={'TRAIN':[e for e in eps if e['label'] in ('A','B','C','D')]}
    split.update({k:[e for e in eps if e['label']==k] for k in ('VAL','TEST1','TEST2','TEST3')})
    if not all(split.values()) or len(eps)!=sum(map(len,split.values())) or report['native']!=8*report['controls'] or sum(e['controls'] for e in eps)!=report['controls']:
        raise ValueError('Incomplete native policy split or accounting')
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(schema=SCHEMA,data=src,reference=ref_pin,source=file_pin(Path(__file__)),
        implementation=file_pin(ROOT/'src/supermimic_research/mechanics/pen_capacity_forecast.py'),
        split={k:[[e['label'],e['seed']] for e in group] for k,group in split.items()},
        optimization=dict(seed=7411,epochs=60,batch=128,lr=.001,validation='min VAL absolute error over all 72 outputs and physically valid rows'),
        target='Physical label is the mean of eight per-tick BODY support vectors. Phase p indexes advisory reference capacity and eligibility at source interval p-1. Phase, stage, reference values, friction forces and same-control capacity are excluded from network inputs.',
        persistence='Previous completed valid control mean. Cold start or preceding invalid window uses the TRAIN-only mean vector. Cold and known-history rows reported separately.'))
    torch.set_num_threads(2)
    train=load_role(split['TRAIN'],target,eligible);val=load_role(split['VAL'],target,eligible)
    models={};variants={}
    for name in ('state','history'):
        models[name],variants[name]=train_variant(train,val,name=='history',out)
        variants[name]['cohorts']={}
    persistence={};training_mean=train['y'].mean(0)
    for name,group in split.items():
        data=train if name=='TRAIN' else val if name=='VAL' else load_role(group,target,eligible)
        persistence[name]=evaluate(torch.where(data['present'][:,None],data['previous'],training_mean),
                                   data,stages)
        for key,net in models.items():
            x=data['x'] if net.use_history else data['x'][:,:75]
            with torch.no_grad():
                pred=net.forward_features(x)
            variants[key]['cohorts'][name]=evaluate(pred,data,stages)
        print(name,[(k,round(v['cohorts'][name]['components']['mae'],6)) for k,v in variants.items()],
              'persistence',round(persistence[name]['components']['mae'],6),flush=True)
    write_json(out/'report.json',dict(schema=SCHEMA,intent=file_pin(out/'intent.json'),
               variants=variants,persistence=persistence))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    main(args.root,args.out)
