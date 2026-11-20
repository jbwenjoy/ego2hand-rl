"""Twelve-generator support-function ablation with the existing 72-probe protocol."""
import argparse
import copy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import fit_pen_capacity_forecast as base
import review_pen_capacity_action_response as reviewer
from supermimic_research.mechanics.pen_generator_capacity import (
    GENERATOR_SCHEMA, GeneratorCapacityHead, load_generator_capacity_head, pair_bound_violations)


def bound_audit(capacity):
    upper,lower=pair_bound_violations(capacity.double())
    def stats(error):
        violated=error>1e-6
        return dict(count=int(violated.sum()),rows=int(violated.any(1).sum()),fraction=float(violated.double().mean()),
                    max=float(error.max()))
    return dict(n=len(capacity),signed_pairs=upper.numel(),tol=1e-6,upper=stats(upper),lower=stats(lower))


def fit(train,val,out):
    torch.manual_seed(7411)
    net=GeneratorCapacityHead()
    x,y=train['x'][:,:75],train['y']
    net.set_training_stats(x,y)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    curve=[];best=float('inf')
    for epoch in range(1,61):
        for ids in torch.randperm(len(x)).split(128):
            loss=((net.forward_features(x[ids])-y[ids])/net.target_scale).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            score=float((net.forward_features(val['x'][:,:75])-val['y']).abs().mean())
        if not np.isfinite(score):
            raise ValueError('Nonfinite validation MAE')
        curve.append(score)
        if score<best:
            best,chosen,weights=score,epoch,copy.deepcopy(net.state_dict())
    path=out/'state.pt'
    torch.save(dict(schema=GENERATOR_SCHEMA,use_history=False,state_dict=weights),path)
    return (load_generator_capacity_head(torch.load(path,map_location='cpu',weights_only=True)),
            dict(model=base.file_pin(path),val_mae=best,selected_epoch=chosen,val_curve=curve))


def main(prior,prepared,out):
    torch.set_num_threads(2)
    out.mkdir(exist_ok=False,parents=True)
    baseline_pin=base.file_pin(prior/'report.json')
    baseline=base.read(baseline_pin)
    protocol=base.read(baseline['intent'])
    data=base.read(protocol['data'])
    reference=base.read(base.read(data['intent'])['reference_capacity'])['refs']['clip_001']['series']
    if reference!=protocol['reference']:
        raise ValueError('Baseline reference or collection mismatch')
    with np.load(base.verify_file_pin(reference)) as f:
        target,eligible,stages=f['capacity'],f['eligible'],f['stage']
        np.testing.assert_array_equal(f['directions'],base.balanced_directions())

    eps={(e['label'],e['seed']):e for e in data['episodes']}
    split={k:[eps[tuple(v)] for v in rows] for k,rows in protocol['split'].items()}
    if len(eps)!=sum(map(len,split.values())) or len(eps)!=len(data['episodes']):
        raise ValueError('Episode ownership mismatch')
    intent=out/'intent.json'
    base.write_json(intent,dict(data=protocol['data'],reference=reference,baseline=baseline_pin,
        schema=GENERATOR_SCHEMA,source=base.file_pin(__file__),
        implementation=base.file_pin(base.ROOT/'src/supermimic_research/mechanics/pen_generator_capacity.py'),
        split=protocol['split'],optimization=protocol['optimization'],target=protocol['target'],
        design='12 learned source BODY six-component generators and origin, 72 scalar latent outputs, direct state-only head as control. Support is max over 13 points. Initial generator centers use only TRAIN axis means. Signed pair upper/lower triangle checks are necessary, not a complete contact-physics constraint.',
        diagnostic_scope='Reanalysis of previously inspected benchmark families; fixed TEST ownership.'))

    train=base.load_role(split['TRAIN'],target,eligible);val=base.load_role(split['VAL'],target,eligible)
    net,variant=fit(train,val,out)
    old=base.load_capacity_head(torch.load(base.verify_file_pin(baseline['variants']['state']['model']),
                                          map_location='cpu',weights_only=True))
    variant['cohorts']={};bounds={}
    with torch.no_grad():
        for role,group in split.items():
            rows=train if role=='TRAIN' else val if role=='VAL' else base.load_role(group,target,eligible)
            x=rows['x'][:,:75]
            direct=old.forward_features(x);generated=net.forward_features(x)
            variant['cohorts'][role]=base.evaluate(generated,rows,stages)
            # A mean of support functions is also subadditive.
            bounds[role]={name:bound_audit(y) for name,y in (
                ('native_mean',rows['y']),('direct',direct),('generator',generated))}
            direct_mae=base.evaluate(direct,rows,stages)['components']['mae']
            if abs(direct_mae-baseline['variants']['state']['cohorts'][role]['components']['mae'])>1e-6:
                raise ValueError('Direct control readback differs')
            print(role,'direct',direct_mae,'structured',variant['cohorts'][role]['components']['mae'],flush=True)
    base.write_json(out/'report.json',dict(intent=base.file_pin(intent),variants={'state':variant},bounds=bounds))
    # Apply precisely the same reviewer and exclusions to the new state-only loader.
    with patch.object(reviewer,'load_capacity_head',load_generator_capacity_head):
        reviewer.evaluate(prepared,out,out/'paired_response')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('baseline',type=Path);p.add_argument('prepared',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.baseline,a.prepared,a.out.resolve())
