"""Matched CPU comparison of absolute capacity labels with paired secant supervision."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from fit_pen_capacity_forecast import ROOT, evaluate, load_role
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.mechanics.pen_capacity_forecast import (
    CapacityHead, SCHEMA, capacity_features, capacity_mismatch, load_capacity_head)
from supermimic_research.strict_json import write_json

SEED = 7411
EPOCHS = 60
PAIR_BATCH = 32
PD_EPS = 1e-6
TRAIN_ROLES = ('TRAIN_A', 'TRAIN_B')


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def load_group(raw, include_mixed):
    selected = raw['selected']
    if len(selected) != 8 or len(set(selected)) != 8:
        raise ValueError('Expected eight distinct controlled axes')
    groups = {}
    for row in raw['rows']:
        name, phase = row['name'], row['phase']
        if name.startswith('mixed_') and not include_mixed:
            continue
        phase_rows = groups.setdefault(phase, {})
        if name in phase_rows:
            raise ValueError('Duplicate source and command')
        with np.load(verify_file_pin(row['packet'])) as f:
            x, pd, cap = (np.asarray(f[key], dtype=np.float64) for key in ('x','pd','capacity'))
        if x.shape != (102,) or pd.shape != (20,) or cap.shape != (72,):
            raise ValueError('Unexpected stored response shape')
        np.testing.assert_array_equal(x[81:101], pd)
        if not np.isfinite(np.r_[x, cap]).all() or np.min(cap) < 0:
            raise ValueError('Invalid source or physical label')
        phase_rows[name] = (x, cap, row['source'], row['valid'] and not row['terminal'])
    expected={'sham','sham_repeat'} | {f'j{i}_{sign}' for i in selected for sign in ('minus','plus')}
    if include_mixed:
        expected.update(('mixed_minus','mixed_plus'))
    for rows in groups.values():
        if set(rows)!=expected:
            raise ValueError('Missing or extra native actions in source group')
        for x, _, pin, _ in rows.values():
            np.testing.assert_array_equal(x[:81], rows['sham'][0][:81])
            if pin != rows['sham'][2] or x[-1] != rows['sham'][0][-1]:
                raise ValueError('Branches have different source receipts or time steps')
    return groups


def to_feature(x):
    return np.r_[x[:55], x[81:101] - x[:20]]


def training_pairs(roles):
    endpoints, contrasts, exclusions = [], [], []
    for role in TRAIN_ROLES:
        raw = roles[role]
        groups = load_group(raw, include_mixed=False)
        for phase, rows in groups.items():
            for name, (x, cap, _, ok) in rows.items():
                if ok:
                    endpoints.append((to_feature(x), cap))
            for axis in raw['selected']:
                minus, plus = (rows[f'j{axis}_{sign}'] for sign in ('minus','plus'))
                dx = plus[0][81:101] - minus[0][81:101]
                scale = np.linalg.norm(dx)
                if not (minus[3] and plus[3]) or scale <= PD_EPS:
                    exclusions.append(dict(role=role,phase=phase,axis=axis,pd_norm=scale,
                                           reason='physics_or_terminal' if not (minus[3] and plus[3]) else 'zero_pd'))
                    continue
                if np.any(np.delete(dx,axis) != 0):
                    raise ValueError('Basis probe moves additional transmitted PD axes')
                contrasts.append((to_feature(minus[0]),to_feature(plus[0]),(plus[1]-minus[1])/scale,scale))
    tensor = lambda rows: [torch.tensor(np.asarray(z),dtype=torch.float32) for z in zip(*rows)]
    return tensor(endpoints), tensor(contrasts), exclusions


def fit(train, val, endpoints, pairs, weight, out, name):
    torch.manual_seed(SEED)
    net = CapacityHead(False)
    x, y = train['x'][:,:75], train['y']
    vx, vy = val['x'][:,:75], val['y']
    ex, ey = endpoints
    minus, plus, dy_per_rad, length = pairs
    net.set_training_stats(x,y)
    scale = dy_per_rad.square().mean().sqrt()
    if scale <= 0 or not torch.isfinite(scale):
        raise ValueError('Training pair labels contain no finite response')
    opt = torch.optim.Adam(net.parameters(), lr=.001)
    rng = torch.Generator().manual_seed(SEED+1)
    audit = hashlib.sha256()
    init = hashlib.sha256(b''.join(v.detach().numpy().tobytes() for v in net.state_dict().values())).hexdigest()
    curve, steps = [], 0
    for epoch in range(EPOCHS):
        for ids in torch.randperm(len(x),generator=rng).split(128):
            ei = torch.randint(len(ex),(2*PAIR_BATCH,),generator=rng)
            pi = torch.randint(len(minus),(PAIR_BATCH,),generator=rng)
            for v in (ids,ei,pi):
                audit.update(v.numpy().tobytes())
            # Both arms execute all three terms; only the response coefficient differs.
            batch = torch.cat((x[ids], ex[ei], minus[pi], plus[pi]))
            pred, epred, pred_minus, pred_plus = net.forward_features(batch).split([len(ids),len(ei),len(pi),len(pi)])
            seq_loss = ((pred-y[ids])/net.target_scale).square().mean()
            absolute_loss = ((epred-ey[ei])/net.target_scale).square().mean()
            response = (pred_plus-pred_minus)/length[pi,None]
            pair_loss = ((response-dy_per_rad[pi])/scale).square().mean()
            loss=seq_loss+absolute_loss+weight*pair_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fit loss')
            opt.zero_grad()
            loss.backward()
            opt.step()
            steps += 1
        with torch.no_grad():
            curve.append(float((net.forward_features(vx)-vy).abs().mean()))
    path=out/(name+'.pt')
    torch.save(dict(schema=SCHEMA,use_history=False,state_dict=net.state_dict()),path)
    net=load_capacity_head(torch.load(path,map_location='cpu',weights_only=True)).requires_grad_(False)
    return net,dict(model=file_pin(path),epochs=EPOCHS,updates=steps,response_weight=weight,
                    initial_hash=init,sampling_hash=audit.hexdigest(),val_mae_by_epoch=curve)


def relative_summary(actual, estimate):
    truth=np.atleast_2d(actual);pred=np.atleast_2d(estimate)
    size=np.linalg.norm(truth)
    return dict(rows=len(truth),error_over_zero=float(np.linalg.norm(pred-truth)/size) if size else None,
                predicted_over_observed=float(np.linalg.norm(pred)/size) if size else None,
                rmse=float(np.sqrt(np.mean((pred-truth)**2))),
                positive_dot=int(np.count_nonzero((pred*truth).sum(1)>0)),
                nonzero_truth=int(np.count_nonzero(np.linalg.norm(truth,axis=1)>0)))


def evaluate_mixed(model, roles, target, eligible):
    model=copy.deepcopy(model).double().requires_grad_(False)
    results={}
    t=lambda a:torch.as_tensor(a,dtype=torch.float64)
    for role in (*TRAIN_ROLES,'TEST'):
        groups=load_group(roles[role],include_mixed=True)
        records=[];exclusions=[]
        for phase,rows in groups.items():
            m,p=rows['mixed_minus'],rows['mixed_plus']
            minus,plus=(t(endpoint[0][81:101]) for endpoint in (m,p))
            direction=plus-minus
            length=float(direction.norm())
            if not (m[3] and p[3]) or length<=PD_EPS:
                exclusions.append(dict(phase=phase,pd_norm=length,reason='unavailable_pair'))
                continue
            state=t(m[0][:55])
            midpoint=(minus+plus)/2
            direction=direction/length
            def predict(u):
                return model(state,u)
            _,jvp=torch.autograd.functional.jvp(predict,midpoint,direction)
            ym,yp=predict(minus),predict(plus)
            ref=t(target[phase-1])
            _,ljvp=torch.autograd.functional.jvp(lambda u:capacity_mismatch(predict(u),ref),midpoint,direction)
            truth=(p[1]-m[1])/length
            actual_loss=float(capacity_mismatch(t(p[1]),ref)-capacity_mismatch(t(m[1]),ref))/length
            loss_sec=float(capacity_mismatch(yp,ref)-capacity_mismatch(ym,ref))/length
            sham,repeat=rows['sham'],rows['sham_repeat']
            sham_delta=float(abs(capacity_mismatch(t(sham[1]),ref)-capacity_mismatch(t(repeat[1]),ref))) if sham[3] and repeat[3] else None
            records.append(dict(phase=phase,pd_norm=length,eligible=bool(eligible[phase-1]),
                native=truth.tolist(),jvp=jvp.tolist(),secant=((yp-ym)/length).tolist(),
                native_loss=actual_loss,loss_jvp=float(ljvp),loss_secant=loss_sec,
                exceeds_sham=bool(abs(actual_loss)*length>sham_delta) if sham_delta is not None else None,
                endpoint_mae=float(torch.cat([abs(ym-t(m[1])),abs(yp-t(p[1]))]).mean())))
        scores={}
        for vector,keys in [('native',('jvp','secant')),('native_loss',('loss_jvp','loss_secant'))]:
            pick=[r for r in records if vector=='native' or r['eligible']]
            if not pick:continue
            for key in keys:
                scores[key]=relative_summary(np.array([r[vector] for r in pick]).reshape(len(pick),-1),
                                             np.array([r[key] for r in pick]).reshape(len(pick),-1))
        results[role]=dict(records=records,exclusions=exclusions,declared=len(groups),summary=scores,
                           endpoint_mae=float(np.mean([r['endpoint_mae'] for r in records])) if records else None)
    return results


def main(bank,out):
    torch.set_num_threads(2)
    sequences=file_pin(ROOT/'runs/pen_capacity_sequences_v1/report.json')
    pair_pin=file_pin(bank/'report.json')
    seq=read(sequences);meta=read(seq['intent'])
    ref_pin=read(meta['reference_capacity'])['refs']['clip_001']['series']
    with np.load(verify_file_pin(ref_pin)) as f:target,eligible,stage=(f[k] for k in ('capacity','eligible','stage'))
    report=read(pair_pin);roles=report['roles']
    if set(roles)!=set((*TRAIN_ROLES,'TEST')) or any(roles[k]['selected']!=roles['TEST']['selected'] for k in TRAIN_ROLES):
        raise ValueError('Missing or mismatched roles')
    actor_pins=read(read(seq['collection'])['intent'])['actors']
    probe_pins=read(read(report['native_report'])['intent'])['actors']
    if {k:v['sha256'] for k,v in probe_pins.items()}!={k:actor_pins[a]['sha256'] for k,a in [('TRAIN_A','A'),('TRAIN_B','B'),('TEST','TEST2')]}:
        raise ValueError('Pair-source actors do not match registered policy split')
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(source=file_pin(Path(__file__)),data=sequences,pairs=pair_pin,reference=ref_pin,
               roles='TRAIN_A/B basis-only contrasts and valid basis/sham endpoints; no mixed endpoints used in fitting. Absolute sequence TRAIN uses only A/B/C/D; previously inspected VAL and TEST policies.',
               schedule=dict(seed=SEED,epochs=EPOCHS,lr=.001,sequence_batch=128,pair_batch=PAIR_BATCH,endpoint_batch=2*PAIR_BATCH),
               fit='Fresh shared initialization. Fixed final epoch; VAL absolute MAE recorded for both. Both arms fit identical sequence/endpoint labels. Paired loss predicts centered capacity change per radian of actual transmitted PD. Residual scaled by pooled TRAIN-only derivative RMS. Unit coefficient versus zero. JVP assessed at actual endpoint midpoint.',
               min_pd_contrast=PD_EPS))
    split=seq['episodes']; seq_group=lambda keys:load_role([e for e in split if e['label'] in keys],target,eligible)
    tr=seq_group(('A','B','C','D'));val=seq_group(('VAL',))
    endpoints,pairs,excluded=training_pairs(roles)
    variants={};models={}
    for name,weight in [('absolute',0.),('response',1.)]:
        models[name],variants[name]=fit(tr,val,endpoints,pairs,weight,out,name)
    if any(variants['response'][k]!=variants['absolute'][k] for k in ('initial_hash','sampling_hash','updates')):
        raise ValueError('Unmatched training work')
    for role in ('VAL','TEST1','TEST2','TEST3'):
        data=val if role=='VAL' else seq_group((role,))
        for name,model in models.items():
            with torch.no_grad():pred=model.forward_features(data['x'][:,:75])
            variants[name].setdefault('absolute_cohorts',{})[role]=evaluate(pred,data,stage)
    for name,model in models.items():
        mixed=evaluate_mixed(model,roles,target,eligible)
        write_json(out/(name+'_mixed.json'),mixed)
        variants[name]['mixed']=file_pin(out/(name+'_mixed.json'))
        print(name, {k:round(v['jvp']['error_over_zero'],3) for k,m in mixed.items() if (v:=m['summary'])},flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),variants=variants,matched=True,
               sequence_rows=len(tr['x']),endpoint_rows=len(endpoints[0]),training_contrasts=len(pairs[0]),
               response_rms_per_actual_pd_rad=float(pairs[2].square().mean().sqrt()),excluded_basis_pairs=excluded))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('bank',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();main(args.bank.resolve(),args.out.resolve())

