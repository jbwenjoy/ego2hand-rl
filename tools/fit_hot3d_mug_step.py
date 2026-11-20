"""Fit matched single-window and shared mug response heads; hold out an entire proposal seed."""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import torch
from torch.nn import functional as F
from compare_hot3d_mug_wrench import physics,file_pin,verify_file_pin,write_json
from supermimic_research.mechanics.mug_step_response import MugStepChart,MugStepHead
from supermimic_research.learning.world_model import atomic_checkpoint,raw_tree_equal


def groups_for(rows,seed,chart,exclude):
    batches=[];excluded=0
    for row in rows:
        if row['seed']!=seed:continue
        with np.load(verify_file_pin(row['bank'])) as f:bank=dict(f)
        for t,parents in enumerate(bank['parent']):
            for i in np.unique(parents):
                keep=parents==i
                pre=bank['source'][t,keep][0];warm=bank['warmstart'][t,keep][0]
                np.testing.assert_array_equal(bank['source'][t,keep],np.broadcast_to(pre,(keep.sum(),len(pre))))
                np.testing.assert_array_equal(bank['warmstart'][t,keep],np.broadcast_to(warm,(keep.sum(),len(warm))))
                key=hashlib.sha256(pre[1:].tobytes()+warm.tobytes()).hexdigest()
                if key in exclude:
                    excluded+=1;continue
                cmd=bank['cmd'][t,keep]
                x=chart.features(pre,warm,cmd)
                y=chart.targets(pre,bank['end'][t,keep],bank['net_wrench'][t,keep])
                batches.append(dict(clip=row['clip'],phase=t+1,width=row['width'],key=key,x=x,y=y))
    return batches,excluded


def evaluate(net,groups):
    per={}
    for g in groups:
        x,y=g['x'],g['y']
        with torch.inference_mode(): yh=net(torch.from_numpy(x)).numpy()
        dim_scale=np.array([10]*3+[10]*3+[1]*6)
        # Group-centered errors isolate the response to changing the command.
        true=y-y.mean(0);estimate=yh-yh.mean(0)
        key=g['clip']
        stats=per.setdefault(key,dict(n=0,groups=0,ss=np.zeros(12),cv=np.zeros(6),err=np.zeros(12),base=np.zeros(12)))
        stats['groups']+=1;stats['n']+=len(x)
        stats['ss']+=(((yh-y)*dim_scale)**2).sum(0)
        stats['cv']+=((y[:,:6]*dim_scale[:6])**2).sum(0)
        stats['err']+=((estimate-true)**2).sum(0);stats['base']+=(true**2).sum(0)
    results={}
    for key,s in per.items():
        n=s['n']
        result=dict(rows=n,groups=s['groups'],position_vector_RMS_mm=np.sqrt(s['ss'][:3].sum()/n),
            rotation_vector_RMS_deg=np.sqrt(s['ss'][3:6].sum()/n),constant_velocity_position_RMS_mm=np.sqrt(s['cv'][:3].sum()/n),
            constant_velocity_rotation_RMS_deg=np.sqrt(s['cv'][3:6].sum()/n))
        for name,ix in [('translation',slice(0,3)),('rotation',slice(3,6)),('force',slice(6,9)),('torque',slice(9,12))]:
            result[name+'_response_error_over_constant']=np.sqrt(s['err'][ix].sum()/s['base'][ix].sum())
        results[str(key)]=result
    return results


def run(bank_dir,out):
    torch.manual_seed(8317);torch.set_num_threads(1)
    source=file_pin(bank_dir/'report.json');bank=physics.read_json(source)
    cfg=physics.read_json(physics.read_json(bank['intent'])['source']);inputs=cfg['inputs']
    chart=MugStepChart(mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model']))),physics.read_json(inputs['binding']))
    train,_=groups_for(bank['rows'],8311,chart,set())
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(source=source,inputs=inputs,code=[file_pin(__file__),
        file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/mug_step_response.py')],
        models=['only1884','shared'],seed=8317,updates=600,first_checkpoint=30,groups_per_step=8,actions_per_group=12,
        scope='Frozen 20 ms state/command response heads. Both fitted on seed8311; normalization only on each fit subset. Shared batches balance clips; the single-window fit uses the same total draws. Evaluate seed8312 after freezing, exclude identical physical source plus warmstart appearing anywhere in TRAIN. Relative response errors measured against within-source constant.'))
    models={};opts={};data={};schedules={}
    for arm in ('only1884','shared'):
        subset=[t for t in train if arm=='shared' or t['clip']==1884]
        X=np.concatenate([t['x'] for t in subset]);Y=np.concatenate([t['y'] for t in subset]);offset=np.cumsum([0]+[len(t['x']) for t in subset])
        x=torch.tensor(X,device='cuda');y=torch.tensor(Y,device='cuda')
        torch.manual_seed(8317);net=MugStepHead(X.shape[1]).cuda()
        net.mean.copy_(x.mean(0));net.scale.copy_(x.std(0,unbiased=False).clamp_min(.0005))
        net.ymean.copy_(y.mean(0));net.yscale.copy_(y.std(0,unbiased=False).clamp_min(.001))
        models[arm]=net;opts[arm]=torch.optim.Adam(net.parameters(),lr=.001);data[arm]=(x,y)
        rng=np.random.default_rng(8317);schedule=[]
        by_clip={c:[i for i,t in enumerate(subset) if t['clip']==c] for c in (1884,1888)}
        for _ in range(600):
            chosen=(rng.choice(by_clip[1884],8) if arm=='only1884' else
                    np.r_[rng.choice(by_clip[1884],4),rng.choice(by_clip[1888],4)])
            schedule.append([rng.integers(offset[i],offset[i+1],size=12) for i in chosen])
        schedules[arm]=torch.tensor(np.asarray(schedule),device='cuda')
    start=perf_counter()
    for step in range(600):
        for arm,net in models.items():
            x,y=data[arm];ix=schedules[arm][step];est=net(x[ix]);target=y[ix]
            diff=(est-target)/net.yscale
            loss=F.smooth_l1_loss(diff,torch.zeros_like(diff))+F.smooth_l1_loss(diff-diff.mean(1,keepdim=True),torch.zeros_like(diff))
            opt=opts[arm];opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
        if step==29:
            cp=out/'first.pt'
            payload=dict(step=30,schedules=schedules,arms={a:dict(net=n.state_dict(),adam=opts[a].state_dict()) for a,n in models.items()})
            atomic_checkpoint(cp,payload);restored=torch.load(cp,map_location='cuda',weights_only=True)
            assert raw_tree_equal(payload,restored)
            for a,n in models.items():
                n.load_state_dict(restored['arms'][a]['net']);opts[a].load_state_dict(restored['arms'][a]['adam'])
            elapsed=perf_counter()-start
            write_json(out/'health.json',dict(checkpoint=file_pin(cp),completed_steps=30,weight_Adam_schedule_exact=True,
                fit_elapsed_s=elapsed,estimated_remaining_fit_s=elapsed*570/30))
    models={a:n.cpu().eval() for a,n in models.items()};secs=perf_counter()-start
    pins={}
    for a,n in models.items():
        path=out/(a+'.pt');atomic_checkpoint(path,dict(n=len(n.mean),state=n.state_dict()));pins[a]=file_pin(path)
    write_json(out/'frozen.json',dict(pins=pins,fit_s=secs,train_groups={str(c):sum(t['clip']==c for t in train) for c in [1884,1888]}))
    test,excluded=groups_for(bank['rows'],8312,chart,{g['key'] for g in train})
    scores={}
    for a,pin in pins.items():
        bundle=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True)
        n=MugStepHead(bundle['n']);n.load_state_dict(bundle['state']);n.eval()
        scores[a]=evaluate(n,test)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),frozen=file_pin(out/'frozen.json'),
        first=file_pin(out/'health.json'),test_groups_excluded=excluded,heldout=scores))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();run(args.bank.resolve(),args.out.resolve())
