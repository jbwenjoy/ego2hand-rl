"""Fit two shared block-allocation models; test the timing policy on independent scenarios."""
import argparse
from pathlib import Path
from time import perf_counter
import torch
from torch.nn import functional as F
import numpy as np
from test_mug_event_budget import h3
from supermimic_research.mechanics.mug_search_schedule import MugSlotHead
from supermimic_research.learning.world_model import atomic_checkpoint,raw_tree_equal


def fit(source,out):
    pin=h3.file_pin(source/'report.json');report=h3.physics.read_json(pin);data=h3.physics.load_arrays(report['data'])
    train=data['role']=='train';seen=set(data['key'][train]);test=np.asarray([not v and k not in seen for v,k in zip(train,data['key'])])
    assert data['x'].shape[1]==177 and np.isfinite(data['x']).all()
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(source=pin,code=[h3.file_pin(__file__),h3.file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/mug_search_schedule.py')],
        update_budget=300,seed=8562,
        scope='Predict within-state centered passing counts for four search slots. Clip-balanced samples, half of each minibatch deliberately from states whose label changes with slot. Fit a reference-only comparator by masking all measured error and previous action residual features. No held-out labels in fit/scales, fixed final checkpoint. Revert to periodic slot if predicted advantage <0.5 controls.'))
    to=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda')
    x=to(data['x'][train]);y=to(data['values'][train]);target=y-y.mean(1,keepdim=True)
    clip=data['clip'][train];changed=np.ptp(data['values'][train],axis=1)>0
    rng=np.random.default_rng(8562);schedule=[]
    for _ in range(300):
        batch=[]
        for c in (1884,1888):
            batch.extend(rng.choice(np.flatnonzero(clip==c),8))
            batch.extend(rng.choice(np.flatnonzero((clip==c)&changed),8))
        schedule.append(batch)
    schedule=torch.as_tensor(np.array(schedule),device='cuda')
    nets={};optim={}
    for arm in ('reference','measured'):
        torch.manual_seed(8562);n=MugSlotHead(x.shape[1],arm=='measured').cuda()
        n.center.copy_(x.mean(0));n.scale.copy_(x.std(0,unbiased=False).clamp_min(.001))
        nets[arm]=n;optim[arm]=torch.optim.AdamW(n.parameters(),lr=.0007,weight_decay=.01)
    start=perf_counter()
    for i,ix in enumerate(schedule,1):
        for arm,n in nets.items():
            loss=F.smooth_l1_loss(n(x[ix]),target[ix]);o=optim[arm]
            o.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),2.,error_if_nonfinite=True);o.step()
        if i==25:
            path=out/'first.pt';payload=dict(step=i,schedule=schedule,arms={a:dict(model=n.state_dict(),adam=optim[a].state_dict()) for a,n in nets.items()})
            atomic_checkpoint(path,payload);restored=torch.load(path,map_location='cuda',weights_only=True)
            assert raw_tree_equal(payload,restored)
            for a,n in nets.items():
                n.load_state_dict(restored['arms'][a]['model']);optim[a].load_state_dict(restored['arms'][a]['adam'])
            torch.cuda.synchronize();elapsed=perf_counter()-start
            h3.write_json(out/'health.json',dict(step=i,first=h3.file_pin(path),exact_weights_adam_schedule=True,loss=float(loss.detach()),
                estimated_remaining_fit_s=elapsed*(300-i)/i))
    torch.cuda.synchronize();duration=perf_counter()-start;pins={}
    for a,n in nets.items():
        path=out/f'{a}.pt';atomic_checkpoint(path,dict(model=n.cpu().state_dict(),error=n.use_error,width=x.shape[1]))
        pins[a]=h3.file_pin(path)
    h3.write_json(out/'frozen.json',dict(weights=pins,fit_s=duration))
    values=data['values'][test];ix=np.arange(len(values));period=data['periodic'][test]
    scores={}
    for a,pin in pins.items():
        ck=torch.load(h3.verify_file_pin(pin),map_location='cpu',weights_only=True)
        n=MugSlotHead(ck['width'],ck['error']);n.load_state_dict(ck['model']);n.eval()
        with torch.inference_mode():pred=n(torch.from_numpy(data['x'][test])).numpy()
        raw=np.argmax(pred,axis=1);pick=np.where(pred[ix,raw]-pred[ix,period]>=.5,raw,period)
        scores[a]={str(c):dict(n=int(mask.sum()),baseline=int(values[ix,period][mask].sum()),best=int(values.max(1)[mask].sum()),
            choice_sum=int(values[ix,pick][mask].sum()),switches=int((pick[mask]!=period[mask]).sum()),
            wins=int((values[ix,pick][mask]>values[ix,period][mask]).sum()),losses=int((values[ix,pick][mask]<values[ix,period][mask]).sum()))
            for c in (1884,1888) for mask in [data['clip'][test]==c]}
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),health=h3.file_pin(out/'health.json'),
        frozen=h3.file_pin(out/'frozen.json'),counts=[int(train.sum()),int(test.sum())],heldout=scores))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();fit(a.source.resolve(),a.out.resolve())
