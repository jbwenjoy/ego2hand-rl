"""Train a candidate value head and measure held-out within-state choice quality."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.nn import functional as F
from collect_mug_handoff_labels import h3
from supermimic_research.mechanics.mug_handoff_value import HandoffValue
from supermimic_research.learning.world_model import atomic_checkpoint,raw_tree_equal


def choice(values,gate,order,margin):
    base=order[0]
    if gate[base]>1:return int(base)
    best=max((j for j in order if gate[j]<=1),key=lambda j:values[j])
    return int(best if values[best]-values[base]>=margin else base)


def fit(source,out):
    pin=h3.file_pin(source/'report.json');report=h3.physics.read_json(pin)
    with np.load(h3.verify_file_pin(report['data'])) as z:data=dict(z)
    train=data['seed']<=8436
    # Match action-independent state/warmstart columns of the logged feature chart.
    keys=[x[0,:111].tobytes() for x in data['x']]
    seen={k for k,t in zip(keys,train) if t}
    test=np.array([not t and k not in seen for k,t in zip(keys,train)])
    assert train.any() and test.any()
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(source=pin,code=[h3.file_pin(__file__),h3.file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/mug_handoff_value.py')],
        seed=84301,train_seeds=list(range(8431,8437)),test_seeds=[8437,8438],updates=350,batch_groups_per_clip=4,
        scope='Predict up to six consecutive native passes under exact H3 handoff. Normalize with training states only, learn absolute and within-state differences. Fixed last checkpoint; no test-based tuning. Gate all overrides by native first pose; compare best predicted, one-control margin, H3 teacher, and oracle.'))
    x=torch.as_tensor(data['x'][train],device='cuda')
    y=torch.as_tensor(data['passing'][train],dtype=torch.float32,device='cuda')
    torch.manual_seed(84301);net=HandoffValue(x.shape[-1]).cuda()
    net.mean.copy_(x.mean((0,1)));net.scale.copy_(x.flatten(0,1).std(0,unbiased=False).clamp_min(.0005))
    groups=[np.flatnonzero(data['clip'][train]==c) for c in (1884,1888)]
    rng=np.random.default_rng(84301)
    batches=torch.as_tensor(np.stack([np.r_[rng.choice(groups[0],4),rng.choice(groups[1],4)] for _ in range(350)]),device='cuda')
    opt=torch.optim.Adam(net.parameters(),lr=.0007)
    start=perf_counter()
    for step,idx in enumerate(batches,1):
        delta=net(x[idx])-y[idx]
        loss=F.smooth_l1_loss(delta,torch.zeros_like(delta)) + F.smooth_l1_loss(delta-delta.mean(1,keepdim=True),torch.zeros_like(delta))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),3,error_if_nonfinite=True);opt.step()
        if step==25:
            first=out/'first.pt';payload=dict(model=net.state_dict(),adam=opt.state_dict(),step=step,schedule=batches)
            atomic_checkpoint(first,payload);back=torch.load(first,map_location='cuda',weights_only=True)
            assert raw_tree_equal(back,payload)
            net.load_state_dict(back['model']);opt.load_state_dict(back['adam']);torch.cuda.synchronize()
            elapsed=perf_counter()-start
            h3.write_json(out/'health.json',dict(first=h3.file_pin(first),step=step,exact_weights_adam_schedule=True,
                finite_loss=float(loss.detach()),estimated_remaining_fit_s=elapsed*(350-step)/step))
    net=net.cpu().eval();wall=perf_counter()-start
    final=out/'value.pt';atomic_checkpoint(final,dict(model=net.state_dict(),width=len(net.mean),updates=350))
    frozen=h3.file_pin(final);ck=torch.load(h3.verify_file_pin(frozen),weights_only=True,map_location='cpu')
    net.load_state_dict(ck['model']);rows=[]
    with torch.inference_mode():preds=net(torch.from_numpy(data['x'])).numpy()
    for i,values in enumerate(preds):
        base=data['baseline'][i]; order=np.r_[base,np.flatnonzero(np.arange(8)!=base)]
        v=data['passing'][i]
        picked=[choice(values,data['gate'][i],order,m) for m in (0.,1.)]
        rows.append(dict(clip=int(data['clip'][i]),seed=int(data['seed'][i]),phase=int(data['phase'][i]),
            role='train' if train[i] else ('test' if test[i] else 'overlap'),baseline=int(v[base]),best=int(v.max()),
            worst=int(v.min()),always=int(v[picked[0]]),margin=int(v[picked[1]]),mae=float(abs(v-values).mean()),
            action_response_sse=float(np.sum(((v-v.mean())-(values-values.mean()))**2)),zero_response_sse=float(np.sum((v-v.mean())**2))))
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),checkpoint=frozen,health=h3.file_pin(out/'health.json'),
        fit_s=wall,counts=dict(train=int(train.sum()),test=int(test.sum()),overlap=int((~train&~test).sum())),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();fit(a.source.resolve(),a.out.resolve())
