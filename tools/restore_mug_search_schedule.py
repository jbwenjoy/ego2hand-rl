"""Verify exact final replication of both timing heads from the first saved checkpoint."""
import argparse
from pathlib import Path
import torch
from torch.nn import functional as F
from fit_mug_search_schedule import MugSlotHead,h3,raw_tree_equal


def restore(run,out):
    pin=h3.file_pin(run/'report.json');fit=h3.physics.read_json(pin)
    data=h3.physics.load_arrays(h3.physics.read_json(h3.physics.read_json(fit['intent'])['source'])['data'])
    is_train=data['role']=='train'
    x=torch.as_tensor(data['x'][is_train],dtype=torch.float32,device='cuda')
    y=torch.as_tensor(data['values'][is_train],dtype=torch.float32,device='cuda');y=y-y.mean(1,keepdim=True)
    first=h3.physics.read_json(fit['health'])['first'];ck=torch.load(h3.verify_file_pin(first),weights_only=True,map_location='cuda')
    frozen=h3.physics.read_json(fit['frozen'])['weights'];nets={};opts={}
    for name,states in ck['arms'].items():
        n=MugSlotHead(x.shape[1],name=='measured').cuda();n.load_state_dict(states['model'])
        opt=torch.optim.AdamW(n.parameters(),lr=.0007,weight_decay=.01);opt.load_state_dict(states['adam'])
        nets[name]=n;opts[name]=opt
    for ix in ck['schedule'][ck['step']:]:
        for name,n in nets.items():
            loss=F.smooth_l1_loss(n(x[ix]),y[ix])
            opt=opts[name];opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(n.parameters(),2.,error_if_nonfinite=True);opt.step()
    checks={}
    for name,n in nets.items():
        expected=torch.load(h3.verify_file_pin(frozen[name]),map_location='cpu',weights_only=True)['model']
        actual=n.cpu().state_dict()
        checks[name]=dict(exact=raw_tree_equal(actual,expected),max_abs=max(float(abs(actual[k]-v).max()) for k,v in expected.items()))
    out.mkdir(exist_ok=False)
    h3.write_json(out/'report.json',dict(fit=pin,first=first,start=ck['step'],final=len(ck['schedule']),checks=checks,code=h3.file_pin(__file__)))
    assert all(c['exact'] for c in checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();restore(a.run.resolve(),a.out.resolve())
