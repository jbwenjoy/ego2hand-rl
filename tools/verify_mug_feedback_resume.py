"""Resume paired actors from update 30 and compare the final parameters byte for byte."""
import argparse
from pathlib import Path
import torch
from distill_mug_witness_feedback import MugResidualActor,h3,raw_tree_equal


def verify(run,out):
    source=h3.file_pin(run/'report.json');result=h3.physics.read_json(source)
    frozen=h3.physics.read_json(result['frozen']);health=h3.physics.read_json(result['health'])
    ck=torch.load(h3.verify_file_pin(health['checkpoint']),weights_only=True,map_location='cuda')
    data=h3.physics.load_arrays(frozen['teacher'])
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
    x=tensor(data['context']);e=tensor(data['error']);nets={};opts={};y={}
    for name,bundle in ck['models'].items():
        n=MugResidualActor(x.shape[1],e.shape[1]).cuda();n.load_state_dict(bundle['state'])
        opt=torch.optim.Adam(n.parameters(),lr=.0007);opt.load_state_dict(bundle['optimizer'])
        nets[name]=n;opts[name]=opt;y[name]=tensor(data[name])
    for ix in ck['schedule'][ck['step']:]:
        for name,n in nets.items():
            loss=((n(x[ix],e[ix])-y[name][ix])/.06).square().mean()
            optim=opts[name];optim.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(n.parameters(),2.,error_if_nonfinite=True);optim.step()
    checks={}
    for name,n in nets.items():
        final=torch.load(h3.verify_file_pin(frozen['weights'][name]),map_location='cpu',weights_only=True)['state']
        rerun=n.cpu().state_dict()
        checks[name]=dict(exact=raw_tree_equal(rerun,final),max_abs=max(float((rerun[k]-v).abs().max()) for k,v in final.items()))
    out.mkdir(exist_ok=False);h3.write_json(out/'report.json',dict(source=source,first=health['checkpoint'],
        resume_cursor=ck['step'],terminal_cursor=len(ck['schedule']),recovery=checks,code=h3.file_pin(__file__)))
    assert all(v['exact'] for v in checks.values())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();verify(a.run.resolve(),a.out.resolve())
