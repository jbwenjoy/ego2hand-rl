"""Matched-update re-fit with post-switch training rows and pre-action mask accounting."""
import copy,json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from fit_pen_load_forecast import batch,rollout,scores
from fit_pen_load_increment import LoadIncrement,gradients
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def episode(row):
    with np.load(verify_file_pin(row['trajectory'])) as f:x,y,v=(f[k] for k in ['x','y','valid'])
    return x,y,v,json.loads(verify_file_pin(row['audit']).read_bytes())['physics']


def mask_codes(x):
    return ((x[:,-6:]>.1).long()*(2**torch.arange(6))).sum(1).numpy()


def main(root,extra,out):
    out.mkdir(exist_ok=False)
    main_data=json.loads((root/'report.json').read_bytes())
    new_data=json.loads((extra/'report.json').read_bytes())
    write_json(out/'intent.json',dict(base=file_pin(root/'report.json'),augmentation=file_pin(extra/'report.json'),
        target='Same additive LoadIncrement, same absolute-output scaling convention',sampling='Uniform without replacement, original training count per epoch. Each extra example owns both its source and label and uses an interval at/after switch. Repeated source prefixes omitted.',
        validation='identical 60 epoch maximum and contact-based VAL selection. Previously inspected holdout policies; exploratory comparison.'))
    data={}
    for r in main_data['episodes']:data.setdefault(r['label'],[]).append(episode(r))
    switches=[]
    for r in new_data['episodes']:
        x,y,v,ph=episode(r)
        # examples() uses the preceding row as history. First target phase is phase0.
        i=r['phase0']-2
        if len(x)>i+1:switches.append((x[i:],y[i:],v[i:],ph))
    x,y=batch(sum([data[k] for k in 'ABCD'],[]),True);n=len(x)
    sx,sy=batch(switches,True)
    before=set(mask_codes(x));x=torch.cat([x,sx]);y=torch.cat([y,sy]);after=set(mask_codes(x))
    vx,vy=batch(data['VAL'],True)
    torch.set_num_threads(2);torch.manual_seed(7411)
    net=LoadIncrement(x,y);opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');curve=[]
    for epoch in range(1,61):
        for ids in torch.randperm(len(x))[:n].split(128):
            err=((net(x[ids])-y[ids])/net.out_scale).square();loss=err[:,:12].mean()+err[:,12:].mean()
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():val=float((((net(vx)-vy)/net.out_scale)[:,:12]**2).mean())
        curve.append(val)
        if val<best:best=val;state=copy.deepcopy(net.state_dict());chosen=epoch
    net.load_state_dict(state);path=out/'increment_augmented.pt';torch.save(state,path);tests={}
    with torch.no_grad():
        for k in ['TEST1','TEST2','TEST3']:
            xx,yy=batch(data[k],True);codes=mask_codes(xx)
            rec=rollout(net,data[k][0],True)
            summary={str(ms):np.sqrt(np.mean(np.square([[r['mm'],r['deg']] for r in rec if r['ms']==ms]),axis=0)).tolist() for ms in (20,40,80,160,320)}
            tests[k]=dict(errors=scores(net(xx)-yy),previous=scores(xx[:,-12:]-yy[:,:12]),open_loop_rms_mm_deg=summary,
                controls=len(xx),unseen_old=sum(int(z not in before) for z in codes),unseen_new=sum(int(z not in after) for z in codes),records=rec)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),original=n,extra=len(sx),new_masks=sorted(map(int,after-before)),outcomes={'history':dict(model=file_pin(path),chosen_epoch=chosen,val_curve=curve,tests=tests)}))
    with patch.object(gradients,'LoadForecast',LoadIncrement):gradients.evaluate(out,out/'native_gradient')
    print(n,len(sx),chosen,{k:(r['errors'],r['open_loop_rms_mm_deg']['80'],r['unseen_old'],r['unseen_new']) for k,r in tests.items()})
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
