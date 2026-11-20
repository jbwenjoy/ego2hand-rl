"""Paired persistence-residual contact ablation and pre-action coverage diagnostics."""
import sys,json,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from fit_pen_load_forecast import batch,rollout,scores
import check_load_forecast_gradients as gradients
from supermimic_research.mechanics.pen_load_increment import LoadIncrement
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(root,out):
    out.mkdir(exist_ok=False)
    src=root/'report.json';master=json.loads(src.read_bytes())
    write_json(out/'intent.json',dict(data=file_pin(src),method='previous source-frame load plus learned correction; same absolute-label output scales and validation objective as history MLP',seed=7411,epochs=60,lr=.001,batch=128,split='A-D training, VAL epoch selection, TEST1-3 previously inspected heldout actors',coverage='Training 6-digit load masks at >0.1 N, plus raw componentwise training extrema; descriptive only.'))
    datasets={}
    for e in master['episodes']:
        with np.load(verify_file_pin(e['trajectory'])) as f:
            xs,ys,ok=[f[k] for k in ('x','y','valid')]
        datasets.setdefault(e['label'],[]).append((xs,ys,ok,json.loads(verify_file_pin(e['audit']).read_bytes())['physics']))
    x,y=batch(sum([datasets[k] for k in 'ABCD'],[]),True);vx,vy=batch(datasets['VAL'],True)
    torch.set_num_threads(2);torch.manual_seed(7411)
    net=LoadIncrement(x,y);opt=torch.optim.Adam(net.parameters(),lr=.001);curve=[];best=float('inf')
    for epoch in range(1,61):
        for inds in torch.randperm(len(x)).split(128):
            e=((net(x[inds])-y[inds])/net.out_scale).square()
            loss=e[:,:12].mean()+e[:,12:].mean()
            opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad(): val=float((((net(vx)-vy)/net.out_scale)[:,:12]**2).mean())
        curve.append(val)
        if val<best:best=val;chosen=epoch;st=copy.deepcopy(net.state_dict())
    net.load_state_dict(st);model_path=out/'increment.pt';torch.save(st,model_path)
    masks=lambda inputs: ((inputs[:,-6:]>.1).long()*(2**torch.arange(6))).sum(1).tolist()
    train_masks=set(masks(x));lo=x.min(0).values;hi=x.max(0).values;tests={}
    with torch.no_grad():
        for label in ('TEST1','TEST2','TEST3'):
            xx,yy=batch(datasets[label],True)
            yh=net(xx);error=yh-yy
            outside=((xx<lo-1e-6)|(xx>hi+1e-6))
            novel=torch.tensor([m not in train_masks for m in masks(xx)])
            records=rollout(net,datasets[label][0],True)
            summary={str(ms):np.sqrt(np.mean(np.square([[r['mm'],r['deg']] for r in records if r['ms']==ms]),axis=0)).tolist() for ms in (20,40,80,160,320)}
            novelty=dict(unseen_masks=int(novel.sum()),hist_load_outside=int(outside[:,-12:].any(1).sum()),max_prev_digit_N=xx[:,-6:].max(0).values.tolist(),mask_strata={})
            for name,part in [('familiar',~novel),('novel',novel)]:
                if part.any():novelty['mask_strata'][name]=scores(error[part])
            tests[label]=dict(controls=len(xx),errors=scores(error),last_load_errors=scores(xx[:,-12:]-yy[:,:12]),coverage=novelty,open_loop_rms_mm_deg=summary,records=records)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),training_masks=sorted(train_masks),max_prev_digit_train_N=hi[-6:].tolist(),
        outcomes={'history':dict(architecture='LoadIncrement',model=file_pin(model_path),chosen_epoch=chosen,train_controls=len(x),val_controls=len(vx),val_curve=curve,tests=tests)}))
    with patch.object(gradients,'LoadForecast',LoadIncrement):
        gradients.evaluate(out,out/'native_gradient')
    print('epoch',chosen)
    for k,v in tests.items():print(k,v['errors'],v['open_loop_rms_mm_deg']['80'],v['open_loop_rms_mm_deg']['320'],v['coverage']['unseen_masks'],v['coverage']['hist_load_outside'])
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
