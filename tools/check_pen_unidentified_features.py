"""Counterfactual on fixed network weights, recomputing the mask on fitting inputs only."""
import json,sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from fit_pen_switch_augmented import episode
from fit_pen_load_forecast import batch,scores,rollout
from supermimic_research.mechanics.pen_load_increment import LoadIncrement
from supermimic_research.mechanics.pen_supported_increment import SupportedIncrement
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
import check_load_forecast_gradients as gradients


def main(root,fitdir,out,extra=None):
    out.mkdir(exist_ok=False)
    data={}
    for e in json.loads((root/'report.json').read_bytes())['episodes']:
        data.setdefault(e['label'],[]).append(episode(e))
    x,y=batch(sum([data[k] for k in 'ABCD'],[]),True)
    if extra:
        augmented=[]
        for r in json.loads((extra/'report.json').read_bytes())['episodes']:
            xx,yy,v,ph=episode(r);i=r['phase0']-2
            if len(xx)>i+1:augmented.append((xx[i:],yy[i:],v[i:],ph))
        sx,sy=batch(augmented,True);x=torch.cat([x,sx]);y=torch.cat([y,sy])
    pin=json.loads((fitdir/'report.json').read_bytes())['outcomes']['history']['model']
    weights=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True)
    old=LoadIncrement(x,y);old.load_state_dict(weights);old.eval()
    net=SupportedIncrement(x,y);weights['learnable_input']=net.learnable_input.clone();net.load_state_dict(weights);net.eval()
    torch.set_num_threads(2)
    with torch.no_grad():
        # Equality for the fitted function over every observed training input.
        torch.testing.assert_close(net(x),old(x),atol=1e-7,rtol=1e-6)
        tests={}
        for label in ('VAL','TEST1','TEST2','TEST3'):
            xx,yy=batch(data[label],True);a=old(xx);b=net(xx)
            unseen=((xx[:,~net.learnable_input]-net.center[~net.learnable_input]).abs()>1e-6).any(1)
            records=rollout(net,data[label][0],True)
            summary={str(ms):np.sqrt(np.mean(np.square([[r['mm'],r['deg']] for r in records if r['ms']==ms]),axis=0)).tolist() for ms in (20,40,80,160,320)}
            tests[label]=dict(controls=len(xx),old=scores(a-yy),new=scores(b-yy),changed_rows=int(unseen.sum()),open_loop_rms_mm_deg=summary,records=records)
    path=out/'supported_increment.pt';torch.save(net.state_dict(),path)
    write_json(out/'report.json',dict(parent=pin,training_controls=len(x),invariant_columns=torch.where(~net.learnable_input)[0].tolist(),outcomes={'history':dict(model=file_pin(path),tests=tests)},
        scope='Post-hoc heldout reanalysis; unchanged trained weights, training-invariant columns masked in nonlinear branch only. Includes validation and prior-inspected test actors.'))
    with patch.object(gradients,'LoadForecast',SupportedIncrement):gradients.evaluate(out,out/'native_gradient')
    for k,v in tests.items():print(k,v['old']['force_N'],v['new']['force_N'],v['changed_rows'],v['open_loop_rms_mm_deg']['80'],v['open_loop_rms_mm_deg']['320'])
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
