"""Directional derivatives wrt actual PD targets on held-out native plus/minus pairs."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from fit_guarded_finger_response import groups
from check_pen_momentum_closure import state_rotation
from supermimic_research.mechanics.pen_load_forecast import LoadForecast,features,rotate_load
from supermimic_research.artifacts import verify_file_pin,file_pin
from supermimic_research.strict_json import write_json


def evaluate(fitdir,out):
    out.mkdir(exist_ok=False)
    root=fitdir.parent; models=json.loads((fitdir/'report.json').read_bytes())['outcomes']
    sources=[('residual','guarded_finger_bank_v1','TEST'),('support','guarded_finger_response_v1','HOLDOUT')]
    scores={}
    for typ,model_info in models.items():
        weights=torch.load(verify_file_pin(model_info['model']),map_location='cpu',weights_only=True)
        net=LoadForecast(torch.zeros(3,len(weights['center'])),torch.zeros(3,len(weights['out_center'])))
        net.load_state_dict(weights);net=net.double().eval()
        for label,folder,role in sources:
            src=json.loads((root/folder/'report.json').read_bytes());bank=json.loads(verify_file_pin(src['arms'][role]).read_bytes())
            data=groups(bank);pins={r['phase']:r['source'] for r in bank['rows']};rows=[]
            for phase,group in data.items():
                m,p=group['mixed_minus'],group['mixed_plus']
                if not (m['ok'] and p['ok']):continue
                snap=torch.load(verify_file_pin(pins[phase]),map_location='cpu',weights_only=False)
                t=snap['python']['tensors'];get=lambda k:t[(('attr','contact_views'),('item',k))][0].numpy()
                state=m['x'][:55];assert np.array_equal(state,p['x'][:55]) and get('counts')[0]==8
                prev=rotate_load(np.r_[get('wrench_sum')[0],get('Fn6_sum')]/8,state_rotation(state).T)
                pd=.5*(m['pd'].astype(float)+p['pd']);d=p['pd'].astype(float)-m['pd']
                feat=features(state,pd,prev if typ=='history' else None)
                tangent=np.zeros_like(feat);tangent[55:75]=d
                J=torch.autograd.functional.jvp(net,torch.from_numpy(feat),torch.from_numpy(tangent))[1].detach().numpy()[:12]
                measured=p['y'][78:90]-m['y'][78:90]
                fd=(net(torch.from_numpy(feat+tangent*.5))-net(torch.from_numpy(feat-tangent*.5))).detach().numpy()[:12]
                rows.append(dict(phase=phase,jvp=J.tolist(),finite_change=fd.tolist(),native=measured.tolist()))
            summary={}
            for head,(a,b) in {'force':(0,3),'torque':(3,6),'normal':(6,12)}.items():
                pred=np.array([r['jvp'][a:b] for r in rows]);actual=np.array([r['native'][a:b] for r in rows])
                summary[head]=dict(error_over_zero=float(np.linalg.norm(pred-actual)/np.linalg.norm(actual)),positive_dot=int((np.sum(pred*actual,axis=1)>0).sum()),pairs=len(rows))
            scores[typ+'_'+label]=dict(source=src['arms'][role],rows=rows,summary=summary)
            print(typ,label,summary)
    write_json(out/'report.json',dict(fit=file_pin(fitdir/'report.json'),scores=scores,scope='Single 20ms applied-PD mixed direction. Frozen pre-action states/history. Both native endpoints admitted.'))
if __name__=='__main__':evaluate(*map(Path,sys.argv[1:]))
