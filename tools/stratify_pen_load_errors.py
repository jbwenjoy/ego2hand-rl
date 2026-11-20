"""Compare held-out forecast errors at contact changes and on identical pose windows."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from fit_pen_load_forecast import LoadForecast,batch,state_rotation,PHASES
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(root,fitdir,out):
    report=json.loads((fitdir/'report.json').read_bytes())
    master=json.loads((root/'report.json').read_bytes())
    models={}
    for mode,details in report['outcomes'].items():
        w=torch.load(verify_file_pin(details['model']),map_location='cpu',weights_only=True)
        net=LoadForecast(torch.zeros(2,len(w['center'])),torch.zeros(2,len(w['out_center'])))
        net.load_state_dict(w);models[mode]=net.eval()
    result={}
    for e in master['episodes']:
        if not e['label'].startswith('TEST'):continue
        with np.load(verify_file_pin(e['trajectory'])) as z:x,y,v=(z[k] for k in ('x','y','valid'))
        eps=[(x,y,v,None)]
        xx,yy=batch(eps,True);prev=xx[:,-12:]
        switch=((yy[:,6:12]>.1)!=(prev[:,6:12]>.1)).any(1)
        predictions={'last_load':prev}
        with torch.no_grad():
            for mode,net in models.items():
                xf,_=batch(eps,mode=='history')
                predictions[mode]=net(xf)[:,:12]
        parts={}
        for name,mask in [('changed',switch),('unchanged',~switch)]:
            if not mask.any():continue
            parts[name]=dict(n=int(mask.sum()),models={})
            for key,pred in predictions.items():
                err=pred[mask]-yy[mask,:12]
                parts[name]['models'][key]={head:float(err[:,a:b].square().sum(1).mean().sqrt()) for head,a,b in [('force_N',0,3),('torque_Nm',3,6),('normal_N',6,12)]}
        baselines={}
        for k in [1,2,4,8,16]:
            metrics={'constant_pose':[],'constant_velocity':[]}
            for phase in PHASES:
                i=phase-1
                if i+16>=len(x) or not np.all(v[i-1:i+17]):continue
                s=x[i,:55];truth=x[i+k,:55];ri=Rotation.from_matrix(state_rotation(s));rt=Rotation.from_matrix(state_rotation(truth))
                for mode,pos,r in [('constant_pose',s[40:43],ri),('constant_velocity',s[40:43]+.02*k*s[49:52],Rotation.from_rotvec(.02*k*s[52:55])*ri)]:
                    metrics[mode].append([1000*np.linalg.norm(pos-truth[40:43]),np.rad2deg((r*rt.inv()).magnitude())])
            baselines[str(k*20)]={name:np.sqrt(np.mean(np.square(a),axis=0)).tolist() for name,a in metrics.items() if a}
        result[e['label']]=dict(contact_groups=parts,matched_horizon_baselines=baselines)
    out.mkdir(exist_ok=False);write_json(out/'report.json',dict(source=file_pin(fitdir/'report.json'),tests=result,scope='Contact group change defined post hoc by changes in the prior/current six mean group normal force threshold 0.1 N. Never an input to fitting.'))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
