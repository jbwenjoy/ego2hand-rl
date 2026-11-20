"""Test fixed pre-action contact descriptors without fitting held-out response labels."""
import json
from pathlib import Path
import sys
import numpy as np
import mujoco
import torch
from fit_guarded_finger_response import groups,fit,evaluate,HEADS
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.mechanics.pen_contact_groups import normal_group
from supermimic_research.strict_json import write_json


def descriptor(source,model):
    snap=torch.load(verify_file_pin(source),map_location='cpu',weights_only=False)
    t=snap['python']['tensors'];d=snap['native']['data']['arrays']
    get=lambda k:t[(('attr','contact_views'),('item',k))][0].numpy()
    count=get('counts')[0]
    if count!=8:raise ValueError('Require full preceding control')
    loads=get('Fn6_sum')/count
    body=model.body('object/cube').id
    contact_points=[[] for _ in range(6)]
    for i in range(int(d['nacon'][0])):
        g,h=d['contact.geom'][i]
        on=model.geom_bodyid[[g,h]]==body
        if on.sum()!=1:continue
        digit=normal_group(model.geom(int(h if on[0] else g)).name or '')
        if digit<0:continue
        local=d['xmat'][0,body].T@(d['contact.pos'][i]-d['xipos'][0,body])
        contact_points[digit].append(local)
    centers=np.array([np.mean(p,axis=0) if p else np.zeros(3) for p in contact_points])
    return loads,centers,np.array([bool(p) for p in contact_points])


def main(root,out):
    root=root.resolve();out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(source=file_pin(root/'report.json'),method='Nearest load-mask Hamming distance (>0.1N preceding mean), then mean normalized state distance + mean squared log1p normal force difference (N) + mean squared shared digit centroid separation / (1cm)^2. Fixed existing full-rank TRAIN anchors. Same source rollouts and mixed-action pairs as previous study; exploratory reanalysis of previously inspected holdout.'))
    manifest=json.loads((root/'report.json').read_bytes())
    reports={k:json.loads(verify_file_pin(pin).read_bytes()) for k,pin in manifest['arms'].items()}
    datasets={k:groups(r) for k,r in reports.items()}
    feat={}
    for k,r in reports.items():
        model=mujoco.MjModel.from_binary_path(str(root/k/'model.mjb'))
        sources={row['phase']:row['source'] for row in r['rows']}
        feat[k]={p:descriptor(pin,model) for p,pin in sources.items()}
    ids=reports['TRAIN']['selected']
    anchors=fit(datasets['TRAIN'],ids)
    states=np.stack([a[1] for a in anchors]);scale=np.maximum(states.std(0),1e-3)
    all_results={}
    for role,data in datasets.items():
        rows=[]
        for p,g in data.items():
            load,centers,present=feat[role][p]
            dist=[]
            for anchor in anchors:
                l,c,v=feat['TRAIN'][anchor[0]];both=v&present
                state_dist=float(np.mean(((anchor[1]-g['sham']['x'][:55])/scale)**2))
                load_dist=float(np.mean((np.log1p(load)-np.log1p(l))**2))
                point_dist=float(np.mean(np.sum((centers[both]-c[both])**2,axis=1))/.01**2) if both.any() else 0.
                dist.append((int(np.count_nonzero((load>.1)!=(l>.1))),state_dist+load_dist+point_dist))
            idx=min(range(len(anchors)),key=dist.__getitem__)
            response=evaluate({p:g},[anchors[idx]],ids)['rows']
            if response:
                rows.append(dict(**response[0],mask_difference=dist[idx][0],load_N=load.tolist(),distances=dist))
        base=evaluate(data,anchors,ids)
        errors={}
        for name in HEADS:
            heads=[r['heads'][name] for r in rows]
            errors[name]=float(np.sqrt(sum(h['error']**2 for h in heads)/sum(h['observed_delta']**2 for h in heads)))
        all_results[role]=dict(rows=rows,errors=errors,state_only=base['summary'],switches=sum(a['anchor']!=b['anchor'] for a,b in zip(rows,base['rows'])),
            exact_masks=sum(r['mask_difference']==0 for r in rows))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),anchor_phases=[a[0] for a in anchors],results=all_results))
    print({k:{n:v[n] for n in ('errors','switches','exact_masks')} for k,v in all_results.items()})
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
