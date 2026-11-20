"""Fixed comparisons for the expanded bank using an unseen residual behavior."""
from pathlib import Path
import json,sys
import numpy as np
import mujoco
from check_guarded_contact_selection import descriptor
from fit_guarded_finger_response import groups,fit,evaluate,HEADS
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def load_role(root,role):
    master=json.loads((root/'report.json').read_bytes())
    r=json.loads(verify_file_pin(master['arms'][role]).read_bytes())
    data=groups(r);m=mujoco.MjModel.from_binary_path(str(root/role/'model.mjb'))
    features={p:descriptor(pin,m) for p,pin in {x['phase']:x['source'] for x in r['rows']}.items()}
    return data,features,r['selected']


def compare(anchors,examples,ids,contact):
    states=np.stack([a[1][1] for a in anchors]);scale=np.maximum(states.std(0),1e-3)
    rows=[]
    data,features,_=examples
    for p,g in data.items():
        load,centers,present=features[p]
        distances=[]
        for _,anchor,(l,c,v) in anchors:
            both=present&v
            d=np.mean(((g['sham']['x'][:55]-anchor[1])/scale)**2)
            if contact:
                d+=np.mean((np.log1p(load)-np.log1p(l))**2)
                if both.any():d+=np.mean(np.sum((centers[both]-c[both])**2,axis=1))/.01**2
            distances.append((np.count_nonzero((load>.1)!=(l>.1)) if contact else 0,d))
        best=min(range(len(anchors)),key=distances.__getitem__)
        role,anchor,_=anchors[best]
        output=evaluate({p:g},[anchor],ids)['rows']
        if output:rows.append(dict(role=role,**output[0]))
    summary={}
    for name in HEADS:
        h=[r['heads'][name] for r in rows];sig=sum(i['observed_delta']**2 for i in h)
        summary[name]=dict(pairs=len(h),positive_dot=sum(i['signed_dot']>0 for i in h),
            error_over_zero=float(np.sqrt(sum(i['error']**2 for i in h)/sig)) if sig else None)
    return dict(rows=rows,summary=summary)


def main(root,out):
    out.mkdir(exist_ok=False)
    prior=root.parent/'guarded_finger_response_v1'
    roles={k:load_role(root if k!='ORIGINAL' else prior,k if k!='ORIGINAL' else 'TRAIN') for k in ['ORIGINAL','TRAIN_A','TRAIN_B','TEST']}
    ids=roles['ORIGINAL'][2];assert all(v[2]==ids for v in roles.values())
    anchors={k:[(k,a,v[1][a[0]]) for a in fit(v[0],ids)] for k,v in roles.items() if k!='TEST'}
    assert all(anchors.values())
    result=dict(source=file_pin(root/'report.json'),prior=file_pin(prior/'report.json'),full_rank_phases={k:[a[1][0] for a in v] for k,v in anchors.items()},tests={})
    for test,examples in [('new_policy',roles['TEST']),('previously_inspected_support',load_role(prior,'HOLDOUT'))]:
        rows={}
        for label,bank in [('original',anchors['ORIGINAL']),('expanded',anchors['TRAIN_A']+anchors['TRAIN_B'])]:
            for contact in (False,True):
                key=label+('_contact' if contact else '_state')
                rows[key]=compare(bank,examples,ids,contact)
        result['tests'][test]=rows
    write_json(out/'report.json',result)
    for name,methods in result['tests'].items():
        print(name,{k:{h:round(a['error_over_zero'],3) for h,a in v['summary'].items()} for k,v in methods.items()})
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
