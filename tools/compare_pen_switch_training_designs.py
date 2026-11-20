"""Compare contact-response labels obtained using high-risk and uniform source acquisition."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def main(root,out):
    out.mkdir(exist_ok=False)
    src=file_pin(root/'report.json');bank=json.loads(verify_file_pin(src).read_bytes())
    write_json(out/'intent.json',dict(source=src,model='Fixed same-command contrast estimator, three inverse-distance weighted TRAIN sources. Inputs: source-only CPU gap and normal velocity across fixed D2/D3-hand/pen geom ID pairs + actual applied PD unit direction. Distances based on 1mm gap, 0.05m/s normal speed, normalized gap range +/-5mm. No phase or endpoint-derived inputs. Target: capacity difference / actual PD contrast length. k and metric fixed across designs; no fitted hyperparameters.',scope='Previously inspected single-reference behaviors. Two source-acquisition designs have equal declared strata budgets. Separate TEST outcomes inspected only after computing predictions.'))
    pools={}
    for role,arm in bank['actors'].items():
        screen=json.loads(verify_file_pin(arm['screen']).read_bytes())
        pools[role]={c['phase']:c for c in screen['candidates']}
    keys=sorted({(r['hand_geom'],r['object_geom']) for role in ('TRAIN_A','TRAIN_B') for c in pools[role].values() for r in c['pairs']})
    keys={k:i for i,k in enumerate(keys)}
    examples=[]
    for s in bank['rows']:
        b={r['name']:r for r in s['branches']}
        if not all(v['valid'] and not v['terminal'] for v in b.values()):continue
        pd=np.subtract(b['plus']['pd'],b['minus']['pd']);norm=np.linalg.norm(pd)
        if norm<1e-7:continue
        state=np.zeros((len(keys),2));state[:,0]=5.
        for r in pools[s['actor']][s['phase']]['pairs']:
            j=keys.get((r['hand_geom'],r['object_geom']))
            if j is not None:state[j]=[np.clip(r['gap_m']/.001,-5,5),np.clip(r['normal_speed_m_s']/.05,-5,5)]
        examples.append(dict(role=s['actor'],phase=s['phase'],risk=s['risk_selected'],uniform=s['uniform_selected'],
            key=np.r_[state.flatten()/np.sqrt(len(keys)),pd/norm],norm=norm,dy=np.subtract(b['plus']['capacity'],b['minus']['capacity'])))
    result={}
    for design in ('risk','uniform'):
        anchors=[e for e in examples if e['role'].startswith('TRAIN') and e[design]]
        features=np.stack([a['key'] for a in anchors]);slopes=np.stack([a['dy']/a['norm'] for a in anchors]);rows=[]
        for e in examples:
            if not e['role'].startswith('TEST'):continue
            dist=np.linalg.norm(features-e['key'],axis=1);near=np.argsort(dist,kind='stable')[:3]
            pred=np.average(slopes[near],axis=0,weights=1/(.001+dist[near]))*e['norm']
            rows.append(dict(role=e['role'],phase=e['phase'],risk=e['risk'],uniform=e['uniform'],native_L2=np.linalg.norm(e['dy']),
                             err_L2=np.linalg.norm(e['dy']-pred),positive_dot=pred@e['dy']>0,closest=float(dist[near[0]])))
        roll={}
        for role in ('TEST_RESIDUAL','TEST_SUPPORT'):
            for subset in ('risk','uniform','all'):
                sel=[r for r in rows if r['role']==role and (subset=='all' or r[subset])]
                signal=sum(r['native_L2']**2 for r in sel)
                roll[f'{role}_{subset}']=dict(n=len(sel),relative_error=np.sqrt(sum(r['err_L2']**2 for r in sel)/signal) if signal else None,
                     positive_dot=sum(bool(r['positive_dot']) for r in sel))
        result[design]=dict(train_sources=len(anchors),rows=rows,summary=roll)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),results=result))
    print({k:v['summary'] for k,v in result.items()})
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
