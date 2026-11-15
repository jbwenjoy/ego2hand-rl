"""Descriptive amplitude/contact-pattern readback, never a causal claim."""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.runtime.controlled_evaluation import write


def pattern(c):
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(c['host_topology'])));z=np.load(verify_file_pin(c['trace']));raw=z['raw_contact_tick_plus23'];body=c['model_layout']['object_body']
    rows=[]
    for tick in range(len(z['post_qpos'])):
        ids=np.flatnonzero(raw[:,0]==tick);geometry=set();active=set();positive=set();edge_count=0;Fn=np.zeros(6)
        for i in ids:
            row=raw[i];g,h=map(int,row[1:3]);ob=m.geom_bodyid[[g,h]]==body
            if ob.sum()!=1:continue
            key=tuple(sorted((g,h)));geometry.add(key)
            if row[4]>=0:
                active.add(key);dim=int(row[3]);count=1 if dim==1 else 2*(dim-1)
                addresses=z['GPU_contact_efc_addresses'][i,:count]
                forces=z['GPU_efc_force_flat'][z['GPU_efc_ranges'][tick]+addresses]
                edge_count+=int(np.count_nonzero(forces>0))
            if row[18]>0:positive.add(key)
            name=m.geom(h if ob[0] else g).name or ''
            if name.startswith('robot/'):
                digit=next((j-1 for j in range(1,6) if name.startswith(f'robot/right_finger{j}_')),5)
                Fn[digit]+=max(0.,row[18])
        rows.append(dict(object_geometry_pairs=sorted(geometry),constraint_pairs=sorted(active),positive_normal_pairs=sorted(positive),
            positive_pyramid_edge_count=edge_count,Fn6=Fn.tolist()))
    return rows


def main(a):
    r=json.loads(a.report.read_text());cases=json.loads(verify_file_pin(r['responses']).read_text())['cases']
    if a.output.exists():raise ValueError('Fresh contact pattern output')
    if len(cases)!=120:raise ValueError('All fixed120 responses')
    patterns={(c['dt'],c['anchor'],c['epsilon'],c['sign'],c['repeat']):pattern(c) for c in cases}
    comparisons=[]
    for dt in [.005,.0025]:
        for anchor in [0,2,4,6]:
            for sign in [1,-1]:
                for repeat in range(3):
                    low,high=[patterns[(dt,anchor,eps,sign,repeat)] for eps in [.002,.005]]
                    item=dict(dt=dt,anchor=anchor,sign=sign,repeat=repeat,ticks=len(low))
                    for key in ['object_geometry_pairs','constraint_pairs','positive_normal_pairs','positive_pyramid_edge_count']:
                        item[key+'_changed_ticks']=sum(p[key]!=q[key] for p,q in zip(low,high))
                    item['Fn6_max_difference_N']=float(abs(np.array([v['Fn6'] for v in low])-np.array([v['Fn6'] for v in high])).max())
                    comparisons.append(item)
    a.output.mkdir();result=dict(schema='same_source_amplitude_contact_patterns:1',run=file_pin(a.report),source=file_pin(__file__),comparisons=comparisons,
        response_patterns=[dict(dt=key[0],anchor=key[1],epsilon=key[2],sign=key[3],repeat=key[4],ticks=value) for key,value in patterns.items()],
        native_integrations=0,neural_updates=0,
        scope='All48 same-sign/repeat amplitude comparisons. Canonical geom-ID sets, EFC-active and strictly positive-normal/edge indicators avoid contact row-order artifacts. Descriptive association with finite-amplitude response changes, NOT proof of sole cause, smoothness or a trained-policy result.')
    write(a.output/'report.json',result)
    print(json.dumps(dict(report=file_pin(a.output/'report.json'),pairs=len(comparisons),
        any_geometry_change=sum(c['object_geometry_pairs_changed_ticks']>0 for c in comparisons),
        any_constraint_pair_change=sum(c['constraint_pairs_changed_ticks']>0 for c in comparisons),
        any_positive_normal_change=sum(c['positive_normal_pairs_changed_ticks']>0 for c in comparisons),
        any_positive_edge_count_change=sum(c['positive_pyramid_edge_count_changed_ticks']>0 for c in comparisons))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
