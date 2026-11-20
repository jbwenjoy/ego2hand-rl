"""Forecast nominal D2/D3 contact onset/release on full candidate census; CPU only."""
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.mechanics.pen_switch_event import (first_event,source_features,metrics,fit,predict,choose_threshold,classify,DT,BINS)
from supermimic_research.mechanics.pen_switch_risk import PenSwitchRisk
from supermimic_research.mechanics.pen_contact_groups import normal_group

TRAIN=('TRAIN_A','TRAIN_B')
TEST=('TEST_RESIDUAL','TEST_SUPPORT')


def role_data(info,expected_names=None):
    model=mujoco.MjModel.from_binary_path(str(verify_file_pin(info['model'])))
    allowed=PenSwitchRisk(model,info['layout']['object_body']).pairs
    keys=[(o,h) for h,o in allowed if normal_group(model.geom(h).name or '') in (1,2)]
    names=[(model.geom(o).name,model.geom(h).name) for o,h in keys]
    if expected_names is not None and names!=expected_names:raise ValueError('Incompatible cross-actor geometry ordering')
    candidates=json.loads(verify_file_pin(info['screen']).read_bytes())['candidates']
    features=[];labels=[];times=[];an=[];at=[];ap=[];phases=[];indices=[];counts=[0,0,0,0]
    for c in candidates:
        kept={(r['object_geom'],r['hand_geom']):r for r in c['pairs']}
        ticks=[set(map(tuple,ids)) for ids in c['nominal_ticks']]
        if len(ticks)!=8 or not set(kept).issubset(keys):raise ValueError('Pair/clock mismatch')
        for j,key in enumerate(keys):
            r=kept.get(key)
            x,cl,ct,cs=source_features(r,j,len(keys))
            present=np.array([key in t for t in ticks]);y,t=first_event(present)
            counts[0]+=int(r is None);counts[1]+=int((r is not None and r['gap_m']<=0)!=present[0])
            counts[2]+=int(np.sum(np.diff(present))>1);counts[3]+=int(r is None and y!=0)
            features.append(x);labels.append(y);times.append(t);an.append(cl);at.append(ct);ap.append(cs)
            phases.append(c['phase']);indices.append(j)
    arrays={k:np.asarray(val) for k,val in dict(x=features,y=labels,t=times,an=an,at=at,ap=ap,phases=phases,indices=indices).items()}
    return names,arrays,dict(controls=len(candidates),pairs=len(keys),rows=len(labels),censored=int(counts[0]),
        native_cpu_source_presence_disagreements=int(counts[1]),multiple_events=int(counts[2]),events_at_censored_rows=int(counts[3]))


def assess(d,prob,times,threshold):
    y=d['y'];pred=classify(prob,threshold)
    f={};a={}
    analytic_rank=d['ap']+(d['an'][:,None]==np.array([1,2]))
    for j,name in enumerate(('onset','release')):
        kind=j+1
        f[name]=metrics(y==kind,pred==kind,prob[:,j],d['t'],times[:,j])
        a[name]=metrics(y==kind,d['an']==kind,d['ap'][:,j],d['t'],d['at'])
        a[name]['hard_then_soft_AP']=metrics(y==kind,d['an']==kind,analytic_rank[:,j],d['t'],d['at'])['average_precision']
    phases=np.unique(d['phases']);mask=np.stack([d['phases']==p for p in phases])
    events=(mask@(y!=0).astype(int))>0
    def source_metrics(cl,ps):
        decision=(mask@(cl!=0).astype(int))>0
        scores=np.max(np.where(mask,ps[None],-1),axis=1)
        res=metrics(events,decision,scores,np.zeros(len(events)),np.zeros(len(events)))
        res.pop('timing_mae_ms');res.pop('timing_matches')
        return res
    f['source_any']=source_metrics(pred,prob.max(1));a['source_any']=source_metrics(d['an'],d['ap'].max(1))
    a['source_any']['hard_then_soft_AP']=source_metrics(d['an'],analytic_rank.max(1))['average_precision']
    for r,cl in [(f,pred),(a,d['an'])]:
        r['typed_tp']=int(((cl==y)&(y!=0)).sum());r['predicted']=int((cl!=0).sum())
        r['typed_micro_f1']=2*r['typed_tp']/max(r['predicted']+(y!=0).sum(),1)
    return dict(events=int((y!=0).sum()),no_event_fraction=float((y==0).mean()),no_event_source_fraction=float((~events).mean()),analytic=a,trained=f)


def main(root,out):
    out.mkdir(exist_ok=False)
    parent=file_pin(root/'report.json');master=json.loads(verify_file_pin(parent).read_bytes());info=master['actors']
    write_json(out/'intent.json',dict(parent=parent,train=list(TRAIN),test=list(TEST),strict_source_inputs='Current CPU signed gap and normal velocity, censored distance flag and pair identity. No future controls, observed tick0, labels, phase index or actual next-tick state supplied.',
       target='Earliest onset/release of pair presence between eight nominal native pre-step geometries. 7 observable gaps, exclude the unobserved final post-step endpoint. Ignore multiplicity-only events.',
       fit='Two regularized logits and per-type ridge event time (TRAIN only). Single global maximum probability threshold via leave-one-training-actor-out typed micro-F1; ties prefer fewer positive rows.',
       unit='One eligible D2/D3 geom pair × candidate phase. Include 30 mm-censored distances as a feature with no velocity; report their event count. No bootstrap claim across correlated phases or actors.'))
    data={};summaries={};names=None
    for role in TRAIN:
        names,data[role],summaries[role]=role_data(info[role],names)
    def join(keys):return {k:np.concatenate([data[r][k] for r in keys]) for k in data[keys[0]]}
    oof=[];oot=[]
    for ia,k in enumerate(TRAIN):
        other=data[TRAIN[1-ia]]
        pr,ts=predict(data[k]['x'],fit(other['x'],other['y'],other['t']))
        oof.append(pr);oot.append(ts)
    joint=join(TRAIN);threshold=choose_threshold(joint['y'],np.concatenate(oof))
    oof_score=assess(joint,np.concatenate(oof),np.concatenate(oot),threshold)
    weights=fit(joint['x'],joint['y'],joint['t'])
    path=out/'model.npz';np.savez_compressed(path,logits=weights[0],timing=weights[1],threshold=threshold)
    sealed=file_pin(path);report={}
    for role in TEST:
        _,d,summaries[role]=role_data(info[role],names)
        pred,t=predict(d['x'],weights)
        report[role]=assess(d,pred,t,threshold)
        np.savez_compressed(out/(role+'.npz'),phase=d['phases'],pair=d['indices'],type=d['y'],first_tick=d['t'],prob=pred,predicted_tick=t,
                            decision=classify(pred,threshold),analytic=d['an'],analytic_time=d['at'],analytic_scores=d['ap'])
    verify_file_pin(sealed)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),code=[file_pin(__file__),file_pin(ROOT/'src/supermimic_research/mechanics/pen_switch_event.py')],
        model=sealed,threshold=threshold,pair_names=names,counts=summaries,oof=oof_score,test=report,results={k:file_pin(out/(k+'.npz')) for k in TEST}))
    for k,v in report.items():
        print(k,summaries[k],v['events'],'analytic',v['analytic']['typed_tp'],v['analytic']['predicted'],'learned',v['trained']['typed_tp'],v['trained']['predicted'])
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
