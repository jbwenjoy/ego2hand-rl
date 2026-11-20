"""Fit local contact-transition Jacobians on admitted TRAIN labels only."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

HEADS={'rotation_rad':(43,46),'force_N':(78,81),'torque_Nm':(81,84),'digit_normal_N':(84,90)}


def groups(report):
    g={}
    for row in report['rows']:
        with np.load(verify_file_pin(row['packet'])) as p:
            packet=dict(x=p['x'][0],y=p['y'][0],pd=p['pd'],ok=row['valid'] and not row['terminal'])
        g.setdefault(row['phase'],{})[row['name']]=packet
    return g


def fit(data,selected):
    anchors=[]
    for phase,rows in data.items():
        zero=rows['sham']
        if not zero['ok']:continue
        dx=[];dy=[]
        for j in selected:
            m,p=(rows[f'j{j}_{side}'] for side in ('minus','plus'))
            if not (m['ok'] and p['ok']):continue
            dx.append((p['pd']-m['pd'])[selected]);dy.append(p['y']-m['y'])
        if len(dx)<8 or np.linalg.matrix_rank(dx,tol=1e-6)<8:continue
        jac=np.linalg.lstsq(dx,dy,rcond=1e-6)[0]
        anchors.append((phase,zero['x'][:55],zero['pd'][selected],zero['y'],jac))
    return anchors


def evaluate(data,anchors,selected):
    state=np.stack([a[1] for a in anchors])
    scale=np.maximum(state.std(0),1e-3)
    rows=[]
    for phase,group in data.items():
        m,p=group['mixed_minus'],group['mixed_plus']
        if not (m['ok'] and p['ok']):continue
        idx=np.argmin(np.mean(((state-m['x'][:55])/scale)**2,axis=1))
        train_phase,_,base_pd,base_y,jac=anchors[idx]
        pred=lambda packet:base_y+(packet['pd'][selected]-base_pd)@jac
        yp,ym=pred(p),pred(m)
        pair={}
        for name,(lo,hi) in HEADS.items():
            s=slice(lo,hi); diff=(p['y']-m['y'])[s]; estimate=(yp-ym)[s]
            sham=group['sham'];again=group['sham_repeat']
            noise=float(np.linalg.norm((again['y']-sham['y'])[s])) if sham['ok'] and again['ok'] else None
            pair[name]=dict(observed_delta=float(np.linalg.norm(diff)),error=float(np.linalg.norm(estimate-diff)),
                signed_dot=float(estimate@diff),sham_difference=noise,
                absolute_prediction_rms=float(np.sqrt(np.mean(np.r_[yp[s]-p['y'][s],ym[s]-m['y'][s]]**2))))
        rows.append(dict(phase=phase,anchor=int(train_phase),heads=pair))
    summary={}
    for name in HEADS:
        h=[r['heads'][name] for r in rows];signal=sum(i['observed_delta']**2 for i in h)
        summary[name]=dict(pairs=len(h),positive_dot=sum(i['signed_dot']>0 for i in h),
            pooled_error_over_zero_response=float(np.sqrt(sum(i['error']**2 for i in h)/signal)) if signal>0 else None,
            exceeds_sham=sum(i['sham_difference'] is not None and i['observed_delta']>i['sham_difference'] for i in h))
    return dict(rows=rows,summary=summary)


def main(root,out):
    report=json.loads((root/'report.json').read_bytes())
    if report['native']!=6800:raise ValueError('Incomplete collector')
    arms={key:json.loads(verify_file_pin(pin).read_bytes()) for key,pin in report['arms'].items()}
    selected=arms['TRAIN']['selected'];assert selected==arms['HOLDOUT']['selected']
    training=groups(arms['TRAIN'])
    anchors=fit(training,selected)
    out.mkdir(exist_ok=False)
    result=dict(source=file_pin(root/'report.json'),training_phases=[a[0] for a in anchors],
        valid_counts={k:sum(r['valid'] for r in v['rows']) for k,v in arms.items()},
        domain='One 20ms transition, measured PD Jacobian in eight loaded-finger coordinates; mixed direction excluded from fit.')
    if anchors:
        np.savez_compressed(out/'model.npz',phases=np.array([a[0] for a in anchors]),states=np.stack([a[1] for a in anchors]),
            pd=np.stack([a[2] for a in anchors]),intercepts=np.stack([a[3] for a in anchors]),jacobians=np.stack([a[4] for a in anchors]),selected=selected)
        result['model']=file_pin(out/'model.npz')
        result['train_ranks']={str(phase):int(np.linalg.matrix_rank(np.stack([(r[f'j{j}_plus']['pd']-r[f'j{j}_minus']['pd'])[selected] for j in selected]),tol=1e-6)) for phase,r in training.items()}
        result['training_mixed']=evaluate(training,anchors,selected)
        result['same_source_mixed']=evaluate({a[0]:training[a[0]] for a in anchors},anchors,selected)
        result['holdout_mixed']=evaluate(groups(arms['HOLDOUT']),anchors,selected)
    write_json(out/'report.json',result)
    print(json.dumps({k:v.get('summary') if isinstance(v,dict) and 'rows' in v else v for k,v in result.items()}))
if __name__=='__main__':main(*map(Path,sys.argv[1:]))

