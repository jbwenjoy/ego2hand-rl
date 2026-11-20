"""Separate diagnostic: fit per-source pose responses directly to independent native constraints."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from fit_pen_substep_poses import (ROOT,READ,inputs_and_labels,contrast,pose_errors,
                                   load_substep_pose,file_pin,verify_file_pin,write_json)
from supermimic_research.mechanics.pen_substep_secant import LocalSecantPose,load_local_secant,SCHEMA


def main(bank,prior,out):
    out.mkdir(exist_ok=False)
    parent=file_pin(bank/'report.json');data=READ(parent);p=READ(file_pin(prior/'report.json'))
    base_pin=p['models']['absolute']['checkpoint']
    write_json(out/'intent.json',dict(bank=parent,absolute=base_pin,train='Valid TRIAD sham and signed same-state TRAIN_A/B basis endpoints only',
        scope='Post-hoc architecture diagnostic on previously inspected cohorts. No TEST-derived thresholds or fitter. Only fit full-rank measured-PD Jacobians, interpolate linearly relative to nearest source servo offsets under train-standardized state55 distance. No clipping of extrapolation.'))
    absolute=load_substep_pose(torch.load(verify_file_pin(base_pin),map_location='cpu',weights_only=True),dtype=torch.float64)
    states,offsets,slopes,owners=[],[],[],[];chosen=data['roles']['TRAIN_A']['selected']
    for role in ['TRAIN_A','TRAIN_B']:
        arm=data['roles'][role];assert arm['selected']==chosen
        by_phase={}
        for row in arm['rows']:
            if not row['valid'] or row['terminal'] or row['name'].startswith('mixed'):continue
            x,y,_=inputs_and_labels(row)
            by_phase.setdefault(row['phase'],{})[row['name']]=(x,y)
        for phase,rows in by_phase.items():
            if 'sham' not in rows:continue
            x0,_=rows['sham'];G=[];S=[]
            for j in chosen:
                if not all(f'j{j}_{s}' in rows for s in ('minus','plus')):continue
                xm,ym=rows[f'j{j}_minus'];xp,yp=rows[f'j{j}_plus']
                np.testing.assert_array_equal(xm[:81],x0[:81]);np.testing.assert_array_equal(xp[:81],x0[:81])
                G.append((xp[81:101]-xm[81:101])[chosen]);S.append((yp-ym).flatten())
            if len(G)!=8 or np.linalg.matrix_rank(G,tol=1e-6)!=8:continue
            J=np.linalg.solve(G,S).reshape(8,7,26)
            states.append(x0[:55]);offsets.append((x0[81:101]-x0[:20])[chosen]);slopes.append(J);owners.append((role,phase))
    T=lambda x:torch.as_tensor(np.asarray(x),dtype=torch.float64)
    model=LocalSecantPose(absolute,T(states),T(offsets),T(slopes),torch.tensor(chosen))
    target=out/'local_secant.pt';torch.save(dict(schema=SCHEMA,state_dict=model.state_dict()),target)
    model=load_local_secant(torch.load(target,map_location='cpu',weights_only=True))
    results={}
    for role in ['TRAIN_A','TRAIN_B','TEST']:
        rows=data['roles'][role]['rows']
        items=[inputs_and_labels(row) for row in rows]
        x=T([z[0] for z in items]);valid=np.asarray([r['valid'] and not r['terminal'] for r in rows]);truth=np.array([z[2] for z in items])
        pred=model.predict_qpos(x[:,:55],x[:,55:81],x[:,81:101]).numpy();a=model.anchor_index(x[:,:55]).numpy()
        results[role]=dict(absolute=pose_errors(pred[valid],truth[valid]),contrasts=contrast(pred,truth,rows,chosen),anchors_used={str(owners[i]):int((a==i).sum()) for i in np.unique(a)})
        if role=='TEST':
            np.savez_compressed(out/'predictions.npz',phase=[r['phase'] for r in rows],name=[r['name'] for r in rows],valid=valid,local_secant=pred)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),checkpoint=file_pin(target),predictions=file_pin(out/'predictions.npz'),training_anchors=owners,results=results))
    print('anchors', len(owners), 'TEST RMS', {k:v['t1_to_t7_rms'] for k,v in results['TEST']['absolute'].items()},'mixed',results['TEST']['contrasts']['mixed'])
if __name__=='__main__':main(*map(Path,sys.argv[1:]))
