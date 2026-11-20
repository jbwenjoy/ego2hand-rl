"""Offline supervised load forecast; parse splits by policy, not by neighboring rows."""
import json,sys,copy
from pathlib import Path
import numpy as np
import torch
from check_pen_wrench_oracle import rigid
from check_pen_momentum_closure import state_rotation,PHASES
from scipy.spatial.transform import Rotation
from supermimic_research.mechanics.pen_load_forecast import LoadForecast,examples,features,rotate_load
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def batch(eps,hist):
    data=[examples(x,y,v,hist) for x,y,v,_ in eps]
    return [torch.from_numpy(np.concatenate(col)).float() for col in zip(*data)]


def scores(err):
    return {name:float(err[:,lo:hi].square().sum(1).mean().sqrt()) for name,lo,hi in [('force_N',0,3),('torque_Nm',3,6),('normal_N',6,12)]}


@torch.no_grad()
def rollout(net,episode,hist):
    x,y,v,ph=episode
    results=[]
    for phase in PHASES:
        start=phase-1
        if start+16>=len(x) or not all(v[start-1:start+17]):continue
        s=x[start,:55].copy()
        h=rotate_load(y[start-1,78:90],state_rotation(s).T@state_rotation(x[start-1,:55]))
        for k in range(16):
            pd=x[start+k,81:101]
            estimate=net(torch.from_numpy(features(s,pd,h if hist else None)).float()).numpy().astype(float)
            R=state_rotation(s)
            new_obj=rigid(s,R@estimate[:3],R@estimate[3:6],ph['mass'],np.array(ph['inertia_BODY']),np.array(ph['gravity']))
            nxt=np.r_[s[:40]+estimate[12:52],new_obj]
            h=rotate_load(estimate[:12],state_rotation(nxt).T@R)
            s=nxt
            if k+1 in (1,2,4,8,16):
                truth=x[start+k+1,:55]
                results.append(dict(phase=phase,ms=20*(k+1),mm=float(1000*np.linalg.norm(s[40:43]-truth[40:43])),
                    deg=float(np.rad2deg(Rotation.from_matrix(state_rotation(s)@state_rotation(truth).T).magnitude()))))
    return results


def train(root,out):
    out.mkdir(exist_ok=False)
    master=json.loads((root/'report.json').read_bytes())
    write_json(out/'intent.json',dict(source=file_pin(root/'report.json'),modes=['state','history'],seeds=[7411],max_epochs=60,batch_size=128,lr=.001,
        epoch_selection='lowest VAL mean standardized squared error over 12 contact outputs; identical protocol both modes',
        heads='Source BODY moving COM hand/object net load and six normals; auxiliary 40 hand-coordinate increments. One complete held-out continuous rollout per test Actor.'))
    datasets={}
    for ep in master['episodes']:
        with np.load(verify_file_pin(ep['trajectory'])) as f:x,y,v=(f[n] for n in ['x','y','valid'])
        physics=json.loads(verify_file_pin(ep['audit']).read_bytes())['physics']
        datasets.setdefault(ep['label'],[]).append((x,y,v,physics))
    torch.set_num_threads(2);outcomes={}
    for hist in (False,True):
        torch.manual_seed(7411)
        x,y=batch(sum([datasets[label] for label in 'ABCD'],[]),hist)
        vx,vy=batch(datasets['VAL'],hist)
        net=LoadForecast(x,y);opt=torch.optim.Adam(net.parameters(),lr=.001);best=float('inf');losses=[]
        for epoch in range(1,61):
            for ids in torch.randperm(len(x)).split(128):
                pred=net(x[ids]);e=((pred-y[ids])/net.out_scale).square()
                loss=e[:,:12].mean()+e[:,12:].mean()
                opt.zero_grad();loss.backward();opt.step()
            with torch.no_grad():value=float((((net(vx)-vy)/net.out_scale)[:,:12]**2).mean())
            losses.append(value)
            if value<best:best=value;chosen=epoch;state=copy.deepcopy(net.state_dict())
        net.load_state_dict(state);name='history' if hist else 'state'
        path=out/(name+'.pt');torch.save(state,path)
        tests={}
        with torch.no_grad():
            for label in ['TEST1','TEST2','TEST3']:
                xx,yy=batch(datasets[label],hist)
                hx,_=batch(datasets[label],True)
                records=rollout(net,datasets[label][0],hist)
                summary={str(ms):np.sqrt(np.mean(np.square([[r['mm'],r['deg']] for r in records if r['ms']==ms]),axis=0)).tolist() for ms in [20,40,80,160,320]}
                tests[label]=dict(controls=len(xx),errors=scores(net(xx)-yy),last_load_errors=scores(hx[:,-12:]-yy[:,:12]),open_loop_rms_mm_deg=summary,records=records)
        outcomes[name]=dict(model=file_pin(path),train_controls=len(x),val_controls=len(vx),chosen_epoch=chosen,val_curve=losses,tests=tests)
        print(name,chosen,{k:(v['errors'],v['open_loop_rms_mm_deg']['80']) for k,v in tests.items()},flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),outcomes=outcomes))
if __name__=='__main__':train(*map(Path,sys.argv[1:]))
