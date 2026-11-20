"""Score the existing proposal bank with learned substep poses and compiled collision geometry."""
import json,sys
from pathlib import Path
import mujoco
import numpy as np
import torch
from check_pen_capacity_geometry_oracle import geometry,balanced_directions
from check_pen_native_load_suggestions import native_state55
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def read(pin):
    return json.loads(verify_file_pin(pin).read_bytes())


def main(fit,source,out):
    from supermimic_research.mechanics.pen_substep_pose import load_substep_pose
    out.mkdir(exist_ok=False);torch.set_num_threads(2)
    train=read(file_pin(fit/'report.json'));bank=read(file_pin(source/'report.json'))
    models={}
    for k in ('absolute','paired'):
        pin=train['models'][k]['checkpoint']
        models[k]=load_substep_pose(torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)).eval()
    target_pin=read(bank['intent'])['targets']
    with np.load(verify_file_pin(target_pin)) as f:target=f['capacity'];dirs=f['directions']
    np.testing.assert_array_equal(dirs,balanced_directions())
    write_json(out/'intent.json',dict(fit=file_pin(fit/'report.json'),bank=file_pin(source/'report.json'),scope='Held-out earlier executed candidates; choose among all declared actions using only the common start and proposed joint targets. No native endpoint observations in selection. W1 CPU collision queries on predicted poses, zero native integration.'))
    results={}
    for arm,detail in bank['results'].items():
        source_cases=read(detail['report']);decl=read(source_cases['proposals']);folder=Path(detail['report']['path']).parent
        m=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));body=m.body('object/cube').id
        qa=m.jnt_qposadr[m.body_jntadr[body]];da=m.jnt_dofadr[m.body_jntadr[body]]
        qids=m.jnt_qposadr[m.actuator_trnid[:,0]];dofs=m.jnt_dofadr[m.actuator_trnid[:,0]]
        arms={k:[] for k in models}
        for item in decl['entries']:
            phase=item['phase'];trials=[r for r in source_cases['observations'] if r['phase']==phase]
            with np.load(verify_file_pin(next(r for r in trials if r['slot']==0)['trace'])) as z:
                state=native_state55(z['pre_qpos'][0],z['pre_qvel'][0],qids,dofs,qa,da)
                warm=z['initial_warmstart'];d=mujoco.MjData(m)
                d.mocap_pos[:]=z['post_mocap_pos'][0];d.mocap_quat[:]=z['post_mocap_quat'][0]
            for k,net in models.items():
                scores={}
                dtype=next(net.parameters()).dtype
                for n,action in item['actions'].items():
                    with torch.no_grad():
                        p=net.predict_qpos(*(torch.as_tensor(v,dtype=dtype) for v in [state,warm,np.asarray(action['pd'])])).cpu().numpy()
                    assert p.shape==(8,27) and np.isfinite(p).all()
                    h=np.mean([geometry(m,d,q,body,dirs) for q in p],axis=0)
                    scores[n]=float(np.abs(h-target[phase-1]).mean())
                choice=min(scores,key=scores.get)
                pairs={n:np.array([e['mismatch'] for e in trials if e['name']==n]) for n in scores}
                complete=all(len(vals)==2 for vals in pairs.values()) and all(e['valid'] and not e['terminal'] for e in trials)
                b=pairs['BASE'];fitness=pairs[choice]
                opposite=(choice[:-4]+'up' if choice.endswith('down') else choice[:-2]+'down') if choice!='BASE' else 'BASE'
                win=bool(complete and fitness.max()+np.ptp(b)<min(b.min(),pairs[opposite].min()))
                arms[k].append(dict(phase=phase,scores=scores,choice=choice,valid=complete,strict_win=win,gain=float(b.mean()-fitness.mean()),worse=bool(complete and fitness.min()-np.ptp(b)>b.max())))
        result={}
        for k,rows in arms.items():
            valid=[r for r in rows if r['valid']]
            result[k]=dict(rows=rows,total=len(valid),wins=sum(r['strict_win'] for r in valid),worse=sum(r['worse'] for r in valid),
                           base=sum(r['choice']=='BASE' for r in valid),median_gain=float(np.median([r['gain'] for r in valid])) if valid else None)
        results[arm]=result
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),results=results))
    print({k:{j:(v['wins'],v['total'],v['median_gain']) for j,v in arm.items()} for k,arm in results.items()})
if __name__=='__main__':main(*map(lambda s:Path(s).resolve(),sys.argv[1:]))
