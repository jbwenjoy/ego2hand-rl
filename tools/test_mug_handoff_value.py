"""Closed-loop evaluation of learned handoff returns and exact truncated policy improvement."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
from mujoco.rollout import Rollout
import numpy as np
import torch
from collect_mug_handoff_labels import configuration,h3
from fit_mug_handoff_value import choice,HandoffValue


def trial(P,head,seed,mode):
    d=P.initial();fork=mujoco.MjData(P.m);rows=[];cmds=[];qpos=[];wall=0.;extra_ticks=0
    with Rollout(nthread=4) as roll:
        for i in range(len(P.ik)):
            clock=perf_counter();o=P.options(d,i,seed,roll,include_features=True)
            base=int(o['order'][0]);selected=base;labels={}
            if mode=='exact6':
                horizon=min(6,len(P.ik)-i)
                best=-1
                for idx in o['order']:
                    k=int(idx);count=0;mujoco.mj_copyData(fork,P.m,d)
                    for h in range(horizon):
                        if h==0:
                            cmd=o['cmd'][k];end=o['end'][k]
                        else:
                            child=P.options(fork,i+h,seed,roll);pick=child['order'][0]
                            cmd=child['cmd'][pick];end=child['end'][pick];extra_ticks+=8*64
                        ok,_=P.advance(fork,i+h,cmd,end);extra_ticks+=64
                        if not ok:break
                        count+=1
                    labels[k]=count
                    if count>best:best=count;selected=k
                    if count==horizon:break
            elif mode!='baseline':
                with torch.inference_mode():y=head(torch.from_numpy(o['x'])).numpy()
                selected=choice(y,o['gate'],o['order'],1. if mode=='margin' else 0.)
            wall+=perf_counter()-clock
            cmd=o['cmd'][selected];ok,metrics=P.advance(d,i,cmd,o['end'][selected])
            cmds.append(cmd.copy());qpos.append(d.qpos.copy())
            rows.append(dict(phase=i+1,baseline=base,pick=selected,changed=not np.array_equal(cmd,o['cmd'][base]),
                tested=labels,passed=bool(ok),mm=metrics[0],deg=metrics[1],table_N=metrics[2]))
            if not ok:break
    z={**P.old,'COM_world_m':P.pos,'quaternion_wxyz':P.rot.as_quat()[:,[3,0,1,2]],'elapsed_s':.02*np.arange(len(P.ik)+1)}
    detail,tr=h3.physics.replay(P.path,P.binding,z,P.base,np.asarray(cmds),1.,1.)
    np.testing.assert_array_equal(qpos,tr['qpos'])
    return dict(seed=seed,mode=mode,controls=len(rows),valid_s=detail['valid_s'],full=len(rows)==len(P.ik) and detail['right_censored'],
        first_failure=detail['first_breach'],extra_native_ticks=extra_ticks,primary_planning_ticks=512*len(rows),
        decision_wall_s=wall,rows=rows),tr


def run(out):
    source=h3.file_pin(Path(__file__).resolve().parents[1]/'runs/mug_handoff_value_v1/report.json')
    fit=h3.physics.read_json(source);frozen=fit['checkpoint']
    ck=torch.load(h3.verify_file_pin(frozen),weights_only=True,map_location='cpu')
    head=HandoffValue(ck['width']);head.load_state_dict(ck['model']);head.eval()
    _,_,planners=configuration()
    seeds=tuple(range(8441,8449));out.mkdir(exist_ok=False,parents=True)
    h3.write_json(out/'intent.json',dict(source=source,model=frozen,code=h3.file_pin(__file__),seeds=seeds,
        scope='Eight fresh proposal seeds, two known human references. Baseline H3, learned unconditionally highest value or require predicted gain >=1 control, and native six-control rollout oracle followed by H3 inside each fork. All check the same eight first commands. Exact6 uses extra native steps, ties prefer baseline order; cease when the horizon is attained. No optimizer steps. Primary learned test: median paired gain >=1 passing control on both clips.'))
    rows=[]
    for clip,P in planners.items():
        for seed in seeds:
            for mode in ('baseline','always','margin','exact6'):
                result,tr=trial(P,head,seed,mode)
                file=out/f'{clip}_{seed}_{mode}.npz';np.savez_compressed(file,**tr)
                rows.append(dict(clip=clip,trace=h3.file_pin(file),**result))
            h3.write_json(out/'progress.json',rows)
            print(clip,seed,[r['valid_s'] for r in rows[-4:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
