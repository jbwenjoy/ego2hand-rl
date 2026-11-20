"""Offline bounded segment shooting, with from-start native validation of accepted paths."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
from collect_mug_handoff_labels import configuration,h3


def measure(P,cmds):
    d=P.initial();qs=[];warm=[];states=[];metrics=[];valid=0
    for i,cmd in enumerate(cmds):
        states.append(P.state(d));warm.append(d.qacc_warmstart.copy())
        ok,info=P.advance(d,i,cmd)
        qs.append(d.qpos.copy());metrics.append(info)
        if not ok:break
        valid+=1
    return dict(qpos=np.array(qs),warm=np.array(warm),state=np.array(states),metrics=np.array(metrics),valid=valid)


def propose(P,cmds,phase,horizon,iteration,seed,count):
    rng=np.random.default_rng(np.random.SeedSequence([seed,iteration]))
    baseline=cmds[phase:phase+horizon]
    grid=np.broadcast_to(baseline,(count,horizon,20)).copy()
    grid[1]=P.ik[phase:phase+horizon];grid[2]=P.base
    for j in range(3,count):
        knots=1+j%3
        noise=rng.normal(size=(knots,20))
        # Include coordinated motion of finger joints as well as independent directions.
        if j%4==0:
            noise=(noise.reshape(knots,5,4)*.3+rng.normal(size=(knots,5,1))).reshape(knots,20)
        norm=np.sqrt((noise**2).mean(1,keepdims=True));noise/=np.maximum(norm,1e-12)
        scale=(.025,.05,.09,.15,.22)[(j//3+iteration)%5]
        if knots==1:delta=noise
        else:
            lin=np.linspace(0,knots-1,horizon);lo=np.minimum(lin.astype(int),knots-2)
            frac=(lin-lo)[:,None];delta=noise[lo]*(1-frac)+noise[lo+1]*frac
        grid[j]+=scale*delta
    return np.clip(grid,*P.m.actuator_ctrlrange.T)


def case(P,prior,seed,folder,iterations):
    arc=h3.physics.load_arrays(prior['trace']);commands=P.ik.copy()
    commands[:len(arc['ctrl'])]=arc['ctrl']
    result=measure(P,commands)
    np.testing.assert_array_equal(result['qpos'],arc['qpos'])
    original=result['valid'];history=[];ticks=0;replays=0
    start=perf_counter()
    with Rollout(nthread=4) as engine:
        for i in range(iterations):
            if result['valid']==len(commands):break
            # Search from several distances before failure. The actual initial
            # pose and the entire committed command history before the anchor are fixed.
            before=(1,3,6,10,4,2,8,16)[i%8]
            phase=max(0,result['valid']-before)
            horizon=min(len(commands)-phase,result['valid']-phase+6)
            grid=propose(P,commands,phase,horizon,i,seed,48)
            pre=result['state'][phase];warm=result['warm'][phase]
            st,_=engine.rollout(P.m,P.workers,np.tile(pre,(len(grid),1)),initial_warmstart=np.tile(warm,(len(grid),1)),
                control=np.repeat(grid,64,axis=1))
            ticks+=len(grid)*horizon*64
            end=st[:,63::64].reshape(-1,st.shape[-1])
            pos,rot,_,_=P.chart.motion(end)
            mm=1000*np.linalg.norm(pos.reshape(len(grid),horizon,3)-P.pos[None,phase+1:phase+1+horizon],axis=2)
            rq=np.tile(P.rot[phase+1:phase+1+horizon].as_quat(),(len(grid),1))
            deg=np.rad2deg((rot*Rotation.from_quat(rq).inv()).magnitude()).reshape(len(grid),horizon)
            risk=np.maximum(mm,deg)/10
            ok=(risk<=1)&np.isfinite(risk)
            passing=np.cumprod(ok,axis=1).sum(1)
            # Prefer strict prefix length, then a smaller distance from the
            # gate around the frontier; use all candidates including the incumbent.
            last=np.minimum(passing,horizon-1)
            near=np.mean(np.minimum(risk,4.)**2,axis=1)
            score=risk[np.arange(len(grid)),last]+.35*near
            order=np.lexsort((np.arange(len(grid)),score,-passing))
            candidate=0;old=result['valid']
            for j in order:
                if j==0:break
                if passing[j]<passing[0]:break
                trial=commands.copy();trial[phase:phase+horizon]=grid[j]
                observed=measure(P,trial);replays+=len(observed['qpos'])
                if observed['valid']<old:continue
                n=min(horizon,len(observed['qpos'])-phase)
                np.testing.assert_array_equal(observed['qpos'][phase:phase+n],st[j,63::64,1:1+P.m.nq][:n])
                # Reject a nominally good pose bank member if the table gate
                # broke earlier than the control rated by the endpoint model.
                if observed['valid']-phase<min(int(passing[j]),horizon):continue
                commands=trial;result=observed;candidate=int(j);break
            history.append(dict(iteration=i+1,phase=phase+1,horizon=horizon,candidate=candidate,
                before=old,after=result['valid'],batch_best=int(passing.max()),elapsed=perf_counter()-start))
            if (i+1)%10==0 or result['valid']==len(commands):
                ck=folder/f'iteration_{i+1:03}.npz'
                np.savez_compressed(ck,commands=commands,qpos=result['qpos'],valid=result['valid'],iteration=i+1,seed=seed)
                # Deterministic iteration seeds replace mutable RNG state.
                with np.load(ck) as loaded:check=measure(P,loaded['commands'])
                np.testing.assert_array_equal(check['qpos'],result['qpos']);assert check['valid']==result['valid']
                elapsed=perf_counter()-start
                h3.write_json(folder/'progress.json',dict(iteration=i+1,budget=iterations,passing=result['valid'],
                    resumed_prefix_exact=True,checkpoint=h3.file_pin(ck),seconds=elapsed,remaining_estimate_s=elapsed*(iterations-i-1)/(i+1)))
    best=folder/'best.npz';np.savez_compressed(best,commands=commands,qpos=result['qpos'],metrics=result['metrics'])
    z={**P.old,'COM_world_m':P.pos,'quaternion_wxyz':P.rot.as_quat()[:,[3,0,1,2]],'elapsed_s':.02*np.arange(len(commands)+1)}
    detail,tr=h3.physics.replay(P.path,P.binding,z,P.base,commands,1.,1.)
    np.testing.assert_array_equal(tr['qpos'][:len(result['qpos'])],result['qpos'])
    assert np.isclose(detail['valid_s'],result['valid']*.02)
    trace=folder/'replay.npz';np.savez_compressed(trace,**tr)
    return dict(clip=P.clip,seed=seed,original=original,valid=result['valid'],requested=len(commands),
        full=result['valid']==len(commands),rounds=len(history),native_search_ticks=ticks,validation_ticks_excluding_checkpoints=64*replays,
        wall_s=perf_counter()-start,history=history,best=h3.file_pin(best),trace=h3.file_pin(trace),replay=detail)


def run(out,iterations):
    _,_,planners=configuration();root=Path(__file__).resolve().parents[1]/'runs'
    parent=h3.file_pin(root/'mug_handoff_decisions_v1/report.json');cases=h3.physics.read_json(parent)['rows']
    out.mkdir(exist_ok=False,parents=True)
    h3.write_json(out/'intent.json',dict(source=parent,code=h3.file_pin(__file__),seeds=[8451,8452],iterations=iterations,batch=48,
        scope='Offline per-reference path optimization, seeded by the first new baseline rollout (8441) only. Remaining commands start at direct IK+preload. Move the optimization anchor around the first failed control. Keep global valid duration monotone, replay accepted segments and checkpoint prefixes from start. No time warp, changed initial grasp, force assistance or shared policy claim.'))
    rows=[]
    for clip,P in planners.items():
        prior=next(r for r in cases if r['clip']==clip and r['seed']==8441 and r['mode']=='baseline')
        for seed in (8451,8452):
            folder=out/f'{clip}_{seed}';folder.mkdir()
            r=case(P,prior,seed,folder,iterations);rows.append(r)
            h3.write_json(out/'progress.json',rows)
            print(clip,seed,r['original'],r['valid'],r['requested'],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--iterations',type=int,default=80)
    a=p.parse_args();run(a.out.resolve(),a.iterations)
