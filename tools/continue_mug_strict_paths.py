"""Continue a geometry-qualified mug path search with replayable complete command checkpoints."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
from repair_mug_path_geometry import StrictCheck,LIMITS,configuration,propose,h3


def run(source,out,iterations):
    pin=h3.file_pin(source/'report.json');prior=h3.physics.read_json(pin);clip=prior['clip']
    _,_,planners=configuration();P=planners[clip];audit=StrictCheck(P)
    cmds=h3.physics.load_arrays(prior['trace'])['ctrl'];cur=audit.measure(cmds)
    assert cur['valid']==prior['strict_controls']
    out.mkdir(exist_ok=False,parents=True)
    seed=8462
    h3.write_json(out/'intent.json',dict(parent=pin,code=h3.file_pin(__file__),base_code=h3.file_pin(Path(__file__).with_name('repair_mug_path_geometry.py')),
        clip=clip,seed=seed,budget=iterations,limits=LIMITS.tolist(),
        scope='Start at the complete command vector of the 32-round geometry repair. Extend monotonically by strict prefix length. Vary anchor distance and perturbation width. From-start exact replay at checkpoints and final, preserve prior runs.'))
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS;d=mujoco.MjData(P.m)
    start=perf_counter();hist=[];ticks=0;last_gain=0
    for i in range(iterations):
        if cur['valid']==len(cmds):break
        gap=(2,5,8,1,12,4,3,16,6,10)[i%10]
        if i-last_gain>=24 and i%3==0:gap+=8
        phase=max(0,cur['valid']-gap);h=min(len(cmds)-phase,cur['valid']-phase+5)
        grid=propose(P,cmds,phase,h,i,seed,32)
        # Include straight-line excursions toward and away from the previously
        # sent servo targets; both retain legal absolute actuator bounds.
        last=cmds[max(phase-1,0)] if phase else P.base
        beta=np.linspace(.8,0,h)[:,None]
        grid[-2]=np.clip((1-beta)*cmds[phase:phase+h]+beta*last,*P.m.actuator_ctrlrange.T)
        grid[-1]=np.clip(cmds[phase:phase+h]+.25*(cmds[phase:phase+h]-last),*P.m.actuator_ctrlrange.T)
        best=None;old=cur['valid']
        for j,seq in enumerate(grid):
            mujoco.mj_resetData(P.m,d);mujoco.mj_setState(P.m,d,cur['state'][phase],sig);d.qacc_warmstart[:]=cur['warm'][phase]
            worst=[];pose=[]
            for k,u in enumerate(seq):
                metric=audit.step(d,phase+k,u)/LIMITS;ticks+=64
                worst.append(float(metric.max()));pose.append(float(metric[:2]@metric[:2]))
                if worst[-1]>1:break
            count=len(worst)-int(worst[-1]>1)
            key=(-count,worst[-1],np.mean(pose),j)
            if best is None or key<best[0]:best=(key,j)
        selected=best[1]
        if selected:
            trial=cmds.copy();trial[phase:phase+h]=grid[selected];checked=audit.measure(trial)
            if checked['valid']>=old:cmds=trial;cur=checked
            else:selected=0
        if old<cur['valid']:last_gain=i
        hist.append(dict(iteration=i+1,passing=cur['valid'],anchor=phase+1,choice=selected))
        if (i+1)%10==0 or cur['valid']==len(cmds):
            p=out/f'iteration_{i+1:03}.npz'
            np.savez_compressed(p,commands=cmds,passing=cur['valid'],iteration=i+1,seed=seed,last_gain=last_gain)
            with np.load(p) as f:verified=audit.measure(f['commands'])
            np.testing.assert_array_equal(verified['qpos'],cur['qpos'])
            elapsed=perf_counter()-start
            h3.write_json(out/'progress.json',dict(checkpoint=h3.file_pin(p),replayed_exact=True,iteration=i+1,
                passing=cur['valid'],budget=iterations,elapsed_s=elapsed,remaining_estimate_s=elapsed*(iterations-i-1)/(i+1)))
    z={**P.old,'elapsed_s':.02*np.arange(len(cmds)+1),'COM_world_m':P.pos,'quaternion_wxyz':P.rot.as_quat()[:,[3,0,1,2]]}
    detail,tr=h3.physics.replay(P.path,P.binding,z,P.base,cmds,1.,1.)
    np.testing.assert_array_equal(cur['qpos'],tr['qpos'][:len(cur['qpos'])])
    p=out/'trace.npz';np.savez_compressed(p,**tr,strict_prefix_metrics=cur['metrics'])
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),clip=clip,initial_strict=prior['strict_controls'],
        strict_controls=cur['valid'],requested=len(cmds),full_strict=cur['valid']==len(cmds),pose_valid_s=detail['valid_s'],
        final_metrics=cur['metrics'][-1].tolist(),history=hist,seconds=perf_counter()-start,ticks=ticks,trace=h3.file_pin(p)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--iterations',type=int,default=120)
    a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.iterations)
