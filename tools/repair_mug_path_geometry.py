"""Repair archived mug commands against per-substep penetration and joint-limit gates."""
import argparse
from pathlib import Path
from time import perf_counter
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from optimize_hot3d_mug_sequence import propose,configuration,h3


LIMITS=np.array([10.,10.,.1,2.,.2,.001])


class StrictCheck:
    def __init__(self,P):
        self.P=P;self.m=P.m;self.mug=P.mug
        self.hand=np.isin(np.arange(P.m.ngeom),[P.m.geom(n).id for n in P.binding['hand_collision_names']])
        j=P.m.actuator_trnid[:,0];self.qids=P.m.jnt_qposadr[j]
        self.lo,self.hi=P.m.jnt_range[j].T
        assert np.all(P.m.jnt_limited[j])
        self.force=np.empty(6)

    def step(self,d,i,command):
        m=self.m;P=self.P;d.ctrl[:]=command;peaks=np.zeros(4)
        assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
        for _ in range(64):
            mujoco.mj_step(m,d);a,b=d.contact.geom.T
            depth=-1000*d.contact.dist
            hm=(self.hand[a]&self.mug[b])|(self.hand[b]&self.mug[a])
            table=0.
            for ic in np.flatnonzero(((a==P.table)&self.mug[b])|((b==P.table)&self.mug[a])):
                mujoco.mj_contactForce(m,d,ic,self.force);table=max(table,self.force[0])
            q=d.qpos[self.qids]
            metrics=[table,depth[hm].max(initial=0.),depth[self.hand[a]&self.hand[b]].max(initial=0.),
                     max(0.,(self.lo-q).max(),(q-self.hi).max())]
            np.maximum(peaks,metrics,out=peaks)
        p,R,_,_=P.chart.motion(P.state(d))
        return np.r_[1000*np.linalg.norm(p-P.pos[i+1]),np.rad2deg((R*P.rot[i+1].inv()).magnitude()),peaks]

    def measure(self,cmds):
        P=self.P;d=P.initial();states=[];warm=[];qs=[];metrics=[]
        for i,cmd in enumerate(cmds):
            states.append(P.state(d));warm.append(d.qacc_warmstart.copy());v=self.step(d,i,cmd)
            qs.append(d.qpos.copy());metrics.append(v)
            if (v>LIMITS).any():break
        valid=len(metrics)-int((metrics[-1]>LIMITS).any())
        return dict(valid=valid,state=np.asarray(states),warm=np.asarray(warm),qpos=np.asarray(qs),metrics=np.asarray(metrics))


def run(input_folder,out,clip,iterations):
    _,_,planners=configuration();P=planners[clip];check=StrictCheck(P);m=P.m
    path=input_folder/f'{clip}_8451/best.npz'
    source=h3.file_pin(path)
    out.mkdir(parents=True,exist_ok=False)
    h3.write_json(out/'intent.json',dict(source=source,code=h3.file_pin(__file__),iterations=iterations,candidates=24,
        seed=8461,limits=LIMITS.tolist(),scope='Repair seed8451 pose-only optimized path. Named gate units: mm COM, deg SO3, newtons table, mm hand/mug, mm hand/self, radians hinge excursions. All retained prefixes are replayed from start with 64 substep checks per command. Exploratory thresholds, nominal simulator only.'))
    with np.load(h3.verify_file_pin(source)) as f:cmds=f['commands']
    curr=check.measure(cmds);initial=curr['valid'];folder=out;hist=[];ticks=0;t0=perf_counter()
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS;d=mujoco.MjData(m)
    for it in range(iterations):
        if curr['valid']==len(cmds):break
        anchor=max(0,curr['valid']-(1,3,6,2,8,4)[it%6])
        n=min(len(cmds)-anchor,curr['valid']-anchor+4)
        pool=propose(P,cmds,anchor,n,it,8461,24)
        prev=curr['valid'];best=None
        for j,seq in enumerate(pool):
            mujoco.mj_resetData(m,d);mujoco.mj_setState(m,d,curr['state'][anchor],sig);d.qacc_warmstart[:]=curr['warm'][anchor]
            scores=[]
            for k,u in enumerate(seq):
                s=check.step(d,anchor+k,u)/LIMITS;scores.append(s);ticks+=64
                if np.any(s>1):break
            passing=len(scores)-int(np.any(scores[-1]>1))
            key=(-passing,float(np.max(scores[-1])),float(np.mean(np.minimum(scores,4)**2)),j)
            if best is None or key<best[0]:best=(key,j)
        chosen=best[1]
        if chosen:
            trial=cmds.copy();trial[anchor:anchor+n]=pool[chosen]
            replay=check.measure(trial)
            if replay['valid']>=prev:cmds=trial;curr=replay
            else:chosen=0
        hist.append(dict(round=it+1,anchor=anchor+1,chosen=chosen,passing=curr['valid']))
        if (it+1)%8==0:
            h3.write_json(out/'progress.json',dict(iteration=it+1,passing=curr['valid'],wall_s=perf_counter()-t0))
    confirmed=check.measure(cmds);np.testing.assert_array_equal(curr['qpos'],confirmed['qpos'])
    z={**P.old,'elapsed_s':.02*np.arange(len(cmds)+1),'COM_world_m':P.pos,'quaternion_wxyz':P.rot.as_quat()[:,[3,0,1,2]]}
    detail,tr=h3.physics.replay(P.path,P.binding,z,P.base,cmds,1.,1.);np.testing.assert_array_equal(tr['qpos'][:len(curr['qpos'])],curr['qpos'])
    dest=folder/'trace.npz';np.savez_compressed(dest,**tr,strict_prefix_metrics=curr['metrics'])
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),clip=clip,initial_strict=initial,
        strict_controls=curr['valid'],requested=len(cmds),pose_valid_s=detail['valid_s'],pose_full=detail['right_censored'],
        last_metrics=curr['metrics'][-1].tolist(),search_ticks=ticks,history=hist,wall_s=perf_counter()-t0,trace=h3.file_pin(dest)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--clip',type=int,required=True);p.add_argument('--iterations',type=int,default=32)
    a=p.parse_args();run(a.source.resolve(),a.out.resolve(),a.clip,a.iterations)
