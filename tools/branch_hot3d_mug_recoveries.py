"""Probe reachability by branching the second and third commands on failed fixed futures."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
from test_hot3d_model_horizon import physics,file_pin,verify_file_pin,write_json,propose,MugStepChart


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    audit=file_pin(root/'mug_alternative_futures_v1/report.json')
    report=physics.read_json(audit)
    oracle=physics.read_json(physics.read_json(report['intent'])['audit'])
    cases=physics.read_json(physics.read_json(oracle['intent'])['source'])
    fit=physics.read_json(physics.read_json(cases['intent'])['fit']);inputs=physics.read_json(fit['intent'])['inputs']
    path=verify_file_pin(inputs['model']);bind=physics.read_json(inputs['binding'])
    m=mujoco.MjModel.from_binary_path(str(path));chart=MugStepChart(m,bind)
    old=physics.load_arrays(inputs['reference'])
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(failed=audit,code=file_pin(__file__),scope=
        'Post-hoc, first state per H3 seed with any passing first option but zero fully feasible fixed sequences; choose using archived geometric diagnostic before search. Keep all 8 first controls. For every passing first, try 16 fresh PDs. For at most 4 passing second options per first, try 16 fresh PDs. Freeze one passing branch with lowest maximum normalized pose error then sum squared error. From-start frozen prefix and selected controls independently replayed; no off-path recovery credit. Finite action bank.'))
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS;work=[mujoco.MjData(m) for _ in range(4)]
    def native_state(d):
        s=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,s,sig);return s
    def propagate(pool,d,cmds,pos,rot):
        st,_=pool.rollout(m,work,np.tile(native_state(d),(len(cmds),1)),control=np.repeat(cmds[:,None],64,axis=1),
            initial_warmstart=np.tile(d.qacc_warmstart,(len(cmds),1)))
        end=st[:,-1].copy();p,R,_,_=chart.motion(end)
        error=np.column_stack([100*np.linalg.norm(p-pos,axis=1),np.rad2deg((R*rot.inv()).magnitude())/10])
        return end,error
    def child(parent,cmd,expected):
        d=mujoco.MjData(m);mujoco.mj_copyData(d,m,parent);d.ctrl[:]=cmd;mujoco.mj_step(m,d,nstep=64)
        np.testing.assert_array_equal(native_state(d),expected);return d
    rows=[]
    for group in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=physics.read_json(file_pin(root/group/'report.json'))['rows'];ref=physics.load_arrays(item['reference'])
        clip=item['clip'];T=ref['body'];pos=T[:,:3,3]+T[:,:3,:3]@chart.com;rot=Rotation.from_matrix(T[:,:3,:3])
        z={**old,'elapsed_s':ref['t'],'COM_world_m':pos,'quaternion_wxyz':rot.as_quat()[:,[3,0,1,2]]}
        base=old['feedforward_PD_target20'][0];ik=np.clip(ref['q'][1:]+base-old['initial_qpos27'][:20],*m.actuator_ctrlrange.T)
        for case in cases['rows']:
            seed=case['seed']
            if case['clip']!=clip or case['mode']!='h3':continue
            eligible=[e['phase'] for e in report['rows'] if e['clip']==clip and e['seed']==seed and e['first_feasible']>0]
            if not eligible:
                rows.append(dict(clip=clip,seed=seed,status='no_eligible_source'));continue
            start=perf_counter();i=min(eligible)-1;arc=physics.load_arrays(case['trace'])
            d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=base;mujoco.mj_forward(m,d)
            for j in range(i):
                d.ctrl[:]=arc['ctrl'][j];mujoco.mj_step(m,d,nstep=64)
                np.testing.assert_array_equal(d.qpos,arc['qpos'][j])
            prior=arc['ctrl'][i-1] if i else base
            residual=prior-ik[i-1] if i else np.zeros(20)
            a=propose(np.random.default_rng(np.random.SeedSequence([seed,i,0])),64,ik[i],base,prior,residual,m.actuator_ctrlrange)[case['records'][i]['shortlist']]
            n1,n2,n3=8,0,0;pass1=pass2=pass3=0;win=None
            with Rollout(nthread=4) as engine:
                e1,c1=propagate(engine,d,a,pos[i+1],rot[i+1])
                for k in np.flatnonzero(c1.max(1)<=1):
                    pass1+=1;d1=child(d,a[k],e1[k])
                    a2=propose(np.random.default_rng(np.random.SeedSequence([seed,i,int(k),1])),16,ik[i+1],base,a[k],a[k]-ik[i],m.actuator_ctrlrange)
                    e2,c2=propagate(engine,d1,a2,pos[i+2],rot[i+2]);n2+=len(a2)
                    good=np.flatnonzero(c2.max(1)<=1);pass2+=len(good)
                    for h in sorted(good,key=lambda h:(c2[h].max(),c2[h]@c2[h],h))[:4]:
                        d2=child(d1,a2[h],e2[h])
                        a3=propose(np.random.default_rng(np.random.SeedSequence([seed,i,int(k),int(h),2])),16,
                            ik[i+2],base,a2[h],a2[h]-ik[i+1],m.actuator_ctrlrange)
                        e3,c3=propagate(engine,d2,a3,pos[i+3],rot[i+3]);n3+=len(a3)
                        for l in np.flatnonzero(c3.max(1)<=1):
                            pass3+=1
                            scores=np.array([c1[k],c2[h],c3[l]])
                            key=(scores.max(),(scores**2).sum(),int(k),int(h),int(l))
                            if win is None or key<win[0]:
                                win=(key,np.array([a[k],a2[h],a3[l]]),np.array([e1[k],e2[h],e3[l]]))
            valid=False;witness=None;previous_valid=int(round(case['valid_s']/.02))
            if win is not None:
                cmd=np.vstack([arc['ctrl'][:i],win[1]])
                detail,tr=physics.replay(path,bind,z,base,cmd,1.,1.)
                np.testing.assert_array_equal(tr['qpos'][-3:],win[2][:,1:1+m.nq])
                valid=detail['right_censored']
                f=out/f'{clip}_{seed}.npz';np.savez_compressed(f,**tr);witness=file_pin(f)
            rows.append(dict(clip=clip,seed=seed,phase=i+1,status='searched',passing=[pass1,pass2,pass3],
                generated=[n1,n2,n3],branch_validated=valid,witness=witness,prior_passing=previous_valid,
                new_verified_passing=i+3 if valid else None,witness_ends_beyond_original_stop=bool(valid and i+3>previous_valid),
                seconds=perf_counter()-start))
        write_json(out/'progress.json',rows)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
