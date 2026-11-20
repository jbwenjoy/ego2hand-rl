"""Pace native simulation at 50 Hz; deliver a preplanned correction without waiting for a worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter,sleep
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch
from test_mug_event_budget import setup,initial,Selector,NativeStrictWindow,context,plant,LIMITS,h3
from test_strict_mug_feedback import tangent_errors,correct


def selected_rule(P,fit,s,i,previous,index):
    """Evaluate only the already chosen branch on the current observation."""
    ref=fit['state'][i];cmd0=fit['cmd'][i]
    dx=tangent_errors(P.m,ref,s[None])[0]
    if index==0 or np.max(abs(dx))<1e-10:delta=np.zeros(20)
    elif index==1:delta=.65*previous
    elif index in (2,3):
        delta=(.5 if index==2 else 1.)*correct('kinematic',P.chart,s,ref,dx,fit,i)
    elif index in (4,5):
        delta=(.25 if index==4 else .65)*correct('native_gain',P.chart,s,ref,dx,fit,i)
    else:
        rng=np.random.default_rng(np.random.SeedSequence([8541,i]))
        noise=rng.normal(size=(3,20));noise/=np.sqrt(np.mean(noise**2,axis=1,keepdims=True))
        amp=min(.09,.018+.003*np.linalg.norm(dx[P.chart.da:P.chart.da+3])/.001)
        delta=.6*previous+amp*noise[(index-6)//2]*(1 if index%2==0 else -1)
    return np.clip(cmd0+np.clip(delta,-.12,.12),*P.m.actuator_ctrlrange.T)


def plan_ahead(selector,fit,s,first,due,submitted):
    # Use only the measurement available at submission. Future controls through
    # the dispatch slot are the already scheduled nominal commands.
    P=selector.P;m=P.m;d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,s,mujoco.mjtState.mjSTATE_FULLPHYSICS);d.qacc_warmstart[:]=0
    for phase in range(first,due):
        d.ctrl[:]=fit['cmd'][phase];mujoco.mj_step(m,d,nstep=64)
    predicted=P.state(d)
    cmd,idx,ticks=selector.choose('dense12',predicted,due,np.zeros(20))
    np.testing.assert_array_equal(cmd,selected_rule(P,fit,predicted,due,np.zeros(20),idx))
    return dict(cmd=cmd,index=idx,ticks=ticks+64*(due-first),predicted=predicted,
                completion=perf_counter(),submitted=submitted)


def run_case(P,checker,cheap,selector,worker,old,fit,case,offset,mode):
    d=initial(P.m,P,old,offset)
    pending=None;pending_due=-1;discarded=[];records=[];qs=[];cmds=[];metrics=[]
    controls=fit['cmd'];anchor=perf_counter();budget=0
    for i,cmd0 in enumerate(controls):
        tick=anchor+.02*i
        if mode!='instant':
            wait=tick-perf_counter()
            if wait>0:sleep(wait)
        start=perf_counter();s=P.state(d);cmd=cmd0;status='base';job_ms=None;pred_error_mm=None
        due=i%6==5
        if mode=='instant' and due:
            cmd,_,used=selector.choose('dense12',s,i,np.zeros(20));budget+=used;status='instant'
        elif mode in ('ahead_raw','ahead_rule'):
            if pending is not None and pending_due<i and pending.done():
                old_job=pending.result();budget+=old_job['ticks'];discarded.append(old_job['completion']-old_job['submitted']);pending=None
            if i%6==0 and i+5<len(controls) and pending is None:
                # Exact nominal states follow the demonstration without extra computation.
                dx=tangent_errors(cheap.m,fit['state'][i],s[None])[0]
                if np.max(abs(dx))>=1e-10:
                    pending_due=i+5;pending=worker.submit(plan_ahead,selector,fit,s.copy(),i,pending_due,perf_counter())
            if due and pending is not None and pending_due==i:
                if pending.done():
                    p=pending.result()
                    if p['completion']<=tick:
                        applied=selected_rule(cheap,fit,s,i,np.zeros(20),p['index']) if mode=='ahead_rule' else p['cmd']
                        cmd=applied;status='received'
                    else:status='too_late'
                    job_ms=1000*(p['completion']-p['submitted']);budget+=p['ticks']
                    pred_error_mm=1000*np.linalg.norm(cheap.chart.motion(s)[0]-cheap.chart.motion(p['predicted'])[0])
                    pending=None
                else:status='not_ready'
        # The nominal-state guard belongs to all arms, including predicted RAW.
        if np.max(abs(tangent_errors(cheap.m,fit['state'][i],s[None])[0]))<1e-10:cmd=cmd0
        ready=perf_counter()
        v=checker.step(d,i,cmd)
        finished=perf_counter()
        records.append(dict(phase=i+1,status=status,decision_ms=1000*(ready-start),
            scheduled_to_ready_ms=1000*(ready-tick) if mode!='instant' else None,
            scheduled_to_simulated_ms=1000*(finished-tick) if mode!='instant' else None,
            plan_ms=job_ms,predicted_COM_error_mm=pred_error_mm))
        qs.append(d.qpos.copy());cmds.append(cmd.copy());metrics.append(v)
        if np.any(v>LIMITS):break
    if pending is not None:
        p=pending.result();budget+=p['ticks'];discarded.append(p['completion']-p['submitted'])
    passed=len(metrics)-int(np.any(metrics[-1]>LIMITS))
    return dict(condition=case,mode=mode,passing=passed,full=passed==len(controls),model_ticks=budget,
                unconsumed_job_s=discarded,records=records),dict(qpos=qs,ctrl=cmds,metrics=metrics)


def main(out):
    torch.set_num_threads(1)
    root,parent,models,asset,bind,old,lib,nets=setup()
    samples=qmc.scale(qmc.LatinHypercube(d=4,seed=8563).random(8),[.95,.95,.9,-.001],[1.05,1.05,1.1,.001])
    conditions={'nominal':[1,1,1,0],**{f'new_{i:02}':v.tolist() for i,v in enumerate(samples)}}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(source=parent,code=h3.file_pin(__file__),conditions=conditions,
        scope='Fixed six-control cycle. Instant is an unpaced diagnostic allowed to block at slots 6,12,etc. Open and asynchronous arms follow absolute 20ms wall-clock ticks. Submit once at block start for slot five periods later, predict through five feedforward commands with private nominal physics, discard late output and immediately use frozen command. Late scheduling cannot turn a late job into an on-time job. No claim about hardware real-time behavior.'))
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool,ThreadPoolExecutor(max_workers=1) as worker:
        for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
            fit=h3.physics.load_arrays(models['clips'][str(clip)]['data'])
            item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows'];ref=h3.physics.load_arrays(item['reference']);T=ref['body']
            pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
            m0=plant(asset,bind,1,1,1);Pn,_=context(m0,bind,pos,rot);cheap,_=context(m0,bind,pos,rot)
            selector=Selector(Pn,NativeStrictWindow(Pn,lib),fit,nets,pool)
            for n,(case,(mass,mu,tau,offset)) in enumerate(conditions.items()):
                m=plant(asset,bind,mass,mu,tau);P,py=context(m,bind,pos,rot);native=NativeStrictWindow(P,lib)
                modes=['open','instant','ahead_raw','ahead_rule'];modes=modes[n%4:]+modes[:n%4]
                for mode in modes:
                    result,tr=run_case(P,native,cheap,selector,worker,old,fit,case,offset,mode)
                    replay=initial(m,P,old,offset)
                    for i,cmd in enumerate(tr['ctrl']):
                        np.testing.assert_array_equal(py.step(replay,i,cmd),tr['metrics'][i])
                        np.testing.assert_array_equal(replay.qpos,tr['qpos'][i])
                    if case=='nominal':np.testing.assert_array_equal(tr['qpos'],fit['state'][1:,1:1+m.nq])
                    dest=out/f'{clip}_{case}_{mode}.npz';np.savez_compressed(dest,**tr)
                    results.append(dict(clip=clip,trace=h3.file_pin(dest),**result))
                h3.write_json(out/'progress.json',results);print(clip,case,[r['passing'] for r in results[-4:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
