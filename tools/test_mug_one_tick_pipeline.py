"""Paced 50Hz delivery of four-option feedback computed one tick ahead."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter,sleep
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch
from test_centered_mug_screen import CenteredScreen,MugHorizonHead
from test_mug_delayed_feedback import selected_rule
from test_mug_event_budget import setup,initial,NativeStrictWindow,context,plant,LIMITS,h3


def preview(screen,s,i,cmd,mode,submitted):
    P=screen.P;m=P.m;d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,s,mujoco.mjtState.mjSTATE_FULLPHYSICS);d.qacc_warmstart[:]=0.;d.ctrl[:]=cmd
    mujoco.mj_step(m,d,nstep=64);pred=P.state(d)
    _,idx,ticks=screen.choose(mode,pred,i+1,cmd-screen.fit['cmd'][i])
    return dict(idx=idx,finished=perf_counter(),submitted=submitted,ticks=ticks+64,pred=pred)


def run_case(P,fast,cheap,screen,worker,old,fit,case,offset,mode):
    d=initial(P.m,P,old,offset);cmds=[];qs=[];metrics=[];events=[];jobs=[];pending=None;due=-1
    previous=np.zeros(20);anchor=perf_counter()
    for i,cmd0 in enumerate(fit['cmd']):
        tick=anchor+.02*i
        if mode.startswith('pipe') or mode=='open':
            wait=tick-perf_counter()
            if wait>0:sleep(wait)
        start=perf_counter();s=P.state(d);cmd=cmd0;status='base';gap=None
        if mode.startswith('sync'):
            route='analytic4' if mode.endswith('analytic') else 'centered4'
            cmd,_,_=screen.choose(route,s,i,previous)
        elif mode.startswith('pipe'):
            if pending is not None and pending.done():
                job=pending.result();jobs.append(dict(ms=1000*(job['finished']-job['submitted']),ticks=job['ticks']))
                if due==i and job['finished']<=tick:
                    cmd=selected_rule(cheap,fit,s,i,previous,job['idx']);status='applied'
                    gap=1000*np.linalg.norm(cheap.chart.motion(s)[0]-cheap.chart.motion(job['pred'])[0])
                else:status='discarded'
                pending=None
            elif pending is not None:status='busy'
            if pending is None and i+1<len(fit['cmd']) and np.max(abs(fit['state'][i]-s))>1e-10:
                route='analytic4' if mode.endswith('analytic') else 'centered4'
                due=i+1;pending=worker.submit(preview,screen,s.copy(),i,cmd.copy(),route,perf_counter())
        ready=perf_counter();previous=cmd-cmd0
        v=fast.step(d,i,cmd);finish=perf_counter()
        cmds.append(cmd.copy());qs.append(d.qpos.copy());metrics.append(v)
        events.append(dict(i=i,status=status,decision_ms=1000*(ready-start),pred_gap_mm=gap,
            ready_ms=1000*(ready-tick) if not mode.startswith('sync') else None,
            simulated_ms=1000*(finish-tick) if not mode.startswith('sync') else None))
        if np.any(v>LIMITS):break
    if pending is not None:
        job=pending.result();jobs.append(dict(ms=1000*(job['finished']-job['submitted']),ticks=job['ticks']))
    passed=len(qs)-int(np.any(metrics[-1]>LIMITS))
    return dict(mode=mode,condition=case,passing=passed,full=passed==len(fit['cmd']),events=events,jobs=jobs),dict(qpos=qs,ctrl=cmds,metrics=metrics)


def run(out):
    torch.set_num_threads(1);root,_,models,asset,bind,old,lib,_=setup()
    source=h3.file_pin(root/'mug_centered_screen_v1/report.json')
    pin=h3.physics.read_json(h3.physics.read_json(source)['intent'])['fit'];weights=h3.physics.read_json(pin)['model']
    ck=torch.load(h3.verify_file_pin(weights),map_location='cpu',weights_only=True)
    net=MugHorizonHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    samples=qmc.scale(qmc.LatinHypercube(d=4,seed=8591).random(8),[.95,.95,.90,-.001],[1.05,1.05,1.10,.001])
    conditions={'nominal':[1.,1.,1.,0.],**{f'new_{i:02}':v.tolist() for i,v in enumerate(samples)}}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(source=source,code=h3.file_pin(__file__),seed=8591,conditions=conditions,
        scope='Eight new combinations plus nominal. Open/pipe paced at 20ms; sync arms measure immediate feedback at same physical clock without wall pacing. At most one outstanding job, one nominal control prediction followed by four native first-window verifications. Apply only a result completed by the next absolute tick, rebind chosen option to current observed state; otherwise use frozen command. No waiting on worker in live decision. Report deadline misses separately for issuing commands and completing simulated physics. Primary criterion is positive paired median gain over open in both clips and no >20ms command-readiness events.'))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as ex,ThreadPoolExecutor(max_workers=1) as worker:
        for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
            fit=h3.physics.load_arrays(models['clips'][str(clip)]['data'])
            item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows'];ref=h3.physics.load_arrays(item['reference']);T=ref['body']
            pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
            nominal=plant(asset,bind,1,1,1);Pn,_=context(nominal,bind,pos,rot);cheap,_=context(nominal,bind,pos,rot)
            screen=CenteredScreen(Pn,NativeStrictWindow(Pn,lib),fit,net,ex)
            for number,(case,(mass,mu,tau,offset)) in enumerate(conditions.items()):
                m=plant(asset,bind,mass,mu,tau);P,py=context(m,bind,pos,rot);native=NativeStrictWindow(P,lib)
                modes=['open','sync_analytic','pipe_analytic','sync_centered','pipe_centered']
                modes=modes[number%5:]+modes[:number%5]
                for mode in modes:
                    r,tr=run_case(P,native,cheap,screen,worker,old,fit,case,offset,mode)
                    replay=initial(m,P,old,offset)
                    for i,cmd in enumerate(tr['ctrl']):
                        np.testing.assert_array_equal(py.step(replay,i,cmd),tr['metrics'][i]);np.testing.assert_array_equal(replay.qpos,tr['qpos'][i])
                    if case=='nominal':np.testing.assert_array_equal(tr['qpos'],fit['state'][1:,1:1+m.nq])
                    p=out/f'{clip}_{case}_{mode}.npz';np.savez_compressed(p,**tr)
                    rows.append(dict(clip=clip,trace=h3.file_pin(p),**r))
                h3.write_json(out/'progress.json',rows);print(clip,case,[r['passing'] for r in rows[-5:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
