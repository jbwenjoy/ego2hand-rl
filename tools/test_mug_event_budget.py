"""Give event-triggered and periodic mug feedback one search slot per four controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc
import torch

from compare_mug_feedback_shortlists import (Selector,context,h3,plant,NativeStrictWindow,LIMITS,
                                            MugResidualActor,observation)


def setup():
    root=Path(__file__).resolve().parents[1]/'runs'
    parent=h3.file_pin(root/'mug_feedback_shortlists_v1/report.json')
    contract=h3.physics.read_json(h3.physics.read_json(parent)['intent'])
    mpc=h3.physics.read_json(h3.physics.read_json(contract['prior'])['intent'])
    fit=h3.physics.read_json(mpc['fit'])
    inputs=h3.physics.read_json(fit['intent'])['inputs']
    asset=h3.verify_file_pin(inputs['model']);bind=h3.physics.read_json(inputs['binding'])
    old=h3.physics.load_arrays(inputs['reference'])
    build=h3.physics.read_json(h3.physics.read_json(h3.physics.read_json(contract['qualification'])['intent'])['build'])
    lib=h3.verify_file_pin(build['helper'])
    students=h3.physics.read_json(h3.physics.read_json(contract['students'])['frozen'])['weights']
    models={}
    for a,pin in students.items():
        ck=torch.load(h3.verify_file_pin(pin),weights_only=True,map_location='cpu')
        net=MugResidualActor(*ck['dims']);net.load_state_dict(ck['state']);models[a]=net.eval()
    return root,parent,fit,asset,bind,old,lib,models


def deviation(chart,s,ref):
    p,r,v,w=chart.motion(s);p0,r0,v0,w0=chart.motion(ref)
    translation=np.linalg.norm((p-p0)+.04*(v-v0))/.003
    rotation=np.linalg.norm((r*r0.inv()).as_rotvec()+.04*(r.apply(w)-r0.apply(w0)))/np.deg2rad(3)
    return float(max(translation,rotation))


def initial(m,P,old,offset):
    d=mujoco.MjData(m);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26']
    d.ctrl[:]=old['feedforward_PD_target20'][0];qa=P.chart.qa
    d.qpos[qa:qa+3]+=Rotation.from_quat(d.qpos[qa+np.array([4,5,6,3])]).apply([offset,0.,0.])
    mujoco.mj_forward(m,d);return d


def run_case(P,fast,reference,selector,old,fit,case,offset,mode,seed,override=None):
    d=initial(P.m,P,old,offset);last=np.zeros(20);records=[];cmds=[];qs=[];metrics=[];spent=False
    for i,base in enumerate(fit['cmd']):
        start=perf_counter();s=P.state(d)
        if i%4==0:spent=False
        risk=deviation(reference.chart,s,fit['state'][i]);index=i%4
        period=(seed+i//4)%4
        final_slot=index==3 or i+1==len(fit['cmd'])
        if mode=='dense':request=True
        elif mode=='periodic':request=not spent and (index==period or final_slot)
        elif mode=='event':request=not spent and (risk>=1. or final_slot)
        elif mode=='open':request=False
        else:request=not spent and (override(s,i,last,risk,index) or final_slot)
        if request:
            cmd,j,ticks=selector.choose('dense12',s,i,last);spent=True
        else:
            cmd=base;j=0;ticks=0
        elapsed=perf_counter()-start;last=cmd-base
        v=fast.step(d,i,cmd);records.append(dict(i=i,risk=risk,requested=bool(request),ticks=ticks,choice=j,seconds=elapsed))
        qs.append(d.qpos.copy());metrics.append(v);cmds.append(cmd.copy())
        if np.any(v>LIMITS):break
    passed=len(qs)-int(np.any(metrics[-1]>LIMITS))
    return dict(condition=case,mode=mode,passing=passed,full=passed==len(fit['cmd']),decisions=records),dict(qpos=qs,ctrl=cmds,metrics=metrics)


def main(out):
    torch.set_num_threads(1)
    root,parent,models,asset,bind,old,lib,nets=setup()
    draws=qmc.scale(qmc.LatinHypercube(d=4,seed=8561).random(12),[.95,.95,.90,-.001],[1.05,1.05,1.10,.001])
    conditions={'nominal':[1.,1.,1.,0.],**{f'new_{i:02}':v.tolist() for i,v in enumerate(draws)}}
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(parent=parent,code=h3.file_pin(__file__),conditions=conditions,seed=8561,
        scope='Fresh parameter draws on two previously inspected clips. One dense12 call per finished four-control block for event and periodic, possibly unequal counts in final partial blocks. Periodic phase rotates across blocks and conditions. Event triggers on current trajectory error extrapolated 40ms above 3mm/3deg or last block slot. Other controls use original witness commands. No student or parameter-specific retuning. Primary criterion positive median paired valid controls vs periodic on BOTH clips, no loss of nominal qualification.'))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as ex:
        for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
            fit=h3.physics.load_arrays(models['clips'][str(clip)]['data'])
            item,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows'];ref=h3.physics.load_arrays(item['reference']);T=ref['body']
            pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
            m0=plant(asset,bind,1,1,1);Pn,_=context(m0,bind,pos,rot)
            selector=Selector(Pn,NativeStrictWindow(Pn,lib),fit,nets,ex)
            for case_index,(key,(mass,mu,tau,offset)) in enumerate(conditions.items()):
                m=plant(asset,bind,mass,mu,tau);P,checker=context(m,bind,pos,rot);fast=NativeStrictWindow(P,lib)
                for mode in ('open','periodic','event','dense'):
                    result,trace=run_case(P,fast,Pn,selector,old,fit,key,offset,mode,case_index)
                    replay=initial(m,P,old,offset)
                    for i,(cmd,goal,v) in enumerate(zip(trace['ctrl'],trace['qpos'],trace['metrics'],strict=True)):
                        np.testing.assert_array_equal(checker.step(replay,i,cmd),v)
                        np.testing.assert_array_equal(replay.qpos,goal)
                    if key=='nominal':np.testing.assert_array_equal(trace['qpos'],fit['state'][1:,1:1+m.nq])
                    flags=[r['requested'] for r in result['decisions']]
                    if mode in ('periodic','event'):
                        for j in range(0,len(flags)-3,4):assert sum(flags[j:j+4])==1
                    p=out/f'{clip}_{key}_{mode}.npz';np.savez_compressed(p,**trace)
                    rows.append(dict(clip=clip,trace=h3.file_pin(p),**result))
                h3.write_json(out/'progress.json',rows);print(clip,key,[r['passing'] for r in rows[-4:]],flush=True)
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('out',type=Path)
    main(parser.parse_args().out.resolve())
