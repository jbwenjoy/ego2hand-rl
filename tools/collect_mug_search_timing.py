"""Counterfactual value of moving one search call within four controls of a mug trajectory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from test_mug_event_budget import setup,initial,Selector,NativeStrictWindow,context,plant,LIMITS,h3,observation,deviation


def block_input(reference,s,fit,i,prev):
    # Only the reference and information observed before this block enter the head.
    c,e=observation(reference.m,fit['state'][i],s,fit['cmd'][i],prev)
    end=min(i+4,len(fit['cmd']))
    p0,r0,v0,w0=reference.chart.motion(fit['state'][i])
    p1,r1,v1,w1=reference.chart.motion(fit['state'][end])
    return np.r_[c,e,fit['cmd'][end-1]-fit['cmd'][i],r0.inv().apply(p1-p0),
                 (r0.inv()*r1).as_rotvec(),r0.inv().apply(v1-v0),w1-w0].astype(np.float32)


def collect(out):
    torch.set_num_threads(1)
    root,oldpin,models,asset,bind,old,lib,nets=setup()
    newpin=h3.file_pin(root/'mug_event_budget_v1/report.json')
    new=h3.physics.read_json(newpin);newconds=h3.physics.read_json(new['intent'])['conditions']
    oldr=h3.physics.read_json(oldpin);oldconds=h3.physics.read_json(oldr['intent'])['conditions']
    out.mkdir(exist_ok=False)
    h3.write_json(out/'intent.json',dict(source_train=oldpin,source_test=newpin,code=h3.file_pin(__file__),
        policy='Frozen dense12 nominal planner applied exactly once in the current block. In the following block use the identically timed periodic search in every branch; stop after eight controls or first breach. Four first-block slot labels from identical data and warmstart.',
        partition='Train from the old dense teacher, Latin hypercube seed8551; one source at every eighth control starting at frame 1. Test on all block starts of periodic paths in seed8561. Never pool scenarios across the split. Baseline first-block slot rotates with block and case ID.'))
    records=[];parity=0
    with ThreadPoolExecutor(max_workers=4) as ex:
        for clip,group in ((1884,'mug_annotated_windows_original_v1'),(1888,'mug_unilateral_interval_v1')):
            fit=h3.physics.load_arrays(models['clips'][str(clip)]['data'])
            row,=h3.physics.read_json(h3.file_pin(root/group/'report.json'))['rows'];ref=h3.physics.load_arrays(row['reference']);T=ref['body']
            pos=T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'];rot=Rotation.from_matrix(T[:,:3,:3])
            m0=plant(asset,bind,1,1,1);Pn,_=context(m0,bind,pos,rot);policy=Selector(Pn,NativeStrictWindow(Pn,lib),fit,nets,ex)
            for role,src,conds in [('train',oldr,oldconds),('test',new,newconds)]:
                cases=[r for r in src['rows'] if r['clip']==clip and r.get('mode')==('dense12' if role=='train' else 'periodic')
                       and r['condition'].startswith('generated_' if role=='train' else 'new_')]
                for case in cases:
                    name=case['condition'];seed=1+int(name.split('_')[1])
                    mass,mu,tau,offset=conds[name];m=plant(asset,bind,mass,mu,tau);P,_=context(m,bind,pos,rot);fast=NativeStrictWindow(P,lib)
                    d=initial(m,P,old,offset);arc=h3.physics.load_arrays(case['trace']);prev=np.zeros(20);fork=mujoco.MjData(m)
                    for i,cmd in enumerate(arc['ctrl']):
                        if i%(8 if role=='train' else 4)==0 and i+8<=len(fit['cmd']):
                            s=P.state(d);x=block_input(Pn,s,fit,i,prev);labels=[];costs=[]
                            for slot in range(4):
                                mujoco.mj_copyData(fork,m,d);last=prev.copy();passed=0;penalty=0.
                                reference_slot=(i//4+seed)%4
                                for h in range(8):
                                    ix=i+h;cmd0=fit['cmd'][ix]
                                    if h==slot or h==4+(reference_slot+1)%4:
                                        u,_,_=policy.choose('dense12',P.state(fork),ix,last)
                                    else:u=cmd0
                                    last=u-cmd0;v=fast.step(fork,ix,u)
                                    penalty+=float(np.minimum(v/LIMITS,3.).max())
                                    if role=='test' and slot==reference_slot:
                                        np.testing.assert_array_equal(fork.qpos,arc['qpos'][ix])
                                        parity+=1
                                    if np.any(v>LIMITS):break
                                    passed+=1
                                labels.append(passed);costs.append(penalty/max(1,min(8,passed+1)))
                            records.append((role,clip,name,i,reference_slot,x,labels,costs,
                                hashlib.sha256(s.tobytes()+d.qacc_warmstart.tobytes()+prev.tobytes()).hexdigest()))
                        fast.step(d,i,cmd);np.testing.assert_array_equal(d.qpos,arc['qpos'][i]);prev=cmd-fit['cmd'][i]
                    print(clip,role,name,len(records),flush=True)
    data=out/'labels.npz'
    np.savez_compressed(data,**{k:np.asarray(v) for k,v in zip(
        ('role','clip','condition','start','periodic','x','values','avg_ratio','key'),zip(*records))})
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),data=h3.file_pin(data),groups=len(records),matched_native_controls=parity))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);collect(p.parse_args().out.resolve())
