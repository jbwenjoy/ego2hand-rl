"""Spend eight native verifications with learned scores for one or three intervals."""
import argparse
from pathlib import Path
from time import perf_counter

import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from test_mug_learned_planning import physics,file_pin,verify_file_pin,write_json,kinematics,propose,mean_reference_wrench,MugStepChart
from supermimic_research.mechanics.mug_horizon import features,continuations,decode,MugHorizonHead


def order_horizon(first_gate,future_gate,score):
    return np.lexsort((np.arange(len(score)),score,np.maximum(first_gate,future_gate),first_gate))


def run_case(m,path,bind,z,direct,base,net,seed,mode):
    chart=MugStepChart(m,bind);b=chart.body
    Q=Rotation.from_quat(m.body_iquat[b,[1,2,3,0]]).as_matrix()
    I=Q@np.diag(m.body_inertia[b])@Q.T;mass=m.body_mass[b]
    target=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],mass,I,m.opt.gravity)
    refrot=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    def score(p,R,w,i,h,cmd):
        t=i+h
        mm=1000*np.linalg.norm(p-z['COM_world_m'][t],axis=1)
        deg=np.rad2deg((R*refrot[t].inv()).magnitude())
        cost=(mm/10)**2+(deg/10)**2+.01*np.mean(((cmd-direct[t-1])/.12)**2,axis=1)
        if target['valid_interval'][i:t].all():
            diff=w-target['wrench_world'][i:t].mean(0)
            sq=np.sum(diff[:,:3]**2,axis=1)/chart.weight**2+np.sum(diff[:,3:]**2,axis=1)/(.05*chart.weight)**2
            cost+=.2*sq/(1+sq)
        return np.maximum(1.,np.maximum(mm,deg)/10),cost
    d=mujoco.MjData(m);d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=base;mujoco.mj_forward(m,d)
    data=[mujoco.MjData(m) for _ in range(4)];sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    previous=base.copy();res=np.zeros(20);is_mug=m.geom_bodyid==b;table=m.geom('mug_task_table').id;f=np.empty(6)
    cmds=[];qs=[];records=[];wall=0.
    with Rollout(nthread=4) as roll:
        for i,ik in enumerate(direct):
            clock=perf_counter();pre=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,pre,sig)
            raw=propose(np.random.default_rng(np.random.SeedSequence([seed,i,0])),64,ik,base,previous,res,m.actuator_ctrlrange)
            seq=continuations(raw,direct,i,m.actuator_ctrlrange)
            with torch.inference_mode():y=net(torch.from_numpy(features(chart,pre,d.qacc_warmstart,seq))).numpy().reshape(-1,3,12)
            g1,c1=score(*decode(chart,pre,y[:,0],1),i,1,raw)
            fg=np.ones_like(g1);fc=np.zeros_like(c1)
            for h in range(2,min(3,len(direct)-i)+1):
                g,c=score(*decode(chart,pre,y[:,h-1],h),i,h,seq[:,h-1])
                fg=np.maximum(fg,g);fc+=c
            if mode=='shifted':
                fg=np.roll(fg,-17);fc=np.roll(fc,-17)
            if mode=='h1':
                fg.fill(1.);fc.fill(0.)
            ranked=order_horizon(g1,fg,c1+fc)
            ids=np.r_[np.arange(4),ranked[ranked>=4][:4]];cs=raw[ids]
            sim,_=roll.rollout(m,data,np.tile(pre,(8,1)),control=np.repeat(cs[:,None],64,axis=1),
                initial_warmstart=np.tile(d.qacc_warmstart,(8,1)))
            end=sim[:,-1]
            p,R,v1,L1=kinematics(end,chart.qa,chart.da,chart.com,I,m.nq)
            _,_,v0,L0=kinematics(pre[None],chart.qa,chart.da,chart.com,I,m.nq)
            w=np.column_stack([mass*((v1-v0)/.02-m.opt.gravity),(L1-L0)/.02])
            eg,ec=score(p,R,w,i,1,cs)
            greedy=int(order_horizon(eg,np.ones(8),ec)[0])
            selected=int(order_horizon(eg,fg[ids],ec+fc[ids])[0]) if mode in ('h3','shifted') else greedy
            k=int(ids[selected]);cmd=cs[selected];wall+=perf_counter()-clock;d.ctrl[:]=cmd
            assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
            peak=0.
            for _ in range(64):
                mujoco.mj_step(m,d);a,bb=d.contact.geom.T
                for ic in np.flatnonzero(((a==table)&is_mug[bb])|((bb==table)&is_mug[a])):
                    mujoco.mj_contactForce(m,d,ic,f);peak=max(peak,f[0])
            obs=np.empty_like(pre);mujoco.mj_getState(m,d,obs,sig);np.testing.assert_array_equal(obs,end[selected])
            mm=1000*np.linalg.norm(p[selected]-z['COM_world_m'][i+1])
            deg=np.rad2deg((R[selected]*refrot[i+1].inv()).magnitude())
            records.append(dict(phase=i+1,choice=k,greedy_same_shortlist=int(ids[greedy]),shortlist=ids.tolist(),
                mm=mm,deg=deg,peak_table_N=peak));cmds.append(cmd.copy());qs.append(d.qpos.copy())
            previous=cmd;res=cmd-ik
            if mm>10 or deg>10 or peak>.1:break
    detail,trace=physics.replay(path,bind,z,base,np.array(cmds),1.,1.)
    np.testing.assert_array_equal(qs,trace['qpos'])
    return dict(mode=mode,seed=seed,valid_s=detail['valid_s'],full=detail['right_censored'] and len(direct)==len(cmds),
        first_failure=detail['first_breach'],controls=len(cmds),plan_wall_s=wall,plan_ticks=512*len(cmds),records=records),trace


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs';fit=file_pin(root/'mug_shared_horizon_v1/report.json');report=physics.read_json(fit)
    inputs=physics.read_json(report['intent'])['inputs'];binding=physics.read_json(inputs['binding']);old=physics.load_arrays(inputs['reference'])
    ck=torch.load(verify_file_pin(report['model']),map_location='cpu',weights_only=True)
    torch.set_num_threads(1);net=MugHorizonHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    path=verify_file_pin(inputs['model']);m=mujoco.MjModel.from_binary_path(str(path))
    seeds=tuple(range(8431,8439));out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(fit=fit,code=file_pin(__file__),seeds=seeds,
        scope='Matched architecture, 64 predictions, 8 native first-window verifications in every arm. h1 uses first horizon only. h3_pre shortlists over 3 predicted windows, chooses by verified first. h3 also uses predicted 40/60ms when choosing from the eight. shifted reassigns only predicted future gate and cost by 17 slots throughout. All execute only first action and replan. Primary test is positive median paired gain of at least one control for h3 over h1 on BOTH clips.'))
    rows=[]
    for group in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=physics.read_json(file_pin(root/group/'report.json'))['rows'];ref=physics.load_arrays(item['reference']);T=ref['body']
        z={**old,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
           'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=z['feedforward_PD_target20'][0];ik=np.clip(ref['q'][1:]+base-z['initial_qpos27'][:20],*m.actuator_ctrlrange.T)
        for seed in seeds:
            for mode in ('h1','h3_pre','h3','shifted'):
                summary,trace=run_case(m,path,binding,z,ik,base,net,seed,mode)
                p=out/f"{item['clip']}_{seed}_{mode}.npz";np.savez_compressed(p,**trace)
                rows.append(dict(clip=item['clip'],trace=file_pin(p),**summary))
            write_json(out/'progress.json',rows)
            print(item['clip'],seed,[x['valid_s'] for x in rows[-4:]],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
