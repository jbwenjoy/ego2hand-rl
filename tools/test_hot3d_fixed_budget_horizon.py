"""Allocate eight native control windows to width or depth using a frozen proposal scorer."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from test_mug_learned_planning import physics,file_pin,verify_file_pin,write_json,MugStepChart,MugStepHead,propose,kinematics,mean_reference_wrench


def costs(p,rot,wrench,commands,ik,z,reference_rotation,load,phase,weight):
    mm=1000*np.linalg.norm(p-z['COM_world_m'][phase+1],axis=1)
    deg=np.rad2deg((rot*reference_rotation[phase+1].inv()).magnitude())
    error=wrench-load['wrench_world'][phase]
    square=np.sum(error[:,:3]**2,axis=1)/weight**2+np.sum(error[:,3:]**2,axis=1)/(.05*weight)**2
    support=square/(1+square) if load['valid_interval'][phase] else np.zeros(len(commands))
    score=(mm/10)**2+(deg/10)**2+.2*support+.01*np.mean(((commands-ik)/.12)**2,axis=1)
    return np.maximum(1.,np.maximum(mm,deg)/10),score


def run_case(m,path,binding,z,direct,base,seed,mode,net):
    chart=MugStepChart(m,binding);b=chart.body
    Q=Rotation.from_quat(m.body_iquat[b,[1,2,3,0]]).as_matrix()
    I=Q@np.diag(m.body_inertia[b])@Q.T
    load=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],m.body_mass[b],I,m.opt.gravity)
    rot=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    live=mujoco.MjData(m);live.qpos[:]=z['initial_qpos27'];live.qvel[:]=z['initial_qvel26'];live.ctrl[:]=base
    mujoco.mj_forward(m,live)
    scratch=[mujoco.MjData(m) for _ in range(8)]
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    def state(d):
        v=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,v,sig);return v
    def propose_rank(d,phase,last,res):
        pre=state(d);ik=direct[phase]
        candidates=propose(np.random.default_rng(np.random.SeedSequence([seed,phase,0])),64,ik,base,last,res,m.actuator_ctrlrange)
        with torch.inference_mode():
            y=net(torch.from_numpy(chart.features(pre,d.qacc_warmstart,candidates))).numpy()
        violation,score=costs(*chart.decode(pre,y),candidates,ik,z,rot,load,phase,chart.weight)
        return candidates,np.lexsort((np.arange(64),score,violation))
    def rollout(slot,source,command,phase):
        d=scratch[slot];pre=state(source)
        mujoco.mj_copyData(d,m,source);d.ctrl[:]=command
        assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
        mujoco.mj_step(m,d,nstep=64)
        end=state(d)
        _,_,v0,L0=kinematics(pre[None],chart.qa,chart.da,chart.com,I,m.nq)
        p,R,v1,L1=kinematics(end[None],chart.qa,chart.da,chart.com,I,m.nq)
        w=np.column_stack([m.body_mass[b]*((v1-v0)/.02-m.opt.gravity),(L1-L0)/.02])
        gate,score=costs(p,R,w,command[None],direct[phase],z,rot,load,phase,chart.weight)
        return float(gate[0]),float(score[0]),end
    previous=base.copy();res=np.zeros(20);records=[];cmds=[];qs=[];wall=0.;inferences=0
    table=m.geom('mug_task_table').id
    mug=m.geom_bodyid==b
    force=np.empty(6)
    with ThreadPoolExecutor(max_workers=4) as executor:
        for t,ik in enumerate(direct):
            start=perf_counter();cand,order=propose_rank(live,t,previous,res);inferences+=64
            shortlist=np.r_[np.arange(4),order[order>=4][:4]]
            h2=mode!='wide8' and t+1<len(direct)
            ids=order[np.isin(order,shortlist)][:4] if h2 else shortlist
            first=[f.result() for f in [executor.submit(rollout,j,live,cand[k],t) for j,k in enumerate(ids)]]
            g=np.array([x[0] for x in first]);c=np.array([x[1] for x in first])
            order1=np.lexsort((ids,c,g));pick=order1[0];alt=int(ids[pick]);child_ids=[]
            if h2:
                calls=[]
                for j,k in enumerate(ids):
                    child,co=propose_rank(scratch[j],t+1,cand[k],cand[k]-ik);inferences+=64
                    k2=int(co[0]);child_ids.append(k2)
                    calls.append(executor.submit(rollout,4+j,scratch[j],child[k2],t+1))
                second=[f.result() for f in calls]
                g2=np.array([x[0] for x in second]);c2=np.array([x[1] for x in second])
                two=int(np.lexsort((ids,c+c2,np.maximum(g,g2),g))[0]);alt=int(ids[two])
                if mode=='depth2':pick=two
            elapsed=perf_counter()-start;wall+=elapsed
            k=int(ids[pick]);cmd=cand[k];live.ctrl[:]=cmd;peak=0.
            for _ in range(64):
                mujoco.mj_step(m,live)
                a,b2=live.contact.geom.T
                for ic in np.flatnonzero(((a==table)&mug[b2])|((b2==table)&mug[a])):
                    mujoco.mj_contactForce(m,live,ic,force);peak=max(peak,force[0])
            s=state(live);np.testing.assert_array_equal(s,first[pick][2])
            p,R,_,_=chart.motion(s);mm=1000*np.linalg.norm(p-z['COM_world_m'][t+1]);deg=np.rad2deg((R*rot[t+1].inv()).magnitude())
            cmds.append(cmd.copy());qs.append(live.qpos.copy());records.append(dict(phase=t+1,choice=k,one=int(ids[order1[0]]),
                two=alt,children=child_ids,mm=mm,deg=deg,table_N=peak,first_gate=g.tolist(),first_cost=c.tolist(),
                second_gate=g2.tolist() if h2 else [],second_cost=c2.tolist() if h2 else [],ids=ids.tolist()))
            previous=cmd;res=cmd-ik
            if mm>10 or deg>10 or peak>.1:break
    detail,trace=physics.replay(path,binding,z,base,np.array(cmds),1.,1.)
    np.testing.assert_array_equal(qs,trace['qpos'])
    return dict(seed=seed,mode=mode,valid_s=detail['valid_s'],full=detail['right_censored'] and len(cmds)==len(direct),
                controls=len(cmds),ticks=len(cmds)*8*64,predictions=inferences,planning_wall_s=wall,records=records),trace


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs';fit=file_pin(root/'mug_shared_step_v1/report.json');report=physics.read_json(fit)
    inputs=physics.read_json(report['intent'])['inputs'];binding=physics.read_json(inputs['binding']);old=physics.load_arrays(inputs['reference'])
    path=verify_file_pin(inputs['model']);m=mujoco.MjModel.from_binary_path(str(path))
    pin=physics.read_json(report['frozen'])['pins']['shared'];ck=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True)
    torch.set_num_threads(1);net=MugStepHead(ck['n']);net.load_state_dict(ck['state']);net.eval()
    seeds=tuple(range(8411,8419));modes=('wide8','shadow4','depth2')
    out.mkdir(exist_ok=False,parents=True)
    write_json(out/'intent.json',dict(fit=fit,model=pin,code=file_pin(__file__),seeds=seeds,modes=modes,
        scope='Same 64 starting options and frozen scorer. Wide8 verifies 4 anchors and top 4 other options. Other arms verify the top 4 of that shortlist and one learned continuation from the native child state of each. Shadow ignores second scores; depth2 uses both. Exactly 8 native control forecasts per decision; only first command applied. Primary contrast depth2 versus shadow4; check per-clip median paired gain >=1 passing control versus both shadow4 and wide8. Gate table contact on actual execution and exact independent replay.'))
    rows=[]
    for name in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=physics.read_json(file_pin(root/name/'report.json'))['rows'];ref=physics.load_arrays(item['reference']);T=ref['body']
        z={**old,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
           'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=z['feedforward_PD_target20'][0];direct=np.clip(ref['q'][1:]+base-z['initial_qpos27'][:20],*m.actuator_ctrlrange.T)
        for seed in seeds:
            for mode in modes:
                s,tr=run_case(m,path,binding,z,direct,base,seed,mode,net)
                f=out/f"{item['clip']}_{seed}_{mode}.npz";np.savez_compressed(f,**tr)
                rows.append(dict(clip=item['clip'],trace=file_pin(f),**s))
            write_json(out/'progress.json',rows)
            print(item['clip'],seed,[v['valid_s'] for v in rows[-3:]],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
