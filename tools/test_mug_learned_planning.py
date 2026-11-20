"""Fresh closed-loop trials of learned mug response ranking and eight-proposal verification."""
import argparse
from pathlib import Path
from time import perf_counter

import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from search_mug_feasible_paths import kinematics,propose
from compare_hot3d_mug_wrench import physics,file_pin,verify_file_pin,write_json,mean_reference_wrench,MugContactImpulse
from supermimic_research.mechanics.mug_step_response import MugStepChart,MugStepHead


def rank(pos,rot,wrench,target_pos,target_rot,target_load,eligible,weight,src_rot,commands,ik):
    mm=1000*np.linalg.norm(pos-target_pos,axis=1)
    deg=np.rad2deg((rot*target_rot.inv()).magnitude())
    e=np.zeros(len(commands))
    if eligible:
        delta=wrench-target_load
        e=(src_rot.inv().apply(delta[:,:3])**2).sum(1)/weight**2+(src_rot.inv().apply(delta[:,3:])**2).sum(1)/(weight*.05)**2
    score=(mm/10)**2+(deg/10)**2+.2*e/(1+e)+.01*np.mean(((commands-ik)/.12)**2,axis=1)
    return np.lexsort((np.arange(len(commands)),score,np.maximum(1.,np.maximum(mm,deg)/10)))


def trial(m,path,binding,z,direct,base,seed,mode,models,threads):
    sensor=MugContactImpulse(m,binding);chart=MugStepChart(m,binding)
    b=sensor.body
    Q=Rotation.from_quat(m.body_iquat[b,[1,2,3,0]]).as_matrix()
    I=Q@np.diag(m.body_inertia[b])@Q.T
    ref=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],m.body_mass[b],I,m.opt.gravity)
    refrot=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    d=mujoco.MjData(m);d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=base
    mujoco.mj_forward(m,d)
    data=[mujoco.MjData(m) for _ in range(threads)]
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    previous=base.copy();residual=np.zeros(20)
    log=[];commands=[];qs=[];budget=0;inferences=0;wall=0.
    with Rollout(nthread=threads) as rollout:
        for i,ik in enumerate(direct):
            clock=perf_counter()
            state=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,state,sig)
            proposals=propose(np.random.default_rng(np.random.SeedSequence([seed,i,0])),64,ik,base,previous,residual,m.actuator_ctrlrange)
            ids=np.arange(64)
            _,source_rot,v0,L0=kinematics(state[None],chart.qa,chart.da,chart.com,I,m.nq)
            def order(pos,rot,wrench,candidates):
                return rank(pos,rot,wrench,z['COM_world_m'][i+1],refrot[i+1],ref['wrench_world'][i],ref['valid_interval'][i],
                            chart.weight,source_rot,candidates,ik)
            if mode in ('single_model','shared_model','shared_verify8'):
                name='only1884' if mode=='single_model' else 'shared'
                with torch.inference_mode():
                    x=chart.features(state,d.qacc_warmstart,proposals)
                    y=models[name](torch.from_numpy(x)).numpy()
                pred=chart.decode(state,y)
                ranked=order(*pred,proposals);inferences+=64
                if mode=='shared_verify8':ids=np.r_[np.arange(4),ranked[ranked>=4][:4]]
            if mode=='random_verify8':
                rng=np.random.default_rng(np.random.SeedSequence([seed,i,810]))
                ids=np.r_[np.arange(4),rng.choice(np.arange(4,64),4,replace=False)]
            if mode in ('native64','shared_verify8','random_verify8'):
                cs=proposals[ids];n=len(ids)
                states,_=rollout.rollout(m,data,np.tile(state,(n,1)),control=np.repeat(cs[:,None],64,axis=1),
                    initial_warmstart=np.tile(d.qacc_warmstart,(n,1)))
                end=states[:,-1]
                pos,R,v1,L1=kinematics(end,chart.qa,chart.da,chart.com,I,m.nq)
                wrench=np.column_stack([m.body_mass[b]*((v1-v0)/.02-m.opt.gravity),(L1-L0)/.02])
                sel=order(pos,R,wrench,cs)[0];chosen=int(ids[sel]);budget+=n*64
            else:chosen=int(ranked[0])
            wall+=perf_counter()-clock
            command=proposals[chosen];d.ctrl[:]=command
            assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
            table=0.
            for _ in range(64):
                mujoco.mj_step(m,d)
                _,peak,_=sensor.hand_wrench(m,d);table=max(peak,table)
            observed=np.empty_like(state);mujoco.mj_getState(m,d,observed,sig)
            if mode in ('native64','shared_verify8','random_verify8'):
                np.testing.assert_array_equal(observed,end[sel])
            p,R,_,_=chart.motion(observed)
            mm=1000*np.linalg.norm(p-z['COM_world_m'][i+1]);deg=np.rad2deg((R*refrot[i+1].inv()).magnitude())
            log.append(dict(phase=i+1,choice=chosen,mm=mm,deg=deg,peak_table_N=table))
            qs.append(d.qpos.copy());commands.append(command.copy())
            previous=command.copy();residual=command-ik
            if mm>10 or deg>10 or table>.1:break
    result,trace=physics.replay(path,binding,z,base,np.asarray(commands),1.,1.)
    np.testing.assert_array_equal(qs,trace['qpos'])
    return dict(valid_s=result['valid_s'],full=result['right_censored'] and len(log)==len(direct),
        first_breach=result['first_breach'],controls=len(log),plan_native_ticks=budget,predictions=inferences,
        planning_wall_s=wall,decision_ms=1000*wall/len(log),log=log),trace


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    fit=file_pin(root/'mug_shared_step_v1/report.json')
    report=physics.read_json(fit);cfg=physics.read_json(report['intent'])
    inputs=cfg['inputs'];old=physics.load_arrays(inputs['reference']);binding=physics.read_json(inputs['binding'])
    path=verify_file_pin(inputs['model']);model=mujoco.MjModel.from_binary_path(str(path))
    torch.set_num_threads(1)
    models={}
    for name,pin in physics.read_json(report['frozen'])['pins'].items():
        ck=torch.load(verify_file_pin(pin),weights_only=True,map_location='cpu')
        net=MugStepHead(ck['n']);net.load_state_dict(ck['state']);models[name]=net.eval()
    regimes=('native64','single_model','shared_model','random_verify8','shared_verify8')
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(fit=fit,code=file_pin(__file__),seeds=[8391,8392],arms=regimes,
        scope='Fresh random direction seeds on same known HOT3D references, fixed nominal models, full continuous trajectories. Each decision generates 64 legal PD proposals. Native verification compares 64 or 8 using four CPU workers. Both shortlist arms retain first four options. Learned heads use CPU batch inference on current measured state, without future data. Exclude post-selection execution/replay from per-decision wall.'))
    rows=[]
    for name in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=physics.read_json(file_pin(root/name/'report.json'))['rows']
        ref=physics.load_arrays(item['reference']);T=ref['body']
        z={**old,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
            'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=z['feedforward_PD_target20'][0];direct=np.clip(ref['q'][1:]+base-z['initial_qpos27'][:20],*model.actuator_ctrlrange.T)
        for seed in (8391,8392):
            for mode in regimes:
                summary,trace=trial(model,path,binding,z,direct,base,seed,mode,models,4)
                p=out/f"{item['clip']}_{seed}_{mode}.npz";np.savez_compressed(p,**trace)
                rows.append(dict(clip=item['clip'],seed=seed,mode=mode,trace=file_pin(p),**summary))
                write_json(out/'progress.json',rows)
                print(item['clip'],seed,mode,summary['valid_s'],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
