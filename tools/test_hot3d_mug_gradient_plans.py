"""Test model-gradient proposals against matched random proposals at eight native verifications."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from check_hot3d_mug_gradients import differentiable_cost
from test_mug_learned_planning import (rank,kinematics,propose,MugStepChart,MugStepHead,MugContactImpulse,
    physics,file_pin,verify_file_pin,write_json,mean_reference_wrench)


def run_case(m,path,binding,z,direct,base,net,clip,seed,mode):
    sensor=MugContactImpulse(m,binding);chart=MugStepChart(m,binding)
    b=chart.body;mass=m.body_mass[b]
    basis=Rotation.from_quat(m.body_iquat[b,[1,2,3,0]]).as_matrix()
    I=basis@np.diag(m.body_inertia[b])@basis.T
    ref=mean_reference_wrench(z['COM_world_m'],z['quaternion_wxyz'],mass,I,m.opt.gravity)
    refrot=Rotation.from_quat(z['quaternion_wxyz'][:,[1,2,3,0]])
    d=mujoco.MjData(m);d.qpos[:]=z['initial_qpos27'];d.qvel[:]=z['initial_qvel26'];d.ctrl[:]=base
    mujoco.mj_forward(m,d);previous=base.copy();residual=np.zeros(20)
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS;workers=[mujoco.MjData(m) for _ in range(4)]
    to_tensor=lambda x:torch.as_tensor(x,dtype=torch.float64)
    records=[];qpos=[];commands=[];wall=0.
    with Rollout(nthread=4) as roll:
        for i,ik in enumerate(direct):
            clock=perf_counter()
            state=np.empty(mujoco.mj_stateSize(m,sig));mujoco.mj_getState(m,d,state,sig)
            pos,r0,v,wb=chart.motion(state)
            wtarget=ref['wrench_world'][i];valid=ref['valid_interval'][i]
            raw=propose(np.random.default_rng(np.random.SeedSequence([seed,i,0])),64,ik,base,previous,residual,m.actuator_ctrlrange)
            # The input map is affine in commanded PD at a fixed measured state.
            # Evaluate its exact local linear map via twenty basis displacements.
            sampled=chart.features(state,d.qacc_warmstart,np.vstack([raw,ik,ik+np.eye(20)]))
            with torch.no_grad():pred=net(to_tensor(sampled[:64])).numpy()
            def order(poses,rots,loads,commands):
                return rank(poses,rots,loads,z['COM_world_m'][i+1],refrot[i+1],wtarget,valid,chart.weight,r0,commands,ik)
            ordering=order(*chart.decode(state,pred),raw)
            extra=ordering[ordering>=4][:4]
            cs=raw[np.r_[np.arange(4),extra]].copy()
            if mode!='subset':
                centers=cs[4:6].copy()
                now=centers.copy()
                x0=to_tensor(sampled[64]);jac=to_tensor(sampled[65:].astype(float)-sampled[64].astype(float))
                target_pos=to_tensor(r0.inv().apply(z['COM_world_m'][i+1]-pos-.02*v)/.01)
                target_q=to_tensor(((r0*Rotation.from_rotvec(.02*wb)).inv()*refrot[i+1]).as_quat())
                target_w=to_tensor((r0.inv().apply(wtarget.reshape(2,3))/np.array([[chart.weight],[chart.weight*.05]])).ravel())
                rng=np.random.default_rng(np.random.SeedSequence([seed,i,611]))
                # Random control has the same opportunity for a persistent direction.
                random_dir=rng.normal(size=centers.shape);random_dir/=np.linalg.norm(random_dir,axis=1,keepdims=True)
                for _ in range(4):
                    ctrl=to_tensor(now).requires_grad_(True);delta=ctrl-to_tensor(ik)
                    estimate=net(x0+delta@jac)
                    pose,wrench=differentiable_cost(estimate,target_pos,target_q,target_w,float(valid))
                    total=pose+.2*wrench+.01*((delta/.12)**2).mean(1)
                    grad=torch.autograd.grad(total.sum(),ctrl)[0].numpy()
                    direction=(-grad/np.maximum(np.linalg.norm(grad,axis=1,keepdims=True),1e-12)
                               if mode=='gradient' else random_dir)
                    now=np.clip(now+.03*direction,*m.actuator_ctrlrange.T)
                cs[-2:]=now
            states,_=roll.rollout(m,workers,np.tile(state,(8,1)),control=np.repeat(cs[:,None],64,axis=1),
                initial_warmstart=np.tile(d.qacc_warmstart,(8,1)))
            end=states[:,-1];ep,eR,v1,L1=kinematics(end,chart.qa,chart.da,chart.com,I,m.nq)
            _,_,v0,L0=kinematics(state[None],chart.qa,chart.da,chart.com,I,m.nq)
            wrench=np.column_stack([mass*((v1-v0)/.02-m.opt.gravity),(L1-L0)/.02])
            selected=int(order(ep,eR,wrench,cs)[0]);cmd=cs[selected]
            wall+=perf_counter()-clock
            d.ctrl[:]=cmd
            table=0.
            assert not d.xfrc_applied.any() and not d.qfrc_applied.any()
            for _ in range(64):
                mujoco.mj_step(m,d)
                _,peak,_=sensor.hand_wrench(m,d);table=max(table,peak)
            live=np.empty_like(state);mujoco.mj_getState(m,d,live,sig)
            np.testing.assert_array_equal(live,end[selected])
            mm=1000*np.linalg.norm(ep[selected]-z['COM_world_m'][i+1])
            deg=np.rad2deg((eR[selected]*refrot[i+1].inv()).magnitude())
            records.append(dict(phase=i+1,slot=selected,modified=mode!='subset' and selected>=6,mm=mm,deg=deg,table_N=table))
            qpos.append(d.qpos.copy());commands.append(cmd.copy())
            previous=cmd;residual=cmd-ik
            if mm>10 or deg>10 or table>.1:break
    detail,trace=physics.replay(path,binding,z,base,np.array(commands),1.,1.)
    np.testing.assert_array_equal(qpos,trace['qpos'])
    return dict(clip=clip,seed=seed,mode=mode,valid_s=detail['valid_s'],full=detail['right_censored'] and len(records)==len(direct),
                controls=len(records),decision_wall_s=wall,native_plan_ticks=512*len(records),records=records),trace


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs'
    fit=file_pin(root/'mug_shared_step_v1/report.json');record=physics.read_json(fit)
    cfg=physics.read_json(record['intent']);inputs=cfg['inputs']
    pin=physics.read_json(record['frozen'])['pins']['shared']
    ck=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=True)
    net=MugStepHead(ck['n']);net.load_state_dict(ck['state']);net.double().requires_grad_(False).eval();torch.set_num_threads(1)
    path=verify_file_pin(inputs['model']);m=mujoco.MjModel.from_binary_path(str(path))
    old=physics.load_arrays(inputs['reference']);binding=physics.read_json(inputs['binding'])
    seeds=tuple(range(8401,8407))
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(fit=fit,gradient_check=file_pin(root/'mug_native_gradient_check_v1/report.json'),
        model=pin,code=file_pin(__file__),seeds=seeds,scope='New seeds fixed after the local gradient test. Common four anchor actions and top two learned non-anchor options. Slots 7/8 either next two by learned rank, four 0.03-rad steps from the top two in a random unit direction, or four recomputed model-gradient steps. Same eight native forecasts. Compare paired survival per clip, below/above chance and first breach type. No test-time model update.'))
    results=[]
    for name in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=physics.read_json(file_pin(root/name/'report.json'))['rows']
        ref=physics.load_arrays(item['reference']);T=ref['body']
        z={**old,'elapsed_s':ref['t'],'COM_world_m':T[:,:3,3]+T[:,:3,:3]@old['COM_BODY_m'],
           'quaternion_wxyz':Rotation.from_matrix(T[:,:3,:3]).as_quat()[:,[3,0,1,2]]}
        base=old['feedforward_PD_target20'][0];ik=np.clip(ref['q'][1:]+base-z['initial_qpos27'][:20],*m.actuator_ctrlrange.T)
        for seed in seeds:
            for mode in ('subset','random','gradient'):
                case,trace=run_case(m,path,binding,z,ik,base,net,item['clip'],seed,mode)
                dest=out/f"{item['clip']}_{seed}_{mode}.npz";np.savez_compressed(dest,**trace)
                results.append(dict(**case,trace=file_pin(dest)))
            write_json(out/'progress.json',results)
            print(item['clip'],seed,[x['valid_s'] for x in results[-3:]],flush=True)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),rows=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
