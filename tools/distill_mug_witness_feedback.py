"""Distill pose and wrench planners on identical conditional correction examples."""
import argparse
from pathlib import Path
from time import perf_counter
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from test_strict_mug_feedback import VARIANTS,plant,correct,h3,MugStepChart
from supermimic_research.mechanics.mug_local_feedback import tangent_errors
from supermimic_research.mechanics.mug_residual_actor import MugResidualActor,observation
from supermimic_research.learning.world_model import atomic_checkpoint,raw_tree_equal


TRAIN={'mass105','mass095','mu095','mu105','start_x_plus_mm','start_x_minus_mm'}


def dataset(prior,fit,inputs):
    old=h3.physics.load_arrays(inputs['reference'])
    bind=h3.physics.read_json(inputs['binding']);asset=h3.verify_file_pin(inputs['model'])
    nom=mujoco.MjModel.from_binary_path(str(asset));chart=MugStepChart(nom,bind)
    sig=mujoco.mjtState.mjSTATE_FULLPHYSICS
    rows={k:[] for k in ('context','error','pose','wrench','clip','phase','condition')}
    for clip in (1884,1888):
        fp=h3.physics.load_arrays(fit['clips'][str(clip)]['data'])
        for rec in prior['rows']:
            if rec['clip']!=clip or rec['mode'] not in ('pose','wrench') or rec['condition']=='nominal':continue
            mass,mu,tau,offset=VARIANTS[rec['condition']]
            actual=plant(asset,bind,mass,mu,tau)
            d=mujoco.MjData(actual);d.qpos[:]=old['initial_qpos27'];d.qvel[:]=old['initial_qvel26'];d.ctrl[:]=old['feedforward_PD_target20'][0]
            d.qpos[chart.qa:chart.qa+3]+=Rotation.from_quat(d.qpos[chart.qa+np.array([4,5,6,3])]).apply([offset,0,0])
            mujoco.mj_forward(actual,d)
            trace=h3.physics.load_arrays(rec['trace']);previous=np.zeros(20)
            for i,(logged,issued) in enumerate(zip(rec['log'],trace['ctrl'],strict=True)):
                ref=fp['state'][i];cmd0=fp['cmd'][i];s=np.empty_like(ref);mujoco.mj_getState(actual,d,s,sig)
                ctx,err=observation(nom,ref,s,cmd0,previous)
                if logged['comparison'] is not None:
                    dx=tangent_errors(nom,ref,s[None])[0]
                    kin=correct('kinematic',chart,s,ref,dx,fp,i);lin=correct('native_gain',chart,s,ref,dx,fp,i)
                    pool=[np.zeros(20),.65*previous,.5*kin,kin,.25*lin,.65*lin]
                    rng=np.random.default_rng(np.random.SeedSequence([8541,i]))
                    noise=rng.normal(size=(3,20));noise/=np.sqrt(np.mean(noise**2,axis=1,keepdims=True))
                    amp=min(.09,.018+.003*np.linalg.norm(dx[chart.da:chart.da+3])/.001)
                    for n in noise:pool.extend([.6*previous+amp*n,.6*previous-amp*n])
                    cs=np.clip(cmd0+np.clip(pool,-.12,.12),*nom.actuator_ctrlrange.T)
                    labels=cs[logged['comparison']]
                else:labels=np.tile(cmd0,(2,1))
                np.testing.assert_array_equal(labels[int(rec['mode']=='wrench')],issued)
                for key,value in [('context',ctx),('error',err),('pose',labels[0]-cmd0),('wrench',labels[1]-cmd0),
                                  ('clip',clip),('phase',i+1),('condition',rec['condition'])]:
                    rows[key].append(value)
                d.ctrl[:]=issued;mujoco.mj_step(actual,d,nstep=64);np.testing.assert_array_equal(d.qpos,trace['qpos'][i])
                previous=issued-cmd0
    return {k:np.asarray(v) for k,v in rows.items()}


def run(out):
    root=Path(__file__).resolve().parents[1]/'runs';source=h3.file_pin(root/'mug_witness_mpc_v1/report.json')
    prior=h3.physics.read_json(source);cfg=h3.physics.read_json(prior['intent'])
    fit=h3.physics.read_json(cfg['fit']);inputs=h3.physics.read_json(fit['intent'])['inputs']
    out.mkdir(exist_ok=False);steps=600
    h3.write_json(out/'intent.json',dict(source=source,training_conditions=sorted(TRAIN),seed=8545,steps=steps,code=[
        h3.file_pin(__file__),h3.file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/mug_residual_actor.py')],
        scope='Rebuild both planner labels on the same trajectories and assert all executed commands/states. Only six single-factor scenarios enter fitting. Both models share clip-balanced minibatches. One CPU forward with no online planning, exact zero residual when state and previous command agree with the nominal witness. Last step fixed in advance.'))
    data=dataset(prior,fit,inputs);file=out/'teacher.npz';np.savez_compressed(file,**data)
    train=np.isin(data['condition'],list(TRAIN))
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device='cuda')
    x=tensor(data['context']);e=tensor(data['error']);models={};opts={};Y={}
    for arm in ('pose','wrench'):
        torch.manual_seed(8545);net=MugResidualActor(x.shape[1],e.shape[1]).cuda()
        net.mean.copy_(x[train].mean(0));net.scale.copy_(x[train].std(0,unbiased=False).clamp_min(.005))
        net.error_scale.copy_(e[train].square().mean(0).sqrt().clamp_min(.001))
        models[arm]=net;opts[arm]=torch.optim.Adam(net.parameters(),lr=.0007);Y[arm]=tensor(data[arm])
    assert raw_tree_equal(models['pose'].state_dict(),models['wrench'].state_dict())
    groups=[np.flatnonzero(train&(data['clip']==c)) for c in (1884,1888)]
    rng=np.random.default_rng(8545);schedule=torch.as_tensor(np.stack([np.r_[rng.choice(groups[0],32),rng.choice(groups[1],32)] for _ in range(steps)]),device='cuda')
    t0=perf_counter()
    for step,idx in enumerate(schedule,1):
        for a,n in models.items():
            pred=n(x[idx],e[idx]);loss=((pred-Y[a][idx])/.06).square().mean()
            opt=opts[a];opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(n.parameters(),2.,error_if_nonfinite=True);opt.step()
        if step==30:
            path=out/'first.pt';bundle=dict(step=step,schedule=schedule,models={a:dict(state=n.state_dict(),optimizer=opts[a].state_dict()) for a,n in models.items()})
            atomic_checkpoint(path,bundle);restored=torch.load(path,map_location='cuda',weights_only=True)
            assert raw_tree_equal(restored,bundle)
            for a,n in models.items():
                n.load_state_dict(restored['models'][a]['state']);opts[a].load_state_dict(restored['models'][a]['optimizer'])
            torch.cuda.synchronize();elapsed=perf_counter()-t0
            h3.write_json(out/'health.json',dict(step=step,checkpoint=h3.file_pin(path),exact_restore=True,elapsed_s=elapsed,remaining_fit_s=elapsed*(steps-step)/step))
    torch.cuda.synchronize();elapsed=perf_counter()-t0
    pins={};stats={}
    for a,n in models.items():
        path=out/(a+'.pt');atomic_checkpoint(path,dict(state=n.cpu().state_dict(),dims=[x.shape[1],e.shape[1]]))
        pins[a]=h3.file_pin(path)
    h3.write_json(out/'frozen.json',dict(weights=pins,teacher=h3.file_pin(file),fit_s=elapsed))
    for a,pin in pins.items():
        ck=torch.load(h3.verify_file_pin(pin),weights_only=True,map_location='cpu');net=MugResidualActor(*ck['dims']);net.load_state_dict(ck['state'])
        with torch.inference_mode():
            val=net(x.cpu(),e.cpu()).numpy()
            assert torch.count_nonzero(net(x.cpu(),torch.zeros_like(e.cpu())))==0
        stats[a]={}
        for c in (1884,1888):
            mask=(~train)&(data['clip']==c);err=val[mask]-data[a][mask]
            stats[a][str(c)]=dict(n=int(mask.sum()),student_rmse_rad=float(np.sqrt(np.mean(err**2))),
                open_rmse_rad=float(np.sqrt(np.mean(data[a][mask]**2))))
    h3.write_json(out/'report.json',dict(intent=h3.file_pin(out/'intent.json'),frozen=h3.file_pin(out/'frozen.json'),
        health=h3.file_pin(out/'health.json'),training_rows=int(train.sum()),heldout_rows=int((~train).sum()),heldout=stats))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);run(p.parse_args().out.resolve())
