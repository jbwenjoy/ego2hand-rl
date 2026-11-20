"""Fit a shared action-conditioned 20/40/60 ms model using contiguous native continuations."""
import argparse
from pathlib import Path
from time import perf_counter
import hashlib
import mujoco
from mujoco.rollout import Rollout
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.nn import functional as F
from supermimic_research.mechanics.mug_horizon import continuations,features,targets,MugHorizonHead
from fit_hot3d_mug_step import atomic_checkpoint,raw_tree_equal,MugStepChart,physics,file_pin,verify_file_pin,write_json
from search_mug_feasible_paths import kinematics


def collect(rows,seed,chart,old,refs,skip):
    m=chart.model;b=chart.body
    inertial=Rotation.from_quat(m.body_iquat[b,[1,2,3,0]]).as_matrix()
    I=inertial@np.diag(m.body_inertia[b])@inertial.T
    buffer=[];n_excluded=0;ticks=0;workers=[mujoco.MjData(m) for _ in range(4)]
    with Rollout(nthread=4) as engine:
        for row in rows:
            if row['seed']!=seed:continue
            direct=refs[row['clip']]
            with np.load(verify_file_pin(row['bank'])) as f:ar=dict(f)
            for i,parent in enumerate(ar['parent']):
                states=ar['source'][i];warm=ar['warmstart'][i];cmd=continuations(ar['cmd'][i],direct,i,m.actuator_ctrlrange)
                sequence,_=engine.rollout(m,workers,states,control=np.repeat(cmd,64,axis=1),initial_warmstart=warm)
                np.testing.assert_array_equal(sequence[:,63],ar['end'][i]);ticks+=len(cmd)*192
                _,_,v0,L0=kinematics(states,chart.qa,chart.da,chart.com,I,m.nq)
                labels=[]
                for h in (1,2,3):
                    end=sequence[:,64*h-1]
                    _,_,v,L=kinematics(end,chart.qa,chart.da,chart.com,I,m.nq)
                    net=np.column_stack([m.body_mass[b]*((v-v0)/(.02*h)-m.opt.gravity),(L-L0)/(.02*h)])
                    if h==1:np.testing.assert_allclose(net,ar['net_wrench'][i],rtol=0,atol=1e-11)
                    labels.append((h,end,net))
                for p in np.unique(parent):
                    mask=parent==p;s=states[mask][0];w=warm[mask][0]
                    key=hashlib.sha256(s[1:].tobytes()+w.tobytes()).hexdigest()
                    if key in skip:n_excluded+=1;continue
                    x=features(chart,s,w,cmd[mask]);y=np.column_stack([targets(chart,s,end[mask],net[mask],h) for h,end,net in labels])
                    buffer.append(dict(x=x,y=y,clip=row['clip'],key=key))
    return buffer,n_excluded,ticks


def main(out):
    r=Path(__file__).resolve().parents[1]/'runs'
    src=file_pin(r/'mug_feasible_paths_v2/report.json')
    bank=physics.read_json(src)
    cfg=physics.read_json(physics.read_json(bank['intent'])['source']);inputs=cfg['inputs']
    old=physics.load_arrays(inputs['reference']);binding=physics.read_json(inputs['binding'])
    m=mujoco.MjModel.from_binary_path(str(verify_file_pin(inputs['model'])));chart=MugStepChart(m,binding)
    refs={}
    for folder in ('mug_annotated_windows_original_v1','mug_unilateral_interval_v1'):
        item,=physics.read_json(file_pin(r/folder/'report.json'))['rows'];ref=physics.load_arrays(item['reference'])
        refs[item['clip']]=np.clip(ref['q'][1:]+old['feedforward_PD_target20'][0]-old['initial_qpos27'][:20],*m.actuator_ctrlrange.T)
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(bank=src,inputs=inputs,train_seed=8311,test_seed=8312,training_rng=8420,
        code=[file_pin(__file__),file_pin(Path(__file__).resolve().parents[1]/'src/supermimic_research/mechanics/mug_horizon.py')],
        scope='Simulate contiguous 20/40/60ms endpoints per command with 0.8 residual decay and known IK references. Shared source-relative pose residuals and accumulated net wrenches about COM. Seed8312 validation excluded from fitting, remove identical train state/warm starts. Full final weights frozen before test. 900 Adam updates, same-state centered response supervision.'))
    tr,_,ticks=collect(bank['rows'],8311,chart,old,refs,set())
    torch.set_num_threads(1);torch.manual_seed(8420)
    T=lambda x:torch.as_tensor(x,device='cuda')
    x=T(np.concatenate([g['x'] for g in tr]));y=T(np.concatenate([g['y'] for g in tr]))
    offsets=np.cumsum([0]+[len(g['x']) for g in tr]);groups=[[i for i,g in enumerate(tr) if g['clip']==c] for c in (1884,1888)]
    rng=np.random.default_rng(8420)
    schedule=T(np.array([[rng.integers(offsets[i],offsets[i+1],size=12) for i in np.r_[rng.choice(groups[0],4),rng.choice(groups[1],4)]] for _ in range(900)]))
    net=MugHorizonHead(x.shape[1]).cuda();net.mean.copy_(x.mean(0));net.scale.copy_(x.std(0,unbiased=False).clamp_min(.0005))
    net.ymean.copy_(y.mean(0));net.yscale.copy_(y.std(0,unbiased=False).clamp_min(.001))
    opt=torch.optim.Adam(net.parameters(),lr=.001);t0=perf_counter()
    for k,idx in enumerate(schedule,1):
        delta=(net(x[idx])-y[idx])/net.yscale
        loss=F.smooth_l1_loss(delta,torch.zeros_like(delta))+F.smooth_l1_loss(delta-delta.mean(1,keepdim=True),torch.zeros_like(delta))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5.,error_if_nonfinite=True);opt.step()
        if k==30:
            first=out/'first.pt';payload=dict(net=net.state_dict(),opt=opt.state_dict(),step=k,schedule=schedule)
            atomic_checkpoint(first,payload);copy=torch.load(first,map_location='cuda',weights_only=True)
            assert raw_tree_equal(copy,payload)
            net.load_state_dict(copy['net']);opt.load_state_dict(copy['opt']);torch.cuda.synchronize()
            elapsed=perf_counter()-t0
            write_json(out/'health.json',dict(first=file_pin(first),step=k,weights_optimizer_schedule_exact=True,loss=float(loss),
                estimated_remaining_fit_seconds=elapsed*(900-k)/k))
    net=net.cpu().eval();elapsed=perf_counter()-t0
    path=out/'final.pt';atomic_checkpoint(path,dict(state=net.state_dict(),n=len(net.mean),step=900))
    frozen=file_pin(path)
    te,excl,te_ticks=collect(bank['rows'],8312,chart,old,refs,{g['key'] for g in tr})
    scores={}
    with torch.no_grad():
        for clip in (1884,1888):
            errors=[];targets_all=[]
            for g in te:
                if g['clip']==clip:
                    errors.extend(net(torch.from_numpy(g['x'])).numpy()-g['y']);targets_all.extend(g['y'])
            e=np.array(errors).reshape(-1,3,12);cv=np.array(targets_all).reshape(-1,3,12)
            scores[str(clip)]=dict(n=len(e),
                pos_RMS_mm=(10*np.sqrt((e[:,:,:3]**2).sum(2).mean(0))).tolist(),
                cv_pos_RMS_mm=(10*np.sqrt((cv[:,:,:3]**2).sum(2).mean(0))).tolist(),
                rot_vector_RMS_deg=(10*np.sqrt((e[:,:,3:6]**2).sum(2).mean(0))).tolist(),
                cv_rot_vector_RMS_deg=(10*np.sqrt((cv[:,:,3:6]**2).sum(2).mean(0))).tolist())
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),model=frozen,first_health=file_pin(out/'health.json'),
        train_steps=900,fit_wall_s=elapsed,training_native_ticks=ticks,test_native_ticks=te_ticks,excluded_groups=excl,heldout=scores))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);main(p.parse_args().out.resolve())
