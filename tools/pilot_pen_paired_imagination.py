"""Exploratory paired WM Actor pilot, including full native, unassisted endpoints."""
import argparse,copy,json,subprocess,sys,time,tomllib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from benchmark_pen_scaled import ROOT,OBJECTIVE,COLLISION
from supermimic_research.mechanics.pen_paired_imagination import imagined_objectives
from supermimic_research.mechanics.pen_load_forecast import LoadForecast
from supermimic_research.mechanics.pen_capacity_forecast import load_capacity_head
from supermimic_research.mechanics.pen_imagined_observation import pen_imagined_observation
from supermimic_research.learning.native_pen_ppo import model_pair,cpu_tree
from supermimic_research.learning.pen_matched_recovery import saved_bytes_equal
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources
from supermimic_research.strict_json import write_json

READ=lambda p:json.loads(verify_file_pin(p).read_bytes())
SEEDS=(7411,7412,7413)


def export(actor,base,path):
    st={k:v.detach().cpu().to(base['actor'][k].dtype) for k,v in actor.state_dict().items()}
    torch.save(dict(actor=st,config=base['config'],batch_backend=base['batch_backend'],iteration=base['iteration'],
                    scope='Frozen 2781 PPO warm start plus 24 offline model-gradient steps; iteration labels the PPO parent'),path)


def main(out):
    r=ROOT/'runs';out.mkdir(exist_ok=False);start=time.perf_counter()
    pins={'actor':file_pin(r/'pen_scaled_train_v1/actor_02781.pt'),'load':READ(file_pin(r/'pen_load_forecast_v1/report.json'))['outcomes']['state']['model'],
          'capacity':READ(file_pin(r/'pen_capacity_fit_v1/report.json'))['variants']['state']['model'],'starts':READ(file_pin(r/'pen_imagined_actor_starts_v1/report.json'))['pack']}
    base=torch.load(verify_file_pin(pins['actor']),weights_only=False,map_location='cpu')
    w=torch.load(verify_file_pin(pins['load']),weights_only=True,map_location='cpu')
    load=LoadForecast(torch.zeros(2,len(w['center'])),torch.zeros(2,len(w['out_center'])));load.load_state_dict(w);load=load.double().eval().requires_grad_(False)
    cap=load_capacity_head(torch.load(verify_file_pin(pins['capacity']),weights_only=True,map_location='cpu'),dtype=torch.float64,device='cpu').requires_grad_(False)
    reward_cfg=tomllib.loads((ROOT/'configs/pen_full_pose_native_v1.toml').read_text())['reward']
    with np.load(verify_file_pin(pins['starts'])) as f:
        t={k:torch.as_tensor(f[k]).double() for k in ['state55','previous_raw','previous_ema','ref_joints','ref_pose_wxyz','ref_links','mount_wxyz','encoder_bias','default_joint_pos','default_joint_vel','lower','upper','capacity_start','capacity_eligible','observation161']}
        for k in ('mass_kg','inertia_BODY','gravity_WORLD'):np.testing.assert_allclose(f[k],np.broadcast_to(f[k][0],f[k].shape))
        mass=torch.tensor(f['mass_kg'][0],dtype=torch.float64);inertia=torch.from_numpy(f['inertia_BODY'][0]).double();gravity=torch.from_numpy(f['gravity_WORLD'][0]).double()
        assert len(t['state55'])==30 and np.array_equal(f['qids'],np.broadcast_to(np.arange(7,27),f['qids'].shape))
    obs=pen_imagined_observation(t['state55'],t['previous_raw'],t['ref_joints'][:,0],t['ref_pose_wxyz'][:,0],t['ref_links'][:,0],t['mount_wxyz'],t['encoder_bias'],t['default_joint_pos'],t['default_joint_vel'])
    torch.testing.assert_close(obs,t['observation161'],rtol=0,atol=1e-6)
    frozen=lambda actor:imagined_objectives(actor,load,cap,t,reward_cfg,mass,inertia,gravity)
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/evaluate_pen_scaled.py',ROOT/'tools/benchmark_pen_scaled.py',OBJECTIVE,COLLISION]
    receipt=freeze_sources(files,out/'snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    write_json(out/'intent.json',dict(pins=pins,seeds=SEEDS,minibatch=10,updates=24,bptt_controls=3,starts=30,raw_trust_norm=.02,lr=.00003,
        arm_A1='Mean original COM/SO3/hand/action tracking reward minus 0.001 raw smoothness and 0.0001 normalized-digit-load squared.',
        arm_A2='A1 minus 0.2 geometry-capacity MAE, only at preregistered eligible phase windows.',
        scope='Diagnostic paired frozen-model fine-tuning. Same initialization, sampled ordering within pair, frozen torso/normalizer/other actions, eight mean output rows D2/D3, no critic updates. Sampling seeds are not independent pretraining runs. One reference, post-inspection actors. Two native numerical repeats per endpoint. No policy promotion.',native_case_order='alternating arm order by seed; original frozen baseline evaluated first'))
    pending=[('parent',pins['actor'])]
    torch.set_num_threads(2)
    for seed in SEEDS:
        rng=torch.Generator().manual_seed(seed)
        order=torch.cat([torch.randperm(30,generator=rng).view(3,10) for _ in range(8)])
        for label in (('A1','A2') if seed%2 else ('A2','A1')):
            actor,_,_=model_pair('cpu',SimpleNamespace(**base['config']));actor.load_state_dict(base['actor']);actor=actor.double().eval().requires_grad_(False)
            layer=actor.net[-1];layer.requires_grad_(True)
            with torch.no_grad():
                init=actor.distribution(obs).mean.clone();W=layer.weight.clone();B=layer.bias.clone()
                before={k:float(v) for k,v in frozen(actor).items()}
            mask=torch.zeros_like(layer.bias);mask[4:12]=1.
            opt=torch.optim.Adam([layer.weight,layer.bias],lr=.00003);history=[]
            for step,idx in enumerate(order,1):
                subset={k:v[idx] for k,v in t.items()}
                terms=imagined_objectives(actor,load,cap,subset,reward_cfg,mass,inertia,gravity)
                loss=-terms['base']+(0.2 if label=='A2' else 0.)*terms['capacity_error']
                opt.zero_grad();loss.backward()
                layer.weight.grad.mul_(mask[:,None]);layer.bias.grad.mul_(mask)
                grad=torch.nn.utils.clip_grad_norm_([layer.weight,layer.bias],1.,error_if_nonfinite=True)
                opt.step()
                # Exact projection on the training-state trust ball since only the final affine map changes.
                with torch.no_grad():
                    drift=float((actor.distribution(obs).mean-init).norm(dim=-1).max())
                    factor=min(1.,.02/max(drift,1e-15))
                    if factor<1:
                        layer.weight.copy_(W+(layer.weight-W)*factor);layer.bias.copy_(B+(layer.bias-B)*factor)
                    drift=float((actor.distribution(obs).mean-init).norm(dim=-1).max());assert drift<=.02000001
                history.append(dict(step=step,loss=float(loss.detach()),grad=float(grad),drift=drift))
                if step==1:
                    ck=out/f'{label}_{seed}_first.pt';torch.save(dict(actor=cpu_tree(actor.state_dict()),optimizer=cpu_tree(opt.state_dict()),step=1,schedule=order),ck)
                    recovered=torch.load(ck,map_location='cpu',weights_only=False)
                    assert saved_bytes_equal(recovered['actor'],cpu_tree(actor.state_dict())) and saved_bytes_equal(recovered['optimizer'],cpu_tree(opt.state_dict()))
                    actor.load_state_dict(recovered['actor']);opt.load_state_dict(recovered['optimizer'])
            with torch.no_grad():after={k:float(v) for k,v in frozen(actor).items()}
            key=f'{label}_{seed}';path=out/(key+'.pt');export(actor,base,path)
            write_json(out/(key+'.json'),dict(label=label,seed=seed,order=order.tolist(),before=before,after=after,history=history,actor=file_pin(path)))
            pending.append((key,file_pin(path)))
    write_json(out/'trained.json',dict(actors=pending,seconds=time.perf_counter()-start))
    evaluated=[]
    for key,pin in pending:
        # The evaluator reads the local source receipt; give the baseline the identical evaluation environment.
        if key=='parent':
            path=out/'parent.pt';torch.save(base,path)
        else:path=verify_file_pin(pin)
        target=out/('eval_'+key)
        with (out/('eval_'+key+'.log')).open('x') as f:
            subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/evaluate_pen_scaled.py'),str(path),str(target),'2'],stdout=f,stderr=subprocess.STDOUT,check=True)
        ev=json.loads((target/'report.json').read_bytes())
        evaluated.append(dict(key=key,actor=pin,evaluation=file_pin(target/'report.json'),cases=ev['cases']))
        write_json(out/'progress.json',dict(evaluated=evaluated))
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),training=file_pin(out/'trained.json'),evaluated=evaluated,seconds=time.perf_counter()-start))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);args=p.parse_args();main(args.out.resolve())
