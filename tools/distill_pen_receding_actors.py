"""Amortize two matched CPU planning teachers in the same frozen policy representation."""
import argparse
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
from scipy.stats import qmc
import torch
from audit_pen_filter_action_coverage import original_raw
from evaluate_pen_dev_clip import b,evaluate_cases,file_pin,read,verify_file_pin,write_json
from supermimic_research.learning.world_model import atomic_checkpoint,raw_tree_equal


ARMS=('pose','pose_support')
UPDATES=400


def pack(rows,frames):
    table={k:[] for k in ('obs','actor','pose','pose_support','clip','phase','seed')}
    for trial in rows:
        s=trial['case']; clip=trial['clip']; seed=trial['seed']
        n=frames[clip]
        dirs=(2*qmc.Sobol(20,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(8*n))))[:8*n]-1).reshape(-1,8,20)
        decisions=read(s['decisions'])
        with np.load(verify_file_pin(s['inputs'])) as data:
            for i,d in enumerate(decisions):
                actor=data['actor_raw'][i].clip(-1,1)
                pool=np.vstack([original_raw(actor,data['previous_raw'][i],list(range(12,16))),
                    np.clip(actor+.35*dirs[i],-1,1),np.clip(actor-.35*dirs[i],-1,1)]).astype(np.float32)
                np.testing.assert_array_equal(pool[int(d['choice'])],data['selected_raw'][i])
                for a in ARMS:table[a].append(pool[d['picks'][a]])
                for key,val in [('obs',data['obs'][i]),('actor',actor),('clip',clip),('phase',i+1),('seed',seed)]:
                    table[key].append(val)
    return {k:np.asarray(v) for k,v in table.items()}


def run(out):
    root=b.ROOT/'runs'
    first=file_pin(root/'pen_receding_support_v1/report.json')
    extra=file_pin(root/'pen_receding_seeds_v1/report.json')
    original=read(first)
    contract=read(read(read(original['intent'])['baseline'])['intent'])
    configs=read(contract['configs'])['clips']
    selected=[dict(seed=112025,clip=c,case=s) for c,cases in original['clips'].items() for s in cases if s['rep']==0]
    selected+=[v for v in read(extra)['rows'] if v['seed']==112026 and v['mode'] in ARMS]
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(original=first,extra=extra,actor=contract['actor'],code=file_pin(__file__),
        source_seeds=[112025,112026],updates=UPDATES,lr=.001,seed=112031,batch_per_clip=32,
        scope='Same visited states, both teacher labels recomputed from the logged ranks and verified proposal pool. Balance clips 002/003 per optimizer step. Fit only final mean rows, preserve torso/normalizer/std. Held-out phase blocks in both clips, one-control border omitted. No native outcome used to choose a checkpoint. W1 parent and students from frame zero, including original clip001 for retention.'))
    dataset=pack(selected,{k:v['item']['frames'] for k,v in configs.items()})
    path=out/'teacher.npz'
    np.savez_compressed(path,**dataset)
    phase=dataset['phase']
    block=(phase-1)%20
    val=(block>=6)&(block<=9)
    train=(block<5)|(block>10)
    rng=np.random.default_rng(112031)
    ids={c:np.flatnonzero(train&(dataset['clip']==c)) for c in configs}
    batches=np.stack([np.r_[rng.choice(ids['clip_002'],32),rng.choice(ids['clip_003'],32)] for _ in range(UPDATES)])
    device='cuda'
    tensor=lambda x:torch.as_tensor(x,device=device)
    st=torch.load(verify_file_pin(contract['actor']),map_location='cpu',weights_only=False)
    modes={};opts={};targets={a:tensor(dataset[a]) for a in ARMS}
    for a in ARMS:
        actor,_,_=b.model_pair(device,SimpleNamespace(**st['config']))
        actor.load_state_dict(st['actor']);actor.eval().requires_grad_(False)
        layer=actor.net[-1];layer.requires_grad_(True)
        modes[a]=actor;opts[a]=torch.optim.Adam(layer.parameters(),lr=.001)
    # Fixed torso permits one resident feature evaluation for all supervised updates.
    actor=modes['pose'];obs=tensor(dataset['obs'])
    with torch.no_grad():
        features=actor.net[:-1](actor.norm(obs))
        base_mean=actor.net[-1](features)[:,:20]
        np.testing.assert_allclose(base_mean.clamp(-1,1).cpu().numpy(),dataset['actor'],atol=1e-4,rtol=0)
    def predict(a,ix):
        return modes[a].net[-1](features[ix])[:,:20]
    def update(a,ix):
        opt=opts[a]
        loss=(predict(a,ix)-targets[a][ix]).square().mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(modes[a].net[-1].parameters(),1.,error_if_nonfinite=True);opt.step()
    start=time.perf_counter();idx=tensor(batches)
    for i,ix in enumerate(idx,1):
        for a in ARMS:update(a,ix)
        if i==20:
            paired={a:dict(layer=modes[a].net[-1].state_dict(),adam=opts[a].state_dict()) for a in ARMS}
            cp=out/'first.pt'
            atomic_checkpoint(cp,dict(step=i,paired=paired,batches=torch.as_tensor(batches),parent=contract['actor'],teacher=file_pin(path)))
            restored=torch.load(cp,map_location=device,weights_only=True)
            assert raw_tree_equal(restored['paired'],paired)
            for a in ARMS:
                modes[a].net[-1].load_state_dict(restored['paired'][a]['layer'])
                opts[a].load_state_dict(restored['paired'][a]['adam'])
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            write_json(out/'health.json',dict(first=file_pin(cp),step=i,weight_and_adam_exact=True,estimated_remaining_s=elapsed*(UPDATES-i)/i))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    actors={'parent':contract['actor']};metrics={}
    for a,actor in modes.items():
        ck=out/(a+'.pt')
        last_params={f'net.{len(actor.net)-1}.weight',f'net.{len(actor.net)-1}.bias'}
        for name,v in actor.state_dict().items():
            before=st['actor'][name]
            if name not in last_params:
                assert torch.equal(v.cpu(),before)
            else:assert torch.equal(v[20:].cpu(),before[20:])
        atomic_checkpoint(ck,{**st,'actor':{k:v.cpu() for k,v in actor.state_dict().items()}})
        actors[a]=file_pin(ck)
        with torch.no_grad():
            pred=predict(a,slice(None)).clamp(-1,1)
            metrics[a]={}
            for clip in configs:
                mask=tensor(val & (dataset['clip']==clip))
                metrics[a][clip]=dict(n=int(mask.sum()),parent_rmse=float((base_mean[mask].clamp(-1,1)-targets[a][mask]).square().mean().sqrt()),
                    student_rmse=float((pred[mask]-targets[a][mask]).square().mean().sqrt()))
    write_json(out/'frozen.json',dict(actors=actors,seconds=elapsed,teacher=file_pin(path),metrics=metrics,train_rows=int(train.sum()),val_rows=int(val.sum())))
    results={}
    for clip in ('clip_002','clip_003','clip_001'):
        folder=out/clip;folder.mkdir()
        if clip=='clip_001':
            obj=b.OBJECTIVE;item=read(file_pin(root/'pen_catalog_screen_v2/report.json'))['results'][clip]
            evaluation=b.ROOT/'configs/controlled_evaluation_v4.toml'
        else:
            cfg=configs[clip];obj=verify_file_pin(cfg['objective']);item=cfg['item'];evaluation=verify_file_pin(cfg['evaluation'])
        shutil.copyfile(evaluation,folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(frozen=file_pin(out/'frozen.json'),clip=clip))
        rows=evaluate_cases(folder,obj,item,actors,dict.fromkeys(actors,0.),2,custom_start=(clip!='clip_001'),refit_export=True)
        results[clip]=[dict(arm=r['arm'],rep=r['rep'],complete=r['stop']=='REFERENCE_COMPLETION',strict=r['summary']['qualified'],
            passing_s=.02*(r['controls']-int(r['stop']!='REFERENCE_COMPLETION')),stop=r['stop'],native=r['case']) for r in rows]
        write_json(out/'progress.json',results)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),frozen=file_pin(out/'frozen.json'),native=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
