"""Matched support-teacher distillation with original-policy action rehearsal."""
import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from distill_pen_receding_actors import b,read,file_pin,verify_file_pin,write_json,evaluate_cases,atomic_checkpoint,raw_tree_equal,UPDATES


def run(out):
    root=b.ROOT/'runs'
    reference=file_pin(root/'pen_distilled_receding_v1/report.json')
    prior=read(reference)
    frozen=read(prior['frozen'])
    original=read(read(read(read(prior['intent'])['original'])['intent'])['baseline'])
    configs=read(read(original['intent'])['configs'])['clips']
    legacy=read(file_pin(root/'pen_catalog_screen_v2/report.json'))['results']['clip_001']
    parent=frozen['actors']['parent']
    out.mkdir(exist_ok=False)
    write_json(out/'intent.json',dict(previous=reference,teacher=frozen['teacher'],parent=parent,code=file_pin(__file__),
        penalties=[1,10],updates_per_arm=UPDATES,scope='Same support teacher, development split and minibatches as unregularized last-layer fit. Add mean-squared raw-output drift on 32 original clip001 parent observations per update. Fresh W1 on all three clips, no evaluation checkpoint selection.'))
    folder=out/'rehearsal_source';folder.mkdir()
    shutil.copyfile(b.ROOT/'configs/controlled_evaluation_v4.toml',folder/'controlled_evaluation_v4.toml')
    write_json(folder/'intent.json',dict(master=file_pin(out/'intent.json')))
    obs=[];raw=[]
    old_pair=b.model_pair
    def record(*args,**kwargs):
        actor,critic,opt=old_pair(*args,**kwargs)
        distribution=actor.distribution
        def sample(x):
            dist=distribution(x)
            obs.append(x[0].detach().cpu().numpy().copy())
            raw.append(dist.mean[0].detach().cpu().numpy().copy())
            return dist
        actor.distribution=sample
        return actor,critic,opt
    with patch.object(b,'model_pair',record):
        row,=evaluate_cases(folder,b.OBJECTIVE,legacy,{'parent':parent},{'parent':0.},1,custom_start=False,refit_export=True)
    if row['stop']!='REFERENCE_COMPLETION' or len(obs)!=row['controls']:raise ValueError('Incomplete rehearsal trajectory')
    sample=folder/'parent.npz';np.savez_compressed(sample,obs=obs,raw=raw)
    st=torch.load(verify_file_pin(parent),map_location='cpu',weights_only=False)
    with np.load(verify_file_pin(frozen['teacher'])) as f:data=dict(f)
    block=(data['phase']-1)%20; train=(block<5)|(block>10)
    ids={c:np.flatnonzero(train&(data['clip']==c)) for c in configs}
    rng=np.random.default_rng(112031)
    batches=np.stack([np.r_[rng.choice(ids['clip_002'],32),rng.choice(ids['clip_003'],32)] for _ in range(UPDATES)])
    anchor=np.random.default_rng(112032).integers(0,len(obs),size=(UPDATES,32))
    def tensor(x):return torch.as_tensor(np.asarray(x),device='cuda')
    schedule=tensor(batches);keep=tensor(anchor)
    actors={'parent':parent,'no_rehearsal':frozen['actors']['pose_support']}
    fits={}
    for weight in (1,10):
        a,_,_=b.model_pair('cuda',SimpleNamespace(**st['config']))
        a.load_state_dict(st['actor']);a.eval().requires_grad_(False)
        layer=a.net[-1];layer.requires_grad_(True)
        with torch.no_grad():
            X=a.net[:-1](a.norm(tensor(data['obs'])))
            K=a.net[:-1](a.norm(tensor(obs)))
            anchor_raw=layer(K)[:,:20].detach()
            torch.testing.assert_close(anchor_raw,tensor(raw),atol=1e-4,rtol=0.)
        opt=torch.optim.Adam(layer.parameters(),lr=.001)
        Y=tensor(data['pose_support'])
        name=f'rehearsal_{weight}'
        for i,(train_ids,keep_ids) in enumerate(zip(schedule,keep),1):
            loss=(layer(X[train_ids])[:,:20]-Y[train_ids]).square().mean()+weight*(layer(K[keep_ids])[:,:20]-anchor_raw[keep_ids]).square().mean()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(layer.parameters(),1.,error_if_nonfinite=True);opt.step()
            if i==20:
                path=out/f'{name}_first.pt'
                payload=dict(step=i,layer=layer.state_dict(),optimizer=opt.state_dict(),teacher=frozen['teacher'],parent=parent,
                             train_schedule=torch.from_numpy(batches),rehearsal_schedule=torch.from_numpy(anchor))
                atomic_checkpoint(path,payload)
                loaded=torch.load(path,weights_only=True,map_location='cuda')
                assert raw_tree_equal(loaded,payload)
                layer.load_state_dict(loaded['layer']);opt.load_state_dict(loaded['optimizer'])
        with torch.no_grad():
            drift=layer(K)[:,:20].clamp(-1,1)-anchor_raw.clamp(-1,1)
            fits[name]=dict(rms_on_rehearsal=float(drift.square().mean().sqrt()),max_on_rehearsal=float(abs(drift).max()),
                            first=file_pin(out/f'{name}_first.pt'))
        path=out/f'{name}.pt'
        atomic_checkpoint(path,{**st,'actor':{k:v.cpu() for k,v in a.state_dict().items()}});actors[name]=file_pin(path)
    write_json(out/'frozen.json',dict(actors=actors,source=file_pin(sample),fits=fits))
    results={}
    for clip in ('clip_002','clip_003','clip_001'):
        folder=out/clip;folder.mkdir()
        if clip=='clip_001':
            obj=b.OBJECTIVE;item=legacy;ev=b.ROOT/'configs/controlled_evaluation_v4.toml'
        else:
            cfg=configs[clip];obj=verify_file_pin(cfg['objective']);item=cfg['item'];ev=verify_file_pin(cfg['evaluation'])
        shutil.copyfile(ev,folder/'controlled_evaluation_v4.toml')
        write_json(folder/'intent.json',dict(frozen=file_pin(out/'frozen.json'),clip=clip))
        rows=evaluate_cases(folder,obj,item,actors,dict.fromkeys(actors,0.),2,custom_start=(clip!='clip_001'),refit_export=True)
        results[clip]=[dict(arm=r['arm'],rep=r['rep'],passing_s=.02*(r['controls']-int(r['stop']!='REFERENCE_COMPLETION')),
            complete=r['stop']=='REFERENCE_COMPLETION',strict=r['summary']['qualified'],native=r['case']) for r in rows]
        write_json(out/'progress.json',results)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),native=results,frozen=file_pin(out/'frozen.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    run(p.parse_args().out.resolve())
