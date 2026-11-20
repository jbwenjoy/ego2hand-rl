"""Compare a single same-parent PPO update on each task and inspect its cross-task effect."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from dex_supermimic.rl.dex_supermimic.ppo import ppo_update
from supermimic_research.learning.native_pen_ppo import model_pair,cpu_tree,tensor_tree_equal
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json

SEED=8421
CLIPS=('clip_001','clip_003')


def dist_stats(old,new,obs,actions=None,clip=.2):
    with torch.no_grad():
        a=old.distribution(obs,validate_args=False)
        b=new.distribution(obs,validate_args=False)
        kl=torch.distributions.kl_divergence(a,b).double().sum(-1)
        delta=(b.mean-a.mean).double()
        act_delta=(b.mean.clamp(-1,1)-a.mean.clamp(-1,1)).double()
        new_sd=b.stddev.double()
        result=dict(kl=float(kl.mean()),kl_p95=float(torch.quantile(kl,.95)),raw_mean_l2=float(delta.norm(dim=-1).mean()),
                    applied_mean_l2=float(act_delta.norm(dim=-1).mean()),std_l2=float((new_sd-a.stddev).norm(dim=-1).mean()),
                    mean_std_before=float(a.stddev.mean()),mean_std_after=float(b.stddev.mean()),n=len(obs))
        if actions is not None:
            ratio=(b.log_prob(actions)-a.log_prob(actions)).sum(-1).exp()
            result['ratio_outside_clip']=float(((ratio-1).abs()>clip).double().mean())
        else:
            # All start observations are saved; action components here describe world zero.
            result['first_mean_before']=a.mean[0].tolist();result['first_mean_after']=b.mean[0].tolist()
            result['first_std_before']=a.stddev[0].tolist();result['first_std_after']=b.stddev[0].tolist()
        assert torch.isfinite(kl).all() and torch.isfinite(delta).all()
        return result


def norm(parameters):
    return torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters if p.grad is not None]))


def main(data_dir,out):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    report_pin=file_pin(data_dir/'report.json')
    report=json.loads(verify_file_pin(report_pin).read_bytes())
    intent=json.loads(verify_file_pin(report['intent']).read_bytes())
    state=torch.load(verify_file_pin(intent['parent']),map_location='cpu',weights_only=False)
    cfg=SimpleNamespace(**state['config'])
    assert state['iteration']==2781 and state['optimizer_steps']==88992
    write_json(out/'intent.json',dict(source=report_pin,parent=intent['parent'],source_code=file_pin(__file__),ppo_source=file_pin(Path(ppo_update.__code__.co_filename)),
               variants=['inherited_joint','inherited_separate','fresh_joint','fresh_separate'],permutation_seed=SEED,
               scope='One independent rollout update (32 AdamW calls) on CPU from the verified parent per variant and source clip. Same action samples and permutation indices on each within-clip comparison. Other-clip metrics are changes on fixed observations, not new-policy rollouts. Fresh Adam clears the entire state and restarts its counter.'))
    memory={}
    old,base_critic,_=model_pair('cpu',cfg)
    old.load_state_dict(state['modules']['actor']);base_critic.load_state_dict(state['modules']['critic'])
    pre={}
    noise=[]
    for name in CLIPS:
        assert all(report['clips'][name]['health']['checks'].values())
        saved=torch.load(verify_file_pin(report['clips'][name]['batch']),map_location='cpu',weights_only=False)
        m=saved['memory']
        assert tuple(m['reward'].shape)==(64,32) and tuple(saved['bootstrap_obs'].shape)==(32,188) and tuple(saved['start_obs'].shape)==(32,161)
        with torch.no_grad():
            logs=[];values=[];eps=[]
            for o,c,x in zip(m['obs'],m['critic_obs'],m['actions']):
                dist=old.distribution(o,validate_args=False)
                logs.append(dist.log_prob(x).sum(-1));values.append(base_critic(c).reshape(32));eps.append((x-dist.mean)/dist.stddev)
            logerr=float((torch.stack(logs)-m['log_prob']).abs().max())
            verr=float((torch.stack(values)-m['value']).abs().max())
            noise.append(torch.stack(eps))
        if logerr>.002 or verr>.002:raise ValueError(f'Parent does not match stored rollout: {logerr} / {verr}')
        pre[name]=dict(logprob_max_error=logerr,value_max_error=verr)
        memory[name]=saved
    noise_max=float((noise[0]-noise[1]).abs().max())
    if noise_max>.005:raise ValueError(f'Noise mismatch {noise_max}')
    results={}
    original_clip=torch.nn.utils.clip_grad_norm_
    original_permutation=torch.randperm
    hashes=[]
    for train in CLIPS:
        saved=memory[train];variants={}
        for inherited in (True,False):
            for separate in (False,True):
                tag=('inherited' if inherited else 'fresh')+('_separate' if separate else '_joint')
                actor,critic,opt=model_pair('cpu',cfg);actor.load_state_dict(state['modules']['actor']);critic.load_state_dict(state['modules']['critic'])
                if inherited:opt.load_state_dict(copy.deepcopy(state['optimizer']))
                history=[];perm=hashlib.sha256()
                def permutation(*args,**kwargs):
                    index=original_permutation(*args,**kwargs);perm.update(index.numpy().tobytes());return index
                def clipping(params,max_norm,*args,**kwargs):
                    actor_p=tuple(actor.parameters());critic_p=tuple(critic.parameters());ps=tuple(params)
                    assert tuple(map(id,ps))==tuple(map(id,actor_p+critic_p))
                    an,cn=float(norm(actor_p)),float(norm(critic_p))
                    if separate:
                        original_clip(actor_p,max_norm,*args,**kwargs);original_clip(critic_p,max_norm,*args,**kwargs)
                        total=(an**2+cn**2)**.5
                    else:total=float(original_clip(ps,max_norm,*args,**kwargs))
                    history.append(dict(actor=an,critic=cn,actor_multiplier=min(1.,max_norm/((an if separate else total)+1e-6))))
                    return torch.tensor(total)
                torch.manual_seed(SEED)
                with patch.object(torch,'randperm',permutation),patch.object(torch.nn.utils,'clip_grad_norm_',clipping):
                    stats=ppo_update(actor,critic,opt,saved['memory'],saved['bootstrap_obs'],cfg)
                if stats['optimizer_steps']!=32 or len(history)!=32 or stats['omitted_transitions_per_epoch']:raise ValueError('Unexpected update count')
                if not tensor_tree_equal(old.norm.state_dict(),actor.norm.state_dict()):raise ValueError('Changed normalization')
                hashes.append(perm.hexdigest())
                model_path=out/f'{train}_{tag}.pt'
                torch.save(dict(actor=cpu_tree(actor.state_dict()),config=state['config'],metadata=state['metadata'],parent=intent['parent'],source_clip=train,tag=tag,
                                batch_backend=state['metadata']['termination_backend'],iteration=2782),model_path)
                scores={}
                for name,ds in memory.items():
                    m=ds['memory'];scores[name]=dict(start=dist_stats(old,actor,ds['start_obs']),rollout=dist_stats(old,actor,m['obs'].reshape(-1,161),m['actions'].reshape(-1,20),cfg.ppo_clip))
                variants[tag]=dict(weights=file_pin(model_path),effect=scores,gradients=history,permutation_hash=hashes[-1],ppo_stats={k:float(val) if isinstance(val,torch.Tensor) else val for k,val in stats.items()})
                print(train,tag,[(c,round(v['start']['applied_mean_l2'],4),round(v['rollout']['kl'],4)) for c,v in scores.items()],flush=True)
        results[train]=variants
    assert len(set(hashes))==1
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),inputs=pre,max_cross_clip_reconstructed_noise_error=noise_max,
                                     outcomes=results,optimizer_steps_per_branch=32,native_integrations=0,all_branch_permutation_hash=hashes[0]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    main(args.data.resolve(),args.output.resolve())
