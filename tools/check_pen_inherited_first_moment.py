"""Separate stored Adam first-moment drift from gradients of one new rollout."""
import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
import json
import torch

from compare_pen_single_update_interference import CLIPS,ROOT,SEED,dist_stats
from dex_supermimic.rl.dex_supermimic.ppo import ppo_update
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.learning.native_pen_ppo import cpu_tree,model_pair
from supermimic_research.strict_json import write_json


def make(parent,cfg):
    a,c,opt=model_pair('cpu',cfg)
    a.load_state_dict(parent['modules']['actor']);c.load_state_dict(parent['modules']['critic'])
    opt.load_state_dict(copy.deepcopy(parent['optimizer']))
    return a,c,opt


def drift_vector(actor,old):
    return torch.cat([(p-q).detach().view(-1).double() for p,q in zip(actor.parameters(),old.parameters())])


def main(data_dir,comparison,out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    compare_pin=file_pin(comparison/'report.json');report=json.loads(verify_file_pin(compare_pin).read_bytes())
    meta=json.loads(verify_file_pin(report['intent']).read_bytes());source=file_pin(data_dir/'report.json')
    if meta['source']!=source or meta['permutation_seed']!=SEED:raise ValueError('Require first matched update comparison')
    parent=torch.load(verify_file_pin(meta['parent']),weights_only=False,map_location='cpu');cfg=SimpleNamespace(**parent['config'])
    assert cfg.weight_decay==0.
    inputs=json.loads(verify_file_pin(source).read_bytes());inputs={n:torch.load(verify_file_pin(inputs['clips'][n]['batch']),map_location='cpu',weights_only=False) for n in CLIPS}
    write_json(out/'intent.json',dict(previous=compare_pin,source=source,script=file_pin(__file__),ablations=['zero_actor_first','zero_critic_first','zero_both_first'],
        scope='Preserve inherited second moments and step counter. One independent CPU PPO update from the parent, same fixed 32 minibatches and joint clipping. Null branch manually supplies zero Actor gradients for 32 AdamW steps, no decay or Critic update.'))
    old,_,_=make(parent,cfg);null,_,null_opt=make(parent,cfg)
    for _ in range(32):
        null_opt.zero_grad(set_to_none=True)
        for param in null.parameters():param.grad=torch.zeros_like(param)
        null_opt.step()
    null_vector=drift_vector(null,old)
    def scores(model):
        return {name:dict(start=dist_stats(old,model,data['start_obs']),rollout=dist_stats(old,model,data['memory']['obs'].reshape(-1,161),data['memory']['actions'].reshape(-1,20),cfg.ppo_clip)) for name,data in inputs.items()}
    results={}
    for name in CLIPS:
        baseline_pin=report['outcomes'][name]['inherited_joint']['weights']
        base,_,_=make(parent,cfg);base.load_state_dict(torch.load(verify_file_pin(baseline_pin),map_location='cpu',weights_only=False)['actor'])
        native_update=drift_vector(base,old);magnitude=native_update.norm()
        results[name]=dict(base=baseline_pin,baseline_parameter_delta_norm=float(magnitude),null_fraction=float(null_vector.norm()/magnitude),
            null_alignment=float(torch.dot(null_vector,native_update)/(null_vector.norm()*magnitude)),variants={})
        for tag in ['zero_actor_first','zero_critic_first','zero_both_first']:
            a,c,opt=make(parent,cfg)
            for module in ([a] if tag=='zero_actor_first' else [c] if tag=='zero_critic_first' else [a,c]):
                for parameter in module.parameters():opt.state[parameter]['exp_avg'].zero_()
            torch.manual_seed(SEED)
            stats=ppo_update(a,c,opt,inputs[name]['memory'],inputs[name]['bootstrap_obs'],cfg)
            assert stats['optimizer_steps']==32 and stats['omitted_transitions_per_epoch']==0
            delta=drift_vector(a,old)
            results[name]['variants'][tag]=dict(scores=scores(a),actual_Adam_steps=32,parameter_delta_norm=float(delta.norm()),
                cosine_with_baseline=float(torch.dot(delta,native_update)/(delta.norm()*magnitude)))
    data=dict(intent=file_pin(out/'intent.json'),null_scores=scores(null),null_parameter_delta_norm=float(null_vector.norm()),sources=results,native_steps=0)
    write_json(out/'report.json',data)
    print('null KL',[(n,v['rollout']['kl']) for n,v in data['null_scores'].items()])
    for n,entry in results.items():
        print(n,entry['null_fraction'],entry['null_alignment'],[(k,[(c,round(v['rollout']['kl'],3),round(v['start']['applied_mean_l2'],3)) for c,v in ab['scores'].items()]) for k,ab in entry['variants'].items()])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('comparison',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    main(a.data.resolve(),a.comparison.resolve(),a.output.resolve())
