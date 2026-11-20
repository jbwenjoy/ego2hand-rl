"""CPU pilot: reconstruct the first sampling policy, then branch the same saved rollout."""
from pathlib import Path
import copy,json,sys
from types import SimpleNamespace
import torch
from pen_ppo_shift import policy_shift
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning.native_pen_ppo import model_pair
from dex_supermimic.rl.dex_supermimic.ppo import ppo_update
from supermimic_research.artifacts import verify_file_pin


def main():
    torch.set_num_threads(2)
    run=ROOT/'runs/pen_scaled_train_v1'
    pin=json.loads((run/'first_checkpoint.json').read_bytes())['checkpoint']
    s=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
    cfg=SimpleNamespace(**s['config']);m=s['memory']
    torch.manual_seed(cfg.seed)
    old,critic,_=model_pair('cpu',cfg)
    raw=next(v for k,v in s['native']['python']['tensors'].items() if k==(('attr','initial_raw'),))
    with torch.no_grad():
        old.net[-1].weight[:20].zero_();old.net[-1].bias[:20].copy_(raw)
        for module,key in [(old,'actor'),(critic,'critic')]:
            module.norm.load_state_dict({k[5:]:v for k,v in s['modules'][key].items() if k.startswith('norm.')})
        errors=[]
        for obs,act,crit,lp,val in zip(m['obs'],m['actions'],m['critic_obs'],m['log_prob'],m['value']):
            lp2=old.distribution(obs).log_prob(act).sum(-1);val2=critic(crit)
            errors.append([float((lp2-lp).abs().max()),float((val2.reshape_as(val)-val).abs().max())])
    bounds=list(map(max,zip(*errors)))
    if bounds[0]>1e-4 or bounds[1]>1e-4:raise ValueError(f'Sampling policy reconstruction failed {bounds}')
    result={}
    for steps in (32,64,128):
        a,c,opt=model_pair('cpu',cfg);a.load_state_dict(old.state_dict());c.load_state_dict(critic.state_dict())
        torch.manual_seed(7411);cfg.ppo_minibatches=steps//4
        stats=ppo_update(a,c,opt,m,s['observations']['critic'],cfg)
        result[steps]=policy_shift(old,a,m,cfg.ppo_clip)
        assert stats['optimizer_steps']==steps
    out=ROOT/'runs/pen_update_cpu_pilot_v1';out.mkdir(exist_ok=False)
    (out/'report.json').write_text(json.dumps(dict(source=pin,device='cpu',worlds=256,bounds=bounds,results=result),indent=2))
    print(json.dumps(result))
if __name__=='__main__': main()
