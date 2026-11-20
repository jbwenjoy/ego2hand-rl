"""CPU equivalence of the first partial-sharing update to unchanged native PPO math."""
from pathlib import Path
import copy
import json
import sys
from types import SimpleNamespace
import torch
from train_pen_multiclip_partial import base,wire,PROFILES
from dex_supermimic.rl.dex_supermimic.ppo import ppo_update
from supermimic_research.artifacts import verify_file_pin,file_pin
from supermimic_research.strict_json import write_json


def main(out):
    out.mkdir(exist_ok=False);torch.set_num_threads(2)
    parent_pin,_,parent=base.parent_state()
    receipt=file_pin(base.ROOT/'runs/pen_multiclip_ppo_v1/shared/first_checkpoint.json')
    pin=json.loads(verify_file_pin(receipt).read_bytes())['checkpoint']
    ck=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)['collectors']['clip_001']
    memory=ck['memory'];end=ck['observations']['critic']
    cfg=SimpleNamespace(**parent['config'])
    def blank():
        a,c,o=base.model_pair('cpu',cfg)
        return SimpleNamespace(actor=a,critic=c,optimizer=o,cfg=cfg)
    results={}
    for key in ('ordinary',*PROFILES):
        l=blank()
        if key=='ordinary':
            l.actor.load_state_dict(parent['modules']['actor']);l.critic.load_state_dict(parent['modules']['critic'])
            l.optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        else:
            other=blank();wire(l,other,parent,key)
        torch.manual_seed(base.SEED+1010)
        stats=ppo_update(l.actor,l.critic,l.optimizer,memory,end,cfg)
        final=tuple(base.cpu_tree(obj.state_dict()) for obj in (l.actor,l.critic,l.optimizer))
        if key=='ordinary':baseline=final
        else:
            for i in range(3):base.check_bytes(baseline[i],final[i],f'{key} readback {i}')
        results[key]=dict(optimizer_steps=stats['optimizer_steps'],actor_loss=float(stats['actor_loss']),critic_loss=float(stats['critic_loss']))
    write_json(out/'report.json',dict(parent=parent_pin,native_batch_checkpoint=pin,
        results=results,comparison='Same recorded native data, CPU permutation seed and ordinary PPO objective for first update. CPU Actor, active Critic and active Adam states byte-identical across three runs. CPU verification only.'))
    print(results)


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
