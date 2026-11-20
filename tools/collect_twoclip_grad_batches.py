"""Capture simultaneous paired W32 rollout batches without any weight or normalization update."""
from pathlib import Path
import json,sys
import torch
from train_pen_multiclip_ppo import (ROOT,OBJECTIVE,COLLISION,PRELOAD,clip_config,multiclip_w32,backend_scope,
    factory,parent_state,init_learners,paired_samples,noise_for,cpu_tree,check_bytes,file_pin,write_json)


def main(out):
    out.mkdir(exist_ok=False);parent_pin,actor_pin,parent=parent_state()
    obj,_=clip_config(out,'clip_003',PRELOAD)
    write_json(out/'intent.json',dict(parent=parent_pin,actor=actor_pin,clip003=file_pin(obj),role='Fresh frozen paired-parent diagnostic; identical normal draws on the two tasks; no optimizer steps.'))
    result={}
    torch.set_num_threads(4)
    with torch.jit.optimized_execution(False),backend_scope(COLLISION),multiclip_w32(OBJECTIVE,obj,factory()) as tasks:
        agents=init_learners(tasks,parent,'shared')
        for name,l in agents.items():
            start=cpu_tree(l.observations['policy'])
            eps=noise_for(1).to(l.device)
            with paired_samples(l.actor,eps):m=l.collect(64)
            health=l.consistency()
            if not all(health['checks'].values()):raise ValueError('Collector inconsistent')
            check_bytes(parent['modules']['actor'],cpu_tree(l.actor.state_dict()),'fixed actor')
            check_bytes(parent['modules']['critic'],cpu_tree(l.critic.state_dict()),'fixed critic')
            path=out/(name+'.pt')
            torch.save(dict(memory=cpu_tree(m),bootstrap_obs=cpu_tree(l.observations['critic']),start_obs=start),path)
            result[name]=dict(batch=file_pin(path),health=health,integrations=l.adapter.owned_native_integrations)
    write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),clips=result));print({k:v['integrations'] for k,v in result.items()})
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
