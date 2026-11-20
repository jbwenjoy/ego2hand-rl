"""Separate clip Critics with shared Actor parameters, alternating native PPO update blocks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch
import train_pen_multiclip_ppo as base
from supermimic_research.learning.pen_partial_sharing import PROFILES,wire,capture,restore,validate
from supermimic_research.learning.pen_multiclip_checkpoint import save_checked
from supermimic_research.learning.native_pen_ppo import cpu_tree,model_pair
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json


def cpu_check(parent,profile):
    cfg=SimpleNamespace(**parent['config'])
    def blank():
        actor,critic,opt=model_pair('cpu',cfg)
        return SimpleNamespace(actor=actor,critic=critic,optimizer=opt,cfg=cfg)
    a,b=blank(),blank();wire(a,b,parent,profile)
    for l in (a,b):base.check_bytes(parent['optimizer'],cpu_tree(l.optimizer.state_dict()),'parent Adam')
    # A real CPU Adam step catches broken aliasing in state initialization vs stepping.
    for param in (*a.actor.parameters(),*a.critic.parameters()):param.grad=torch.zeros_like(param)
    a.optimizer.step();validate(a,b,profile)
    bstep=int(b.optimizer.state[next(b.actor.parameters())]['step'])
    assert bstep==(88993 if profile=='shared_adam' else 88992)
    assert int(b.optimizer.state[next(b.critic.parameters())]['step'])==88992


def counts(learners,profile,index):
    a,b=(learners[c] for c in base.CLIPS);validate(a,b,profile)
    for key,l in learners.items():
        local=base.schedule()[:index].count(key)
        for module,num in ((l.actor,index if profile=='shared_adam' else local),(l.critic,local)):
            if {int(l.optimizer.state[p]['step']) for p in module.parameters()}!={88992+32*num}:
                raise ValueError('Unexpected Adam counter')
        if [l.iteration,l.optimizer_steps,l.training_transitions,l.adapter.physics_integrations]!=[local,32*local,2048*local,16384*local]:
            raise ValueError('Unexpected learner counters')


@torch.no_grad()
def starts(learners):
    return {k:l.observations['policy'][0:1].clone() for k,l in learners.items()}


@torch.no_grad()
def distributions(actor,obs):
    return {k:(actor.distribution(x,validate_args=False).mean.clone(),actor.distribution(x,validate_args=False).scale.clone()) for k,x in obs.items()}


def drift(old,new):
    return {k:dict(raw_mean_l2=float((new[k][0]-old[k][0]).norm()),clipped_mean_l2=float((new[k][0].clamp(-1,1)-old[k][0].clamp(-1,1)).norm()),std_l2=float((new[k][1]-old[k][1]).norm())) for k in old}


def run(profile,out,parent,obj3,manifest,receipt,protocol):
    folder=out/profile;folder.mkdir()
    rows=[];ckpt=None
    with torch.jit.optimized_execution(False),base.backend_scope(base.COLLISION) as collision:
        for begin,end in ((1,1),(2,base.BATCHES)):
            with base.multiclip_w32(base.OBJECTIVE,obj3,manifest) as tasks:
                ls={k:base.BackendSupportPenPPO(tasks[k],parent['config'],'RESIDUAL') for k in base.CLIPS}
                a,b=(ls[k] for k in base.CLIPS);wire(a,b,parent,profile)
                norms={k:base.norm_state(l) for k,l in ls.items()}
                if ckpt is None:
                    obs=starts(ls);parent_outputs=distributions(a.actor,obs)
                else:
                    saved=torch.load(verify_file_pin(ckpt),map_location='cpu',weights_only=False)
                    audit=restore(ls,profile,protocol,saved);counts(ls,profile,1)
                    write_json(folder/'first_checkpoint.json',dict(checkpoint=ckpt,load=audit))
                    del saved
                for index in range(begin,end+1):
                    clip=base.schedule()[index-1];before=distributions(a.actor,obs)
                    row=base.batch(ls[clip],clip,index,norms[clip],protocol['clips'][clip]['frames'])
                    counts(ls,profile,index);after=distributions(a.actor,obs)
                    row.update(update_drift=drift(before,after),parent_drift=drift(parent_outputs,after))
                    with (folder/'rows.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
                    rows.append(row)
                state=capture(ls,profile,protocol,receipt,end);path=folder/('first_full.pt' if end==1 else 'final_full.pt')
                loaded=save_checked(path,state);ckpt=file_pin(path);del loaded,state
                if end==base.BATCHES:
                    dest=out/(profile+'.pt')
                    torch.save(dict(actor=cpu_tree(a.actor.state_dict()),iteration=2781+end,config=parent['config'],
                        batch_backend=dict(factory=manifest['generated'],collision_backend=collision.contract['backend_id'],worlds=32,profile=profile,
                                           runtime=a.adapter.metadata()['termination_backend']),
                        parent_iteration=2781,added_policy_updates=end,trained_on=list(base.CLIPS)),dest)
    report=dict(profile=profile,actor=file_pin(dest),iterations=file_pin(folder/'rows.jsonl'),first_restore=file_pin(folder/'first_checkpoint.json'),
                final_checkpoint=ckpt,totals={k:sum(v[k] for v in rows) for k in ('native','transitions','optimizer_calls','valid_count','seconds')},
                valid_per_clip={k:sum(r['valid_count'] for r in rows if r['clip']==k) for k in base.CLIPS})
    write_json(folder/'report.json',report)
    return report


def main(out,execute):
    torch.set_num_threads(4)
    pin,actor_pin,parent=base.parent_state()
    for profile in PROFILES:cpu_check(parent,profile)
    if not execute:
        print('Both partial-sharing parent Adam states and update counters verified on CPU.');return
    out.mkdir(exist_ok=False,parents=True);phase='setup'
    try:
        cfg_dir=out/'clip003_config';cfg_dir.mkdir()
        obj3,_=base.clip_config(cfg_dir,'clip_003',base.PRELOAD)
        inputs=base.multiclip_inputs(base.OBJECTIVE,obj3);manifest=base.factory()
        protocol=dict(profiles=list(PROFILES),parent=pin,parent_actor=actor_pin,base_study=file_pin(base.ROOT/'runs/pen_multiclip_ppo_v1/report.json'),
            clips=inputs,schedule=base.schedule(),seed=base.SEED,steps=base.STEPS,worlds=base.WORLDS,updates=1024,native=524288,
            transitions=65536,scope='Both share Actor, separate Critics and critic Adam. Only actor Adam moments/counters differ in ownership. Original joint clipping and per-active-clip PPO update unchanged; first-state drift is measured after each 32-step PPO update block.')
        write_json(out/'intent.json',protocol)
        files=sorted((base.ROOT/'src').rglob('*.py'))+[Path(__file__),Path(base.__file__),base.ROOT/'tools/evaluate_pen_dev_clip.py',base.ROOT/'tools/benchmark_pen_scaled.py',base.OBJECTIVE,base.COLLISION]
        receipt=base.freeze_sources(files,out/'snapshot',base.ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip())
        write_json(out/'source_receipt.json',receipt)
        reports={}
        expected=[json.loads(l) for l in (base.ROOT/'runs/pen_multiclip_ppo_v1/shared/rows.jsonl').read_bytes().splitlines()]
        for phase in PROFILES:
            report=run(phase,out,parent,obj3,manifest,receipt,protocol)
            observed=[json.loads(l) for l in verify_file_pin(report['iterations']).read_bytes().splitlines()]
            assert [(r['sample_noise_sha256'],r['permutation_seed']) for r in expected]==[(r['sample_noise_sha256'],r['permutation_seed']) for r in observed]
            reports[phase]=file_pin(out/phase/'report.json')
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),source_receipt=file_pin(out/'source_receipt.json'),
            actors={k:read_actor(v) for k,v in reports.items()},profiles=reports))
    except BaseException as exc:
        write_json(out/'failure.json',dict(phase=phase,error=repr(exc)));raise


def read_actor(report):
    return json.loads(verify_file_pin(report).read_bytes())['actor']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--execute',action='store_true')
    a=p.parse_args();main(a.out.resolve(),a.execute)
