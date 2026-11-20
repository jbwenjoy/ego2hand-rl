"""Matched two-reference training with shared and independent PPO models."""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch
import numpy as np
import torch

from benchmark_pen_scaled import ROOT, OBJECTIVE, COLLISION, BackendSupportPenPPO, pilot_settings, factory
from evaluate_pen_dev_clip import config as clip_config
from supermimic_research.runtime.pen_multiclip_batch import multiclip_w32, multiclip_inputs
from supermimic_research.runtime.explicit_collision_backend import backend_scope
from supermimic_research.learning.native_pen_ppo import cpu_tree,model_pair,validate_modules
from supermimic_research.learning.pen_multiclip_checkpoint import CLIPS, owners, validate_identity, snapshot, save_checked, restore, check_bytes
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources
from supermimic_research.strict_json import write_json

PARENT=ROOT/'runs/pen_batch_updates_v1/parent.json'
PRELOAD=ROOT/'runs/pen_refit_dev_preload_v1/report.json'
BATCHES=32
STEPS=64
WORLDS=32
SEED=7411


def parent_state():
    pin=json.loads(PARENT.read_bytes())
    state=torch.load(verify_file_pin(pin),map_location='cpu',weights_only=False)
    if state['iteration']!=2781 or state['metadata']['arm']!='RESIDUAL' or state['optimizer_steps']!=88992:
        raise ValueError('Expected full residual parent at 2781')
    actor_pin=file_pin(ROOT/'runs/pen_scaled_train_v1/actor_02781.pt')
    endpoint=torch.load(verify_file_pin(actor_pin),map_location='cpu',weights_only=False)
    check_bytes(endpoint['actor'],state['modules']['actor'],'parent Actor pin')
    cfg=pilot_settings(OBJECTIVE)[1]
    if cfg!=state['config']:raise ValueError('PPO config differs from parent')
    from types import SimpleNamespace
    a,c,opt=model_pair('cpu',SimpleNamespace(**cfg))
    validate_modules(a,c,opt,state);a.load_state_dict(state['modules']['actor']);c.load_state_dict(state['modules']['critic'])
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    if int(a.norm.count)<=0 or int(a.norm.count)!=int(c.norm.count):raise ValueError('Expected calibrated parent normalization')
    check_bytes(state['optimizer'],cpu_tree(opt.state_dict()),'CPU Adam restore')
    if any(int(v['step'])!=88992 for v in opt.state.values()):raise ValueError('Parent Adam counters differ')
    return pin,actor_pin,dict(modules=state['modules'],optimizer=state['optimizer'],config=cfg)


def schedule():
    return [CLIPS[i%2] for i in range(BATCHES)]


def init_learners(tasks,parent,profile):
    result={}
    for clip in CLIPS:
        learner=BackendSupportPenPPO(tasks[clip],parent['config'],'RESIDUAL')
        if profile=='shared' and result:
            first=result[CLIPS[0]]
            learner.actor,learner.critic,learner.optimizer=first.actor,first.critic,first.optimizer
        else:
            learner.actor.load_state_dict(parent['modules']['actor'])
            learner.critic.load_state_dict(parent['modules']['critic'])
            learner.optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        learner.normalizers_ready=True
        result[clip]=learner
    validate_identity(result,profile)
    return result


def noise_for(index):
    rng=torch.Generator(device='cpu').manual_seed(SEED+1009*index)
    return torch.randn(STEPS,WORLDS,20,dtype=torch.float32,generator=rng,device='cpu')


@contextmanager
def paired_samples(actor,noise):
    distribution=actor.distribution
    observed=0
    def sample_distribution(*args,**kwargs):
        dist=distribution(*args,**kwargs)
        def sample():
            nonlocal observed
            value=dist.loc+dist.scale*noise[observed]
            observed+=1
            return value
        dist.sample=sample
        return dist
    with patch.object(actor,'distribution',sample_distribution):
        yield
    if observed!=STEPS:raise ValueError('Action noise draw count differs')


def norm_state(learner):
    return {k:cpu_tree(getattr(learner,k).norm.state_dict()) for k in ('actor','critic')}


def batch(learner,clip,index,baseline_norm,phase_count):
    before=learner.adapter.physics_integrations
    # Imported parent stats are frozen throughout both profiles. Drop the previous
    # rollout normalization callback inputs before collecting the next batch.
    learner.normalization_batch=None
    eps=noise_for(index);noise_digest=hashlib.sha256(eps.numpy().tobytes()).hexdigest()
    start=time.perf_counter()
    with paired_samples(learner.actor,eps.to(learner.device)):
        m=learner.collect(STEPS)
    health=learner.consistency()
    if not all(health['checks'].values()):raise ValueError(f'Failed consistency: {health}')
    clock=learner.adapter.physics_integrations-before
    if clock!=WORLDS*STEPS*8:raise ValueError('Native step accounting')
    torch.manual_seed(SEED+1009*index+1)
    stats=learner.update()
    learner.normalization_batch=None
    check_bytes(baseline_norm,norm_state(learner),'frozen normalization')
    if stats['optimizer_steps']!=32 or stats['omitted_transitions_per_epoch']!=0:
        raise ValueError('Optimizer budget differs')
    if not all(torch.isfinite(v).all().item() for net in (learner.actor,learner.critic) for v in net.parameters()):
        raise ValueError('Nonfinite network')
    phase=m['phase_used']
    if int(phase.min())<1 or int(phase.max())>=phase_count:raise ValueError('Phase outside declared reference')
    valid=((m['contact_geometry_max']<=learner.adapter.objective_constants['geometry_limits']).all(-1)
           & ~m['engine_overflow'] & m['no_applied_force'] & (m['contact_errors']==0)
           & (m['contact_nonhand_max_sum'][...,0]<=.1))
    hist=torch.bincount(phase[valid],minlength=phase_count).cpu()
    return dict(batch=index,clip=clip,seconds=time.perf_counter()-start,native=clock,transitions=phase.numel(),
                optimizer_calls=stats['optimizer_steps'],sample_noise_sha256=noise_digest,permutation_seed=SEED+1009*index+1,
                valid_count=int(hist.sum()),valid_phase_histogram=hist.tolist(),late_valid=int(hist[(phase_count-1)//2+1:].sum()),
                first_phase_count=int((phase==1).sum()),max_phase=int(phase.max()),terminal=int(m['terminated'].sum()),
                reference_end=int(m['truncated'].sum()),max_object_penetration_mm=float(m['contact_geometry_max'][...,0].max())*1000,
                health=health)


def evaluate_counts(learners,profile,n):
    owner=owners(profile)
    for key in set(owner.values()):
        clips=[c for c in CLIPS if owner[c]==key]
        adds=sum(schedule()[:n].count(c) for c in clips)*32
        actual={int(v['step']) for v in learners[clips[0]].optimizer.state.values()}
        if actual!={88992+adds}:raise ValueError('Unexpected actual Adam counter')
    for clip in CLIPS:
        n_local=schedule()[:n].count(clip);l=learners[clip]
        observed=[l.iteration,l.optimizer_steps,l.training_transitions,l.adapter.physics_integrations]
        expected=[n_local,n_local*32,n_local*2048,n_local*16384]
        if observed!=expected:raise ValueError(f'{clip} {observed} != {expected}')


def export(out,learners,profile,manifest,collision,config):
    result={}
    for clip,key in owners(profile).items():
        if key in result:continue
        learner=learners[clip];relevant=[c for c,k in owners(profile).items() if k==key]
        updates=sum(learners[c].iteration for c in relevant)
        payload=dict(actor=cpu_tree(learner.actor.state_dict()),iteration=2781+updates,config=config,
                     batch_backend=dict(factory=manifest['generated'],collision_backend=collision.contract['backend_id'],
                                        worlds=32,profile=profile,runtime=learner.adapter.metadata()['termination_backend']),
                     trained_on=relevant,parent_iteration=2781,added_policy_updates=updates)
        p=out/(key+'.pt');torch.save(payload,p);result[key]=file_pin(p)
    return result


def run_profile(profile,out,parent,obj3,manifest,receipt,protocol):
    folder=out/profile;folder.mkdir()
    logs=[];ckpt=None
    with torch.jit.optimized_execution(False),backend_scope(COLLISION) as collision:
        for first,last in [(1,1),(2,BATCHES)]:
            with multiclip_w32(OBJECTIVE,obj3,manifest) as tasks:
                learners=init_learners(tasks,parent,profile)
                original_norm=norm_state(learners[CLIPS[0]])
                if ckpt:
                    saved=torch.load(verify_file_pin(ckpt),map_location='cpu',weights_only=False)
                    restored=restore(learners,saved,profile,protocol)
                    evaluate_counts(learners,profile,first-1)
                    write_json(folder/'first_checkpoint.json',dict(checkpoint=ckpt,restored=restored,completed_batches=first-1))
                    del saved
                for idx in range(first,last+1):
                    name=schedule()[idx-1]
                    row=batch(learners[name],name,idx,original_norm,protocol['clips'][name]['frames'])
                    evaluate_counts(learners,profile,idx)
                    logs.append(row)
                    with (folder/'rows.jsonl').open('a',encoding='utf-8') as log:
                        log.write(json.dumps(row,allow_nan=False)+'\n')
                state=snapshot(learners,profile,receipt,last,protocol)
                path=folder/('first_full.pt' if last==1 else 'final_full.pt')
                check=save_checked(path,state)
                ckpt=file_pin(path)
                del state,check
                if last==BATCHES:
                    endpoints=export(out,learners,profile,manifest,collision,parent['config'])
                del learners
    summed={k:sum(row[k] for row in logs) for k in ['native','transitions','optimizer_calls','seconds','valid_count']}
    write_json(folder/'report.json',dict(profile=profile,actors=endpoints,totals=summed,final_checkpoint=ckpt,
                                         first_restore=file_pin(folder/'first_checkpoint.json'),iterations=file_pin(folder/'rows.jsonl')))
    return endpoints,logs


def main(out,execute):
    torch.set_num_threads(4)
    parent_pin,actor_pin,parent=parent_state()
    preload_pin=file_pin(PRELOAD)
    with tempfile.TemporaryDirectory() as temp:
        check,_=clip_config(Path(temp),'clip_003',PRELOAD)
        spec=multiclip_inputs(OBJECTIVE,check)
        print('CPU preflight passed:',{k:v['frames'] for k,v in spec.items()},'parent',parent_pin['sha256'],flush=True)
    if not execute:return
    out.mkdir(parents=True,exist_ok=False)
    stage='configuration'
    try:
        task=out/'clip003_config';task.mkdir()
        obj3,_=clip_config(task,'clip_003',PRELOAD);spec=multiclip_inputs(OBJECTIVE,obj3)
        manifest=factory()
        protocol=dict(schema='pen_alternating_multiclip_PPO:1',parent=parent_pin,parent_actor=actor_pin,preload=preload_pin,
                      clips=spec,seed=SEED,worlds=32,rollout_controls=STEPS,batches=BATCHES,schedule=schedule(),
                      normalization='Freeze imported parent actor and critic normalization buffers. No calibration or rollout-based update.',
                      profiles=['shared','specialists'],native_budget_per_profile=524288,transition_budget_per_profile=65536,
                      optimizer_budget_per_profile=1024,common_randomness='Identical standardized Gaussian arrays and per-update permutation seed per scheduled batch. Divergent trajectories possible.',
                      scope='Controlled native model-free residual PPO. Both physical tasks fresh; no parent native state or world-model update.')
        write_json(out/'intent.json',protocol)
        sources=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/evaluate_pen_dev_clip.py',ROOT/'tools/benchmark_pen_scaled.py',OBJECTIVE,COLLISION]
        receipt=freeze_sources(sources,out/'snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        write_json(out/'source_receipt.json',receipt)
        endpoints={};records={}
        for stage in protocol['profiles']:
            path,rows=run_profile(stage,out,parent,obj3,manifest,receipt,protocol)
            endpoints.update(path);records[stage]=rows
        for a,b in zip(records['shared'],records['specialists']):
            if (a['clip'],a['sample_noise_sha256'],a['permutation_seed'])!=(b['clip'],b['sample_noise_sha256'],b['permutation_seed']):
                raise ValueError('Profile ordering or sampling seed differs')
        write_json(out/'report.json',dict(intent=file_pin(out/'intent.json'),source_receipt=file_pin(out/'source_receipt.json'),
                                          actors=endpoints,profiles={k:file_pin(out/k/'report.json') for k in protocol['profiles']}))
    except BaseException as exc:
        write_json(out/'failure.json',dict(stage=stage,error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);p.add_argument('--execute',action='store_true')
    args=p.parse_args();main(args.out.resolve(),args.execute)
