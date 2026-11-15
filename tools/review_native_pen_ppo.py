"""Fresh-env combined load plus fixed-rollout CUDA optimizer continuity checks."""
import argparse
import copy
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from dex_supermimic.rl.dex_supermimic.ppo import ppo_update
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.runtime.pen_portable_state import restore_rng
from supermimic_research.learning.native_pen_ppo import NativePenPPO,model_pair,cpu_tree,tensor_tree_equal
from train_native_pen_ppo import finite_training_state


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def clone_training_modules(state,device):
    actor,critic,opt=model_pair(device,SimpleNamespace(**state['config']))
    actor.load_state_dict(state['modules']['actor']);critic.load_state_dict(state['modules']['critic'])
    # Adam's CPU step tensors may otherwise alias the caller's in-memory state.
    opt.load_state_dict(copy.deepcopy(state['optimizer']))
    return actor,critic,opt


def main(a):
    hp=a.health.resolve();health=json.loads(hp.read_text());cp=verify_file_pin(health['config']);cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or health['passed']!=health['total']:raise ValueError('Fresh healthy-checkpoint review')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),ROOT/'tools/train_native_pen_ppo.py',cp]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    state=torch.load(verify_file_pin(health['checkpoint']),map_location='cpu',weights_only=False)
    from mjlab.sim.sim import Simulation
    oldstep=Simulation.step;observed=0;adapter=None
    def step(sim):
        nonlocal observed
        oldstep(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            adapter=NativePenPolicyAdapter((cp.parent/cfg['adapter_config']).resolve(),32);trainer=NativePenPPO(adapter,cfg['ppo'])
            audit=trainer.load(copy.deepcopy(state))
            checks={key:bool(v) for key,v in audit.items() if isinstance(v,bool)}
            checks.update(native_array_bytes=audit['native']['all_array_bytes_equal'],native_scalars=audit['native']['all_scalar_values_equal'],
                exact_iteration=trainer.iteration==health['iteration'],exact_optimizer_steps=trainer.optimizer_steps==health['optimizer_steps'],
                exact_transition_count=trainer.training_transitions==health['training_transitions'],finite_modules_optimizer=finite_training_state(trainer))
            initial=health.get('initial_state')
            if not initial and health.get('parent'):
                # The continuation has its own verified parent; compare its
                # parameters rather than searching by a checkpoint basename.
                initial=health['parent']
            if initial:
                before=torch.load(verify_file_pin(initial),map_location='cpu',weights_only=False)
                for name,module in [('actor',trainer.actor),('critic',trainer.critic)]:
                    checks[name+'_parameters_updated']=any(not tensor_tree_equal(before['modules'][name][key],state['modules'][name][key]) for key,_ in module.named_parameters())
            memory={key:value.to(trainer.device) for key,value in state['memory'].items()};final=state['observations']['critic'].to(trainer.device)
            clones=[];source_optimizer=cpu_tree(state['optimizer']);live_optimizer=cpu_tree(trainer.optimizer.state_dict())
            for _ in range(2):
                actor,critic,opt=clone_training_modules(state,trainer.device)
                restore_rng(state['native']['rng'])
                stats=ppo_update(actor,critic,opt,memory,final,SimpleNamespace(**state['config']))
                clones.append(dict(actor=cpu_tree(actor.state_dict()),critic=cpu_tree(critic.state_dict()),optimizer=cpu_tree(opt.state_dict()),stats=cpu_tree(stats)))
            for key in clones[0]:checks['fixed_rollout_CUDA_'+key]=tensor_tree_equal(clones[0][key],clones[1][key])
            checks['source_optimizer_immutable']=tensor_tree_equal(source_optimizer,state['optimizer'])
            checks['live_loaded_optimizer_unchanged']=tensor_tree_equal(live_optimizer,cpu_tree(trainer.optimizer.state_dict()))
            restore_rng(state['native']['rng'])
            checks['live_loaded_model_unchanged']=tensor_tree_equal(cpu_tree(trainer.actor.state_dict()),state['modules']['actor']) and tensor_tree_equal(cpu_tree(trainer.critic.state_dict()),state['modules']['critic'])
            checks['zero_native_integrations']=observed==0
            verify_source_snapshot(receipt,live=True);verify_file_pin(health['checkpoint'])
            result=dict(schema='pen_native_PPO_checkpoint_review:1',health=file_pin(hp),checkpoint=health['checkpoint'],
                source_receipt=file_pin(out/'source_receipt.json'),load_audit=audit,
                checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,
                diagnostic_clone_optimizer_steps=2*state['config']['ppo_epochs']*state['config']['ppo_minibatches'],
                actor_norm_count=int(trainer.actor.norm.count),critic_norm_count=int(trainer.critic.norm.count),
                scope='Fresh-env complete checkpoint readback and two identical fixed-rollout CUDA update clones; no new task learning, native future repeatability, strict task/ego/CHORD-gradient claim.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],clone_optimizer_steps=result['diagnostic_clone_optimizer_steps'])))
            if not all(checks.values()):raise ValueError('Keep failed combined/optimizer resume checks')
    finally:
        Simulation.step=oldstep
        if adapter:adapter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--health',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
