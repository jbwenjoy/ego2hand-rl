"""Two registered arms, shared source, equal pilot budgets, retained failures."""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,pilot_settings,fork_initial_state
from supermimic_research.learning.native_pen_ppo import cpu_tree,tensor_tree_equal
from train_native_pen_ppo import finite_training_state


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def save(learner,receipt,path):
    temporary=path.with_suffix('.tmp');torch.save(learner.checkpoint(receipt),temporary);temporary.replace(path);return file_pin(path)


def main(a):
    cp=a.config.resolve();cfg,ppo=pilot_settings(cp);qp=a.qualification.resolve();qualification=json.loads(qp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['pilot_iterations']!=2 or cfg['num_envs']!=32 or cfg['rollout_controls']!=64:raise ValueError('Fresh declared two-iteration/two-arm pilot')
    if qualification['config']!=file_pin(cp) or qualification['passed']!=qualification['total'] or qualification['optimizer_steps']!=0:raise ValueError('Current completed zero-update qualification required')
    verify_source_snapshot(json.loads(verify_file_pin(qualification['source_receipt']).read_text()),live=True)
    common=torch.load(verify_file_pin(qualification['common_initial']),map_location='cpu',weights_only=False)
    if common['optimizer_steps'] or common['optimizer']['state']:raise ValueError('Untrained common optimizer required')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),ROOT/'tools/train_native_pen_ppo.py',cp,ROOT/'docs/geometry-support-pilot-protocol-v1.md']
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;updates=0;runtime=None;arms=[];phase='construction'
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            for arm in ['RESIDUAL','SUPPORT']:
                phase=arm+'_initialization';folder=out/arm;folder.mkdir();start_integrations=observed
                runtime=SupportPenRuntime(cp,32);learner=SupportPenPPO(runtime,ppo,arm);fork,changes=fork_initial_state(common,learner.metadata());audit=learner.load(fork)
                if not all(v for v in audit.values() if isinstance(v,bool)) or not audit['native']['all_array_bytes_equal']:raise ValueError('Initial common fork load differs')
                initial=save(learner,receipt,folder/'initial_state.pt');write(folder/'initialization.json',dict(common_parent=qualification['common_initial'],initial=initial,patches=changes,
                    modules_identical=tensor_tree_equal(common['modules'],fork['modules']),norm_observations_identical=tensor_tree_equal(common['observations'],fork['observations'])))
                logs=[];parent=initial
                for iteration in [1,2]:
                    phase=f'{arm}_{iteration}';t=time.perf_counter();learner.collect(64);collection_s=time.perf_counter()-t;health=learner.consistency()
                    if health['passed']!=health['total']:raise ValueError('Collector failed before optimizer')
                    m=learner.memory;summary=dict(geometry_fail_fraction=float(m['contact_objective_geometry_failure'].float().mean()),
                        mean_tracking_reward=float(m['contact_objective_tracking_reward'].mean()),mean_raw_reward=float(m['raw_reward'].mean()),
                        mean_support_bonus=float(m['contact_objective_support_bonus'].mean()),support_eligible_fraction=float(m['contact_objective_support_eligible'].float().mean()),
                        max_object_penetration_m=float(m['contact_geometry_max'][...,0].max()),max_self_penetration_m=float(m['contact_geometry_max'][...,1].max()),
                        phase_min=int(m['phase_source'].min()),phase_max=int(m['phase_source'].max()),terminated_worlds=int(m['terminated'].sum()),truncated_worlds=int(m['truncated'].sum()))
                    torch.save(cpu_tree(m),folder/f'memory_{iteration:04d}.pt');t=time.perf_counter();stats=learner.update();update_s=time.perf_counter()-t;updates+=stats['optimizer_steps']
                    if not finite_training_state(learner):raise ValueError('Nonfinite trained modules/optimizer')
                    checkpoint=save(learner,receipt,folder/f'checkpoint_{iteration:04d}.pt');loop_s=collection_s+update_s
                    log=dict(iteration=iteration,arm=arm,parent=parent,checkpoint=checkpoint,memory=file_pin(folder/f'memory_{iteration:04d}.pt'),collector=health,
                        summary=summary,collection_wall_s=collection_s,update_wall_s=update_s,remaining_this_arm_loop_s=(2-iteration)*loop_s,
                        optimizer_steps=learner.optimizer_steps,training_transitions=learner.training_transitions,normalizer_count=int(learner.actor.norm.count))
                    write(folder/f'health_{iteration:04d}.json',log);logs.append(log);parent=checkpoint
                    print(json.dumps(dict(arm=arm,iteration=iteration,checks=[health['passed'],health['total']],checkpoint=checkpoint,summary=summary,remaining_this_arm_loop_s=log['remaining_this_arm_loop_s'])),flush=True)
                actual=observed-start_integrations
                if actual!=32768 or learner.optimizer_steps!=64 or learner.training_transitions!=4096:raise ValueError('Registered per-arm budget differs')
                arms.append(dict(arm=arm,initial=initial,common_parent=qualification['common_initial'],initialization=file_pin(folder/'initialization.json'),checkpoint=parent,health=file_pin(folder/'health_0002.json'),
                    iterations=2,actual_native_integrations=actual,optimizer_steps=64,training_transitions=4096,logs=logs))
                runtime.close();runtime=None
            verify_file_pin(qualification['common_initial']);verify_source_snapshot(receipt,live=True)
            checks=dict(equal_native_budget=all(c['actual_native_integrations']==32768 for c in arms),equal_optimizer_budget=all(c['optimizer_steps']==64 for c in arms),
                equal_transition_budget=all(c['training_transitions']==4096 for c in arms),actual_native_total=observed==65536,actual_optimizer_total=updates==128)
            result=dict(schema='pen_geometry_support_training_pilot:1',config=file_pin(cp),qualification=file_pin(qp),source_receipt=file_pin(out/'source_receipt.json'),arms=arms,
                checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,optimizer_steps=updates,
                scope=cfg['scope']+' Actual new geometry-aware pilot parameters, original timing/physics/no aids; training metrics alone do not establish strict task success or support benefit. Shared calibration charged in qualification.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native_integrations=observed,optimizer_steps=updates)))
            if not all(checks.values()):raise ValueError('Retain pilot budget mismatch')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,actual_native_integrations=observed,optimizer_steps=updates));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
