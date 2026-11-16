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


def registered_budget(cfg):
    long=cfg.get('study')=='LONG_CONTROL_BASELINE'
    n=cfg['pilot_iterations'];interval=cfg.get('checkpoint_interval',1)
    if long:
        if n!=2000 or interval!=100 or cfg.get('fixed_frame_features')!='CACHED' or cfg.get('first_checkpoint_reload') is not True:raise ValueError('Registered2000-round cached control baseline and first reload required')
    elif n not in [2,64]:raise ValueError('Historical pilot cannot silently expand its budget')
    if interval<1 or cfg['num_envs']!=32 or cfg['rollout_controls']!=64:raise ValueError('Registered32-world64-control budget required')
    return long


def main(a):
    cp=a.config.resolve();cfg,ppo=pilot_settings(cp);qp=a.qualification.resolve();qualification=json.loads(qp.read_text());out=a.output.resolve()
    iterations=cfg['pilot_iterations'];interval=cfg.get('checkpoint_interval',1)
    long=registered_budget(cfg)
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh explicitly registered two-arm trial')
    if qualification['config']!=file_pin(cp) or qualification['passed']!=qualification['total'] or qualification['optimizer_steps']!=0:raise ValueError('Current completed zero-update qualification required')
    verify_source_snapshot(json.loads(verify_file_pin(qualification['source_receipt']).read_text()),live=True)
    common=torch.load(verify_file_pin(qualification['common_initial']),map_location='cpu',weights_only=False)
    if common['optimizer_steps'] or common['optimizer']['state']:raise ValueError('Untrained common optimizer required')
    protocol=(cp.parent/cfg.get('protocol','../docs/geometry-support-pilot-protocol-v1.md')).resolve()
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),ROOT/'tools/train_native_pen_ppo.py',cp,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;updates=0;runtime=None;arms=[];phase='construction'
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            if long:torch.set_num_threads(4)
            for arm_index,arm in enumerate(['RESIDUAL','SUPPORT']):
                phase=arm+'_initialization';folder=out/arm;folder.mkdir();start_integrations=observed
                runtime=SupportPenRuntime(cp,32);learner=SupportPenPPO(runtime,ppo,arm);fork,changes=fork_initial_state(common,learner.metadata());audit=learner.load(fork)
                if not all(v for v in audit.values() if isinstance(v,bool)) or not audit['native']['all_array_bytes_equal']:raise ValueError('Initial common fork load differs')
                initial=save(learner,receipt,folder/'initial_state.pt');write(folder/'initialization.json',dict(common_parent=qualification['common_initial'],initial=initial,patches=changes,
                    modules_identical=tensor_tree_equal(common['modules'],fork['modules']),norm_observations_identical=tensor_tree_equal(common['observations'],fork['observations'])))
                logs=[];parent=initial
                for iteration in range(1,iterations+1):
                    phase=f'{arm}_{iteration}';iteration_start=time.perf_counter();t=time.perf_counter();learner.collect(64);collection_s=time.perf_counter()-t;health=learner.consistency()
                    if health['passed']!=health['total']:raise ValueError('Collector failed before optimizer')
                    m=learner.memory;summary=dict(geometry_fail_fraction=float(m['contact_objective_geometry_failure'].float().mean()),
                        mean_tracking_reward=float(m['contact_objective_tracking_reward'].mean()),mean_raw_reward=float(m['raw_reward'].mean()),
                        mean_support_bonus=float(m['contact_objective_support_bonus'].mean()),support_eligible_fraction=float(m['contact_objective_support_eligible'].float().mean()),
                        max_object_penetration_m=float(m['contact_geometry_max'][...,0].max()),max_self_penetration_m=float(m['contact_geometry_max'][...,1].max()),
                        phase_min=int(m['phase_source'].min()),phase_max=int(m['phase_source'].max()),terminated_worlds=int(m['terminated'].sum()),truncated_worlds=int(m['truncated'].sum()))
                    checkpoint_due=iteration==1 or iteration%interval==0 or iteration==iterations;memory_pin=None
                    if checkpoint_due:
                        memory_path=folder/f'memory_{iteration:04d}.pt';torch.save(cpu_tree(m),memory_path);memory_pin=file_pin(memory_path)
                    t=time.perf_counter();stats=learner.update();update_s=time.perf_counter()-t;updates+=stats['optimizer_steps']
                    if not finite_training_state(learner):raise ValueError('Nonfinite trained modules/optimizer')
                    checkpoint=save(learner,receipt,folder/f'checkpoint_{iteration:04d}.pt') if checkpoint_due else None;loop_s=time.perf_counter()-iteration_start
                    reload_audit=None;reload_s=0.
                    if long and iteration==1:
                        reload_started=time.perf_counter()
                        saved=torch.load(verify_file_pin(checkpoint),map_location='cpu',weights_only=False);reload_audit=learner.load(saved)
                        if not all(v for v in reload_audit.values() if type(v) is bool) or not reload_audit['native']['all_array_bytes_equal'] or not reload_audit['native']['all_scalar_values_equal']:raise ValueError('First optimized checkpoint reload differs')
                        reload_s=time.perf_counter()-reload_started
                    log=dict(iteration=iteration,arm=arm,parent=parent,checkpoint=checkpoint,memory=memory_pin,collector=health,
                        summary=summary,collection_wall_s=collection_s,update_wall_s=update_s,iteration_wall_s=loop_s,remaining_this_arm_loop_s=(iterations-iteration)*loop_s,
                        estimated_total_remaining_loop_s=((iterations-iteration)+(1-arm_index)*iterations)*loop_s,
                        optimizer_steps=learner.optimizer_steps,training_transitions=learner.training_transitions,normalizer_count=int(learner.actor.norm.count))
                    if reload_audit is not None:log.update(first_checkpoint_reload=reload_audit,first_checkpoint_reload_wall_s=reload_s)
                    if long:
                        recent=[row['iteration_wall_s'] for row in logs[-49:]]+[loop_s];typical=sorted(recent)[len(recent)//2]
                        log.update(remaining_this_arm_loop_s=(iterations-iteration)*typical,estimated_total_remaining_loop_s=((iterations-iteration)+(1-arm_index)*iterations)*typical,ETA_scope='Median of at most50 recent loop times;startup/reload/next-arm setup/final evaluation excluded.')
                    write(folder/f'health_{iteration:04d}.json',log);logs.append(log)
                    if checkpoint is not None:parent=checkpoint
                    print(json.dumps(dict(arm=arm,iteration=iteration,checks=[health['passed'],health['total']],checkpoint=checkpoint,summary=summary,remaining_this_arm_loop_s=log['remaining_this_arm_loop_s'],estimated_total_remaining_loop_s=log['estimated_total_remaining_loop_s'])),flush=True)
                actual=observed-start_integrations
                if actual!=iterations*16384 or learner.optimizer_steps!=iterations*32 or learner.training_transitions!=iterations*2048:raise ValueError('Registered per-arm budget differs')
                arms.append(dict(arm=arm,initial=initial,common_parent=qualification['common_initial'],initialization=file_pin(folder/'initialization.json'),checkpoint=parent,health=file_pin(folder/f'health_{iterations:04d}.json'),
                    iterations=iterations,actual_native_integrations=actual,optimizer_steps=learner.optimizer_steps,training_transitions=learner.training_transitions,logs=logs))
                runtime.close();runtime=None
            verify_file_pin(qualification['common_initial']);verify_source_snapshot(receipt,live=True)
            checks=dict(equal_native_budget=all(c['actual_native_integrations']==iterations*16384 for c in arms),equal_optimizer_budget=all(c['optimizer_steps']==iterations*32 for c in arms),
                equal_transition_budget=all(c['training_transitions']==iterations*2048 for c in arms),actual_native_total=observed==iterations*32768,actual_optimizer_total=updates==iterations*64)
            result=dict(schema='pen_geometry_support_long_control_training:2' if long else 'pen_geometry_support_training_pilot:1',config=file_pin(cp),qualification=file_pin(qp),source_receipt=file_pin(out/'source_receipt.json'),arms=arms,
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
