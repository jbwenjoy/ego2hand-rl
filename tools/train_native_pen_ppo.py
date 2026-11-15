"""Staged actual native PPO baseline with explicit counts and combined checkpoints."""
import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter
from supermimic_research.learning.native_pen_ppo import NativePenPPO
from supermimic_research.learning.gpu_monitor import GPUMonitor


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def finite_training_state(trainer):
    values=list(trainer.actor.state_dict().values())+list(trainer.critic.state_dict().values())
    values += [v for state in trainer.optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)]
    by_device={}
    for value in values:
        if value.is_floating_point():by_device.setdefault(str(value.device),[]).append(torch.isfinite(value).all())
    return all(bool(torch.stack(items).all()) for items in by_device.values())


def save_checkpoint(trainer,receipt,path):
    state=trainer.checkpoint(receipt);temporary=path.with_suffix('.tmp')
    torch.save(state,temporary);temporary.replace(path)
    return file_pin(path)


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve();qualification=json.loads(a.qualification.read_text())
    if out.exists() or not out.is_relative_to(ROOT/'runs'):raise ValueError('Fresh staged training output')
    if qualification['passed']!=qualification['total'] or qualification['config']!=file_pin(cp) or qualification['actual_optimizer_steps']:
        raise ValueError('Completed current collector qualification required')
    verify_source_snapshot(json.loads(verify_file_pin(qualification['source_receipt']).read_text()),live=True)
    if a.until not in [2,32] or cfg['iterations']!=32 or cfg['first_checkpoint_iteration']!=2 or cfg['num_envs']!=32:
        raise ValueError('Declared2/32 stage budget required')
    if a.resume is None and a.until!=2:raise ValueError('First stage ends at first trained checkpoint for review/ETA')
    out.mkdir();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})
    paths += [Path(__file__).resolve(),cp,(cp.parent/cfg['adapter_config']).resolve()]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,revision);write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    oldstep=Simulation.step;observed=0;adapter=None;monitor=GPUMonitor();monitor_started=False;startup=time.perf_counter()
    def step(sim):
        nonlocal observed
        oldstep(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            adapter=NativePenPolicyAdapter((cp.parent/cfg['adapter_config']).resolve(),32);trainer=NativePenPPO(adapter,cfg['ppo'])
            load_audit=None;initial=None
            if a.resume:
                parent=torch.load(a.resume.resolve(),map_location='cpu',weights_only=False);load_audit=trainer.load(parent)
                if not all(v for k,v in load_audit.items() if isinstance(v,bool)):raise ValueError('Combined checkpoint restore failed')
                if trainer.iteration!=2 or a.until!=32:raise ValueError('Declared continuation starts at2 and ends32')
            else:
                trainer.calibrate(cfg['normalization_calibration_controls'])
                initial=save_checkpoint(trainer,receipt,out/'initial_state.pt')
            start_iteration=trainer.iteration;warmup_integrations=observed
            startup_wall=time.perf_counter()-startup
            logs=[];torch.cuda.reset_peak_memory_stats();monitor.start();monitor_started=True;started=time.perf_counter()
            for it in range(start_iteration,a.until):
                t=time.perf_counter();trainer.collect(cfg['ppo']['ppo_rollout_steps']);torch.cuda.synchronize();collect_wall=time.perf_counter()-t
                t=time.perf_counter();consistency=trainer.consistency();check_wall=time.perf_counter()-t
                if not all(consistency['checks'].values()):
                    torch.save(trainer.checkpoint(receipt),out/f'failed_collect_{it+1:04d}.pt')
                    write(out/f'failed_collect_{it+1:04d}.json',consistency)
                    raise ValueError('Rollout consistency/finite/no-aid/capacity failed')
                t=time.perf_counter();stats=trainer.update();torch.cuda.synchronize();update_wall=time.perf_counter()-t
                if not finite_training_state(trainer):
                    torch.save(trainer.checkpoint(receipt),out/f'failed_update_{it+1:04d}.pt');raise ValueError('Nonfinite modules/optimizer')
                row=dict(iteration=trainer.iteration,collect_wall_s=collect_wall,check_wall_s=check_wall,update_wall_s=update_wall,
                    consistency=consistency,stats={k:float(v) if isinstance(v,torch.Tensor) else v for k,v in stats.items()},
                    optimizer_steps=trainer.optimizer_steps,training_transitions=trainer.training_transitions,observed_stage_integrations=observed)
                logs.append(row)
                with (out/'iterations.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps(dict(iteration=trainer.iteration,rollout_reward=consistency['mean_raw_reward'],
                    collect_s=collect_wall,update_s=update_wall,optimizer_steps=trainer.optimizer_steps,native_integrations=observed)),flush=True)
            torch.cuda.synchronize();wall=time.perf_counter()-started;monitor.stop();monitor_started=False
            verify_source_snapshot(receipt,live=True)
            checkpoint=save_checkpoint(trainer,receipt,out/f'checkpoint_{trainer.iteration:04d}.pt')
            expected=(a.until-start_iteration)*32*cfg['ppo']['ppo_rollout_steps']*8+warmup_integrations
            checks=dict(native_integration_count=observed==expected,
                optimizer_count=trainer.optimizer_steps==trainer.iteration*cfg['ppo']['ppo_epochs']*cfg['ppo']['ppo_minibatches'],
                transition_count=trainer.training_transitions==trainer.iteration*32*cfg['ppo']['ppo_rollout_steps'],
                calibration_count=trainer.calibration_transitions==32*cfg['normalization_calibration_controls'],
                finite_training_state=finite_training_state(trainer),
                all_rollouts_qualified=all(row['consistency']['passed']==row['consistency']['total'] for row in logs),
                no_omitted_samples=all(row['stats']['omitted_transitions_per_epoch']==0 for row in logs),
                exact_actor_dimensions=trainer.actor.net[0].in_features==161 and trainer.actor.action_dim==20,
                exact_critic_dimensions=trainer.critic.net[0].in_features==188,
                checkpoint_terminal=verify_file_pin(checkpoint).is_file())
            times=np.array([row['collect_wall_s']+row['check_wall_s']+row['update_wall_s'] for row in logs])
            remaining=cfg['iterations']-trainer.iteration
            telemetry=dict(samples=monitor.samples,errors=monitor.errors,scope='Device-wide windows over this stage collection/check/update, includes other desktop loads; no speedup claim.')
            if monitor.samples:
                values=np.array([row['values'] for row in monitor.samples]);telemetry.update(
                    GPU_median_percent=float(np.median(values[:,0])),GPU_max_percent=float(values[:,0].max()),
                    VRAM_max_MiB=float(values[:,1].max()),VRAM_total_MiB=float(values[0,2]),VRAM_max_percent=float(values[:,1].max()/values[0,2]*100))
            health=dict(schema='pen_native_PPO_stage_health:1',config=file_pin(cp),qualification=file_pin(a.qualification),
                source_receipt=file_pin(out/'source_receipt.json'),checkpoint=checkpoint,initial_state=initial,
                parent=file_pin(a.resume) if a.resume else None,load_audit=load_audit,
                iteration=trainer.iteration,optimizer_steps=trainer.optimizer_steps,training_transitions=trainer.training_transitions,
                calibration_transitions=trainer.calibration_transitions,stage_actual_native_integrations=observed,
                stage_warmup_integrations=warmup_integrations,logical_adapter_integrations=adapter.physics_integrations,
                startup_and_initial_state_wall_s=startup_wall,stage_loop_wall_s=wall,logs=logs,
                remaining_compute_estimate_s=float(wall/len(logs)*remaining),
                remaining_compute_range_s=[float(times.min()*remaining),float(times.max()*remaining)],
                estimate_scope='Observed iteration extrapolation; excludes future process startup, verification, checkpoint save and evaluation, and changing contact cost.',
                GPU_telemetry=telemetry,Torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),Torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                checks=checks,passed=sum(checks.values()),total=len(checks),training_running=False,pause_goal_requested=False,
                scope='Actual native model-free PPO diagnostic. No strict task/CHORD-gradient/ego/acceleration or deterministic W32 physical replay qualification; first trained checkpoint needs independent restore review.')
            write(out/f'health_{trainer.iteration:04d}.json',health)
            print(json.dumps(dict(health=file_pin(out/f'health_{trainer.iteration:04d}.json'),checks=[health['passed'],health['total']],
                checkpoint=checkpoint,remaining_compute_estimate_s=health['remaining_compute_estimate_s'],GPU={k:v for k,v in telemetry.items() if k not in ['samples','errors','scope']})),flush=True)
            if not all(checks.values()):raise ValueError('Keep unhealthy checkpoint; no automatic retry')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),observed_integrations=observed))
        raise
    finally:
        if monitor_started:monitor.stop()
        Simulation.step=oldstep
        if adapter:adapter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--qualification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--until',type=int,required=True);p.add_argument('--resume',type=Path);main(p.parse_args())
