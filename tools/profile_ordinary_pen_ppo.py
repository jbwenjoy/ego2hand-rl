"""Actual32-world PPO:separate clean timing,Python calls and Torch CPU trace."""
import argparse,cProfile,json,pstats,subprocess,sys,time,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,pilot_settings
from supermimic_research.learning.native_pen_ppo import cpu_tree
from supermimic_research.learning.gpu_monitor import GPUMonitor


def json_value(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    raise TypeError('Unsupported diagnostic value '+str(type(value)))


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,default=json_value)+'\n',encoding='utf-8')


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());op=(cp.parent/cfg['objective_config']).resolve();objective,ppo=pilot_settings(op);out=a.output.resolve()
    comparison=cfg.get('study')=='EMPTY_SIGNAL_DRIVER_COMPARISON'
    declared=['BASE_A','FAST_A','FAST_B','BASE_B','BASE_C','FAST_C','TORCH_BASE','TORCH_FAST'] if comparison else ['CLEAN_A','CLEAN_B','CPROFILE','TORCH_CPU_PROFILE','CLEAN_C']
    if out.exists() or not out.is_relative_to(ROOT/'runs') or cfg['cases']!=declared or cfg['num_envs']!=32 or cfg['warmup_iterations']!=2:raise ValueError('Fresh registered32-world ordinary profile')
    out.mkdir();protocol=(cp.parent/cfg['protocol']).resolve();paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,op,protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;executed_updates=0;cases=[]
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            torch.set_num_threads(4);runtime=SupportPenRuntime(op,32);learner=SupportPenPPO(runtime,ppo,cfg['arm']);learner.calibrate(cfg['calibration_controls']);warm=[]
            for iteration in range(2):
                learner.collect(64);health=learner.consistency()
                if health['passed']!=health['total']:raise ValueError('Warm collector discrepancy')
                stats=learner.update();executed_updates+=stats['optimizer_steps'];warm.append(dict(health=health,stats=stats))
            fixture=learner.checkpoint(receipt);torch.save(fixture,out/'fixture.pt');fixture_pin=file_pin(out/'fixture.pt');warm_native=observed
            write(out/'warmup.json',dict(warmup=warm,actual_native_integrations=warm_native,optimizer_steps=executed_updates,fixture=fixture_pin))
            for name in cfg['cases']:
                driver='EMPTY_SIGNALS' if comparison and 'FAST' in name else 'UPSTREAM'
                folder=out/name;folder.mkdir();t=time.perf_counter();audit=learner.load(torch.load(verify_file_pin(fixture_pin),map_location='cpu',weights_only=False));restore_wall=time.perf_counter()-t
                if not audit['native']['all_array_bytes_equal'] or not audit['native']['all_scalar_values_equal']:raise ValueError('Fixture readback differs')
                before=observed;resets=runtime.explicit_reset_worlds;cursor=learner.optimizer_steps;timer={};events={};pyprof=cProfile.Profile() if name=='CPROFILE' else None;torchprof=None;monitor=GPUMonitor() if cfg.get('telemetry_enabled',True) else None
                if monitor:monitor.start()
                if name.startswith('TORCH_'):torchprof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=False,with_stack=False,profile_memory=False);torchprof.__enter__()
                if pyprof:pyprof.enable()
                try:
                    for phase,fn in [('collect64',lambda:learner.collect(64,control_driver=driver)),('consistency',learner.consistency),('update32',learner.update)]:
                        torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();begin=time.perf_counter()
                        with torch.profiler.record_function(phase):result=fn()
                        end.record();torch.cuda.synchronize();timer[phase]=time.perf_counter()-begin;events[phase]=start.elapsed_time(end)
                        if phase=='consistency':health=result
                        if phase=='update32':stats=result;executed_updates+=stats['optimizer_steps']
                finally:
                    if pyprof:pyprof.disable()
                    if torchprof:torchprof.__exit__(None,None,None)
                    if monitor:monitor.stop()
                artifacts={}
                if pyprof:
                    pyprof.dump_stats(str(folder/'calls.pstats'));records=[]
                    for (filename,line,func),(primitive,total,self_s,cumulative,callers) in pstats.Stats(pyprof).stats.items():records.append(dict(file=filename,line=line,function=func,primitive_calls=primitive,total_calls=total,self_s=self_s,cumulative_s=cumulative))
                    records.sort(key=lambda r:r['cumulative_s'],reverse=True);write(folder/'python_calls.json',records);artifacts.update(pstats=file_pin(folder/'calls.pstats'),python_calls=file_pin(folder/'python_calls.json'))
                if torchprof:
                    torchprof.export_chrome_trace(str(folder/'torch_cpu_trace.json'));operators=[dict(key=e.key,count=e.count,self_cpu_us=e.self_cpu_time_total,total_cpu_us=e.cpu_time_total) for e in torchprof.key_averages()];operators.sort(key=lambda r:r['total_cpu_us'],reverse=True);write(folder/'torch_cpu_operators.json',operators);artifacts.update(torch_trace=file_pin(folder/'torch_cpu_trace.json'),torch_operators=file_pin(folder/'torch_cpu_operators.json'))
                torch.save(cpu_tree(learner.memory),folder/'rollout.pt');torch.save(dict(actor=cpu_tree(learner.actor.state_dict()),critic=cpu_tree(learner.critic.state_dict()),optimizer=cpu_tree(learner.optimizer.state_dict())),folder/'modules.pt')
                m=learner.memory;checks=dict(collector=health['passed']==health['total'],native=observed-before==16384,updates=learner.optimizer_steps-cursor==32,transitions=m['obs'].shape[:2]==(64,32),no_aid=bool(m['no_applied_force'].all()),no_overflow=not bool(m['engine_overflow'].any()))
                row=dict(name=name,control_driver=driver,wall_s=timer,CUDA_ms=events,restore_wall_s=restore_wall,actual_native_integrations=observed-before,optimizer_steps=learner.optimizer_steps-cursor,reset_worlds=runtime.explicit_reset_worlds-resets,
                    terminated_worlds=int(m['terminated'].sum()),truncated_worlds=int(m['truncated'].sum()),collector=health,checks=checks,rollout=file_pin(folder/'rollout.pt'),modules=file_pin(folder/'modules.pt'),
                    artifacts=artifacts,GPU_telemetry=dict(samples=monitor.samples,errors=monitor.errors) if monitor else dict(disabled=True,reason='Isolate ordinary work from background nvidia-smi threads/processes.'),restore_audit=audit)
                cases.append(row);write(folder/'report.json',row);print(json.dumps(dict(name=name,wall_s=timer,checks=checks,reset_worlds=row['reset_worlds'])),flush=True)
                if not all(checks.values()):raise ValueError('Retain ordinary profile discrepancy')
            verify_source_snapshot(receipt,live=True);checks=dict(declared_cases=len(cases)==len(declared),native_total=observed==(165888 if comparison else 116736),optimizer_total=executed_updates==(320 if comparison else 224),all_case_checks=all(all(c['checks'].values()) for c in cases))
            result=dict(schema='ordinary_pen_PPO_driver_comparison:1' if comparison else 'ordinary_pen_PPO_profile:1',config=file_pin(cp),protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),fixture=fixture_pin,warmup=file_pin(out/'warmup.json'),cases=cases,checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,optimizer_steps=executed_updates,
                role=cfg['role'],scope='Fresh early-training32-world actual collector/consistency/update profile. CPU profiler overhead excluded from clean repeats;CUDA event boundaries measured,not GPU-kernel occupancy. No legacy native resume,policy promotion,task/ego or optimization speedup claim.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],native=observed,updates=executed_updates)),flush=True)
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),actual_native_integrations=observed,optimizer_steps=executed_updates));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
