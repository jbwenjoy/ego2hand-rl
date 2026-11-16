"""One real ordinary rollout/update with bounded CPU+CUDA activity recording."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot
from supermimic_research.runtime.pen_support_runtime import SupportPenRuntime
from supermimic_research.learning.pen_support_ppo import SupportPenPPO,pilot_settings
from supermimic_research.learning.native_pen_ppo import cpu_tree


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main(a):
    run=json.loads(a.comparison.read_text());out=a.output.resolve();op=ROOT/'configs/pen_centered_support_training_v1.toml';objective,ppo=pilot_settings(op)
    if out.exists() or not out.is_relative_to(ROOT/'runs') or run['passed']!=run['total'] or torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():raise ValueError('Fresh healthy current-source fixture and CUDA profiler required')
    out.mkdir();protocol=ROOT/'docs/pen-ordinary-gpu-timeline-protocol-v1.md';paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),protocol]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    from mjlab.sim.sim import Simulation
    original=Simulation.step;observed=0;runtime=None;updates=0
    def step(sim):
        nonlocal observed
        original(sim);observed+=sim.num_envs
    Simulation.step=step
    try:
        with torch.jit.optimized_execution(False):
            torch.set_num_threads(4);runtime=SupportPenRuntime(op,32);learner=SupportPenPPO(runtime,ppo,'SUPPORT');audit=learner.load(torch.load(verify_file_pin(run['fixture']),map_location='cpu',weights_only=False))
            wall={};prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False,profile_memory=False)
            with prof:
                for name,fn in [('collect64',lambda:learner.collect(64)),('consistency',learner.consistency),('update32',learner.update)]:
                    with torch.profiler.record_function(name):
                        torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize();wall[name]=time.perf_counter()-start
                    if name=='consistency':health=result
                    if name=='update32':updates=result['optimizer_steps']
            prof.export_chrome_trace(str(out/'trace.json'));averages=[dict(key=e.key,count=e.count,cpu_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,device_us=e.device_time_total,self_device_us=e.self_device_time_total) for e in prof.key_averages()];write(out/'key_averages.json',averages)
            trace=json.loads((out/'trace.json').read_text());events=trace['traceEvents'];kernels=[e for e in events if e.get('cat')=='kernel' and 'dur' in e];groups={}
            for e in kernels:
                g=groups.setdefault(e['name'],dict(name=e['name'],count=0,sum_us=0.));g['count']+=1;g['sum_us']+=e['dur']
            phases={}
            for name in wall:
                label=next(e for e in events if e.get('name')==name and e.get('ph')=='X');selected=[e for e in kernels if label['ts']<=e['ts'] and e['ts']+e['dur']<=label['ts']+label['dur']]
                phases[name]=dict(CPU_label_us=label['dur'],GPU_kernel_count=len(selected),GPU_kernel_sum_us=sum(e['dur'] for e in selected))
            torch.save(cpu_tree(learner.memory),out/'rollout.pt');checks=dict(native=observed==16384,updates=updates==32,collector=health['passed']==health['total'],exact_restore=audit['native']['all_array_bytes_equal'] and audit['native']['all_scalar_values_equal'],CUDA_kernels_present=bool(kernels),no_aid=bool(learner.memory['no_applied_force'].all()),no_overflow=not bool(learner.memory['engine_overflow'].any()))
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='ordinary_pen_GPU_timeline:1',comparison=file_pin(a.comparison),fixture=run['fixture'],protocol=file_pin(protocol),source_receipt=file_pin(out/'source_receipt.json'),trace=file_pin(out/'trace.json'),key_averages=file_pin(out/'key_averages.json'),rollout=file_pin(out/'rollout.pt'),
                instrumented_wall_s=wall,phases=phases,kernel_groups=sorted(groups.values(),key=lambda g:g['sum_us'],reverse=True),kernel_count=len(kernels),categories={cat:sum(e.get('cat')==cat for e in events) for cat in sorted({e['cat'] for e in events if 'cat' in e})},
                checks=checks,passed=sum(checks.values()),total=len(checks),actual_native_integrations=observed,optimizer_steps=updates,collector=health,
                scope='One current-source instrumented UPSTREAM rollout/update. Kernel durations can overlap;not clean speed,occupancy or main-policy promotion. Original source/state/gates intact;no installed-package/driver/CUDA changes.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],phases=phases,kernels=len(kernels))),flush=True)
            if not all(checks.values()):raise ValueError('Retain GPU timeline qualification discrepancy')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),actual_native_integrations=observed,optimizer_steps=updates));raise
    finally:
        if runtime:runtime.close()
        Simulation.step=original


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
