"""Profile actual native collection and discarded update at matched budgets."""
import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import functools
import json
import statistics
import subprocess
import sys
import time
import tomllib
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
import warp as wp
from torch.utils._python_dispatch import TorchDispatchMode
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.frozen_sources import freeze_sources,verify_source_snapshot,verify_source_pin
from supermimic_research.runtime.pen_native_policy import NativePenPolicyAdapter,NativePenFeatures
from supermimic_research.learning.native_pen_ppo import NativePenPPO,cpu_tree
from supermimic_research.runtime.resident_contact_metrics import ResidentContactMetrics
from supermimic_research.runtime.pen_portable_state import engine_sources


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


class OperationCounts(TorchDispatchMode):
    def __init__(self):super().__init__();self.calls=Counter();self.sites=Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kwargs=kwargs or {};name=str(func);self.calls[name]+=1
        if name.startswith(('aten._local_scalar_dense.','aten.nonzero.')):
            frames=traceback.extract_stack(limit=30)
            sites=[f'{f.filename}:{f.lineno}:{f.name}' for f in frames if (('supermimic-research' in f.filename or 'site-packages\\mjlab' in f.filename) and f.name!='__torch_dispatch__')]
            self.sites[(name,' > '.join(sites[-5:]))]+=1
        return func(*args,**kwargs)
    def report(self):
        return dict(operations=dict(self.calls),synchronization_sites=[dict(operation=k[0],stack=k[1],calls=v) for k,v in self.sites.most_common()],
            scope='ATen dispatch counts; nonzero has dynamic output and may synchronize. Warp/driver/Python work is not represented by these counts.')


class Ranges:
    def __init__(self):self.restore=[];self.calls=Counter()
    def wrap(self,cls,name,label):
        own=name in cls.__dict__;old=getattr(cls,name)
        @functools.wraps(old)
        def wrapped(*a,**kw):
            self.calls[label]+=1
            with torch.profiler.record_function(label):return old(*a,**kw)
        setattr(cls,name,wrapped);self.restore.append((cls,name,old,own))
    def close(self):
        for cls,name,old,own in reversed(self.restore):
            if own:setattr(cls,name,old)
            else:delattr(cls,name)


def timing(fn):
    torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();begin=time.perf_counter();result=fn();dispatch=time.perf_counter()-begin
    end.record();end.synchronize();wall=time.perf_counter()-begin
    return result,dict(host_dispatch_s=dispatch,host_complete_s=wall,GPU_stream_interval_s=start.elapsed_time(end)/1000)


def main(a):
    cp=a.config.resolve();cfg=tomllib.loads(cp.read_text());out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'runs') or (cfg['num_envs'],cfg['controls'],cfg['warmup_rollouts'],cfg['instrumented_rollouts'])!=(32,64,1,1):raise ValueError('Fresh declared collection profile')
    out.mkdir();trainpath=(cp.parent/cfg['training_config']).resolve();train=tomllib.loads(trainpath.read_text());ap=(trainpath.parent/train['adapter_config']).resolve()
    paths=sorted({p for pkg in ['supermimic_research','dex_supermimic','chord_study'] for p in (ROOT/'src'/pkg).rglob('*.py')})+[Path(__file__).resolve(),cp,trainpath,ap]
    receipt=freeze_sources(paths,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip());write(out/'source_receipt.json',receipt)
    lp=(cp.parent/cfg['prior_ledger']).resolve();prior=json.loads(lp.read_text());first=json.loads(verify_file_pin(prior['roots']['first2']).read_text());initial_pin=first['initial_state']
    rp=(cp.parent/cfg['resident_ledger']).resolve();resident=json.loads(rp.read_text());aliases=json.loads(verify_file_pin(resident['source_resolutions']).read_text())
    manifest=json.loads(verify_file_pin(resident['verified_manifest']).read_text())
    for e in manifest['records']:verify_source_pin(e['observed'],aliases)
    source=torch.load(verify_file_pin(initial_pin),map_location='cpu',weights_only=False)
    from mjlab.sim.sim import Simulation
    from mjlab.envs import ManagerBasedRlEnv
    oldstep=Simulation.step;oldadapter=NativePenPolicyAdapter.step;observed=0;discarded_updates=0;adapter=None;metric=None;phase='construction';rows=[];audits=[]
    def step(sim):
        nonlocal observed
        oldstep(sim);observed+=sim.num_envs
        if metric is not None:metric.sample()
    def adapterstep(self,*args,**kwargs):
        if metric is not None:metric.reset_window()
        return oldadapter(self,*args,**kwargs)
    Simulation.step=step;NativePenPolicyAdapter.step=adapterstep
    try:
        with torch.jit.optimized_execution(False):
            adapter=NativePenPolicyAdapter(ap,32);trainer=NativePenPPO(adapter,train['ppo'])
            w=torch.arange(32,device='cuda:0')[None,:,None];j=torch.arange(20,device='cuda:0')[None,None,:];t=torch.arange(64,device='cuda:0')[:,None,None]
            actions=.05*torch.sin(.2*t+.13*w+.03*j);torch.save(actions.cpu(),out/'forced_actions.pt')
            def restore():
                audit=trainer.load(copy.deepcopy(source));audits.append(dict(modules=audit['modules_equal'],optimizer=audit['optimizer_equal'],observation=audit['observation_equal'],normalization=audit['normalization_equal'],
                    native_arrays=audit['native']['all_array_bytes_equal'],native_scalars=audit['native']['all_scalar_values_equal']))
            def collect():return trainer.collect(64,forced_actions=actions)
            phase='warmup';restore();collect();trainer.update();discarded_updates+=32
            for mode in ['baseline','resident']:
                if mode=='resident':
                    sim=adapter.env.sim;m=sim.mj_model;jids=[m.joint('robot/'+n).id for n in adapter.motion.joint_names]
                    metric=ResidentContactMetrics(m,sim.wp_model,sim.wp_data,adapter.features.body,jids)
                for repeat in range(cfg['repeats']):
                    phase=f'{mode}_{repeat}';restore();before=observed
                    _,collection=timing(collect);consistency=trainer.consistency()
                    memory=out/f'{mode}_{repeat}_memory.pt';torch.save(cpu_tree(trainer.memory),memory)
                    _,update=timing(trainer.update);discarded_updates+=32
                    row=dict(mode=mode,repeat=repeat,collection=collection,update=update,consistency=consistency,memory=file_pin(memory),native_integrations=observed-before,
                        discarded_optimizer_steps=32,terminated=consistency['terminated'],metric_metadata=None if metric is None else metric.metadata())
                    rows.append(row);write(out/'progress.json',dict(phase=phase,observed_native_integrations=observed,discarded_optimizer_steps=discarded_updates,rows=rows))
            metric=None;phase='instrumented_baseline';restore();counts=OperationCounts();ranges=Ranges()
            for cls,names,prefix in [(NativePenPolicyAdapter,['step','reset','observation'],'adapter'),(NativePenFeatures,['policy','critic','reference_pose_world','mount','pose'],'features'),
                (Simulation,['step','forward','sense'],'simulation'),(ManagerBasedRlEnv,['step','reset'],'environment')]:
                for name in names:ranges.wrap(cls,name,'native_profile/'+prefix+'.'+name)
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
                    with counts:collect()
                prof.export_chrome_trace(str(out/'collection_trace.json'))
                summaries=[]
                for event in prof.key_averages():
                    summaries.append(dict(name=event.key,count=event.count,self_CPU_us=event.self_cpu_time_total,CPU_total_us=event.cpu_time_total,
                        device_total_us=getattr(event,'device_time_total',0),self_device_us=getattr(event,'self_device_time_total',0)))
                write(out/'operation_counts.json',dict(**counts.report(),range_calls=dict(ranges.calls),events=sorted(summaries,key=lambda x:x['self_CPU_us'],reverse=True)))
            finally:ranges.close()
            instrumented_consistency=trainer.consistency();_,diagnostic_update=timing(trainer.update);discarded_updates+=32
            checks=dict(all_restores=all(all(x.values()) for x in audits),all_collection_consistent=all(r['consistency']['passed']==r['consistency']['total'] for r in rows),
                instrumented_consistent=instrumented_consistency['passed']==instrumented_consistency['total'],matched_per_rollout_integrations=all(r['native_integrations']==16384 for r in rows),
                actual_native_budget=observed==131072,discarded_update_budget=discarded_updates==256,initial_source_unchanged=file_pin(verify_file_pin(initial_pin))==initial_pin)
            summaries={}
            for mode in ['baseline','resident']:
                selected=[r for r in rows if r['mode']==mode]
                summaries[mode]={key:{field:statistics.median(r[key][field] for r in selected) for field in selected[0][key]} for key in ['collection','update']}
            verify_source_snapshot(receipt,live=True)
            result=dict(schema='native_collection_computational_profile:1',config=file_pin(cp),training_config=file_pin(trainpath),initial_state=initial_pin,resident_evidence=file_pin(rp),source_receipt=file_pin(out/'source_receipt.json'),
                engine_sources=engine_sources(),actions=file_pin(out/'forced_actions.pt'),rows=rows,medians=summaries,load_audits=audits,checks=checks,passed=sum(checks.values()),total=len(checks),
                operation_counts=file_pin(out/'operation_counts.json'),trace=file_pin(out/'collection_trace.json'),actual_native_integrations=observed,discarded_optimizer_steps=discarded_updates,
                retained_policy_optimizer_steps=0,instrumented_consistency=instrumented_consistency,instrumented_update=diagnostic_update,scope=cfg['scope']+' Profile mode has Python dispatch/trace overhead and is excluded from throughput medians; forced actions are identical but W32 future trajectories need not be bitwise equal.')
            write(out/'report.json',result);print(json.dumps(dict(report=file_pin(out/'report.json'),checks=[result['passed'],result['total']],medians=summaries,integrations=observed,discarded_updates=discarded_updates)),flush=True)
            if not all(checks.values()):raise ValueError('Keep computational profiling discrepancies')
    except Exception as error:
        write(out/'failure.json',dict(type=type(error).__name__,message=str(error),phase=phase,observed_native_integrations=observed,discarded_optimizer_steps=discarded_updates))
        raise
    finally:
        Simulation.step=oldstep;NativePenPolicyAdapter.step=oldadapter
        if adapter:adapter.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
