"""Fixed-parent private GPU update benchmark, not a policy learning run.

Each variant uses the same paired100 state, five Adam states, RNG and batches.
Private updates are discarded; only numerical comparison arrays/timings remain.
No resumable checkpoint, native simulation, model selection or long training.
"""
from pathlib import Path
import argparse, hashlib, json, os, statistics, subprocess, sys, threading, time, traceback
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def pin(path):
    path=Path(path).resolve(strict=True);b=path.read_bytes()
    return dict(path=str(path),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())

def write(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,default=ROOT/'src');p.add_argument('--output',type=Path,required=True);p.add_argument('--microbatch',type=int,default=2);p.add_argument('--warmup',type=int,default=1);p.add_argument('--measured',type=int,default=2);p.add_argument('--profile',action='store_true');args=p.parse_args()
    out=args.output.resolve();assert not out.exists();out.mkdir(parents=True)
    sys.path.insert(0,str(args.source_root.resolve(strict=True)))
    from supermimic_research.learning import runner as R,world_model as W,update_step as U
    from supermimic_research.runtime import structured as C,phone_task as T
    config=(ROOT/'configs/phone_offline.toml').read_text('utf8').replace('manifest = "artifacts.toml"','manifest = '+json.dumps(str(ROOT/'configs/artifacts.toml'))).replace('device = "cpu"','device = "cuda:0"')
    config_path=out/'configuration.toml';config_path.write_text(config,'utf8')
    started=time.perf_counter();phase='setup';stop=threading.Event();samples=[]
    def sample_gpu():
        while not stop.is_set():
            q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
            if q.returncode==0:
                try:samples.append(dict(elapsed_s=time.perf_counter()-started,**dict(zip(['utilization_pct','memory_used_MiB','memory_total_MiB','power_W'],[float(x.strip()) for x in q.stdout.strip().split(',')]))))
                except ValueError:pass
            stop.wait(.5)
    sampler=threading.Thread(target=sample_gpu,daemon=True);sampler.start()
    try:
        s=R._setup(config_path,smoke_only=False);torch=s['torch'];assert s['device']=='cuda:0' and torch.cuda.is_available()
        parent_pin=s['registry'].pin('paired100_checkpoint');parent=torch.load(parent_pin['path'],map_location='cpu',weights_only=False)
        assert parent['paired_round']==100 and parent['shared_WM_updates']==200
        model=s['model'];arms=s['arms'];model.load_state_dict(parent['shared_WM']);U.restore_arm_states(arms,parent['arms'])
        norm=W.move_tensors(parent['WM_normalizers'],'cuda:0');physical=W.move_tensors(parent['physical_context'],'cuda:0');scales=W.move_tensors(parent['WM_loss_scales'],'cuda:0')
        opts={'WM':U.make_optimizer(model,learning_rate=.0001)};arm_opts={}
        for mode,a in arms.items():
            arm_opts[mode]={'actor':U.make_optimizer(a['actor'],learning_rate=.0001),'critic':U.make_optimizer(a['critic'],learning_rate=.0003)}
            opts[mode+'_Actor']=arm_opts[mode]['actor'];opts[mode+'_Critic']=arm_opts[mode]['critic']
        for key,opt in opts.items():opt.load_state_dict(parent['optimizers'][key])
        s['generator'].set_state(parent['sample_generator'])
        import random
        random.setstate(parent['RNG']['python']);np.random.set_state(parent['RNG']['numpy']);torch.set_rng_state(parent['RNG']['torch']);torch.cuda.set_rng_state(parent['CUDA_RNG'],'cuda:0')
        d=s['dataset'].canonical;eligible=s['dataset'].actor_indices(2)
        bank={key:torch.tensor(d[source],dtype=dtype,device='cuda:0') for key,source,dtype in [('x','x58',torch.float64),('m','memory83',torch.float64),('obs','obs222',torch.float32),('obs1','next_obs222',torch.float32),('terminated','actor_absorbing_done',torch.bool),('reference_timeout','reference_timeout',torch.bool)]}
        rewards={};reward_scope={}
        with torch.no_grad():
            for mode,task in s['tasks'].items():
                rr=torch.full((len(d['row_id']),),float('nan'),dtype=torch.float64,device='cuda:0')
                for offset in range(0,len(eligible),8):
                    ii=eligible[offset:offset+8];rv=task.rewards(torch.tensor(d['y58'][ii],dtype=torch.float64,device='cuda:0'),task.unflatten_memory(torch.tensor(d['memory83'][ii],dtype=torch.float64,device='cuda:0')),torch.tensor(d['action20'][ii],dtype=torch.float64,device='cuda:0'))
                    assert rv['objective_known'].all() and torch.isfinite(rv['total']).all();rr[torch.tensor(ii,device='cuda:0')]=rv['total']
                rewards[mode]=rr;reward_scope[mode]=len(eligible)
        calls=dict(FK=0,directional=0)
        original_poses=s['geometry'].provider.poses;original_features=s['geometry'].features
        def poses(*a,**kw):
            calls['FK']+=1
            with torch.profiler.record_function('geometry.FK'):return original_poses(*a,**kw)
        def features(*a,**kw):
            calls['directional']+=1
            with torch.profiler.record_function('geometry.directional'):return original_features(*a,**kw)
        s['geometry'].provider.poses=poses;s['geometry'].features=features
        torch.cuda.synchronize();setup_s=time.perf_counter()-started;torch.cuda.reset_peak_memory_stats();rows=[];fixture={}
        update_config=U.UpdateConfig(imagination_horizon=2,microbatch=args.microbatch)
        def step(index,label):
            nonlocal fixture
            before=calls.copy();torch.cuda.synchronize();tick=time.perf_counter()
            with torch.profiler.record_function('paired.WM_supervised'):
                wm=U.supervised_WM_step(model,opts['WM'],R._sample(s['new'],s['old'],s['generator'],s['c'],torch),norm,scales,physical,codec=C,phase='BENCH_WM_'+str(index))
            pick=torch.randint(len(eligible),(8,),generator=s['generator'],device='cpu').numpy();ii=eligible[pick];idx=torch.tensor(ii,device='cuda:0');real={key:value[idx] for key,value in bank.items()}
            with torch.profiler.record_function('paired.Actor_Critic'):
                result=U.matched_actor_critic_step(world_model=model,arms=arms,optimizers=arm_opts,real_batch=real,mode_rewards={m:v[idx] for m,v in rewards.items()},tasks=s['tasks'],geometry=s['geometry'],codec=C,norms=norm,physical=physical,wm_step=T.wm_step_current_only,config=update_config,phase='BENCH_'+str(index))
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            row=dict(index=index,phase=label,seconds=seconds,row_ids=d['row_id'][ii].tolist(),WM_loss=float(wm['loss']),modes={m:{k:float(v[k]) for k in ['real_TD_loss','imagined_Actor_loss']} for m,v in result.items()},geometry_calls={k:calls[k]-before[k] for k in calls})
            rows.append(row);print(json.dumps(row),flush=True)
            if label=='measured' and index==args.warmup+args.measured-1:
                groups={'WM':wm['gradient_snapshot']}|{m+'_'+kind.upper():record['gradient_snapshots'][kind] for m,record in result.items() for kind in ['actor','critic']}
                for group,gr in groups.items():
                    for side in ['pre_clip','post_clip']:
                        for name,v in gr[side].items():fixture[group+'/'+side+'/'+name]=v.detach().cpu().numpy().copy()
                for name,network in [('WM',model),*[(m+'_'+key,a[key]) for m,a in arms.items() for key in ['actor','critic','target_critic']]]:
                    for key,v in network.state_dict().items():fixture[name+'/state/'+key]=v.detach().cpu().numpy().copy()
            return row
        for i in range(args.warmup):phase='warmup';step(i,'warmup')
        for i in range(args.measured):phase='measured';step(args.warmup+i,'measured')
        activities=torch.profiler.supported_activities();profile_rows=[]
        if args.profile:
            phase='profiler'
            with torch.profiler.profile(activities=[v for v in [torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA] if v in activities],record_shapes=False,profile_memory=False,with_stack=False) as prof:step(args.warmup+args.measured,'profiler')
            prof.export_chrome_trace(str(out/'trace.json'))
            for event in sorted(prof.key_averages(),key=lambda x:x.self_cpu_time_total,reverse=True)[:60]:
                profile_rows.append(dict(name=event.key,count=event.count,self_CPU_us=event.self_cpu_time_total,total_CPU_us=event.cpu_time_total,self_device_us=getattr(event,'self_device_time_total',0.)))
            write(out/'profiler_top.json',profile_rows)
        np.savez_compressed(out/'numerical_fixture.npz',**fixture)
        stop.set();sampler.join(timeout=10);write(out/'GPU_samples.json',samples)
        timed=[v['seconds'] for v in rows if v['phase']=='measured'];measurement_samples=[v for v in samples if v['elapsed_s']>=setup_s]
        report=dict(schema='supermimic_fixed_private_GPU_update_benchmark:1',status='COMPLETE_PRIVATE_TIMING_NOT_POLICY_TRAINING',source_root=str(args.source_root.resolve()),source_pins=R.source_pins(),tool=pin(__file__),parent_checkpoint=parent_pin,configuration=pin(config_path),device=torch.cuda.get_device_name(0),NN_dtype='float32',physical_dtype='float64',AMP=False,TF32=False,microbatch=args.microbatch,B=8,H=2,setup_s=setup_s,rows=rows,steady_seconds_mean=statistics.mean(timed),steady_seconds_median=statistics.median(timed),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),GPU_util_median=statistics.median(v['utilization_pct'] for v in measurement_samples) if measurement_samples else None,geometry_calls=calls,profile_activities=[str(v) for v in activities],profiler_top=profile_rows,fixture=pin(out/'numerical_fixture.npz'),real_bank_resident=True,real_bank_scope='Same canonical row bank resident for both variants; measures shared update core, not runner CPU-upload overhead',WM_sources=dict(tripod=153,old=684),private_paired_updates=len(rows),private_Adam_steps=5*len(rows),no_native_or_policy_qualification=True,scope='Short private parameter-update timing/numerical test. No saved policy/resumable checkpoint, long training, data fitting selection, source physics or precision change. Setup/reward precompute and fixture transfers excluded from steady timing; profiler step excluded from timing summary.')
        write(out/'report.json',report);print(json.dumps(dict(report=pin(out/'report.json'),seconds=report['steady_seconds_mean'],GPU=report['GPU_util_median'])),flush=True)
    except BaseException as error:
        stop.set();sampler.join(timeout=10);write(out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
