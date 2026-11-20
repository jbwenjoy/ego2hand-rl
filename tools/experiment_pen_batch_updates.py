"""Equal-sample on-policy PPO continuation and same-rollout optimizer branches."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
import torch
from benchmark_pen_large_batches import batch_runtime,sample_board
from benchmark_pen_scaled import ROOT,OBJECTIVE,COLLISION,factory,pilot_settings,BackendSupportPenPPO
from dex_supermimic.rl.dex_supermimic.ppo import ppo_update
from supermimic_research.learning.native_pen_ppo import model_pair,cpu_tree
from supermimic_research.runtime.explicit_collision_backend import backend_scope
from supermimic_research.artifacts import file_pin,verify_file_pin
from supermimic_research.strict_json import write_json
from supermimic_research.frozen_sources import freeze_sources
from pen_ppo_shift import policy_shift

BUDGET=1048576


def save_actor(out,name,actor,learner,iteration):
    path=out/(name+'.pt')
    torch.save(dict(actor=cpu_tree(actor.state_dict()),iteration=iteration,config=vars(learner.cfg),
                    batch_backend=learner.adapter.metadata()['termination_backend']),path)
    return path


def branches(out,learner,old,index):
    results=[]
    for steps in (32,64,128):
        cfg=copy.copy(learner.cfg);cfg.ppo_minibatches=steps//4
        with torch.random.fork_rng(devices=[0]):
            a,c,opt=model_pair(learner.device,cfg)
            a.load_state_dict(old.state_dict());c.load_state_dict(learner.critic.state_dict())
            opt.load_state_dict(copy.deepcopy(learner.optimizer.state_dict()))
            torch.manual_seed(7411)
            stats=ppo_update(a,c,opt,learner.memory,learner.observations['critic'],cfg)
            assert stats['optimizer_steps']==steps
            shift=policy_shift(old,a,learner.memory,cfg.ppo_clip)
            path=save_actor(out,f'shared_steps_{steps}',a,learner,2781+index)
            results.append(dict(steps=steps,shift=shift,actor=file_pin(path)))
            del a,c,opt
    write_json(out/'branches.json',dict(rollout=index,results=results))
    return [Path(r['actor']['path']) for r in results]


def main(worlds,steps,out,source_pin):
    out.mkdir(exist_ok=False,parents=True)
    files=[Path(__file__),ROOT/'tools/pen_ppo_shift.py',ROOT/'tools/benchmark_pen_scaled.py',ROOT/'tools/benchmark_pen_large_batches.py',
           ROOT/'tools/evaluate_pen_scaled.py',OBJECTIVE,COLLISION]+sorted((ROOT/'src').rglob('*.py'))
    receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    saved=torch.load(verify_file_pin(source_pin),map_location='cpu',weights_only=False)
    if saved['iteration']!=2781 or saved['metadata']['arm']!='RESIDUAL':raise ValueError('Wrong starting checkpoint')
    torch.set_num_threads(4)
    samples,errors,rows,evals=[],[],[],[]
    stop=threading.Event();watch=threading.Thread(target=sample_board,args=(stop,samples,errors),daemon=True)
    watch.start();stage='construction'
    try:
        with torch.jit.optimized_execution(False),backend_scope(COLLISION),batch_runtime(OBJECTIVE,worlds,factory()) as runtime:
            cfg=pilot_settings(OBJECTIVE)[1];cfg['ppo_minibatches']=steps//4
            learner=BackendSupportPenPPO(runtime,cfg,'RESIDUAL')
            learner.actor.load_state_dict(saved['modules']['actor']);learner.critic.load_state_dict(saved['modules']['critic'])
            learner.optimizer.load_state_dict(saved['optimizer']);learner.normalizers_ready=True
            del saved
            for index in range(1,BUDGET//(64*worlds)+1):
                stage=f'collect_{index}'
                torch.cuda.synchronize();start=time.perf_counter()
                m=learner.collect(64);torch.cuda.synchronize();collected=time.perf_counter()
                health=learner.consistency()
                if not all(health['checks'].values()):
                    write_json(out/'failed_health.json',health);raise ValueError('Collector consistency failed')
                check_end=time.perf_counter()
                old=copy.deepcopy(learner.actor).eval().requires_grad_(False)
                if (worlds,steps,index)==(1024,32,4):
                    # Stage branches outside all timed intervals, after collection and before the original update.
                    evals.extend(branches(out,learner,old,index));torch.cuda.empty_cache()
                stage=f'update_{index}'
                torch.cuda.synchronize();upd_start=time.perf_counter()
                torch.cuda.reset_peak_memory_stats()
                stats=learner.update();torch.cuda.synchronize();end=time.perf_counter()
                if stats['optimizer_steps']!=steps or stats['omitted_transitions_per_epoch'] or not all(bool(torch.isfinite(p).all()) for module in (learner.actor,learner.critic) for p in module.parameters()):
                    raise ValueError('Invalid update')
                peak=torch.cuda.max_memory_allocated()/2**20
                shift=policy_shift(old,learner.actor,m,cfg['ppo_clip']);del old
                valid=((m['contact_geometry_max']<=runtime.objective_constants['geometry_limits']).all(-1)&~m['engine_overflow']&
                       m['no_applied_force']&(m['contact_errors']==0)&(m['contact_nonhand_max_sum'][...,0]<=.1))
                row=dict(index=index,total=worlds*64,valid=int(valid.sum()),late=int((valid&(m['phase_used']>=185)).sum()),
                         start=start,collected=collected,check_end=check_end,upd_start=upd_start,end=end,torch_peak_mib=peak,shift=shift)
                rows.append(row)
                with (out/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            evals.append(save_actor(out,'final_actor',learner.actor,learner,2781+learner.iteration))
        timed=[r for r in rows if r['index']!=1 and not (worlds==1024 and steps==32 and r['index']==4)]
        seconds=sum(r['check_end']-r['start']+r['end']-r['upd_start'] for r in timed)
        use=[s for s in samples if any(r['start']<=s['start']<=s['end']<=r['check_end'] or r['upd_start']<=s['start']<=s['end']<=r['end'] for r in timed)]
        result=dict(worlds=worlds,steps_per_rollout=steps,transitions=sum(r['total'] for r in rows),added_updates=learner.optimizer_steps,
                    valid_per_s=sum(r['valid'] for r in timed)/seconds,raw_per_s=sum(r['total'] for r in timed)/seconds,
                    collection_s=sum(r['collected']-r['start'] for r in timed),update_s=sum(r['end']-r['upd_start'] for r in timed),
                    measured_s=seconds,board_peak_mib=max(s['mib'] for s in use),mean_watts=float(np.mean([s['watts'] for s in use])),
                    peak_watts=max(s['watts'] for s in use),torch_update_peak_mib=max(r['torch_peak_mib'] for r in timed),
                    late_count=sum(r['late'] for r in rows),source=source_pin,actors=[file_pin(p) for p in evals])
        write_json(out/'training_report.json',result)
    except BaseException as error:
        write_json(out/'failure.json',dict(stage=stage,error=repr(error),complete_rollouts=len(rows)));raise
    finally:
        stop.set();watch.join(6);write_json(out/'telemetry.json',dict(samples=samples,errors=errors))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('worlds',type=int);p.add_argument('steps',type=int);p.add_argument('out',type=Path);p.add_argument('pin',type=Path)
    a=p.parse_args();main(a.worlds,a.steps,a.out.resolve(),json.loads(a.pin.read_bytes()))
