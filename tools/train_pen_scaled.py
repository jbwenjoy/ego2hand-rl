"""Three hours of measured PPO training with phase coverage and strict W1 evaluations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark_pen_scaled import ROOT, OBJECTIVE, COLLISION, factory, batch_runtime, BackendSupportPenPPO, pilot_settings
import torch
from supermimic_research.artifacts import file_pin
from supermimic_research.frozen_sources import freeze_sources
from supermimic_research.learning.native_pen_ppo import cpu_tree
from supermimic_research.learning.pen_matched_recovery import saved_bytes_equal
from supermimic_research.runtime.explicit_collision_backend import backend_scope
from supermimic_research.strict_json import write_json


def atomic_save(value,path):
    temp=path.with_suffix('.tmp')
    torch.save(value,temp)
    os.replace(temp,path)


def main(worlds,out,hours):
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    files=sorted((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'tools/evaluate_pen_scaled.py',ROOT/'tools/benchmark_pen_scaled.py',OBJECTIVE,COLLISION]
    receipt=freeze_sources(files,out/'source_snapshot',ROOT,subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(out/'source_receipt.json',receipt)
    write_json(out/'intent.json',dict(hours_training=hours,worlds=worlds,seed=7411,arm='RESIDUAL',rollout_steps=64,
        second_half='phases 185..368',last_quarter='277..368',start='phase 1, fresh Actor',evaluation='first, each hour and final; 3 numerical repetitions at final',
        benchmarks=[file_pin(ROOT/f'runs/pen_scale_v2/W{w}/report.json') for w in (32,128,256)]))
    elapsed=0.
    next_eval=3600
    learner=None
    try:
        with torch.jit.optimized_execution(False),backend_scope(COLLISION),batch_runtime(OBJECTIVE,worlds,factory()) as runtime:
            cfg=pilot_settings(OBJECTIVE)[1]
            learner=BackendSupportPenPPO(runtime,cfg,'RESIDUAL')
            learner.calibrate(8)
            total_hist=torch.zeros(369,dtype=torch.long)
            with (out/'iterations.jsonl').open('x',encoding='utf8',buffering=1) as log:
                while elapsed < hours*3600:
                    start=time.perf_counter()
                    m=learner.collect(64)
                    health=learner.consistency()
                    if not all(health['checks'].values()):
                        write_json(out/'failed_health.json',health)
                        raise ValueError([k for k,v in health['checks'].items() if not v])
                    phase=m['phase_used']
                    geometry=m['contact_geometry_max']
                    valid=(geometry<=runtime.objective_constants['geometry_limits']).all(-1) & ~m['engine_overflow'] & m['no_applied_force'] & (m['contact_errors']==0) & (m['contact_nonhand_max_sum'][...,0]<=.1)
                    histogram=torch.bincount(phase[valid],minlength=369).cpu()
                    total_hist+=histogram
                    stats=learner.update()
                    if not all(bool(torch.isfinite(p).all()) for module in (learner.actor,learner.critic) for p in module.parameters()):
                        raise ValueError('Nonfinite parameter')
                    torch.cuda.synchronize()
                    elapsed+=time.perf_counter()-start
                    i=learner.iteration
                    row=dict(iteration=i,training_s=elapsed,transitions=learner.training_transitions,optimizer_steps=learner.optimizer_steps,
                        phase_histogram_valid=histogram.tolist(),valid_second_half=int(histogram[185:].sum()),valid_last_quarter=int(histogram[277:].sum()),
                        terminal_reference_count=int(m['truncated'].sum()),max_phase=int(phase.max()),valid_count=int(histogram.sum()))
                    log.write(json.dumps(row)+'\n')
                    final=elapsed>=hours*3600
                    evaluate=i==1 or elapsed>=next_eval or final
                    if evaluate or i%250==0:
                        snap=learner.checkpoint(receipt)
                        path=out/'latest_full.pt'
                        atomic_save(snap,path)
                        if i==1:
                            recovered=torch.load(path,map_location='cpu',weights_only=False)
                            if not saved_bytes_equal(snap,recovered):
                                raise ValueError('First checkpoint disk round trip differs')
                            audit=learner.load(recovered)
                            os.link(path,out/'first_full.pt')
                            # Preserve the exact file independently of subsequent replacement.
                            write_json(out/'first_checkpoint.json',dict(iteration=i,checkpoint=file_pin(out/'first_full.pt'),load=audit))
                            del recovered
                        del snap
                        write_json(out/'latest.json',dict(iteration=i,training_s=elapsed,valid_second_half_total=int(total_hist[185:].sum()),valid_last_quarter_total=int(total_hist[277:].sum())))
                    if evaluate:
                        actor_path=out/f'actor_{i:05}.pt'
                        atomic_save(dict(actor=cpu_tree(learner.actor.state_dict()),iteration=i,config=cfg,
                                         batch_backend=runtime.metadata()['termination_backend']),actor_path)
                        with (out/f'eval_{i:05}.log').open('x',encoding='utf8') as output:
                            subprocess.run([sys.executable,'-X','utf8',str(ROOT/'tools/evaluate_pen_scaled.py'),str(actor_path),str(out/f'eval_{i:05}'),'3' if final else '1'],
                                           stdout=output,stderr=subprocess.STDOUT,check=True)
                        if elapsed>=next_eval: next_eval+=3600
            write_json(out/'report.json',dict(iterations=learner.iteration,transitions=learner.training_transitions,optimizer_steps=learner.optimizer_steps,
                       seconds=elapsed,valid_phase_counts=total_hist.tolist(),evaluations=[file_pin(path) for path in sorted(out.glob('eval_*/report.json'))]))
    except BaseException as error:
        write_json(out/'failure.json',dict(error=repr(error),training_s=elapsed,iteration=getattr(learner,'iteration',None)))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('worlds',type=int)
    p.add_argument('out',type=Path)
    p.add_argument('--hours',type=float,default=3)
    a=p.parse_args()
    main(a.worlds,a.out.resolve(),a.hours)

