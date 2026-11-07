"""One fixed first-schedule job for engineering qualification, separate cohort.

Default is stdlib CPU preflight. --run requires external GPU authorization.
Task quality never selects/stops/retries a process. No result enters the27jobs.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q',ROOT/'scripts/run_c2_portfolio_queue.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/format103-qualification-v1.json')
    parser.add_argument('--run',action='store_true');a=parser.parse_args()
    started=time.monotonic();plan,worker,life=q.preflight(a.plan.resolve(strict=True))
    job=plan['jobs'][0]
    worker.require(plan['qualification_job_id']==job['job_id']=='repeat1_trainseed4_A1L','Fixed engineering job changed')
    if not a.run:
        print({'status':'passed_CPU_preflight','GPU_started':False,'job_id':job['job_id'],'plan_sha256':worker.sha(a.plan)});return
    root=Path(plan['output_root']);worker.require(not root.exists(),'No overwrite/retry');root.mkdir(parents=True)
    state={'status':'starting','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat(),
      'plan_sha256':worker.sha(a.plan),'job_id':job['job_id'],'scope':'one engineering job, never pooled into confirmation27'}
    write=lambda:life.write(root/'state.json',state)
    write();plan['plan_sha256_runtime']=state['plan_sha256']
    try:
        with life.gpu_lock():
            life.reject_project_gpu_workers()
            state['status']='running';write()
            observer=lambda pid,parent:life.observe_actual_pid(pid,parent,plan['source_pins']['worker']['path'],a.plan,job['output_dir'])
            state['process']=life.watch(job['command'],Path(job['wrapper_metadata_dir']),root/'qualification_process',hard_seconds=45,observer=observer)
            worker.require(state['process']['status']=='completed' and state['process']['exit_code']==0,'Engineering failure; no retry')
            state['health']=q.health(plan,job,worker)
            state['status']='engineering_passed'
    except BaseException as error:
        state.update(status='engineering_failed',error=repr(error));raise
    finally:
        state.update(finished_utc=datetime.now(timezone.utc).isoformat(),whole_wall_seconds=time.monotonic()-started);write()

if __name__=='__main__':main()
