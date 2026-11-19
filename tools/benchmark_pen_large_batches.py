"""Measure usable contact-filtered samples, power and memory during PPO collection."""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from benchmark_pen_scaled import ROOT, COLLISION, OBJECTIVE, factory, pilot_settings, BackendSupportPenPPO
import numpy as np
import torch
from supermimic_research.runtime.explicit_collision_backend import backend_scope
from supermimic_research.runtime.pen_large_batch_runtime import batch_runtime
from supermimic_research.strict_json import write_json


def sample_board(stop, samples, errors):
    args = ['nvidia-smi', '--id=0', '--query-gpu=utilization.gpu,memory.used,power.draw',
            '--format=csv,noheader,nounits']
    while not stop.is_set():
        start = time.perf_counter()
        try:
            raw = subprocess.check_output(args, text=True, timeout=5, creationflags=subprocess.CREATE_NO_WINDOW)
            util, mib, watts = map(float, raw.strip().split(','))
            samples.append(dict(start=start, end=time.perf_counter(), utilization=util, mib=mib, watts=watts))
        except (ValueError, subprocess.SubprocessError, OSError) as error:
            errors.append(str(error))
        stop.wait(.2)


def summarize(rows, samples):
    measured = rows[1:]
    seconds = sum(row['end']-row['start'] for row in measured)
    use = [s for s in samples if any(r['start'] <= s['start'] <= s['end'] <= r['end'] for r in measured)]
    if len(measured)!=4 or not use:
        raise ValueError('Incomplete collection timing or GPU measurements')
    valid = sum(r['valid'] for r in measured)
    total = sum(r['total'] for r in measured)
    return dict(valid_steps_s=valid/seconds, raw_steps_s=total/seconds, valid_fraction=valid/total,
                valid_steps=valid, measured_steps=total, collection_seconds=seconds, measured_samples=len(use),
                board_mean_W=float(np.mean([s['watts'] for s in use])), board_peak_W=max(s['watts'] for s in use),
                board_peak_MiB=max(s['mib'] for s in use), whole_run_board_peak_MiB=max(s['mib'] for s in samples),
                board_median_util_pct=float(np.median([s['utilization'] for s in use])))


def main(worlds,out):
    out.mkdir(parents=True,exist_ok=False)
    rows, samples, errors = [], [], []
    stop = threading.Event()
    watcher = threading.Thread(target=sample_board,args=(stop,samples,errors),daemon=True)
    watcher.start()
    stage='construction'
    try:
        torch.set_num_threads(4)
        with torch.jit.optimized_execution(False), backend_scope(COLLISION), batch_runtime(OBJECTIVE,worlds,factory()) as runtime:
            learner=BackendSupportPenPPO(runtime,pilot_settings(OBJECTIVE)[1],'RESIDUAL')
            stage='calibration'
            learner.calibrate(8)
            for index in range(5):
                stage=f'collect_{index}'
                torch.cuda.synchronize()
                start=time.perf_counter()
                m=learner.collect(64)
                torch.cuda.synchronize()
                end=time.perf_counter()
                health=learner.consistency()
                write_json(out/f'health_{index}.json',health)
                if not all(health['checks'].values()):
                    raise ValueError([k for k,v in health['checks'].items() if not v])
                valid=((m['contact_geometry_max']<=runtime.objective_constants['geometry_limits']).all(-1)
                       & ~m['engine_overflow'] & m['no_applied_force'] & (m['contact_errors']==0)
                       & (m['contact_nonhand_max_sum'][...,0]<=.1))
                rows.append(dict(start=start,end=end,total=64*worlds,valid=int(valid.sum())))
            if learner.optimizer_steps:
                raise ValueError('Unexpected optimizer step')
        result=summarize(rows,samples)
        write_json(out/'report.json',dict(worlds=worlds,**result,rows=rows))
    except BaseException as e:
        write_json(out/'failure.json',dict(stage=stage,error=repr(e),rows=rows))
        raise
    finally:
        stop.set()
        watcher.join(6)
        write_json(out/'telemetry.json',dict(samples=samples,errors=errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('worlds',type=int)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    main(args.worlds,args.out)
