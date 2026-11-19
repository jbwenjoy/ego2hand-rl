"""Fresh-process collector benchmark with the same 64-control PPO rollout."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from supermimic_research.artifacts import file_pin, verify_file_pin
from supermimic_research.learning.backend_pen_ppo import BackendSupportPenPPO
from supermimic_research.learning.pen_support_ppo import pilot_settings
from supermimic_research.learning.gpu_monitor import GPUMonitor
from supermimic_research.runtime.explicit_collision_backend import backend_scope
from supermimic_research.runtime.pen_scaled_runtime import batch_runtime
from supermimic_research.strict_json import write_json

OBJECTIVE = ROOT / 'configs/pen_long_control_training_v1.toml'
COLLISION = ROOT / 'configs/collision_backend_face_normal_v1.json'


def factory():
    verified = json.loads((ROOT / 'runs/pen_guarded_w32_v2/report.json').read_text())
    if verified['actual_native_integrations'] != 1536 or not all(verified['checks'].values()):
        raise ValueError('Missing W32 qualification')
    return verified['contract']['identity']['factory']


def main(worlds, out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    monitor = GPUMonitor()
    monitor.start()
    rows = []
    try:
        with torch.jit.optimized_execution(False), backend_scope(COLLISION), batch_runtime(OBJECTIVE, worlds, factory()) as runtime:
            learner = BackendSupportPenPPO(runtime, pilot_settings(OBJECTIVE)[1], 'RESIDUAL')
            learner.calibrate(8)
            for i in range(5):
                torch.cuda.synchronize()
                start = time.perf_counter()
                learner.collect(64)
                torch.cuda.synchronize()
                end = time.perf_counter()
                health = learner.consistency()
                write_json(out / f'health_{i}.json', health)
                if not all(health['checks'].values()):
                    raise ValueError([k for k,v in health['checks'].items() if not v])
                phase = learner.memory['phase_used']
                rows.append(dict(warmup=i == 0, start=start, end=end, wall_s=end-start,
                                 transitions=worlds*64, native=worlds*512, second_half=int((phase>=185).sum())))
            assert learner.optimizer_steps == 0
    except BaseException as err:
        write_json(out / 'failure.json', dict(error=repr(err), rows=rows))
        raise
    finally:
        monitor.stop()
        write_json(out / 'gpu.json', dict(samples=monitor.samples, errors=monitor.errors))
    measured=rows[1:]
    readings=np.array([s['values'] for s in monitor.samples if any(r['start']<=s['start']<=r['end'] for r in measured)])
    seconds=sum(r['wall_s'] for r in measured)
    report=dict(worlds=worlds, rows=rows, native_per_s=4*worlds*512/seconds, transitions_per_s=4*worlds*64/seconds,
                gpu_median_pct=float(np.median(readings[:,0])), gpu_p95_pct=float(np.percentile(readings[:,0],95)),
                vram_peak_mib=float(readings[:,1].max()), gpu_samples=len(readings), run_peak_mib=max(s['values'][1] for s in monitor.samples),
                torch_peak_mib=torch.cuda.max_memory_allocated()/2**20, compile_excluded=True, source=file_pin(__file__))
    write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('worlds',type=int)
    p.add_argument('out',type=Path)
    args=p.parse_args()
    main(args.worlds,args.out)

