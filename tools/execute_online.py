"""Durable process receipt and GPU samples for a configured online run.

The trainer enforces authorization and first-health stopping. This supervisor
never retries, changes its configuration, or resumes a completed child.
"""
from pathlib import Path
import argparse,datetime,json,statistics,subprocess,sys,threading,time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from supermimic_research.learning import online_contract as C
from supermimic_research.artifacts import file_pin


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=args.config.resolve(strict=True);_,c,_=C.configuration(config)
    out=args.output.resolve();C.need(out.is_relative_to(ROOT/'runs') and not out.exists(),'Fresh process output under unified runs')
    C.need(c['training']['authorized'] is True and c['online']['native_authorized'] is True,'Actual configured run')
    out.mkdir(parents=True);stdout=out/'stdout.log';stderr=out/'stderr.log'
    command=[sys.executable,'-X','utf8',str(ROOT/'scripts/research.py'),'online','--config',str(config),'--mode','run']
    samples=[];stop=threading.Event()
    def sampler():
        while not stop.is_set():
            try:
                value=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=8)
                if value.returncode==0:
                    samples.append(dict(unix_s=time.time(),**dict(zip(['GPU_util_pct','memory_used_MiB','memory_total_MiB','power_W'],map(float,value.stdout.strip().split(','))))))
            except (OSError,ValueError,subprocess.TimeoutExpired):pass
            stop.wait(.5)
    start=datetime.datetime.now(datetime.timezone.utc).isoformat();tick=time.perf_counter()
    with stdout.open('xb') as so,stderr.open('xb') as se:
        child=subprocess.Popen(command,stdout=so,stderr=se,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        C.write(out/'process.json',dict(pid=child.pid,command=command,configuration=file_pin(config),started_UTC=start))
        thread=threading.Thread(target=sampler,daemon=True);thread.start();code=child.wait();stop.set();thread.join(timeout=10)
    C.write(out/'GPU_samples.json',samples)
    run_out=(config.parent/c['training']['output']).resolve();updates=run_out/'updates.jsonl'
    rows=[json.loads(row) for row in updates.read_bytes().splitlines()] if updates.exists() else []
    steady=[s for s in samples if any(r['begin_unix_s']<=s['unix_s']<=r['end_unix_s'] for r in rows)]
    terminal=dict(schema='supermimic_research.online_process_terminal:1',pid=child.pid,returncode=code,command=command,
        configuration=file_pin(config),launcher=file_pin(__file__),sources=C.source_pins(),stdout=file_pin(stdout),stderr=file_pin(stderr),
        GPU_samples=file_pin(out/'GPU_samples.json'),started_UTC=start,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-tick,
        GPU_update_scope='Exact actual update wall windows; setup/CPU collection/cache/health excluded; includes first-update warm-up',
        GPU_update_samples=len(steady),GPU_update_util_median=statistics.median(s['GPU_util_pct'] for s in steady) if steady else None,
        GPU_update_memory_max_MiB=max((s['memory_used_MiB'] for s in steady),default=None),automatic_retry=False)
    C.write(out/'terminal.json',terminal)
    print(json.dumps(dict(terminal=file_pin(out/'terminal.json'),returncode=code,elapsed_s=terminal['elapsed_s'],GPU_update_samples=len(steady))),flush=True)
    return code


if __name__=='__main__':raise SystemExit(main())
