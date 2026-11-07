"""Execute one reviewed pinned CPU assembly protocol and preserve real terminal status."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys, time

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True)
ap.add_argument('--process-dir',type=Path,required=True);args=ap.parse_args()
protocol=json.loads(args.protocol.read_text(encoding='utf8'))
assert protocol['CPU_run_authorized'] is True
runner=protocol['source_pins']['runner'];assert pin(runner['path'])==runner
assert not Path(protocol['output_path']).exists(),'No restart or overwrite'
r=args.process_dir.resolve();r.mkdir()
cmd=[sys.executable,'-X','utf8',runner['path'],'--protocol',str(args.protocol.resolve()),'--run']
t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/'stdout.log').open('x',encoding='utf8') as o,(r/'stderr.log').open('x',encoding='utf8') as e:
    child=subprocess.Popen(cmd,cwd=Path(runner['path']).parent.parent,stdout=o,stderr=e,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps(dict(status='STARTED_ONCE',pid=child.pid,started_UTC=start)),flush=True)
    code=child.wait()
record=dict(schema='source_region_assembly_process_terminal:11',pid=child.pid,returncode=code,
    command=cmd,started_UTC=start,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    elapsed_s=time.perf_counter()-t,source=pin(__file__),runner=runner,protocol=pin(args.protocol),
    stdout=pin(r/'stdout.log'),stderr=pin(r/'stderr.log'),automatic_retry=False)
with (r/'terminal.json').open('x',encoding='utf8') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True);sys.exit(code)
