"""One reviewed CPU-only dynamics run, with a real subprocess terminal receipt."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True)
ap.add_argument('--process-dir',type=Path,required=True);a=ap.parse_args()
j=json.loads(a.protocol.read_text(encoding='utf8'))
assert j['schema']=='hand2_source_mapping_constant_u_diagnostic_protocol:11'
assert j['CPU_run_authorized'] is True and j['training_or_GPU_authorized'] is False
assert not Path(j['output']).exists(),'No retry or overwrite'
runner=j['source_pins']['runner'];assert pin(runner['path'])==runner
r=a.process_dir.resolve();r.mkdir()
cmd=[sys.executable,'-X','utf8',runner['path'],'--protocol',str(a.protocol.resolve()),'--run']
t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/'stdout.log').open('x',encoding='utf8') as o,(r/'stderr.log').open('x',encoding='utf8') as e:
    child=subprocess.Popen(cmd,cwd=Path(runner['path']).parent.parent,stdout=o,stderr=e,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps(dict(status='STARTED_ONCE',pid=child.pid,started_UTC=start)),flush=True)
    code=child.wait()
rec=dict(schema='source_mapping_dynamic_process_terminal:11',pid=child.pid,returncode=code,
    command=cmd,started_UTC=start,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    elapsed_s=time.perf_counter()-t,source=pin(__file__),runner=runner,protocol=pin(a.protocol),
    stdout=pin(r/'stdout.log'),stderr=pin(r/'stderr.log'),automatic_retry=False)
with (r/'terminal.json').open('x',encoding='utf8') as f:json.dump(rec,f,indent=2)
print(json.dumps(rec),flush=True);sys.exit(code)
