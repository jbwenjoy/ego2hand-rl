"""Run one pinned finite-action acquisition and record actual process terminal."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);a=ap.parse_args()
j=json.loads(a.protocol.read_text(encoding='utf8'));h=Path(__file__).resolve().parent
assert j['CPU_run_authorized'] and not j['training_or_parameter_update_authorized'] and not Path(j['output']).exists()
runner=j['source_pins']['runner'];assert pin(runner['path'])==runner
folder=h/'process_native_run1';folder.mkdir()
cmd=[sys.executable,'-X','utf8',runner['path'],'--protocol',str(a.protocol.resolve()),'--run']
t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (folder/'stdout.log').open('x',encoding='utf8') as o,(folder/'stderr.log').open('x',encoding='utf8') as e:
    child=subprocess.Popen(cmd,cwd=h,stdout=o,stderr=e,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    print(json.dumps(dict(status='STARTED_ONCE',pid=child.pid,started_UTC=start)),flush=True);code=child.wait()
r=dict(schema='phone_native_action_branch_process_terminal:1',pid=child.pid,returncode=code,command=cmd,
    started_UTC=start,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-t,
    source=pin(__file__),runner=runner,protocol=pin(a.protocol),stdout=pin(folder/'stdout.log'),stderr=pin(folder/'stderr.log'),automatic_retry=False)
with (folder/'terminal.json').open('x',encoding='utf8') as f:json.dump(r,f,indent=2)
print(json.dumps(r),flush=True);sys.exit(code)
