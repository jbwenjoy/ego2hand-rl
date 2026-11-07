"""Run the fixed fresh phone training budget; checkpoint handoff is Root-owned."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time

def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
ap=argparse.ArgumentParser();ap.add_argument('--protocol',type=Path,required=True);a=ap.parse_args()
p=json.loads(a.protocol.read_text(encoding='utf8'));root=Path(__file__).resolve().parent
assert p['CPU_run_authorized'] is True and p['GPU_run_authorized'] is False
assert p['recipe']['updates']==5000 and p['recipe']['background_continue_to_final'] is True
assert not Path(p['output']).exists()
runner=p['source_pins']['CLI'];assert pin(runner['path'])==runner
folder=root/'process_training_run1';folder.mkdir()
cmd=[sys.executable,'-X','utf8',runner['path'],'--protocol',str(a.protocol.resolve()),'--run']
t=time.perf_counter();start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (folder/'stdout.log').open('x',encoding='utf8') as o,(folder/'stderr.log').open('x',encoding='utf8') as e:
    child=subprocess.Popen(cmd,cwd=root,stdout=o,stderr=e,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    with (folder/'started.json').open('x',encoding='utf8') as f:
        json.dump(dict(pid=child.pid,started_UTC=start,command=cmd,protocol=pin(a.protocol),
            liveness_scope='Launch receipt only; use live exec handle or actual terminal, never this file alone.'),f,indent=2)
    print(json.dumps(dict(status='TRAINING_STARTED_ONCE',pid=child.pid,started_UTC=start)),flush=True)
    code=child.wait()
j=dict(schema='phone_world_model_training_process_terminal:1',pid=child.pid,returncode=code,
    started_UTC=start,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-t,
    command=cmd,source=pin(__file__),runner=runner,protocol=pin(a.protocol),stdout=pin(folder/'stdout.log'),
    stderr=pin(folder/'stderr.log'),automatic_retry=False)
with (folder/'terminal.json').open('x',encoding='utf8') as f:json.dump(j,f,indent=2)
print(json.dumps(j),flush=True);sys.exit(code)
