"""Root's two actual reset+20ms checks; no learned updates or retries."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_CPUFalse_v1.json';p=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='7bc3b594ec35c3d6290fcae02f6680da767994574c7b5146281d2ae6de3c3272'
assert p['CPU_run_authorized'] is False and p['NN_training_GPU_authorized'] is False and p['budget']['maximum_native_steps']==192
for v in [p['source'],*p['sources'].values(),*p['inputs'].values()]:assert pin(v['path'])==v
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(v.exists() for v in [protocol,process,Path(p['output'])])
p['CPU_run_authorized']=True;p['root_review']=dict(parent_protocol=pin(base),launcher=pin(__file__),scope='Two physical reset+single20ms interface checks,192steps maximum, raw300 state and PREcontacts persisted. Zero Actor/WM/GPU/learning, not2s holding.')
write(protocol,p);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',p['source']['path'],'--protocol',str(protocol),'--run'];t=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,process=str(process))),flush=True);rc=child.wait()
write(process/'terminal.json',dict(schema='phone_tripod_actual_bridge_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-t,protocol=pin(protocol),source=pin(p['source']['path']),launcher=pin(__file__),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
