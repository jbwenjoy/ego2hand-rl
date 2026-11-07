"""One reviewed native-contact preload preparation. Zero integration or learning."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_source1mm_CPUFalse_v2.json';c=json.loads(base.read_text(encoding='utf8'))
assert c['native_forward_authorized'] is False and c['integration_training_GPU_authorized'] is False
for r in [*c['source_pins'].values(),*c['inputs'].values()]:assert pin(r['path'])==r
protocol=H/'protocol_source1mm_CPUTrue_run1.json';process=H/'process_source1mm_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['native_forward_authorized']=True;c['root_review']=dict(source_protocol=pin(base),launcher=pin(__file__),
    scope='At most40freshFW for exact compressed geometry; failed1mm retained. Actual native con.pos/frame/J and true PDbox, separate force/moment units. Three-normal proposals and finite static-wrench control responses are not holding or learned physics gradients. No object aid/steps/training.')
write(protocol,c);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['source_pins']['runner']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,output=c['output'])),flush=True);rc=child.wait()
t=dict(schema='native_tripod_preload_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,
    ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,
    source=pin(c['source_pins']['runner']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',t);print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))))
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
