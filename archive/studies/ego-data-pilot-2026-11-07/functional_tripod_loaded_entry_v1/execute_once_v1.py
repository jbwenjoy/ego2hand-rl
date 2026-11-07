"""Root-reviewed one two-state compression search; no retries or integration."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_CPUFalse_v1.json';c=json.loads(base.read_text(encoding='utf8'))
assert c['CPU_static_run_authorized'] is False and c['neural_training_or_integration_authorized'] is False
for r in [*c['sources'].values(),*c['inputs'].values()]:assert pin(r['path'])==r
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['CPU_static_run_authorized']=True;c['root_review']=dict(source_protocol=pin(base),launcher=pin(__file__),
    scope='Same two qualified geometries. Negative compression target separate from unchanged full214 hard geometry. No native load/holding qualification. Keep all failures, originalphone/rest8, no model/material/solver changes.')
write(protocol,c);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['sources']['runner']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,output=c['output'])),flush=True);rc=child.wait()
t=dict(schema='native_loaded_entry_geometry_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,
    ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,
    source=pin(c['sources']['runner']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',t);print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))))
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
