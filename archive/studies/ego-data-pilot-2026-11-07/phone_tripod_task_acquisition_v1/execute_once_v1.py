"""Root collects the frozen tripod train/monitor families once, no NN."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_CPUFalse_v1.json';p=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='c172465221cdac1814335c7a7e6764aa0600a80182ea6842826d21f8f4ddccef'
assert p['CPU_run_authorized'] is False
for v in [*p['inputs'].values(),*p['sources'].values()]:assert pin(v['path'])==v
assert p['sources']['collector']['sha256']=='7c9f9bcfedc56d197ecf8d637ca51ef242ca22751e872b7b8bf290bcddf5b597'
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(v.exists() for v in [protocol,process,Path(p['output'])])
p['CPU_run_authorized']=True;p['root_review']=dict(parent_protocol=pin(base),launcher=pin(__file__),scope='16fixed controlfamilies,32episodes bothdt,at most61440steps/640intervals. Newtripod TRAIN0..11 and monitor12..15, same known loaded state. Rawfsync and allfailure/partial/alias preserved. No learning or policy/task benefit qualification; separate POSTH terminal vs collectiontruncation and old2s native gate.')
write(protocol,p);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',p['sources']['collector']['path'],'--protocol',str(protocol),'--run'];start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,process=str(process),output=p['output'])),flush=True);rc=child.wait()
write(process/'terminal.json',dict(schema='phone_tripod_task_native_acquisition_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,protocol=pin(protocol),source=pin(p['sources']['collector']['path']),launcher=pin(__file__),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
