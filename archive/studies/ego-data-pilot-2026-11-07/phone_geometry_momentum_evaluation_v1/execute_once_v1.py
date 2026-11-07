"""Root's fixed final5000 evaluation, two inference forwards, no updates."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_cpu_false_v1.json';c=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='63743218dbea1a0e091e4a3aad04d75c17b6fc31f3c57446456d95fa563a3fbb'
assert c['CPU_run_authorized'] is False and c['GPU_run_authorized'] is False
assert c['sources']['evaluator']['sha256']=='32f565f0d120272114692505d5ada9455fc2f857b9225ee6b1a3e2b82ca8a302'
for r in [*c['inputs'].values(),*c['sources'].values()]:assert pin(r['path'])==r
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['CPU_run_authorized']=True;c['root_review']=dict(source_protocol=pin(base),launcher=pin(__file__),scope='Fixed5000, CHECK1472 endpoints +VAL114, exactly2NN forwards/1586rows, no fit/AD/native/GPU/TEST. Preserve all4 unavailable CHECK and all20axes/native-zero response strata, original seen states not unseen ego.')
write(protocol,c);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['sources']['evaluator']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,process=str(process),output=c['output'])),flush=True);rc=child.wait()
write(process/'terminal.json',dict(schema='phone_fixed5000_CHECK_evaluation_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,source=pin(c['sources']['evaluator']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
