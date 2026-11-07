"""Root's single final5000 fresh-family inference batch, no updates/native."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_CPUFalse_v1.json';c=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='f73c79c2be5b3a2671ca10b6b3ae2ebefb1a34c2f8e94e1098f86374a49f5314'
assert c['CPU_run_authorized'] is False and c['GPU_run_authorized'] is False
assert c['sources']['evaluator']['sha256']=='eb25256e0383cfc133a486756c1da89e8e976c7c17d681ee28a48016ce49e9b9'
for r in [*c['sources'].values(),*c['inputs'].values()]:assert pin(r['path'])==r
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['CPU_run_authorized']=True;c['root_review']=dict(source_protocol=pin(base),launcher=pin(__file__),scope='One frozen5000 model/one B228 inference/zero updates. Keep240planned/12partial unknown, no target leakage/normalizer fit or checkpoint selection. Same known poses, not independent ego/policy benefit.')
write(protocol,c);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['sources']['evaluator']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,process=str(process))),flush=True);rc=child.wait()
write(process/'terminal.json',dict(schema='phone_fresh12_fixed5000_evaluation_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,source=pin(c['sources']['evaluator']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
