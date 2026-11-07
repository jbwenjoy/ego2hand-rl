"""Root's two zero-action callback/native-loop qualification cases."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'zero_CPUFalse_v1.json';p=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='9af38befd061379d1c147c0e39524524805c73a469e6ef6637aa6d9a9826c18a'
assert p['CPU_run_authorized'] is False and p['learned_policy_or_NN_training_authorized'] is False and p['budget']['NN_calls']==0
for v in [p['source'],*p['sources'].values(),*p['inputs'].values()]:assert pin(v['path'])==v
protocol=H/'zero_CPUTrue_run1.json';process=H/'zero_process_run1'
assert not any(v.exists() for v in [protocol,process,Path(p['output'])])
p['CPU_run_authorized']=True;p['root_review']=dict(parent_protocol=pin(base),launcher=pin(__file__),scope='Qualify new50Hz controller hook on bothdt against saved original constant-GN traces. No learned policy or gradient/holding benefit. Native original gates unchanged, no retry.')
write(protocol,p);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',p['source']['path'],'--protocol',str(protocol),'--run'];start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,process=str(process))),flush=True);rc=child.wait()
write(process/'terminal.json',dict(schema='scheduled_controller_zero_native_qualification_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,protocol=pin(protocol),source=pin(p['source']['path']),launcher=pin(__file__),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
