"""Collect the predeclared new12 families once, without learned updates."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_CPUFalse_v2.json';c=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='282c4bbee0848cc43772e141d8fcb90a36da276545f8ba27439c75c77b9f51fd'
assert c['CPU_run_authorized'] is False and c['new_training_authorized'] is False and c['GPU_run_authorized'] is False
assert c['maximum_steps']==23040 and c['maximum_rows']==240 and c['maximum_episodes']==24
for r in [*c['source_pins'].values(),*c['inputs'].values()]:assert pin(r['path'])==r
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['CPU_run_authorized']=True;c['root_review']=dict(source_protocol=pin(base),launcher=pin(__file__),scope='Predeclared family1000..1011, bothdt, at most23040 native steps. Same two known poses/model and frozen Reader/trace path; no training/model selection/ego or holding qualification. All aborts/partial/unknown and aliases retained; v1 API-global draft preserved, v2 binds actual protocol argument.')
write(protocol,c);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['source_pins']['collector']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,output=c['output'],process=str(process))),flush=True);rc=child.wait()
write(process/'terminal.json',dict(schema='phone_fresh12_native_acquisition_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,source=pin(c['source_pins']['collector']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
