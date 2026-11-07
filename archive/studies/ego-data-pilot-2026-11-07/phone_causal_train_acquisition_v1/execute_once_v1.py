"""Reviewed one all72-baselines-first acquisition; no learned parameter updates."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_cpu_false_v1.json';c=json.loads(base.read_text(encoding='utf8'))
assert c['CPU_run_authorized'] is False and c['training_authorized'] is False and c['fresh_final_acquisition_authorized'] is False
assert c['budget']['maximum_native_steps']==282432
for r in [*c['sources'].values(),*c['inputs'].values()]:assert pin(r['path'])==r
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['CPU_run_authorized']=True;c['root_review']=dict(source_protocol=pin(base),launcher=pin(__file__),
    scope='Known originalTRAIN states. All72 byte-matched baseline barrier before2870new sides,10unavailable kept. Raw300 restore/originalepisode safety origin, actualdu64/du32, all failed/partial results preserved. Familyresponse fit/check only, no VAL/freshfinal acquisition or NN training.')
write(protocol,c);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['sources']['runner']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,output=c['output'],process=str(process))),flush=True);rc=child.wait()
t=dict(schema='causal_TRAIN_action_acquisition_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,
    ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,
    source=pin(c['sources']['runner']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',t);print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))))
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
