"""Root reviewed actual two-source four-control/two-dt native run; no training."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_CPUFalse_v2.json';c=json.loads(base.read_text(encoding='utf8'))
assert c['CPU_run_authorized'] is False and c['budget']['maximum_cases']==16
for r in [*c['source_pins'].values(),*c['inputs'].values()]:assert pin(r['path'])==r
for source in c['preload_sources']:
    for key in ['preload_report','preload_protocol','preload_saved_review']:assert pin(source[key]['path'])==source[key]
protocol=H/'protocol_CPUTrue_run1.json';process=H/'process_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['CPU_run_authorized']=True;c['root_review']=dict(source_protocol=pin(base),launcher=pin(__file__),
    scope='All two-source/four measured controls/bothdt cases preserved. Use raw true-u from native con.pos preload probes/fresh final, no material-A or additional prepareFW. Direct frozen2s evaluator/gates, old2pad parallel. Initial all3Fn does not certify wrench balance/holding. No object aid or new learning.')
write(protocol,c);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['source_pins']['runner']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,output=c['output'])),flush=True);rc=child.wait()
t=dict(schema='loaded_tripod_two_source_dynamic_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,
    ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,
    source=pin(c['source_pins']['runner']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',t);print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))))
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
