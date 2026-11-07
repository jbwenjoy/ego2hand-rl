"""Root's saved-trace label export, no native/FK/NN calls."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_CPUFalse_v1.json';p=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='f8e2f36216f317b2505c4b911c4e7ecb6ed6aa8ead0cdd4e9dd0bb6e03ba6310'
assert p['saved_only_run_authorized'] is False and p['native_NN_FK_authorized'] is False
for v in [*p['inputs'].values(),*p['sources'].values()]:assert pin(v['path'])==v
protocol=H/'protocol_savedTrue_run1.json';process=H/'process_run1'
assert not any(v.exists() for v in [protocol,process,Path(p['output'])])
p['saved_only_run_authorized']=True;p['root_review']=dict(parent_protocol=pin(base),launcher=pin(__file__),scope='All205 actually captured tripod TRAIN/MONITOR rows. Reuse pinned sourceBODY movingCOM signed impulse decoder; completeH futurelabels and partialprefix roles separate. Zero new native/NN/FK, no fitting or model selection.')
write(protocol,p);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',p['sources']['exporter']['path'],'--protocol',str(protocol),'--run'];start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se);print(json.dumps(dict(pid=child.pid,process=str(process))),flush=True);rc=child.wait()
write(process/'terminal.json',dict(schema='phone_tripod_saved_labels_export_terminal:1',pid=child.pid,returncode=rc,command=command,started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,protocol=pin(protocol),source=pin(p['sources']['exporter']['path']),launcher=pin(__file__),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False))
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
