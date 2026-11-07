"""Root's single CPU-only run; stop at first genuine healthy update250."""
from pathlib import Path
import datetime, hashlib, json, subprocess, sys, time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True); b=p.read_bytes()
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f: json.dump(v,f,indent=2,allow_nan=False)
base=H/'protocol_cpu_false_v1.json'; c=json.loads(base.read_text(encoding='utf8'))
assert pin(base)['sha256']=='37a120088ffead520d3d43325624a7778d59271ab763d1bdb11ba54b69928785'
assert c['CPU_run_authorized'] is False and c['GPU_run_authorized'] is False
assert c['sources']['trainer']['sha256']=='224a0c9f47e196c626e862aa52d46402a454a6d16f0b497d3720ae74e25ca539'
assert c['recipe']['updates']==5000 and c['recipe']['stop_after_first_healthy_checkpoint'] is True
assert c['resume_checkpoint'] is None and c['resume_health'] is None
for r in [*c['sources'].values(),*c['inputs'].values(),c['recipe_pin']]: assert pin(r['path'])==r
review=H/'independent_source_review_v1/report.json'
assert pin(review)['sha256']=='83f8d9a8fbb821e4eb40e92426d69e9cec3511e6d1d0ccf6cb389118ed8eba75'
protocol=H/'protocol_CPUTrue_run1.json'; process=H/'process_run1'
assert not any(p.exists() for p in [protocol,process,Path(c['output'])])
c['CPU_run_authorized']=True
c['root_review']=dict(source_protocol=pin(base),independent_SOURCE_review=pin(review),launcher=pin(__file__),
    scope='One fresh CPU-only mixed-F32-NN/F64-geometry-and-physics run, seed1731. FIT2118 absolute/1434 responses, no CHECK/TEST target-array read; fixed VAL114 state monitor only. Budget5000 resumable, worker stops after healthy250 and ETA. No Actor, CHORD policy, native simulation, GPU, or ego performance qualification.')
write(protocol,c); process.mkdir(); stdout=process/'stdout.log'; stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',c['sources']['trainer']['path'],'--protocol',str(protocol),'--run']
started=time.perf_counter(); utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se)
    print(json.dumps(dict(pid=child.pid,output=c['output'],process=str(process))),flush=True)
    rc=child.wait()
terminal=dict(schema='phone_geometry_momentum_training_terminal:1',pid=child.pid,returncode=rc,command=command,
    started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-started,
    source=pin(c['sources']['trainer']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',terminal)
print(json.dumps(dict(returncode=rc,terminal=pin(process/'terminal.json'))),flush=True)
if rc: print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
