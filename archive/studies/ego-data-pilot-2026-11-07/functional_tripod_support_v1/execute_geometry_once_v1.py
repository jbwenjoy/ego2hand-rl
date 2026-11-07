"""Root-reviewed finite geometry execution with an actual terminal handle; no retry."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False)
a=argparse.ArgumentParser();a.add_argument('--reviewed-source-protocol',type=Path,required=True);args=a.parse_args()
c=json.loads(args.reviewed_source_protocol.read_text(encoding='utf8'))
assert c['CPU_static_run_authorized'] is False and c['integration_or_training_authorized'] is False
assert c['designated_geom_ids']==[15,28,41] and c['budget']['global_wall_s']==360
for r in [*c['inputs'].values(),*c['source_pins'].values()]:assert pin(r['path'])==r
process=H/'process_geometry_run1';protocol=H/'protocol_CPUTrue_run1.json'
assert not process.exists() and not protocol.exists() and not Path(c['output']).exists()
c['CPU_static_run_authorized']=True
c['root_review']=dict(reviewed_source_protocol=pin(args.reviewed_source_protocol),
    reason='Actual middle q8:12 freedom, true three-tip convex geometry/full214 constraints, named material/phone pose concessions. No point LP, zero-qacc, object aids or new learning. Preserve all4 entry failures; any geometry pass still requires native controls/dynamics.',
    launcher=pin(__file__))
write(protocol,c);process.mkdir();solver=c['source_pins']['solver']['path']
metadata_cmd=[sys.executable,'-X','utf8',solver,'--protocol',str(protocol)]
meta=subprocess.run(metadata_cmd,capture_output=True,text=True,encoding='utf8')
write(process/'metadata_preflight.json',dict(returncode=meta.returncode,command=metadata_cmd,stdout=meta.stdout,stderr=meta.stderr))
if meta.returncode:print(meta.stderr);raise SystemExit(meta.returncode)
command=metadata_cmd+['--run'];stdout=process/'stdout.log';stderr=process/'stderr.log'
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se)
    print(json.dumps(dict(pid=child.pid,command=command,process=str(process),output=c['output'])),flush=True)
    rc=child.wait()
terminal=dict(schema='bounded_geometry_process_terminal:1',pid=child.pid,returncode=rc,command=command,
    started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,
    source=pin(solver),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',terminal);print(json.dumps(dict(terminal=pin(process/'terminal.json'),returncode=rc)))
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
