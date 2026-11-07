"""Root-reviewed actual three-route/two-dt native evaluation; never new training."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time
H=Path(__file__).resolve().parent
def pin(p):
    p=Path(p).resolve(strict=True);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False)
a=argparse.ArgumentParser();a.add_argument('--reviewed-protocol',type=Path,required=True);args=a.parse_args()
source=json.loads(args.reviewed_protocol.read_text(encoding='utf8'))
assert source['CPU_run_authorized'] is False and source['training_or_GPU_authorized'] is False
assert source['designated_geom_ids']==[15,28,41] and source['control_routes']==['no_preload_ctrl_equal_q','normal_1N_each_pad','normal_3N_each_pad']
for r in [*source['source_pins'].values(),*source['inputs'].values()]:assert pin(r['path'])==r
assert source['source_pins']['runner']['path']==str(H/'run_functional_tripod_dynamics_v3.py')
protocol=H/'protocol_projection_CPUTrue_run1.json';process=H/'process_projection_run1'
assert not protocol.exists() and not process.exists() and not Path(source['output']).exists()
source['CPU_run_authorized']=True
source['root_review']=dict(source_protocol=pin(args.reviewed_protocol),launcher=pin(__file__),
    qualification='Two of four adaptive geometric projections have fresh complete three-pad geometry. This is not native contact/control/holding evidence yet. Preserve all three fixed controls, unavailable routes, failed/partial outcomes, bothdt and old2pad gates. No retuning/extra force/training.')
write(protocol,source);process.mkdir();stdout=process/'stdout.log';stderr=process/'stderr.log'
command=[sys.executable,'-X','utf8',source['source_pins']['runner']['path'],'--protocol',str(protocol),'--run']
start=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
with stdout.open('x',encoding='utf8') as so,stderr.open('x',encoding='utf8') as se:
    child=subprocess.Popen(command,stdout=so,stderr=se)
    print(json.dumps(dict(pid=child.pid,output=source['output'],process=str(process))),flush=True);rc=child.wait()
terminal=dict(schema='functional_tripod_projection_native_terminal:1',pid=child.pid,returncode=rc,command=command,
    started_UTC=utc,ended_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_s=time.perf_counter()-start,
    source=pin(source['source_pins']['runner']['path']),launcher=pin(__file__),protocol=pin(protocol),stdout=pin(stdout),stderr=pin(stderr),automatic_retry=False)
write(process/'terminal.json',terminal);print(json.dumps(dict(terminal=pin(process/'terminal.json'),returncode=rc)))
if rc:print(stderr.read_text(encoding='utf8'))
raise SystemExit(rc)
